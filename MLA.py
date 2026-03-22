import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim_kv, latent_dim_q, rope_dim):
        """
        参数:
        d_model: 模型隐藏层维度 (例如 4096)
        num_heads: 注意力头数 (例如 32)
        latent_dim_kv: KV 联合压缩后的潜向量维度 (d_c) (例如 512，远小于 num_heads * head_dim)
        latent_dim_q: Query 压缩后的潜向量维度 (d_c') (例如 1536)
        rope_dim: 专门用于旋转位置编码的解耦维度 (d_r) (例如 64)
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_dim = rope_dim
        
        # 1. KV 联合压缩路径 (Low-Rank KV Compression)
        # 降维：生成潜向量 c_kv (这就是推理时唯一需要 Cache 的东西！)
        self.W_down_kv = nn.Linear(d_model, latent_dim_kv, bias=False)
        self.kv_norm = nn.LayerNorm(latent_dim_kv)
        
        # 升维：从潜向量恢复出多头的 Key 和 Value 的内容部分 (Content)
        self.W_up_k = nn.Linear(latent_dim_kv, num_heads * self.head_dim, bias=False)
        self.W_up_v = nn.Linear(latent_dim_kv, num_heads * self.head_dim, bias=False)
        
        # 2. Query 压缩路径
        self.W_down_q = nn.Linear(d_model, latent_dim_q, bias=False)
        self.q_norm = nn.LayerNorm(latent_dim_q)
        self.W_up_q = nn.Linear(latent_dim_q, num_heads * self.head_dim, bias=False)
        
        # 3. 解耦的 RoPE 路径 (Decoupled RoPE)
        # 为什么解耦？因为 RoPE 是位置相关的旋转，不能被压缩进内容潜向量里
        self.W_rope_q = nn.Linear(d_model, num_heads * rope_dim, bias=False)
        self.W_rope_k = nn.Linear(d_model, rope_dim, bias=False) # 共享 Key 的 RoPE
        
        # 输出映射
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.size()
        
        # ==========================================
        # 阶段一：生成压缩潜向量 (Latent Vectors)
        # ==========================================
        # 1. 生成 KV 潜向量 [B, T, latent_dim_kv]
        c_kv = self.kv_norm(self.W_down_kv(x)) 
        
        # 2. 生成 Query 潜向量 [B, T, latent_dim_q]
        c_q = self.q_norm(self.W_down_q(x))

        # ==========================================
        # 阶段二：升维与解耦计算 (训练时的完整前向传播)
        # ==========================================
        # 1. 恢复 Content (内容部分)
        # [B, T, num_heads, head_dim]
        q_c = self.W_up_q(c_q).view(B, T, self.num_heads, self.head_dim)
        k_c = self.W_up_k(c_kv).view(B, T, self.num_heads, self.head_dim)
        v_c = self.W_up_v(c_kv).view(B, T, self.num_heads, self.head_dim)
        
        # 2. 计算 RoPE 部分 (单独的分支)
        # [B, T, num_heads, rope_dim]
        q_r = self.W_rope_q(x).view(B, T, self.num_heads, self.rope_dim)
        # 注意：K 的 RoPE 头是所有 Q 头共享的 (类似 MQA)
        # [B, T, 1, rope_dim] -> [B, T, num_heads, rope_dim] (广播)
        k_r = self.W_rope_k(x).view(B, T, 1, self.rope_dim).expand(-1, -1, self.num_heads, -1)
        
        # (假设这里调用了 apply_rotary_pos_emb 函数对 q_r 和 k_r 进行旋转)
        q_r_rot, k_r_rot = apply_rotary_pos_emb(q_r, k_r, rope_cos, rope_sin)
        
        # 3. 拼接 Content 和 RoPE
        # [B, T, num_heads, head_dim + rope_dim]
        q = torch.cat([q_c, q_r_rot], dim=-1).transpose(1, 2)
        k = torch.cat([k_c, k_r_rot], dim=-1).transpose(1, 2)
        v = v_c.transpose(1, 2) # Value 不需要位置信息，保持 [B, num_heads, T, head_dim]

        # ==========================================
        # 阶段三：标准 Attention 计算
        # ==========================================
        # q, k: [B, num_heads, T, head_dim + rope_dim]
        # v: [B, num_heads, T, head_dim]
        scale = math.sqrt(self.head_dim + self.rope_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)
        
        # [B, num_heads, T, head_dim]
        out = torch.matmul(attn, v)
        
        # 还原形状并输出
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.W_out(out)

def rotate_half(x):
    """
    辅助函数：将张量的最后一个维度（即 rope_dim）切分成两半，并进行旋转。
    这是模拟复平面上向量旋转的关键步骤。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将旋转位置编码 (RoPE) 应用于解耦后的 Query 和 Key。
    
    在 MLA 的上下文中：
    q 形状: [B, T, num_heads, rope_dim]
    k 形状: [B, T, num_heads, rope_dim]
    """
    # 假设传入的 cos 和 sin 是预先计算好的，形状为 [T, rope_dim]
    # 我们需要在 Batch (第0维) 和 Heads (第2维) 上增加维度，以便进行广播 (Broadcasting)相乘
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2) # 变为 [1, T, 1, rope_dim]
        sin = sin.unsqueeze(0).unsqueeze(2) # 变为 [1, T, 1, rope_dim]
        
    # 核心公式：x * cos(\theta) + rotate_half(x) * sin(\theta)
    # 利用逐元素乘法完美替代了低效的矩阵乘法
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot