import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        """
        参数:
        d_model: 模型的隐藏层维度
        num_q_heads: Query 头的数量
        num_kv_heads: Key 和 Value 头的数量 (必须能被 num_q_heads 整除)
        """
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "Query heads 必须是 KV heads 的整数倍"
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        
        # 计算每个 KV head 需要服务多少个 Query head (这就是 Group 的概念)
        self.num_groups = num_q_heads // num_kv_heads 

        # 线性映射层
        self.W_q = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, d_model

        # 1. 线性投影并划分多头
        # Q: [B, T, num_q_heads, head_dim] -> [B, num_q_heads, T, head_dim]
        q = self.W_q(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        
        # K, V: [B, T, num_kv_heads, head_dim] -> [B, num_kv_heads, T, head_dim]
        k = self.W_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. GQA 的核心：复制 K 和 V 以匹配 Q 的头数
        # 使用 repeat_interleave 将 K 和 V 在头数维度上进行复制
        # 例如：K 从 [B, 2, T, head_dim] 变成 [B, 8, T, head_dim] (如果 num_groups=4)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # 3. 标准的 Scaled Dot-Product Attention
        # scores: [B, num_q_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        # out: [B, num_q_heads, T, head_dim]
        out = torch.matmul(attn, v)

        # 4. 拼接所有头并进行最终的线性映射
        # [B, num_q_heads, T, head_dim] -> [B, T, num_q_heads, head_dim] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.W_o(out)

# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_q_heads = 8
    num_kv_heads = 2 # 这里体现 GQA: 8个Q头，分为2组，每组4个Q头共享1个K和1个V

    x = torch.randn(batch_size, seq_len, d_model)
    gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
    output = gqa(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")