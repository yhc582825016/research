import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        参数:
        d_model: 模型的隐藏层维度 (例如 512)
        num_heads: 注意力头的数量 (例如 8)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # MHA 的特点：Q, K, V 都有各自完整的独立权重矩阵
        # 为了高效计算，通常将 Q, K, V 的线性映射合并为一个大的矩阵乘法
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # 输出的线性映射
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.size() # Batch size, Sequence length, d_model(C)

        # 1. 一次性计算 Q, K, V 并进行切分
        # qkv 形状: [B, T, 3 * d_model]
        qkv = self.c_attn(x)
        
        # 将 qkv 切分为 q, k, v，每个形状为 [B, T, d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. 划分多头 (Multi-Head)
        # [B, T, d_model] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        # transpose(1, 2) 是为了把 heads 维度提到前面，方便后续不同头之间独立计算
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算 Scaled Dot-Product Attention
        # scores 形状: [B, num_heads, T, T]
        # 这一步计算每个 token 与其他所有 token 的相关性打分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 应用 Mask (如果在 Decoder 中或者处理 Padding)
        if mask is not None:
            # mask 通常是一个下三角矩阵或 padding 掩码
            # 将需要 mask 的地方替换为极小值，这样 softmax 后对应的权重就会变为 0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. Softmax 归一化注意力权重
        attn = F.softmax(scores, dim=-1)

        # 6. 注意力权重与 Value 相乘
        # out 形状: [B, num_heads, T, head_dim]
        out = torch.matmul(attn, v)

        # 7. 拼接所有头并还原形状
        # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, d_model]
        # contiguous() 是必须的，因为 transpose 破坏了内存的连续性
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. 最后的线性映射
        return self.c_proj(out)

# --- 测试代码 ---
if __name__ == "__main__":
    B, T, C = 2, 10, 512
    num_heads = 8
    x = torch.randn(B, T, C)
    
    # 模拟一个 Causal Mask (下三角矩阵，防止看到未来的信息)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)

    mha = MultiHeadAttention(d_model=C, num_heads=num_heads)
    output = mha(x, mask=mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")