import torch

# 1. 预计算频率 (在模型初始化时调用一次即可，节省推理时间)
def precompute_freqs_cis(head_dim, seq_len, theta=10000.0):
    """
    预先计算旋转所需的 cos 和 sin 值。
    head_dim: 每个注意力头的维度 (通常是 d_model // num_heads)
    seq_len: 序列最大长度
    theta: 基底常数，默认 10000.0
    """
    # 计算 \theta_i = 10000^(-2(i-1)/d)
    # torch.arange 步长为 2，取 head_dim 的一半
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # 生成位置序列 m: [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=inv_freq.device).float()
    
    # 计算外积 m * \theta_i，形状为 [seq_len, head_dim // 2]
    freqs = torch.outer(t, inv_freq)
    
    # 复制一份，将形状变为 [seq_len, head_dim]，以便后续与 Q 和 K 逐元素相乘
    # 比如原本是 [f1, f2]，拼接后变成 [f1, f2, f1, f2]
    freqs = torch.cat((freqs, freqs), dim=-1)
    
    # 返回 cos 和 sin
    return torch.cos(freqs), torch.sin(freqs)

# 2. 辅助函数：旋转张量的一半
def rotate_half(x):
    """
    将张量最后一个维度切分成两半，并进行符号反转和交换位置。
    用于模拟复数乘法中的旋转操作。
    """
    # x: [batch_size, num_heads, seq_len, head_dim]
    x1, x2 = x.chunk(2, dim=-1)
    # 返回 [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)

# 3. 核心应用函数 (在每次 Attention 计算前调用)
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将预计算的 cos 和 sin 应用到 Query 和 Key 上。
    """
    # 假设 q, k 的形状是 [batch_size, num_heads, seq_len, head_dim]
    # cos, sin 的初始形状是 [seq_len, head_dim]
    # 需要在第 0 维(batch) 和第 1 维(heads) 上增加维度，以便利用广播机制 (Broadcasting)
    cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]

    # 核心旋转公式:
    # 结果 = x * cos(\theta) + rotate_half(x) * sin(\theta)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot

# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64

    # 模拟 Attention 前的 Q 和 K
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 预计算 cos 和 sin (实际工程中这步会放在 __init__ 里缓存)
    cos, sin = precompute_freqs_cis(head_dim, seq_len)

    # 应用 RoPE
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"原始 Q 形状: {q.shape}")
    print(f"应用 RoPE 后的 Q 形状: {q_out.shape}")