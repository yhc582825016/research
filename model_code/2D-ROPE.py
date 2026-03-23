import torch

# 1. 预计算 2D 频率 (核心逻辑)
def precompute_freqs_cis_2d(head_dim, height, width, theta=10000.0):
    """
    为 2D 图像特征图预计算旋转位置编码的 cos 和 sin。
    head_dim: 每个注意力头的维度 (必须能被 4 整除)
    height: 图像 Patch 的高度 (H)
    width:  图像 Patch 的宽度 (W)
    """
    # 2D RoPE 的核心：将每个头的维度劈成两半
    # 一半用于编码高度 (Y轴)，一半用于编码宽度 (X轴)
    assert head_dim % 4 == 0, "head_dim 必须是 4 的倍数"
    half_dim = head_dim // 2
    
    # 分别计算 H 和 W 的基础频率 (与 1D RoPE 相同，但维度减半)
    # inv_freq 形状: [half_dim // 2]
    inv_freq_h = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
    inv_freq_w = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
    
    # 生成 2D 网格坐标
    grid_h = torch.arange(height, dtype=torch.float32) # [H]
    grid_w = torch.arange(width, dtype=torch.float32)  # [W]
    
    # 计算外积得到每个坐标的旋转角度
    freqs_h = torch.outer(grid_h, inv_freq_h)  # [H, half_dim // 2]
    freqs_w = torch.outer(grid_w, inv_freq_w)  # [W, half_dim // 2]
    
    # --- 维度广播 (Broadcasting) 组合 2D 坐标 ---
    # 将 H 的频率扩展到整个 W 维度: [H, W, half_dim // 2]
    freqs_h = freqs_h.unsqueeze(1).expand(height, width, -1)
    # 将 W 的频率扩展到整个 H 维度: [H, W, half_dim // 2]
    freqs_w = freqs_w.unsqueeze(0).expand(height, width, -1)
    
    # 拼接 H 和 W 的频率: [H, W, half_dim]
    freqs = torch.cat((freqs_h, freqs_w), dim=-1)
    
    # 将 2D 空间展平为 1D 序列 (为了和 Transformer 的输入形状对齐)
    # seq_len = H * W
    freqs = freqs.view(height * width, -1) # [seq_len, half_dim]
    
    # 复制一份以匹配 head_dim，方便后续做逐元素乘法
    freqs = torch.cat((freqs, freqs), dim=-1) # [seq_len, head_dim]
    
    return torch.cos(freqs), torch.sin(freqs)

# 2. 辅助函数：旋转一半 (与 1D RoPE 完全一致！)
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 3. 核心应用函数 (与 1D RoPE 完全一致！)
def apply_rotary_pos_emb(q, k, cos, sin):
    # 假设 q, k 的形状是 [batch_size, num_heads, seq_len, head_dim]
    # 这里的 seq_len 就是 H * W
    cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot

# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 2
    num_heads = 8
    head_dim = 64
    
    # 假设输入一张图像，经过 ViT 切 patch 后，特征图大小为 14 x 14
    H, W = 14, 14 
    seq_len = H * W # 196
    
    # 模拟 Vision Transformer 输出的 Q 和 K
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 预计算 2D RoPE
    cos, sin = precompute_freqs_cis_2d(head_dim, H, W)

    # 应用 2D RoPE
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"原始 Q 形状: {q.shape}")
    print(f"应用 2D RoPE 后的 Q 形状: {q_out.shape}")