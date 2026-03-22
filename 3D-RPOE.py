import torch

# 1. 预计算 3D 时空网格的频率 (针对纯视频模型设计)
def precompute_freqs_cis_3d(head_dim, num_frames, height, width, theta=10000.0):
    """
    为 3D 视频特征图预计算旋转位置编码的 cos 和 sin。
    head_dim: 每个注意力头的维度 (为了平分给 T, H, W，必须能被 6 整除)
    num_frames: 视频帧数 / 时间维度的 Token 数 (T)
    height: 空间高度 Token 数 (H)
    width:  空间宽度 Token 数 (W)
    """
    # 3D RoPE 核心：将 head_dim 平均劈成三份，分别负责 T, H, W
    assert head_dim % 6 == 0, "head_dim 必须是 6 的倍数，以便平分为三份且均为偶数"
    dim_part = head_dim // 3  # 每份的维度
    
    # 计算基础频率 (三种维度使用相同的衰减基底)
    # inv_freq 形状: [dim_part // 2]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_part, 2).float() / dim_part))
    
    # 生成 3D 网格的独立坐标
    grid_t = torch.arange(num_frames, dtype=torch.float32) # [T]
    grid_h = torch.arange(height, dtype=torch.float32)     # [H]
    grid_w = torch.arange(width, dtype=torch.float32)      # [W]
    
    # 分别计算三个方向的外积频率
    freqs_t = torch.outer(grid_t, inv_freq) # [T, dim_part // 2]
    freqs_h = torch.outer(grid_h, inv_freq) # [H, dim_part // 2]
    freqs_w = torch.outer(grid_w, inv_freq) # [W, dim_part // 2]
    
    # --- 3D 维度的广播 (Broadcasting) 魔法 ---
    # 我们需要构建一个 [T, H, W, dim_part // 2] 的统一空间
    
    # 把 T 扩展到 H 和 W: [T, 1, 1, dim] -> [T, H, W, dim]
    freqs_t = freqs_t.view(num_frames, 1, 1, -1).expand(-1, height, width, -1)
    
    # 把 H 扩展到 T 和 W: [1, H, 1, dim] -> [T, H, W, dim]
    freqs_h = freqs_h.view(1, height, 1, -1).expand(num_frames, -1, width, -1)
    
    # 把 W 扩展到 T 和 H: [1, 1, W, dim] -> [T, H, W, dim]
    freqs_w = freqs_w.view(1, 1, width, -1).expand(num_frames, height, -1, -1)
    
    # 拼接三个维度的频率，形成完整的特征向量: [T, H, W, head_dim // 2]
    freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
    
    # 展平 3D 时空，变成 1D 的序列喂给 Transformer
    # seq_len = T * H * W
    freqs = freqs.view(num_frames * height * width, -1) # [seq_len, head_dim // 2]
    
    # 复制以匹配 head_dim，准备用于旋转
    freqs = torch.cat((freqs, freqs), dim=-1) # [seq_len, head_dim]
    
    return torch.cos(freqs), torch.sin(freqs)

# 2. 辅助函数：旋转一半 (与 1D/2D RoPE 完全一致)
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 3. 核心应用函数 (与 1D/2D RoPE 完全一致)
def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0) 
    sin = sin.unsqueeze(0).unsqueeze(0) 
    
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# --- 测试代码 ---
if __name__ == "__main__":
    head_dim = 96  # 必须能被 6 整除
    T, H, W = 4, 14, 14 # 4帧视频，每帧切分成 14x14 个 Patch
    seq_len = T * H * W # 784
    
    q = torch.randn(2, 8, seq_len, head_dim)
    k = torch.randn(2, 8, seq_len, head_dim)
    
    # 预计算 3D 频率并应用
    cos, sin = precompute_freqs_cis_3d(head_dim, T, H, W)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"输入 Q 形状: {q.shape}")
    print(f"输出 Q 形状: {q_out.shape}")