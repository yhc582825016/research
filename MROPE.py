import torch

class MultimodalRoPE:
    def __init__(self, head_dim, dim_t, dim_h, dim_w, theta=10000.0):
        """
        参数:
        head_dim: Attention Head 的总维度 (例如 128)
        dim_t, dim_h, dim_w: 分配给时间、高度、宽度的维度数量 (例如 32, 48, 48)
        """
        assert dim_t + dim_h + dim_w == head_dim, "三个维度的和必须等于 head_dim"
        self.dims = [dim_t, dim_h, dim_w]
        self.theta = theta

    def _compute_1d_freqs(self, pos_ids, dim):
        """通用的 1D 频率计算"""
        # pos_ids: [seq_len]
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(pos_ids.float(), inv_freq) # [seq_len, dim // 2]
        return torch.cat((freqs, freqs), dim=-1)       # [seq_len, dim]

    def forward(self, q, k, pos_ids_3d):
        """
        pos_ids_3d: 形状为 [3, seq_len]，包含了 (t_ids, h_ids, w_ids)
        """
        seq_len = pos_ids_3d.shape[1]
        
        # 1. 核心逻辑：分别计算 T, H, W 三个方向的旋转频率
        freqs_t = self._compute_1d_freqs(pos_ids_3d[0], self.dims[0]) # [seq_len, dim_t]
        freqs_h = self._compute_1d_freqs(pos_ids_3d[1], self.dims[1]) # [seq_len, dim_h]
        freqs_w = self._compute_1d_freqs(pos_ids_3d[2], self.dims[2]) # [seq_len, dim_w]
        
        # 2. 在特征维度上拼接，重组为完整的 head_dim
        # freqs 形状: [seq_len, head_dim]
        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
        
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        
        # 3. 旋转操作 (与标准 RoPE 完全相同)
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
            
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        
        return q_rot, k_rot

# --- 测试与面试讲解代码 ---
if __name__ == "__main__":
    head_dim = 128
    # 假设给时间分配 32 维，高和宽各分配 48 维
    mrope = MultimodalRoPE(head_dim, dim_t=32, dim_h=48, dim_w=48)
    
    seq_len = 4
    q = torch.randn(1, 8, seq_len, head_dim)
    k = torch.randn(1, 8, seq_len, head_dim)
    
    # 【面试官看点：如何用同一个接口处理不同模态？】
    
    # 场景 1: 纯文本 (Text)
    # 对于文本，时间、高度、宽度的 ID 是同步递增的！模型会将其视为一条对角线上的 3D 轨迹。
    text_pos_ids = torch.tensor([
        [0, 1, 2, 3], # T
        [0, 1, 2, 3], # H
        [0, 1, 2, 3]  # W
    ])
    
    # 场景 2: 单张图像 (Image) 假设是一个 2x2 的 Patch 网格
    # 时间是静止的 (全是 0)，H 和 W 在 2D 空间中变化。
    image_pos_ids = torch.tensor([
        [0, 0, 0, 0], # T (同一时刻)
        [0, 0, 1, 1], # H (第0行和第1行)
        [0, 1, 0, 1]  # W (第0列和第1列)
    ])
    
    # 场景 3: 视频 (Video) 假设有 2 帧，每帧是一个 1x2 的 Patch
    # 时间 T 随帧数推移，H 和 W 刻画每一帧内部的空间。
    video_pos_ids = torch.tensor([
        [0, 0, 1, 1], # T (第0帧和第1帧)
        [0, 0, 0, 0], # H (只有第0行)
        [0, 1, 0, 1]  # W (第0列和第1列)
    ])
    
    # 无论什么模态，直接前向传播即可
    q_out, k_out = mrope.forward(q, k, video_pos_ids)
    print(f"M-RoPE 输出形状: {q_out.shape}")