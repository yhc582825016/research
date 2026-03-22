import torch
import torch.nn as nn

# ==========================================
# 1. Layer Normalization (LLM 标准配置)
# ==========================================
class SimpleLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        """
        d_model: 词向量的维度 (特征维度)
        """
        super().__init__()
        self.eps = eps
        # LayerNorm 的可学习参数，维度与特征维度一致
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x 的形状通常是: [Batch, Seq_len, d_model]
        
        # 【核心差异 1：计算维度的不同】
        # LayerNorm 只在最后一个维度 (d_model) 上求均值和方差
        # 这意味着同一个 Token 内的所有特征自己做归一化，与其他 Token 和其他样本无关
        mean = x.mean(dim=-1, keepdim=True) # 形状变为 [Batch, Seq_len, 1]
        var = x.var(dim=-1, unbiased=False, keepdim=True) # 形状变为 [Batch, Seq_len, 1]
        
        # 归一化并应用仿射变换
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ==========================================
# 2. Batch Normalization (视觉模型常用，LLM 中已弃用)
# ==========================================
class SimpleBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        num_features: 特征维度
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 【核心差异 2：BatchNorm 需要维护全局移动平均状态 (Running Stats)】
        # 因为在推理阶段（Batch Size 可能为 1），不能用当前 Batch 的均值，必须用训练时积累的均值
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # NLP 中的 x 形状通常是 [Batch, Seq_len, d_model]
        # 但标准的 BatchNorm1d 期望输入是 [Batch, Features, Seq_len]
        # 为了演示底层逻辑，我们假设这里 x 已经转置为 [Batch, d_model, Seq_len]
        
        if self.training:
            # BatchNorm 在 Batch (dim=0) 和 空间/序列维度 (dim=2) 上求均值
            # 相当于对所有样本在同一个特征通道上的值做归一化
            mean = x.mean(dim=(0, 2), keepdim=True) # 形状变为 [1, d_model, 1]
            var = x.var(dim=(0, 2), unbiased=False, keepdim=True)
            
            # 更新全局状态 (推理时使用)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # 推理阶段使用累积的全局状态
            mean = self.running_mean.view(1, -1, 1)
            var = self.running_var.view(1, -1, 1)
            
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 恢复 gamma 和 beta 的形状进行广播
        gamma = self.gamma.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        
        return gamma * x_norm + beta


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 只有缩放参数 gamma (通常被命名为 weight)，没有偏移参数 beta
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 1. 计算均方根 (RMS)
        # x.pow(2).mean(-1, keepdim=True) 就是特征维度的平方均值
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # 2. 归一化并乘以权重
        x_norm = x / rms
        return self.weight * x_norm