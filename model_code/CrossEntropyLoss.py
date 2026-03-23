import torch
import torch.nn as nn

class ManualCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        """
        参数:
        logits: 模型的原始输出，形状为 [Batch_size, Num_classes]
        target: 真实的标签索引，形状为 [Batch_size]
        """
        batch_size = logits.size(0)
        
        # ==========================================
        # 错误示范 (朴素计算)：面试时可以提一下，但千万别作为最终答案
        # probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdim=True)
        # log_probs = torch.log(probs)
        # ==========================================

        # ==========================================
        # 阶段一：数值稳定的 LogSoftmax 计算 (LogSumExp 技巧)
        # ==========================================
        # 1. 找到每个样本 logits 的最大值
        # detach() 是为了防止这个求最大值的操作影响反向传播的梯度
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        max_logits = max_logits.detach()
        
        # 2. 对 logits 进行平移 (减去最大值)
        # 这一步保证了 safe_logits 里的最大值永远是 0，从而 torch.exp(0) = 1，绝对不会溢出
        safe_logits = logits - max_logits
        
        # 3. 计算 LogSumExp
        # 公式: log(sum(exp(x - max)))
        log_sum_exp = torch.log(torch.sum(torch.exp(safe_logits), dim=-1, keepdim=True))
        
        # 4. 计算最终的 log 概率
        # log(Softmax(x)) = x - max - log(sum(exp(x - max)))
        log_probs = safe_logits - log_sum_exp

        # ==========================================
        # 阶段二：计算 NLL Loss (Negative Log Likelihood)
        # ==========================================
        # 目标是提取 target 对应类别的 log 概率。
        # target 形状从 [Batch_size] 变为 [Batch_size, 1] 以便配合 gather
        target = target.view(-1, 1)
        
        # 使用 gather 函数高效提取对应索引的值
        # 例如，如果 target 是 [2]，就会提取 log_probs 里索引为 2 的那个概率值
        target_log_probs = log_probs.gather(dim=-1, index=target)

        # 交叉熵 = -log(p)
        loss = -target_log_probs

        # ==========================================
        # 阶段三：归一化 (Reduction)
        # ==========================================
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- 测试与验证代码 ---
if __name__ == "__main__":
    batch_size = 4
    num_classes = 10
    
    # 模拟模型的 Logits 输出 (包含故意设置的极大值以测试数值稳定性)
    logits = torch.randn(batch_size, num_classes) * 100 
    # 模拟真实的类别标签
    target = torch.randint(0, num_classes, (batch_size,))
    
    # 1. 使用我们手写的 Loss
    manual_criterion = ManualCrossEntropyLoss()
    manual_loss = manual_criterion(logits, target)
    
    # 2. 使用 PyTorch 官方的 Loss 进行对比验证
    official_criterion = nn.CrossEntropyLoss()
    official_loss = official_criterion(logits, target)
    
    print(f"手写 Cross Entropy Loss: {manual_loss.item():.6f}")
    print(f"官方 Cross Entropy Loss: {official_loss.item():.6f}")
    print(f"误差差异: {abs(manual_loss.item() - official_loss.item()):.6e}")