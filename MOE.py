import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 定义单个专家 (Expert)
# 在标准的 Transformer 中，这就是那个两层的 FeedForward Network (FFN/MLP)
class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# 2. 核心 MoE 层
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k, expert_hidden_dim):
        """
        num_experts: 专家的总数 (例如 8)
        top_k: 每个 token 激活的专家数量 (通常是 2)
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由器 (Router/Gate)：决定每个 token 去哪个专家
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # 专家列表：使用 ModuleList 注册所有的专家网络
        self.experts = nn.ModuleList(
            [Expert(d_model, expert_hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        # x 形状: [Batch, Seq_len, d_model]
        B, T, C = x.size()
        
        # 为了方便路由计算，将 Batch 和 Seq_len 展平
        x_flat = x.view(-1, C) # [B*T, d_model]
        
        # 1. 计算路由分数 (Router Logits)
        router_logits = self.router(x_flat) # [B*T, num_experts]
        
        # 2. 选出 Top-K 个专家，并计算对应的权重
        # routing_weights: [B*T, top_k], selected_experts: [B*T, top_k]
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        
        # 对选出的 Top-K 权重进行 Softmax 归一化
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 初始化最终的输出张量
        final_output = torch.zeros_like(x_flat)
        
        # 3. 将 Token 分发给对应的专家进行计算 (Dispatch & Compute)
        # 面试时，遍历专家的写法最容易讲解清晰
        for i, expert in enumerate(self.experts):
            # 找到哪些 token 被分配给了当前专家 i
            # token_indices 是在 [B*T] 维度上的索引
            # k_indices 是在 [top_k] 维度上的索引 (指明是第 1 选择还是第 2 选择)
            token_indices, k_indices = torch.where(selected_experts == i)
            
            # 如果没有 token 被分配给这个专家，直接跳过，这就是“稀疏计算”的体现
            if token_indices.numel() == 0:
                continue
                
            # 提取出分配给该专家的 token
            tokens_for_expert = x_flat[token_indices]
            
            # 专家进行计算
            expert_output = expert(tokens_for_expert)
            
            # 提取对应的路由权重，并增加一个维度以便广播相乘: [被选中的 token 数量, 1]
            expert_weights = routing_weights[token_indices, k_indices].unsqueeze(-1)
            
            # 4. 加权并累加到最终输出中 (Combine)
            final_output[token_indices] += expert_output * expert_weights
            
        # 还原回原来的形状
        return final_output.view(B, T, C)

# 3. 完整的包含 Attention 和 MoE 的 Transformer Block
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_experts, top_k, expert_hidden_dim):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        # 这里为了演示简便，直接调用原生的 MHA，您可以用上一条回答的手写 MHA 替换
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.moe = SparseMoELayer(d_model, num_experts, top_k, expert_hidden_dim)

    def forward(self, x):
        # Attention 阶段 (带残差连接)
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_out
        
        # MoE 阶段 (替换了传统的 FFN，带残差连接)
        x = x + self.moe(self.ln_2(x))
        return x

# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 4
    num_experts = 8
    top_k = 2  # 经典的 8 选 2 策略
    expert_hidden_dim = 512
    
    x = torch.randn(batch_size, seq_len, d_model)
    moe_block = MoETransformerBlock(d_model, num_heads, num_experts, top_k, expert_hidden_dim)
    
    output = moe_block(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")