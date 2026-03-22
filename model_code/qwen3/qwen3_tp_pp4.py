import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

# =============================================================================
# 1. 配置与并行上下文 (Context)
# =============================================================================

class ModelConfig:
    def __init__(self):
        self.vocab_size = 1000
        self.hidden_size = 64        # 演示用小维度
        self.num_heads = 4           
        self.num_layers = 2          # 总共2层
        self.intermediate_size = 256
        self.seq_len = 10

class ParallelContext:
    def __init__(self, rank, world_size, tp_size, pp_size):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size

        # 坐标计算: Rank 0,1 -> Stage 0 | Rank 2,3 -> Stage 1
        self.pp_rank = rank // tp_size
        self.tp_rank = rank % tp_size

        # 建立 TP 通信组 (用于 All-Reduce)
        self.tp_group = None
        for i in range(pp_size):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                self.tp_group = group

    def get_next_stage_rank(self):
        # 简单路由：发给下一个 Stage 的相同 TP rank
        if self.pp_rank < self.pp_size - 1:
            return self.rank + self.tp_size
        return None

    def get_prev_stage_rank(self):
        if self.pp_rank > 0:
            return self.rank - self.tp_size
        return None

# =============================================================================
# 2. TP 核心通信算子 (Autograd Functions - 关键新增)
# =============================================================================

class CopyToRegion(torch.autograd.Function):
    """
    用于 ColumnParallel 的输入处。
    前向: Identity (不做操作)
    反向: All-Reduce (梯度求和)
    原理: 输入 X 被复制到多卡，反向时各卡计算的 dL/dX 需要累加才是真正的总梯度。
    """
    @staticmethod
    def forward(ctx, input, tp_group):
        ctx.tp_group = tp_group
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone() 
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad, None

class ReduceFromRegion(torch.autograd.Function):
    """
    用于 RowParallel 的输出处。
    前向: All-Reduce (结果求和)
    反向: Identity (不做操作)
    原理: RowParallel 各卡输出只是部分和，需要 Sum 才是完整输出。反向时梯度广播即可。
    """
    @staticmethod
    def forward(ctx, input, tp_group):
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# =============================================================================
# 3. TP 基础层 (Linear Layers)
# =============================================================================

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, context=None):
        super().__init__()
        self.context = context
        self.tp_size = context.tp_size
        self.out_dim_per_part = out_dim // self.tp_size
        
        self.weight = nn.Parameter(torch.empty(self.out_dim_per_part, in_dim))
        self.bias = nn.Parameter(torch.empty(self.out_dim_per_part)) if bias else None
        
        # 固定随机种子初始化，保证同一 TP 组内初始权重逻辑一致
        torch.manual_seed(42) 
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # 1. 插入 Copy 算子 (处理反向梯度聚合)
        x_parallel = CopyToRegion.apply(x, self.context.tp_group)
        # 2. 本地计算
        return F.linear(x_parallel, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, context=None):
        super().__init__()
        self.context = context
        self.tp_size = context.tp_size
        self.in_dim_per_part = in_dim // self.tp_size
        
        self.weight = nn.Parameter(torch.empty(out_dim, self.in_dim_per_part))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # 1. 本地计算
        out = F.linear(x, self.weight)
        # 2. 插入 Reduce 算子 (处理前向结果聚合)
        out = ReduceFromRegion.apply(out, self.context.tp_group)
        
        if self.bias is not None:
            out = out + self.bias
        return out

# =============================================================================
# 4. Vocab Parallel Loss (并行交叉熵)
# =============================================================================

class VocabParallelCrossEntropy(nn.Module):
    def __init__(self, context, vocab_size):
        super().__init__()
        self.context = context
        self.vocab_size = vocab_size
        self.vocab_per_partition = vocab_size // context.tp_size
        self.vocab_start_index = context.tp_rank * self.vocab_per_partition
        self.vocab_end_index = (context.tp_rank + 1) * self.vocab_per_partition

    def forward(self, logits, targets):
        # logits: [batch, seq, vocab_per_part]
        # targets: [batch, seq]
        
        # 1. 全局 Max (数值稳定用)
        local_max, _ = torch.max(logits, dim=-1)
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=self.context.tp_group)
        
        # 2. 全局 Sum Exp (分母)
        local_sum_exp = torch.sum(torch.exp(logits - global_max.unsqueeze(-1)), dim=-1)
        global_sum_exp = local_sum_exp.clone()
        dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM, group=self.context.tp_group)
        global_log_sum_exp = torch.log(global_sum_exp) + global_max

        # 3. 获取 Target Logit (分子)
        target_mask = (targets >= self.vocab_start_index) & (targets < self.vocab_end_index)
        local_target_ids = targets - self.vocab_start_index
        safe_local_target_ids = torch.where(target_mask, local_target_ids, torch.tensor(0, device=targets.device))
        
        target_logits = logits.gather(dim=-1, index=safe_local_target_ids.unsqueeze(-1)).squeeze(-1)
        target_logits = target_logits * target_mask.float()
        dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=self.context.tp_group)

        # 4. 计算 Loss
        loss = global_log_sum_exp - target_logits
        return loss.mean()

# =============================================================================
# 5. Transformer 模型结构
# =============================================================================

class TPLayer(nn.Module):
    def __init__(self, config, context, layer_id):
        super().__init__()
        dim = config.hidden_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Attention
        self.qkv_proj = ColumnParallelLinear(dim, dim * 3, context=context)
        self.o_proj = RowParallelLinear(dim, dim, bias=False, context=context)
        # MLP
        self.gate_up_proj = ColumnParallelLinear(dim, config.intermediate_size * 2, bias=False, context=context)
        self.down_proj = RowParallelLinear(config.intermediate_size, dim, bias=False, context=context)

    def forward(self, x):
        # Attention Block
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x)
        attn_out = qkv.chunk(3, dim=-1)[0] # 简化处理：这里仅取 Q 作为演示
        x = self.o_proj(attn_out)
        x = residual + x
        # MLP Block
        residual = x
        x = self.norm2(x)
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        x = residual + x
        return x

class PipelineTransformer(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.context = context
        self.config = config
        
        # 分配层给当前 Stage
        layers_per_stage = config.num_layers // context.pp_size
        self.my_layers_idx = range(context.pp_rank * layers_per_stage, 
                                   (context.pp_rank + 1) * layers_per_stage)
        
        print(f"[Rank {context.rank}] Building Stage {context.pp_rank}, Layers {list(self.my_layers_idx)}")

        # Stage 0 负责 Embedding
        if context.pp_rank == 0:
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed = None

        self.layers = nn.ModuleList([
            TPLayer(config, context, idx) for idx in self.my_layers_idx
        ])

        # Last Stage 负责 Head
        if context.pp_rank == context.pp_size - 1:
            self.final_norm = nn.LayerNorm(config.hidden_size)
            self.head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, context=context)
        else:
            self.head = None

    def forward(self, x):
        if self.embed:
            x = self.embed(x)
        
        for layer in self.layers:
            x = layer(x)
            
        if self.head:
            x = self.final_norm(x)
            x = self.head(x) 
            
        return x

# =============================================================================
# 6. 训练步执行逻辑 (包含反向传播)
# =============================================================================

def run_training_step(model, context, config, optimizer, input_ids=None):
    device = torch.device(f"cuda:{context.rank}")
    
    # ------------------- 1. Forward Pass -------------------
    if context.pp_rank == 0:
        curr_input = input_ids.to(device)
    else:
        # 接收来自上一个 Stage 的 Tensor
        recv_buffer = torch.zeros(2, config.seq_len, config.hidden_size, device=device)
        src = context.get_prev_stage_rank()
        dist.recv(recv_buffer, src=src)
        
        # [关键] 开启梯度追踪：这是反向传播跨设备的关键节点
        curr_input = recv_buffer.clone().detach().requires_grad_(True)

    # 计算
    output = model(curr_input)

    # ------------------- 2. Transition / Loss -------------------
    loss = None
    if context.pp_rank < context.pp_size - 1:
        # 发送给下一个 Stage
        dst = context.get_next_stage_rank()
        dist.send(output.detach().contiguous(), dst=dst) # detach 发送，不传梯度图
    else:
        # 最后一个 Stage：计算 Loss
        # 确保同一 TP 组 targets 一致
        torch.manual_seed(999) 
        targets = torch.randint(0, config.vocab_size, (2, config.seq_len)).to(device)
        
        loss_fn = VocabParallelCrossEntropy(context, config.vocab_size)
        loss = loss_fn(output, targets)
        print(f"✅ [Rank {context.rank}] Step Loss: {loss.item():.4f}")

    # ------------------- 3. Backward Pass -------------------
    optimizer.zero_grad()

    if context.pp_rank == context.pp_size - 1:
        # [Last Stage]
        # 1. 启动反向传播
        loss.backward()
        
        # 2. 发送梯度回上一级
        if context.pp_rank > 0:
            dst = context.get_prev_stage_rank()
            dist.send(curr_input.grad.contiguous(), dst=dst)
            
    else:
        # [Middle / First Stage]
        # 1. 接收来自下一级的梯度
        grad_recv = torch.zeros_like(output)
        src = context.get_next_stage_rank()
        dist.recv(grad_recv, src=src)
        
        # 2. 连接计算图并反向
        output.backward(grad_recv)
        
        # 3. 如果还有更前面的 Stage，继续回传 (本例只有2个Stage，故不需要这段)
        # if context.pp_rank > 0: ... send curr_input.grad ...

    # ------------------- 4. Optimizer Step -------------------
    optimizer.step()
    # 简单打印部分权重的均值，证明参数更新了
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            if context.rank == 0: # 只让 Rank 0 打印一下避免刷屏
                print(f"✨ [Rank 0] Parameter '{name}' updated. Mean val: {param.data.mean():.6f}")
            break # 只打印一个

# =============================================================================
# 7. Worker 主程序
# =============================================================================

def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' # 修改端口防止冲突
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 配置
    config = ModelConfig()
    context = ParallelContext(rank, world_size, tp_size=2, pp_size=2)
    
    # 模型
    model = PipelineTransformer(config, context).to(f"cuda:{rank}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # 模拟数据
    if context.pp_rank == 0:
        torch.manual_seed(123)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_len))
    else:
        input_ids = None

    print(f"[Rank {rank}] Start Training Step...")
    run_training_step(model, context, config, optimizer, input_ids)
    print(f"[Rank {rank}] Finished.")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"Error: 需要 {WORLD_SIZE} 个 GPU 运行此代码 (TP=2 * PP=2)")
    else:
        print("🚀 Starting Complete Training Loop (TP=2, PP=2)...")
        mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)