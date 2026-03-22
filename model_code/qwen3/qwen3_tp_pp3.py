import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import math

# =============================================================================
# 1. 配置与并行上下文
# =============================================================================

class ModelConfig:
    def __init__(self):
        self.vocab_size = 1000
        self.hidden_size = 64        # 方便演示的小维度
        self.num_heads = 4           # TP=2时，每卡2个头
        self.num_layers = 2          # 总共2层
        self.intermediate_size = 256
        self.seq_len = 10

class ParallelContext:
    def __init__(self, rank, world_size, tp_size, pp_size):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size

        # 核心坐标计算
        # Rank 0,1 -> Stage 0 | Rank 2,3 -> Stage 1
        self.pp_rank = rank // tp_size
        self.tp_rank = rank % tp_size

        # 建立 TP 通信组 (All-Reduce 用)
        # Stage 0 的 TP组是 [0,1], Stage 1 的 TP组是 [2,3]
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
# 2. 核心组件：Vocab Parallel Cross Entropy (新增部分)
# =============================================================================

class VocabParallelCrossEntropy(nn.Module):
    def __init__(self, context, vocab_size):
        super().__init__()
        self.context = context
        self.vocab_size = vocab_size
        # 计算当前 Rank 负责的词表范围
        self.vocab_per_partition = vocab_size // context.tp_size
        self.vocab_start_index = context.tp_rank * self.vocab_per_partition
        self.vocab_end_index = (context.tp_rank + 1) * self.vocab_per_partition

    def forward(self, logits, targets):
        # logits: [batch, seq, vocab_per_part] (Partial)
        # targets: [batch, seq] (Global Labels)

        # --- Step 1: 计算全局 LogSumExp (分母) ---
        # 1.1 找局部最大值
        local_max, _ = torch.max(logits, dim=-1)
        
        # 1.2 全局最大值 (All-Reduce Max)
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=self.context.tp_group)
        
        # 1.3 局部 Sum Exp (数值稳定性处理：x - max)
        local_sum_exp = torch.sum(torch.exp(logits - global_max.unsqueeze(-1)), dim=-1)
        
        # 1.4 全局 Sum Exp (All-Reduce Sum)
        global_sum_exp = local_sum_exp.clone()
        dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM, group=self.context.tp_group)
        
        # 1.5 得到全局 LogSumExp: log(sum) + max
        global_log_sum_exp = torch.log(global_sum_exp) + global_max

        # --- Step 2: 获取 Target 对应的 Logit (分子) ---
        # 2.1 创建掩码：Target 是否在当前 GPU 范围内
        target_mask = (targets >= self.vocab_start_index) & (targets < self.vocab_end_index)
        
        # 2.2 计算本地相对索引
        local_target_ids = targets - self.vocab_start_index
        # 将越界的 ID 设为 0 以防止 gather 报错 (后续会被 mask 过滤)
        safe_local_target_ids = torch.where(target_mask, local_target_ids, torch.tensor(0, device=targets.device))
        
        # 2.3 Gather 取值
        target_logits = logits.gather(dim=-1, index=safe_local_target_ids.unsqueeze(-1)).squeeze(-1)
        
        # 2.4 过滤无效值
        target_logits = target_logits * target_mask.float()
        
        # 2.5 全局求和 (只有持有该 target 的 rank 有值，其他为0，相加即为真实值)
        dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=self.context.tp_group)

        # --- Step 3: 计算 Loss ---
        # Loss = log(sum(exp(x))) - x_target
        loss = global_log_sum_exp - target_logits
        return loss.mean()

# =============================================================================
# 3. TP 基础层
# =============================================================================

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, context=None):
        super().__init__()
        self.tp_size = context.tp_size
        self.tp_group = context.tp_group
        self.out_dim_per_part = out_dim // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.out_dim_per_part, in_dim))
        self.bias = nn.Parameter(torch.empty(self.out_dim_per_part)) if bias else None
        
        torch.manual_seed(42) 
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, context=None):
        super().__init__()
        self.tp_size = context.tp_size
        self.tp_group = context.tp_group
        self.in_dim_per_part = in_dim // self.tp_size
        self.weight = nn.Parameter(torch.empty(out_dim, self.in_dim_per_part))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        out = F.linear(x, self.weight)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out

# =============================================================================
# 4. Transformer 层 & Pipeline
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
        # Attention
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x)
        attn_out = qkv.chunk(3, dim=-1)[0] # 简化模拟
        x = self.o_proj(attn_out)
        x = residual + x
        # MLP
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
        
        layers_per_stage = config.num_layers // context.pp_size
        self.my_layers_idx = range(context.pp_rank * layers_per_stage, 
                                   (context.pp_rank + 1) * layers_per_stage)
        
        print(f"[Rank {context.rank}] Building Stage {context.pp_rank}, Layers {list(self.my_layers_idx)}")

        if context.pp_rank == 0:
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed = None

        self.layers = nn.ModuleList([
            TPLayer(config, context, idx) for idx in self.my_layers_idx
        ])

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
            x = self.head(x) # [batch, seq, vocab/tp]
            
        return x

# =============================================================================
# 5. 执行逻辑 (修改版：集成 Loss 计算)
# =============================================================================

def run_step(model, context, config, input_ids=None):
    device = torch.device(f"cuda:{context.rank}")
    
    # --- 1. Recv ---
    if context.pp_rank == 0:
        curr_input = input_ids.to(device)
    else:
        recv_buffer = torch.zeros(2, config.seq_len, config.hidden_size, device=device)
        src = context.get_prev_stage_rank()
        dist.recv(recv_buffer, src=src)
        curr_input = recv_buffer
        print(f"[Rank {context.rank}] Recv data from Rank {src}")

    # --- 2. Compute ---
    output = model(curr_input)

    # --- 3. Send or Loss ---
    if context.pp_rank < context.pp_size - 1:
        dst = context.get_next_stage_rank()
        dist.send(output.contiguous(), dst=dst)
        print(f"[Rank {context.rank}] Sent data to Rank {dst}")
        return None
    else:
        # === Last Stage: 计算 Loss ===
        
        # 为了演示，我们在这里随机生成 targets
        # 关键点：Rank 2 和 Rank 3 必须生成完全一样的 targets
        torch.manual_seed(999) 
        targets = torch.randint(0, config.vocab_size, (2, config.seq_len)).to(device)
        
        # 初始化并行 Loss
        loss_fn = VocabParallelCrossEntropy(context, config.vocab_size)
        loss = loss_fn(output, targets)
        
        print(f"✅ [Rank {context.rank}] Pipeline Finished! Loss: {loss.item():.4f}")
        return loss

def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 配置: 4卡, TP=2, PP=2
    config = ModelConfig()
    context = ParallelContext(rank, world_size, tp_size=2, pp_size=2)
    
    model = PipelineTransformer(config, context).to(f"cuda:{rank}")
    
    if context.pp_rank == 0:
        torch.manual_seed(123)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_len))
    else:
        input_ids = None

    run_step(model, context, config, input_ids)

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"Error: 需要 {WORLD_SIZE} 个 GPU 运行此代码 (TP=2 * PP=2)")
    else:
        print("🚀 Starting Hybrid Parallel Transformer (TP=2, PP=2) with Loss...")
        mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)