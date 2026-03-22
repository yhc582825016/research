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
        # Rank 0 -> Rank 2, Rank 1 -> Rank 3
        if self.pp_rank < self.pp_size - 1:
            return self.rank + self.tp_size
        return None

    def get_prev_stage_rank(self):
        # Rank 2 -> Rank 0, Rank 3 -> Rank 1
        if self.pp_rank > 0:
            return self.rank - self.tp_size
        return None

# =============================================================================
# 2. TP 基础层 (Column/Row Linear)
# =============================================================================

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, context=None):
        super().__init__()
        self.tp_size = context.tp_size
        self.tp_group = context.tp_group
        
        # 输出维度切分
        self.out_dim_per_part = out_dim // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.out_dim_per_part, in_dim))
        self.bias = nn.Parameter(torch.empty(self.out_dim_per_part)) if bias else None
        
        # 初始化 (这里简单处理，实际需要广播种子确保一致)
        torch.manual_seed(42) 
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # x: [batch, seq, in] -> out: [batch, seq, out_part]
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, context=None):
        super().__init__()
        self.tp_size = context.tp_size
        self.tp_group = context.tp_group
        
        # 输入维度切分
        self.in_dim_per_part = in_dim // self.tp_size
        self.weight = nn.Parameter(torch.empty(out_dim, self.in_dim_per_part))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # x: [batch, seq, in_part]
        out = F.linear(x, self.weight)
        # 核心：TP 内部 All-Reduce，把部分和加起来
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out

# =============================================================================
# 3. Transformer 层 (Attention + MLP)
# =============================================================================

class TPLayer(nn.Module):
    def __init__(self, config, context, layer_id):
        super().__init__()
        self.hidden_size = config.hidden_size
        dim = config.hidden_size
        
        # Norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention (简化版，无 GQA/RoPE细节，专注 TP 结构)
        # QKV: Column Parallel
        self.qkv_proj = ColumnParallelLinear(dim, dim * 3, context=context)
        # Output: Row Parallel
        self.o_proj = RowParallelLinear(dim, dim, bias=False, context=context)

        # MLP
        # Gate/Up: Column Parallel
        self.gate_up_proj = ColumnParallelLinear(dim, config.intermediate_size * 2, bias=False, context=context)
        # Down: Row Parallel
        self.down_proj = RowParallelLinear(config.intermediate_size, dim, bias=False, context=context)

    def forward(self, x):
        # --- Attention Block ---
        residual = x
        x = self.norm1(x)
        
        # QKV Proj
        qkv = self.qkv_proj(x) # Output is Partial
        # (在此处可以加入 RoPE 和 Attention 计算逻辑)
        # 为演示简单，直接将 partial qkv 投影回 partial output
        # 实际代码这里需要 reshape heads -> scaled dot product -> context
        # 这里模拟 Attention 的维度变换：
        attn_out = qkv.chunk(3, dim=-1)[0] # 取 Q 作为模拟输出
        
        x = self.o_proj(attn_out) # All-Reduce inside
        x = residual + x

        # --- MLP Block ---
        residual = x
        x = self.norm2(x)
        
        gate_up = self.gate_up_proj(x) # Output is Partial
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up # Local computation on partial data
        
        x = self.down_proj(x) # All-Reduce inside
        x = residual + x
        
        return x

# =============================================================================
# 4. Pipeline Stage (模型分段)
# =============================================================================

class PipelineTransformer(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.context = context
        self.config = config
        
        # 计算当前 GPU 负责哪一层
        # PP=2, Layers=2 -> Stage 0 负责 Layer 0, Stage 1 负责 Layer 1
        layers_per_stage = config.num_layers // context.pp_size
        self.my_layers_idx = range(context.pp_rank * layers_per_stage, 
                                   (context.pp_rank + 1) * layers_per_stage)
        
        print(f"[Rank {context.rank}] Building Stage {context.pp_rank}, Layers {list(self.my_layers_idx)}")

        # 1. Embedding (只在 Stage 0)
        if context.pp_rank == 0:
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed = None

        # 2. Transformer Layers
        self.layers = nn.ModuleList([
            TPLayer(config, context, idx) for idx in self.my_layers_idx
        ])

        # 3. Output Head (只在 Last Stage)
        if context.pp_rank == context.pp_size - 1:
            self.final_norm = nn.LayerNorm(config.hidden_size)
            # Head 也是 Column Parallel，输出 Partial Logits，通常 CrossEntropyLoss 支持并行计算
            self.head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, context=context)
        else:
            self.head = None

    def forward(self, x):
        # 如果是 Stage 0，x 是 input_ids [batch, seq]
        # 如果是 Stage 1，x 是 hidden_states [batch, seq, dim]
        
        if self.embed:
            x = self.embed(x)
        
        for layer in self.layers:
            x = layer(x)
            
        if self.head:
            x = self.final_norm(x)
            x = self.head(x) # Output: Partial Logits
            
        return x

# =============================================================================
# 5. 执行逻辑 (Simulation)
# =============================================================================

def run_step(model, context, config, input_ids=None):
    device = torch.device(f"cuda:{context.rank}")
    
    # --- 1. Recv (如果不是第一级) ---
    if context.pp_rank == 0:
        # 第一级直接使用输入
        curr_input = input_ids.to(device)
    else:
        # 后续级从上一级接收 Hidden States
        recv_buffer = torch.zeros(2, config.seq_len, config.hidden_size, device=device)
        src = context.get_prev_stage_rank()
        dist.recv(recv_buffer, src=src)
        curr_input = recv_buffer
        print(f"[Rank {context.rank}] Recv data from Rank {src}")

    # --- 2. Compute ---
    output = model(curr_input)

    # --- 3. Send (如果不是最后一级) ---
    if context.pp_rank < context.pp_size - 1:
        dst = context.get_next_stage_rank()
        dist.send(output.contiguous(), dst=dst)
        print(f"[Rank {context.rank}] Sent data to Rank {dst}")
        return None
    else:
        # 最后一级返回结果
        return output

def worker(rank, world_size):
    # 初始化环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 配置: 4卡, TP=2, PP=2
    config = ModelConfig()
    context = ParallelContext(rank, world_size, tp_size=2, pp_size=2)
    
    # 构建模型 (只包含当前 Stage 的层)
    model = PipelineTransformer(config, context).to(f"cuda:{rank}")
    
    # 模拟数据
    if context.pp_rank == 0:
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_len))
    else:
        input_ids = None

    # 运行一次 Forward
    final_output = run_step(model, context, config, input_ids)

    # 验证输出
    if context.pp_rank == context.pp_size - 1:
        # final_output 是 Partial Logits，需要 Gather 才能看到完整形状
        # 但我们这里只验证它跑通了
        print(f"✅ [Rank {rank}] Pipeline Finished! Logits Shape: {final_output.shape} (Partial Vocab)")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"Error: 这里的演示配置需要 {WORLD_SIZE} 个 GPU (TP=2 * PP=2)")
    else:
        print("🚀 Starting 2-Layer Transformer (TP=2, PP=2)...")
        mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)