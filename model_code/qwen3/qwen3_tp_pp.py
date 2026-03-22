import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import math

# =============================================================================
# 1. 配置与上下文管理
# =============================================================================

class Qwen3Config:
    def __init__(self):
        self.hidden_size = 512          # 隐藏层维度
        self.num_attention_heads = 8    # 总头数 (TP=2时每卡4头)
        self.num_key_value_heads = 8
        self.intermediate_size = 2048   # MLP中间维度
        self.vocab_size = 4096          # 词表大小
        self.num_hidden_layers = 4      # 总层数 (PP=2时每Stage 2层)
        self.max_position_embeddings = 128
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0

class ParallelContext:
    def __init__(self, rank, world_size, tp_size, pp_size):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        
        # 拓扑坐标
        # Rank 0,1 -> Stage 0 | Rank 2,3 -> Stage 1
        self.pp_rank = rank // tp_size
        self.tp_rank = rank % tp_size
        
        self.tp_group = None
        self.init_groups()

    def init_groups(self):
        # 初始化 TP 组 (例如 [0,1] 是一个组, [2,3] 是一个组)
        for i in range(self.pp_size):
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group
    
    def get_prev_stage_rank(self):
        # 简单逻辑：找上一个 Stage 的对应 TP rank
        if self.pp_rank == 0: return None
        return self.rank - self.tp_size

    def get_next_stage_rank(self):
        if self.pp_rank == self.pp_size - 1: return None
        return self.rank + self.tp_size

# =============================================================================
# 2. 基础 TP 组件 (Megatron Style)
# =============================================================================

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)
        self.out_per_partition = out_features // self.tp_size
        
        # 权重切分：[out_per_part, in]
        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_partition))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 实际应使用种子同步初始化，这里简化为固定种子
        torch.manual_seed(42 + self.weight.shape[0]) 
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in] (Full) -> output: [batch, seq, out_part] (Partial)
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)
        self.in_per_partition = in_features // self.tp_size
        
        # 权重切分：[out, in_per_part]
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(42 + self.weight.shape[1])
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in_part] (Partial)
        output_parallel = F.linear(x, self.weight)
        # All-Reduce 聚合
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        return output_parallel

# =============================================================================
# 3. 模型组件 (RoPE, Norm, MLP, Attention)
# =============================================================================

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, heads, seq, dim]
    # cos, sin: [1, 1, seq, dim]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class Qwen3AttentionTP(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.local_heads = config.num_attention_heads // context.tp_size
        
        self.q_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=context.tp_group)
        self.k_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=context.tp_group)
        self.v_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=context.tp_group)
        self.o_proj = RowParallelLinear(config.hidden_size, config.hidden_size, bias=False, tp_group=context.tp_group)

    def forward(self, x, freqs_cis):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = freqs_cis
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)

class Qwen3MLPTP(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, tp_group=context.tp_group)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, tp_group=context.tp_group)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, tp_group=context.tp_group)

    def forward(self, x):
        # x是完整的，gate/up出来是切分的，act后也是切分的，down_proj内部会reduce
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen3DecoderLayerTP(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.self_attn = Qwen3AttentionTP(config, context)
        self.mlp = Qwen3MLPTP(config, context)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, freqs_cis):
        r = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, freqs_cis)
        x = r + x
        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return r + x

# =============================================================================
# 4. Pipeline Stage (流水线分段模型)
# =============================================================================

class Qwen3PipelineStage(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.config = config
        self.context = context
        
        # 计算当前Stage负责的层
        layers_per_stage = config.num_hidden_layers // context.pp_size
        self.start_layer = context.pp_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage
        
        print(f"[Rank {context.rank}] Initializing Stage {context.pp_rank}, Layers {self.start_layer}-{self.end_layer-1}")

        # 1. Embedding (仅 Stage 0)
        # 注意：这里为了简化，Embedding 没有做 TP (VocabParallel)，实际大模型通常会做
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) if context.pp_rank == 0 else None
        
        # 2. Layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayerTP(config, context) for _ in range(layers_per_stage)
        ])
        
        # 3. Final Norm & Head (仅 Last Stage)
        self.norm = Qwen3RMSNorm(config.hidden_size) if context.pp_rank == context.pp_size - 1 else None
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, tp_group=context.tp_group) if context.pp_rank == context.pp_size - 1 else None

        # 预计算 RoPE (简单起见，所有 rank 都存一份)
        self.register_buffer("inv_freq", 1.0 / (config.rope_theta ** (torch.arange(0, config.hidden_size // config.num_attention_heads, 2).float() / (config.hidden_size // config.num_attention_heads))))

    def get_rope(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, x):
        # Stage 0 输入是 IDs [batch, seq]，其他是 Hidden [batch, seq, dim]
        if self.context.pp_rank == 0:
            x = self.embed_tokens(x)
        
        batch, seq, _ = x.shape
        freqs_cis = self.get_rope(seq, x.device)
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        if self.context.pp_rank == self.context.pp_size - 1:
            x = self.norm(x)
            x = self.lm_head(x) # 输出 Logits (Partial via ColumnParallel)
            
        return x

# =============================================================================
# 5. 执行引擎 (Engine & Simulation)
# =============================================================================

def run_pipeline_step(model, input_ids, context, config):
    """简单的非流水线步进 (Simple Forward Pass)"""
    device = torch.device(f"cuda:{context.rank}")
    
    # --- 1. Receive ---
    if context.pp_rank == 0:
        # First stage takes real data
        curr_input = input_ids.to(device)
    else:
        # Others receive from previous stage
        # 接收 Hidden States: [Batch, Seq, Hidden]
        recv_shape = (2, 32, config.hidden_size) # 假设 Batch=2, Seq=32
        curr_input = torch.zeros(recv_shape, device=device)
        src_rank = context.get_prev_stage_rank()
        dist.recv(curr_input, src=src_rank)
        # print(f"[Rank {context.rank}] Received data from Rank {src_rank}")

    # --- 2. Compute ---
    output = model(curr_input)

    # --- 3. Send / Output ---
    if context.pp_rank == context.pp_size - 1:
        # Last stage outputs result
        # print(f"[Rank {context.rank}] Pipeline Finished. Output shape: {output.shape}")
        return output
    else:
        # Others send to next stage
        dst_rank = context.get_next_stage_rank()
        dist.send(output.contiguous(), dst=dst_rank)
        # print(f"[Rank {context.rank}] Sent data to Rank {dst_rank}")
        return None

def worker_process(rank, world_size, tp_size, pp_size):
    # 环境初始化
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 上下文与模型
    context = ParallelContext(rank, world_size, tp_size, pp_size)
    config = Qwen3Config()
    model = Qwen3PipelineStage(config, context).to(f"cuda:{rank}")
    
    # 模拟输入数据 (只有 Rank 0/1 (Stage 0) 需要，但其实只有 Rank 0 发起数据)
    if rank < tp_size:
        input_ids = torch.randint(0, config.vocab_size, (2, 32)) # Batch=2, Seq=32
    else:
        input_ids = None

    # 运行
    output = run_pipeline_step(model, input_ids, context, config)
    
    # 验证
    if context.pp_rank == context.pp_size - 1:
        # 最后是 TP 分割的 Logits，我们简单打印均值验证数值存在
        print(f"✅ [Rank {rank}] Final Logits Mean: {output.mean().item():.4f}")

    dist.destroy_process_group()

def main():
    TP_SIZE = 2
    PP_SIZE = 2
    WORLD_SIZE = TP_SIZE * PP_SIZE
    
    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"❌ Need at least {WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
        return

    print(f"🚀 Starting distributed simulation: TP={TP_SIZE}, PP={PP_SIZE}, World={WORLD_SIZE}")
    mp.spawn(worker_process,
             args=(WORLD_SIZE, TP_SIZE, PP_SIZE),
             nprocs=WORLD_SIZE,
             join=True)

if __name__ == "__main__":
    main()