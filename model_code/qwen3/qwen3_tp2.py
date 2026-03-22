import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

# =============================================================================
# 1. 配置与并行上下文 (Pure TP Version)
# =============================================================================

class Qwen3Config:
    def __init__(self):
        self.hidden_size = 64          # 演示用小维度
        self.num_attention_heads = 4   # TP=2时，每卡2个头
        self.num_key_value_heads = 4
        self.intermediate_size = 256
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        self.max_position_embeddings = 128

class ParallelContext:
    """
    纯 TP 上下文管理
    """
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = world_size # 在纯 TP 模式下，TP size 等于 World size
        
        # 在纯 TP 模式下，所有 Rank 都在同一个通信组
        # 实际代码中通常会创建一个 explicit group 以便扩展(如配合 DP)
        # 这里我们将所有 rank [0, 1, ... world_size-1] 作为一个组
        self.tp_group = dist.new_group(list(range(world_size)))
        self.tp_rank = rank # 在这个组内的 rank 就是全局 rank

# =============================================================================
# 2. 基础 TP 线性层 (Megatron Style)
# =============================================================================

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, context=None):
        super().__init__()
        self.tp_group = context.tp_group
        self.tp_size = context.tp_size
        
        # Column Parallel: 切分输出维度
        self.out_per_part = out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.out_per_part, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_part))
        else:
            self.register_parameter('bias', None)
            
        # 初始化 (这里需要确保所有卡上的初始化是同步的，简单起见固定种子)
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None: nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in] (Full) -> out: [batch, seq, out_part] (Partial)
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, context=None):
        super().__init__()
        self.tp_group = context.tp_group
        self.tp_size = context.tp_size
        
        # Row Parallel: 切分输入维度
        self.in_per_part = in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_part))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        torch.manual_seed(42)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None: nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in_part] (Partial)
        output = F.linear(x, self.weight)
        
        # 核心：All-Reduce 求和，恢复完整结果
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        
        if self.bias is not None:
            output = output + self.bias
        return output

# =============================================================================
# 3. 辅助组件 (Norm, RoPE)
# =============================================================================

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, heads, seq, dim]
    # cos, sin: [1, 1, seq, dim]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# =============================================================================
# 4. Qwen3 模型组件 (Pure TP)
# =============================================================================

class Qwen3AttentionTP(nn.Module):
    def __init__(self, config, context, layer_idx=0):
        super().__init__()
        self.config = config
        self.context = context
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.tp_size = context.tp_size
        # 确保头数能整除
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = self.num_heads // self.tp_size
        self.local_num_kv_heads = self.num_kv_heads // self.tp_size
        
        # Column Parallel: Q, K, V
        self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.head_dim, bias=True, context=context)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True, context=context)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True, context=context)
        
        # Row Parallel: Output
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, context=context)
        
        # QK Norm (Local)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Proj
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.local_num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)
        
        # 2. Norm & Transpose
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # 3. RoPE (Local)
        cos, sin = freqs_cis
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 4. Repeat KV (if GQA)
        if self.local_num_kv_heads != self.local_num_heads:
            key_states = repeat_kv(key_states, self.local_num_heads // self.local_num_kv_heads)
            value_states = repeat_kv(value_states, self.local_num_heads // self.local_num_kv_heads)

        # 5. Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None: 
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.local_num_heads * self.head_dim)
        
        # 6. Output (All-Reduce)
        return self.o_proj(attn_output)

class Qwen3MLPTP(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        # Gate & Up: Column Parallel
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, context=context)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, context=context)
        
        # Down: Row Parallel
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, context=context)
        self.act_fn = F.silu

    def forward(self, x):
        # Gate/Up (Partial) -> Silu/Mul (Local) -> Down (All-Reduce Full)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, context, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3AttentionTP(config, context, layer_idx)
        self.mlp = Qwen3MLPTP(config, context)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, freqs_cis=freqs_cis, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

# =============================================================================
# 5. 模拟运行 (Simulation)
# =============================================================================

def get_rope_embeddings(seq_len, head_dim, device, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def worker(rank, world_size):
    # 1. Init
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 2. Context (TP = World Size)
    context = ParallelContext(rank, world_size)
    config = Qwen3Config()

    # 3. Model
    layer = Qwen3DecoderLayer(config, context, layer_idx=0).to(device)

    # 4. Input
    batch, seq = 2, 8
    hidden_states = torch.randn(batch, seq, config.hidden_size, device=device)
    head_dim = config.hidden_size // config.num_attention_heads
    cos, sin = get_rope_embeddings(seq, head_dim, device)

    # 5. Forward
    output = layer(hidden_states, freqs_cis=(cos, sin))

    # 6. Verify (Check sync)
    print(f"[Rank {rank}] Output Mean: {output.mean().item():.5f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 2 # 假设有 2 张卡做 TP
    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"Need {WORLD_SIZE} GPUs for TP simulation.")
    else:
        print(f"🚀 Starting TP-Only Simulation (TP={WORLD_SIZE})...")
        mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)