import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import math
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# 0. 环境设置与模拟 (与你提供的环境一致)
# -----------------------------------------------------------------------------
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'  # 你可以改为 >1 来测试真实多卡，但在单进程脚本中模拟需要 spawn
# 为了演示逻辑，这里我们假设就在 Rank 0 上跑，但代码逻辑是通用的分布式逻辑。
# 如果想真测试，请使用 torch.multiprocessing.spawn

if not dist.is_initialized():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

device = torch.device(f"cuda:{os.environ['RANK']}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(42)

# -----------------------------------------------------------------------------
# 1. 基础 TP 线性层 (Megatron Style)
# -----------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    输入: [batch, seq, in_dim]
    权重: [out_dim // world_size, in_dim] (按列切分)
    输出: [batch, seq, out_dim // world_size] (保持切分状态，不 Gather)
    """
    def __init__(self, input_size, output_size, bias=False):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.output_size_per_partition = output_size // self.world_size

        # 初始化权重片段
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, device=device))
        else:
            self.register_parameter('bias', None)
        
        # 初始化 (这里为了演示简单随机初始化，实际会从完整模型切分加载)
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # x: [batch, seq, in_dim] (每个 rank 都有完整的 x)
        output = F.linear(x, self.weight, self.bias)
        # output: [batch, seq, out_dim / world_size]
        return output

    def load_from_full_layer(self, full_layer):
        # 辅助函数：从完整层切分权重
        full_w = full_layer.weight.data
        start = self.rank * self.output_size_per_partition
        end = (self.rank + 1) * self.output_size_per_partition
        self.weight.data.copy_(full_w[start:end, :])
        
        if self.bias is not None and full_layer.bias is not None:
            full_b = full_layer.bias.data
            self.bias.data.copy_(full_b[start:end])

class RowParallelLinear(nn.Module):
    """
    输入: [batch, seq, in_dim // world_size] (接收切分后的输入)
    权重: [out_dim, in_dim // world_size] (按行切分)
    输出: [batch, seq, out_dim] (经过 All-Reduce，恢复完整)
    """
    def __init__(self, input_size, output_size, bias=False):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.input_size_per_partition = input_size // self.world_size

        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, device=device))
        else:
            self.register_parameter('bias', None)
            
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # x: [batch, seq, in_dim / world_size]
        output_parallel = F.linear(x, self.weight)
        # output_parallel: [batch, seq, out_dim] (Partial Sum)
        
        # All-Reduce: 将所有卡的结果相加
        if self.world_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
        
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel

    def load_from_full_layer(self, full_layer):
        full_w = full_layer.weight.data
        start = self.rank * self.input_size_per_partition
        end = (self.rank + 1) * self.input_size_per_partition
        self.weight.data.copy_(full_w[:, start:end])
        
        # Bias 不切分，通常只需要在 All-Reduce 之后加上即可
        # 或者是每个 rank 加 bias / world_size 然后 reduce。
        # 这里为了简单，我们假设 bias 只由 rank 0 处理或者在 reduce 后处理。
        # 标准做法：RowLinear 的 bias 不需要切分，它是加在最终 reduce 后的结果上的。
        if self.bias is not None and full_layer.bias is not None:
            self.bias.data.copy_(full_layer.bias.data)

# -----------------------------------------------------------------------------
# 2. 改造后的 Transformer 组件 (Attention & MLP)
# -----------------------------------------------------------------------------

# 模拟配置类
class Qwen3Config:
    def __init__(self):
        self.hidden_size = 64  # 示例小参数
        self.num_attention_heads = 4
        self.num_key_value_heads = 4 # 假设 GQA/MQA 没有开启，或者简单处理
        self.intermediate_size = 128
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-6
        self.attention_dropout = 0.0
        self.max_position_embeddings = 1024
        self.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        self.num_hidden_layers = 1
        self.layer_types = ["default"]
        self._attn_implementation = "eager"

# 复用你提供的 RoPE 代码 (略微简化引用)
from torch import nn
# (这里假设上面你提供的 Qwen3RotaryEmbedding, apply_rotary_pos_emb, Qwen3RMSNorm 已经定义好了)
# 为了代码可运行，我这里简单 mock 一下缺失的依赖或直接使用你的类名

class Qwen3MLPTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Gate 和 Up 是 Column Parallel
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        
        # Down 是 Row Parallel
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        # x: [batch, seq, hidden] (Full)
        
        # gate_buf 和 up_buf 都是切分后的 [batch, seq, inter_dim / world_size]
        gate_buf = self.gate_proj(x)
        up_buf = self.up_proj(x)
        
        # 逐元素操作，无需通信，直接在本地切片上做
        intermediate = self.act_fn(gate_buf) * up_buf
        
        # Row Parallel 内部会做 All-Reduce，输出恢复为 [batch, seq, hidden]
        down_proj = self.down_proj(intermediate)
        return down_proj

class Qwen3AttentionTP(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.world_size = dist.get_world_size()
        
        # 逻辑上的总头数
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        
        # 本地由当前 GPU 负责的头数
        self.num_heads = self.total_num_heads // self.world_size
        self.num_kv_heads = self.total_num_kv_heads // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.scaling = self.head_dim ** -0.5

        # Q, K, V 使用 Column Parallel
        # 输出维度缩小为 local_heads * head_dim
        self.q_proj = ColumnParallelLinear(self.hidden_size, self.total_num_heads * self.head_dim, bias=True)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.total_num_kv_heads * self.head_dim, bias=True)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.total_num_kv_heads * self.head_dim, bias=True)
        
        # Output 使用 Row Parallel
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Norms (假设作用在 head_dim 上，需要在 forward 里 reshape)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 投影 (Column Parallel)
        # 结果形状: [batch, seq, local_heads * head_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape 为 [batch, seq, local_heads, head_dim] 并转置为 [batch, local_heads, seq, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 应用 Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # 转置用于 Attention 计算
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # 3. RoPE (本地计算即可，因为 Position ID 是一样的，sin/cos 也是一样的)
        cos, sin = position_embeddings
        # 需要确保 apply_rotary_pos_emb 兼容切分后的 heads。通常 cos/sin 是 [seq, head_dim] 或者广播的，
        # 只要维度匹配，TP 下本地做 RoPE 是安全的。
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 4. Attention 计算 (Scaled Dot Product Attention)
        # 如果是 GQA (Group Query Attention)，可能需要 repeat_kv，这里假设 num_heads == num_kv_heads 简化
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # 5. 准备输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Flatten: [batch, seq, local_heads * head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # 6. 输出投影 (Row Parallel + All-Reduce)
        # 这里会将各个 rank 的部分结果聚合
        output = self.o_proj(attn_output)
        
        return output

class Qwen3DecoderLayerTP(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3AttentionTP(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLPTP(config)
        
        # RMSNorm 通常不切分，每个 Rank 保存完整的权重，输入也是完整的，输出也是完整的
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        residual = hidden_states
        
        # 1. Norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # 2. Attention TP
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        # 3. MLP TP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def load_weights_from_original(self, original_layer):
        """
        核心工具：将单机模型的权重切分并加载到 TP 模型中
        """
        with torch.no_grad():
            # Copy Norms (直接复制，不切分)
            self.input_layernorm.weight.copy_(original_layer.input_layernorm.weight)
            self.post_attention_layernorm.weight.copy_(original_layer.post_attention_layernorm.weight)
            
            # Copy Attention
            # Q, K, V (Column Parallel)
            self.self_attn.q_proj.load_from_full_layer(original_layer.self_attn.q_proj)
            self.self_attn.k_proj.load_from_full_layer(original_layer.self_attn.k_proj)
            self.self_attn.v_proj.load_from_full_layer(original_layer.self_attn.v_proj)
            # Norms inside Attention
            self.self_attn.q_norm.weight.copy_(original_layer.self_attn.q_norm.weight)
            self.self_attn.k_norm.weight.copy_(original_layer.self_attn.k_norm.weight)
            # O Proj (Row Parallel)
            self.self_attn.o_proj.load_from_full_layer(original_layer.self_attn.o_proj)

            # Copy MLP
            # Gate, Up (Column Parallel)
            self.mlp.gate_proj.load_from_full_layer(original_layer.mlp.gate_proj)
            self.mlp.up_proj.load_from_full_layer(original_layer.mlp.up_proj)
            # Down (Row Parallel)
            self.mlp.down_proj.load_from_full_layer(original_layer.mlp.down_proj)

# -----------------------------------------------------------------------------
# 3. 验证脚本
# -----------------------------------------------------------------------------

def test_tp_correctness():
    # 1. 配置
    config = Qwen3Config()
    bsz, seq_len = 2, 8
    
    # 2. 创建原始单机模型
    # 注意：需要你原代码中的 Qwen3DecoderLayer 定义（假设已存在）
    original_layer = Qwen3DecoderLayer(config, layer_idx=0).to(device)
    original_layer.eval()
    
    # 3. 创建 TP 模型
    tp_layer = Qwen3DecoderLayerTP(config, layer_idx=0).to(device)
    tp_layer.eval()
    
    # 4. 加载权重：从 Original -> TP
    tp_layer.load_weights_from_original(original_layer)
    
    # 5. 构造输入
    input_ids = torch.randint(0, 100, (bsz, seq_len)).to(device)
    # 简单的 Embedding 模拟
    hidden_states = torch.randn(bsz, seq_len, config.hidden_size).to(device)
    
    # 构造 Pos Embeddings
    rotary_emb = Qwen3RotaryEmbedding(config).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    cos, sin = rotary_emb(hidden_states, position_ids)
    
    # 6. 运行前向传播
    with torch.no_grad():
        # 原始模型输出
        # 注意: 原始 Qwen3DecoderLayer 接收 position_embeddings 参数
        orig_out = original_layer(
            hidden_states=hidden_states.clone(),
            position_embeddings=(cos, sin)
        )
        
        # TP 模型输出
        tp_out = tp_layer(
            hidden_states=hidden_states.clone(),
            position_embeddings=(cos, sin)
        )

    # 7. 对比结果
    print(f"Rank {os.environ['RANK']} Check:")
    print(f"Original Output Mean: {orig_out.mean().item():.4f}")
    print(f"TP Output Mean:       {tp_out.mean().item():.4f}")
    
    try:
        torch.testing.assert_close(orig_out, tp_out, rtol=1e-5, atol=1e-5)
        print("✅ TP Implementation matches Original Layer exactly!")
    except AssertionError as e:
        print("❌ Mismatch found!")
        print(e)

# 为了让这段代码在你的环境中跑起来，你需要确保 Qwen3DecoderLayer, Qwen3Attention 等
# 原始类在上下文中是可用的。
if __name__ == "__main__":
    # 假设所有 Qwen3 相关类已经在上面定义了
    test_tp_correctness()