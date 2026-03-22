import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

# =============================================================================
# 1. 配置与分布式上下文管理 (Configuration & Distributed Context)
# =============================================================================

class Qwen3Config:
    """模型配置类"""
    def __init__(self):
        self.hidden_size = 512
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.intermediate_size = 2048
        self.vocab_size = 4096
        self.num_hidden_layers = 4
        self.max_position_embeddings = 128
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0


class DistributedContext:
    """
    分布式环境上下文管理器
    负责管理 Rank (进程ID)、World Size (总进程数) 以及 TP/PP 分组。
    """
    def __init__(self, rank, world_size, tp_size, pp_size, sequence_parallel=False):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size  # Tensor Parallel 大小
        self.pp_size = pp_size  # Pipeline Parallel 大小

        # Sequence Parallel (序列并行) 配置
        # 如果开启 SP，Activation 将在 Sequence 维度上切分
        self.sequence_parallel = sequence_parallel
        self.seq_dim = 1  # 默认张量形状 [Batch, Seq, Hidden]，所以在 dim=1 切分

        # 计算当前进程在流水线(PP)和张量并行(TP)中的坐标
        # 假设 4个GPU, TP=2, PP=2:
        # GPU 0,1 -> Stage 0 (TP Group A)
        # GPU 2,3 -> Stage 1 (TP Group B)
        self.pp_rank = rank // tp_size
        self.tp_rank = rank % tp_size

        self.tp_group = None
        self._init_process_groups()

    def _init_process_groups(self):
        """初始化通信组，TP 组内的 GPU 需要频繁通信 (AllReduce/AllGather)"""
        for i in range(self.pp_size):
            # 例如: Stage 0 的 ranks=[0,1], Stage 1 的 ranks=[2,3]
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group

    def get_prev_stage_rank(self):
        """获取流水线上一级对应的 Rank (用于接收激活值)"""
        if self.pp_rank == 0:
            return None
        # 简单的 P2P 映射：Rank 2 接收 Rank 0 的数据，Rank 3 接收 Rank 1 的数据
        return self.rank - self.tp_size

    def get_next_stage_rank(self):
        """获取流水线下一级对应的 Rank (用于发送激活值)"""
        if self.pp_rank == self.pp_size - 1:
            return None
        return self.rank + self.tp_size


# =============================================================================
# 2. Sequence Parallel 通信原语 (SP Communication Primitives)
#    核心思想：在 LayerNorm/Dropout 后将数据在 Seq 维度 Scatter (切分)，
#    在需要全量计算 (如 Attention, MLP) 前 Gather (聚合)。
# =============================================================================

def _permute_dim_to_front(tensor, dim: int):
    """辅助函数：将指定维度移到第0维，方便进行 Scatter/Gather 操作"""
    if dim < 0:
        dim += tensor.dim()
    # 生成排列索引，例如 dim=1, shape=[B, S, H] -> perm=[1, 0, 2] -> shape=[S, B, H]
    perm = [dim] + [i for i in range(tensor.dim()) if i != dim]
    # 生成逆排列索引，用于恢复形状
    inv_perm = [0] * tensor.dim()
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return tensor.permute(perm).contiguous(), perm, inv_perm


def _all_gather_along_dim(tensor, dim: int, group):
    """
    SP 核心操作：Gather
    将切分后的 Tensor 从各个 Rank 收集起来，沿着指定 dim 拼接。
    用于：[Batch, Seq/TP, Hidden] -> [Batch, Seq, Hidden]
    """
    world_size = dist.get_world_size(group=group)
    
    # 1. 把切分维度移到最前面: [Seq/TP, Batch, Hidden]
    tensor_permuted, _, inv_perm = _permute_dim_to_front(tensor, dim)
    
    split_size = tensor_permuted.shape[0]
    rest_shape = tensor_permuted.shape[1:]
    
    # 展平以便通信
    tensor_2d = tensor_permuted.view(split_size, -1)

    # 2. All-Gather
    gather_list = [torch.empty_like(tensor_2d) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_2d, group=group)
    
    # 3. 拼接: [Seq, Batch * Hidden_flat]
    gathered_2d = torch.cat(gather_list, dim=0)

    # 4. 恢复形状: [Seq, Batch, Hidden] -> [Batch, Seq, Hidden]
    gathered_tensor = gathered_2d.view(split_size * world_size, *rest_shape)
    output = gathered_tensor.permute(inv_perm).contiguous()
    return output


def _reduce_scatter_along_dim_sum(tensor, dim: int, group):
    """
    SP 核心操作：Reduce-Scatter
    先对所有 Rank 的 Tensor 求和 (Reduce)，然后沿着 dim 切分 (Scatter) 给各个 Rank。
    用于 RowParallelLinear 的输出聚合：代替 All-Reduce，减少通信量。
    """
    world_size = dist.get_world_size(group=group)
    
    # 1. 移位: [Seq, Batch, Hidden]
    tensor_permuted, _, inv_perm = _permute_dim_to_front(tensor, dim)
    
    full_seq_len = tensor_permuted.shape[0]
    assert full_seq_len % world_size == 0, f"序列长度 {full_seq_len} 必须能被 TP 大小 {world_size} 整除"
    
    shard_seq_len = full_seq_len // world_size
    rest_shape = tensor_permuted.shape[1:]
    
    tensor_2d = tensor_permuted.view(full_seq_len, -1)

    # 2. Reduce-Scatter (Op=SUM)
    output_2d = torch.empty((shard_seq_len, tensor_2d.shape[1]), device=tensor.device, dtype=tensor.dtype)
    
    # 优先使用高效的 reduce_scatter_tensor (PyTorch 新版特性)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(output_2d, tensor_2d, op=dist.ReduceOp.SUM, group=group)
    else:
        # 兼容旧版：先 chunk 再 reduce_scatter
        chunks = list(tensor_2d.chunk(world_size, dim=0))
        dist.reduce_scatter(output_2d, chunks, op=dist.ReduceOp.SUM, group=group)

    # 3. 恢复形状: [Seq/TP, Batch, Hidden] -> [Batch, Seq/TP, Hidden]
    output_tensor = output_2d.view(shard_seq_len, *rest_shape)
    output = output_tensor.permute(inv_perm).contiguous()
    return output


class ScatterToSequenceParallelFunc(torch.autograd.Function):
    """
    自定义 Autograd 函数：Scatter (前向切分，反向聚合)
    Forward: 输入全量 Seq，切分给各 Rank。
    Backward: 各 Rank 梯度 Gather 回全量。
    """
    @staticmethod
    def forward(ctx, input_tensor, dim: int, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

        # 移位并切分
        input_permuted, perm, inv_perm = _permute_dim_to_front(input_tensor, dim)
        full_len = input_permuted.shape[0]
        assert full_len % world_size == 0
        
        part_len = full_len // world_size
        start_idx = rank * part_len
        end_idx = start_idx + part_len
        
        # 取当前 Rank 对应的片段
        sharded_permuted = input_permuted[start_idx:end_idx].contiguous()
        
        # 保存上下文用于反向传播
        ctx.full_len = full_len
        ctx.perm = perm
        ctx.inv_perm = inv_perm

        return sharded_permuted.permute(inv_perm).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：Scatter 的逆操作是 Gather (All-Gather)
        grad_input = _all_gather_along_dim(grad_output, ctx.dim, ctx.group)
        return grad_input, None, None


class GatherFromSequenceParallelFunc(torch.autograd.Function):
    """
    自定义 Autograd 函数：Gather (前向聚合，反向切分)
    Forward: 各 Rank 输入分片，Gather 为全量 Seq。
    Backward: 全量梯度切分 (Scatter) 回各 Rank。
    """
    @staticmethod
    def forward(ctx, sharded_tensor, dim: int, group):
        ctx.dim = dim
        ctx.group = group
        return _all_gather_along_dim(sharded_tensor, dim, group)

    @staticmethod
    def backward(ctx, grad_output_full):
        # 反向传播：Gather 的逆操作是 Scatter (直接切片即可)
        world_size = dist.get_world_size(group=ctx.group)
        rank = dist.get_rank(group=ctx.group)
        
        grad_permuted, _, inv_perm = _permute_dim_to_front(grad_output_full, ctx.dim)
        full_len = grad_permuted.shape[0]
        part_len = full_len // world_size
        
        start = rank * part_len
        end = start + part_len
        
        grad_sharded_permuted = grad_permuted[start:end].contiguous()
        grad_sharded = grad_sharded_permuted.permute(inv_perm).contiguous()
        return grad_sharded, None, None


def scatter_to_sequence_parallel(tensor, dist_ctx: DistributedContext):
    """封装函数：将全量 Tensor 切分为 SP 状态"""
    if (not dist_ctx.sequence_parallel) or dist.get_world_size(group=dist_ctx.tp_group) == 1:
        return tensor
    return ScatterToSequenceParallelFunc.apply(tensor, dist_ctx.seq_dim, dist_ctx.tp_group)


def gather_from_sequence_parallel(sharded_tensor, dist_ctx: DistributedContext):
    """封装函数：将 SP 状态的 Tensor 聚合为全量"""
    if (not dist_ctx.sequence_parallel) or dist.get_world_size(group=dist_ctx.tp_group) == 1:
        return sharded_tensor
    return GatherFromSequenceParallelFunc.apply(sharded_tensor, dist_ctx.seq_dim, dist_ctx.tp_group)


# =============================================================================
# 3. TP 线性层 (Row/Column Parallel Linear)
#    Megatron-LM 风格：
#    - ColumnParallel: 权重按列切分 (输出被切分)
#    - RowParallel: 权重按行切分 (输入被切分，输出需 Reduce)
# =============================================================================

class RowParallelLinearFunc(torch.autograd.Function):
    """RowParallel 核心逻辑：支持 Bias 和 SP 优化"""
    @staticmethod
    def forward(ctx, input_, weight, bias, tp_group, sequence_parallel: bool, seq_dim: int):
        ctx.tp_group = tp_group
        ctx.sequence_parallel = sequence_parallel
        ctx.seq_dim = seq_dim
        ctx.save_for_backward(input_, weight, bias)

        # 1. 本地 MatMul: [..., H/tp] @ [H/tp, Out] -> [..., Out] (此时是部分和)
        output_partial = input_.matmul(weight.t())

        # 2. 聚合结果
        if sequence_parallel:
            # SP 模式优化：不做 All-Reduce，而是 Reduce-Scatter
            # 结果从 [Batch, Seq, Out] 变成 [Batch, Seq/TP, Out]
            output = _reduce_scatter_along_dim_sum(output_partial, seq_dim, tp_group)
        else:
            # 普通 TP 模式：All-Reduce (Sum)
            dist.all_reduce(output_partial, op=dist.ReduceOp.SUM, group=tp_group)
            output = output_partial

        # 3. 加上 Bias
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias = ctx.saved_tensors

        # 如果 Forward 做了 Reduce-Scatter (变小了)，Backward 就要做 All-Gather (变回去)
        if ctx.sequence_parallel:
            grad_output_full = _all_gather_along_dim(grad_output, ctx.seq_dim, ctx.tp_group)
        else:
            grad_output_full = grad_output

        # 计算梯度
        grad_input = grad_output_full.matmul(weight)
        grad_weight = grad_output_full.reshape(-1, grad_output_full.shape[-1]).t().matmul(
            input_.reshape(-1, input_.shape[-1])
        )
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output_full.sum(dim=tuple(range(grad_output_full.dim() - 1)))
            
        return grad_input, grad_weight, grad_bias, None, None, None


class ColumnParallelLinearFunc(torch.autograd.Function):
    """ColumnParallel 核心逻辑：输入复用，输出切分"""
    @staticmethod
    def forward(ctx, input_, weight, bias, tp_group):
        ctx.tp_group = tp_group
        ctx.save_for_backward(input_, weight, bias)
        
        # MatMul: [..., H] @ [Out/tp, H].T -> [..., Out/tp]
        output = input_.matmul(weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias = ctx.saved_tensors
        
        # 反向传播：
        # grad_input 需要聚合所有 Rank 对输入的梯度贡献 -> All-Reduce
        grad_input = grad_output.matmul(weight)
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.tp_group)

        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t().matmul(
            input_.reshape(-1, input_.shape[-1])
        )
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            
        return grad_input, grad_weight, grad_bias, None


def _init_weights_deterministically(weight_part, full_shape, split_dim, start_idx, end_idx, seed=1234):
    """
    确定性初始化：
    为了保证分布式训练一致性，我们生成完整的权重矩阵 (使用相同的 Seed)，
    然后切出当前 Rank 需要的那一部分。
    """
    device = weight_part.device
    with torch.no_grad():
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        
        full_weight = torch.empty(full_shape, device=device, dtype=weight_part.dtype)
        nn.init.xavier_normal_(full_weight, generator=rng)

        # 截取属于当前 Rank 的部分
        if split_dim == 0:  # Row 切分 (用于 ColumnParallel 的权重矩阵形状 [Out, In] 是按 Out 切)
            weight_part.copy_(full_weight[start_idx:end_idx, :])
        else:               # Col 切分 (用于 RowParallel 的权重矩阵形状 [Out, In] 是按 In 切)
            weight_part.copy_(full_weight[:, start_idx:end_idx])


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层
    - 权重形状: [Out_features / TP, In_features]
    - 输出形状: [..., Out_features / TP]
    """
    def __init__(self, in_features, out_features, bias=True, tp_group=None, seed=1234):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)
        self.tp_rank = dist.get_rank(group=tp_group)

        self.out_per_partition = out_features // self.tp_size
        self.out_start = self.tp_rank * self.out_per_partition
        self.out_end = self.out_start + self.out_per_partition

        # 权重切分：切分输出维度 (dim=0)
        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features, device="cuda"))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_partition, device="cuda"))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(seed)

    def reset_parameters(self, seed):
        _init_weights_deterministically(
            self.weight,
            full_shape=(self.out_per_partition * self.tp_size, self.weight.shape[1]),
            split_dim=0,
            start_idx=self.out_start,
            end_idx=self.out_end,
            seed=seed
        )
        # Bias 初始化也需要类似逻辑，略
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return ColumnParallelLinearFunc.apply(x, self.weight, self.bias, self.tp_group)


class RowParallelLinear(nn.Module):
    """
    行并行线性层
    - 权重形状: [Out_features, In_features / TP]
    - 输出形状: [..., Out_features] (SP下为 Scatter 后的形状)
    """
    def __init__(self, in_features, out_features, bias=True, tp_group=None, seed=1234,
                 sequence_parallel=False, seq_dim=1):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)
        self.tp_rank = dist.get_rank(group=tp_group)
        self.sequence_parallel = sequence_parallel
        self.seq_dim = seq_dim

        self.in_per_partition = in_features // self.tp_size
        self.in_start = self.tp_rank * self.in_per_partition
        self.in_end = self.in_start + self.in_per_partition

        # 权重切分：切分输入维度 (dim=1)
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition, device="cuda"))
        if bias:
            # Bias 不切分，每个 Rank 都有完整的 Bias (因为输出被 Reduce 后是完整的)
            self.bias = nn.Parameter(torch.empty(out_features, device="cuda"))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(seed)

    def reset_parameters(self, seed):
        _init_weights_deterministically(
            self.weight,
            full_shape=(self.weight.shape[0], self.in_per_partition * self.tp_size),
            split_dim=1,
            start_idx=self.in_start,
            end_idx=self.in_end,
            seed=seed
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return RowParallelLinearFunc.apply(
            x, self.weight, self.bias, self.tp_group, self.sequence_parallel, self.seq_dim
        )


# =============================================================================
# 4. Vocab Parallel Cross Entropy (词表并行交叉熵)
#    为了节省显存，Logits 层通常不在单卡计算完整的 VocabSize，而是切分计算。
# =============================================================================

class VocabParallelCrossEntropyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits_part, target, tp_group, vocab_start, vocab_end):
        """
        logits_part: [Batch*Seq, VocabSize/TP] (当前 Rank 只负责一部分词表的 Logits)
        target:      [Batch*Seq]
        """
        logits_f = logits_part.float()
        
        # 1. 寻找全局最大值 (用于数值稳定性的 Softmax)
        local_max = logits_f.max(dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

        # 2. 计算分母: sum(exp(x - max))
        logits_shifted = logits_f - global_max.unsqueeze(-1)
        exp_local = logits_shifted.exp()
        sumexp_local = exp_local.sum(dim=-1)
        sumexp_global = sumexp_local.clone()
        dist.all_reduce(sumexp_global, op=dist.ReduceOp.SUM, group=tp_group)
        
        # Log(Sum(Exp))
        log_sum_exp = sumexp_global.log() + global_max

        # 3. 获取目标 Target 对应的 Logit
        # 判断 Target 是否在当前 Rank 负责的 Vocab 范围内
        target_in_range_mask = (target >= vocab_start) & (target < vocab_end)
        
        # 将全局 target 映射到本地索引
        target_local_idx = (target - vocab_start).clamp(min=0)
        
        # Gather 出对应的 logit 值
        gathered_logits = logits_f.gather(dim=-1, index=target_local_idx.unsqueeze(-1)).squeeze(-1)
        gathered_logits = gathered_logits.masked_fill(~target_in_range_mask, 0.0)
        
        # 全局归约，只有负责该 Target 的 Rank 贡献非0值
        true_class_logit = gathered_logits.clone()
        dist.all_reduce(true_class_logit, op=dist.ReduceOp.SUM, group=tp_group)

        # 4. 计算 Loss = log(sum(exp)) - target_logit
        loss = (log_sum_exp - true_class_logit).mean()

        # 5. 保存上下文用于反向传播
        # 计算当前分片的 Softmax 概率: exp(x) / sum(exp)
        softmax_part = exp_local / sumexp_global.unsqueeze(-1)
        
        ctx.tp_group = tp_group
        ctx.vocab_start = vocab_start
        ctx.vocab_end = vocab_end
        ctx.save_for_backward(softmax_part, target, target_in_range_mask)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax_part, target, target_in_range_mask = ctx.saved_tensors
        vocab_start = ctx.vocab_start
        
        # Gradient of CrossEntropy: Softmax(x) - 1 (if x is target)
        grad_logits = softmax_part
        
        # 只有当 Target 落在当前 Rank 范围内时，才需要减 1
        if target_in_range_mask.any():
            local_target_idx = (target[target_in_range_mask] - vocab_start).long()
            row_indices = torch.nonzero(target_in_range_mask, as_tuple=False).squeeze(-1)
            grad_logits[row_indices, local_target_idx] -= 1.0

        # Scale by grad_output and 1/N (since we used mean)
        batch_size = target.numel()
        grad_logits = grad_logits * (grad_output / batch_size)
        
        return grad_logits, None, None, None, None


def vocab_parallel_cross_entropy(logits_part, target, dist_ctx):
    vocab_per_part = logits_part.shape[-1]
    vocab_start = dist_ctx.tp_rank * vocab_per_part
    vocab_end = vocab_start + vocab_per_part
    
    return VocabParallelCrossEntropyFunc.apply(
        logits_part, target, dist_ctx.tp_group, vocab_start, vocab_end
    )


# =============================================================================
# 5. 模型组件 (RoPE, Attention, MLP)
# =============================================================================

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, tp_group=None, sequence_parallel=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device="cuda"))
        self.eps = eps
        self.tp_group = tp_group
        
        # 在 SP 模式下，输入是 Sharded 的，但 LayerNorm 的可学习参数 Weight 是复制的 (Replicated)。
        # 因此，反向传播计算出的 weight 梯度是基于局部数据的，需要 All-Reduce 同步。
        if tp_group is not None and sequence_parallel and dist.get_world_size(group=tp_group) > 1:
            def _grad_sync_hook(grad):
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_group)
                return grad
            self.weight.register_hook(_grad_sync_hook)

    def forward(self, x):
        # x: [Batch, Seq/TP, Hidden] if SP else [Batch, Seq, Hidden]
        # RMSNorm 是逐元素的，不涉及 Seq 维度交互，所以 SP 模式下可以直接算
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    # 简单的复数旋转模拟
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Qwen3AttentionTP(nn.Module):
    def __init__(self, config, dist_ctx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 每个 TP Rank 负责的头数
        self.local_heads = config.num_attention_heads // dist_ctx.tp_size

        # QKV 投影: Column Parallel (输出切分)
        self.q_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=dist_ctx.tp_group, seed=1001)
        self.k_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=dist_ctx.tp_group, seed=1002)
        self.v_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=dist_ctx.tp_group, seed=1003)
        
        # 输出投影: Row Parallel (输入切分，输出聚合)
        # SP 开启时，这里会执行 Reduce-Scatter 而不是 All-Reduce
        self.o_proj = RowParallelLinear(
            config.hidden_size, config.hidden_size, bias=False, tp_group=dist_ctx.tp_group, seed=1004,
            sequence_parallel=dist_ctx.sequence_parallel, seq_dim=dist_ctx.seq_dim
        )

    def forward(self, x, rotary_pos_emb):
        b, s, _ = x.shape
        
        # 1. 投影得到 Q, K, V
        # 结果形状: [Batch, Seq, Local_Heads, Head_Dim]
        q = self.q_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE
        cos, sin = rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Attention 计算 (简化版，未包含 FlashAttn)
        # [B, Local_Heads, Seq, Head_Dim] @ [B, Local_Heads, Head_Dim, Seq] -> [B, Local_Heads, Seq, Seq]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        context_layer = torch.matmul(attn_probs, v)
        
        # 4. 恢复形状: [B, Seq, Hidden/TP]
        context_layer = context_layer.transpose(1, 2).contiguous().view(b, s, -1)
        
        # 5. 输出投影 (RowParallel, 内部处理通信)
        return self.o_proj(context_layer)


class Qwen3MLPTP(nn.Module):
    def __init__(self, config, dist_ctx):
        super().__init__()
        # Gate & Up: Column Parallel
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, tp_group=dist_ctx.tp_group, seed=2001)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, tp_group=dist_ctx.tp_group, seed=2002)
        
        # Down: Row Parallel
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False, tp_group=dist_ctx.tp_group, seed=2003,
            sequence_parallel=dist_ctx.sequence_parallel, seq_dim=dist_ctx.seq_dim
        )

    def forward(self, x):
        # SwiGLU 激活
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """单个 Transformer 解码层"""
    def __init__(self, config, dist_ctx):
        super().__init__()
        self.dist_ctx = dist_ctx
        self.self_attn = Qwen3AttentionTP(config, dist_ctx)
        self.mlp = Qwen3MLPTP(config, dist_ctx)
        
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            tp_group=dist_ctx.tp_group, sequence_parallel=dist_ctx.sequence_parallel
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            tp_group=dist_ctx.tp_group, sequence_parallel=dist_ctx.sequence_parallel
        )

    def forward(self, x, rotary_pos_emb):
        # 输入 x 可能是 SP 切分过的 (Seq/TP) 或者 全量的 (Seq)
        residual = x
        
        # 1. Norm
        x = self.input_layernorm(x)

        # 2. Attention
        # 如果开启 SP，Attention 需要全量 Seq 才能计算 Attention Score (Q @ K.T)
        # 所以先 Gather 收集全量数据
        if self.dist_ctx.sequence_parallel:
            x_full = gather_from_sequence_parallel(x, self.dist_ctx)
            # attn_out 已经是被 RowParallelLinear Reduce-Scatter 过的，所以是 [Seq/TP]
            attn_out = self.self_attn(x_full, rotary_pos_emb)
        else:
            attn_out = self.self_attn(x, rotary_pos_emb)

        # 3. Residual Add
        x = residual + attn_out
        residual = x

        # 4. Norm
        x = self.post_attention_layernorm(x)

        # 5. MLP
        # 同样的，如果 SP 开启，MLP 前需要 Gather (RowParallel 输出是 Scatter 的)
        if self.dist_ctx.sequence_parallel:
            x_full = gather_from_sequence_parallel(x, self.dist_ctx)
            mlp_out = self.mlp(x_full) # 输出再次被 Scatter
        else:
            mlp_out = self.mlp(x)

        return residual + mlp_out


# =============================================================================
# 6. Pipeline Stage (流水线阶段封装)
# =============================================================================

class Qwen3PipelineStage(nn.Module):
    """
    表示流水线中的一个阶段 (Stage)。
    Stage 0: Embedding -> Layers 0~N
    Stage 1...M: Layers ...
    Stage Last: Layers ... -> Norm -> Head
    """
    def __init__(self, config, dist_ctx):
        super().__init__()
        self.config = config
        self.dist_ctx = dist_ctx

        layers_per_stage = config.num_hidden_layers // dist_ctx.pp_size
        self.start_layer = dist_ctx.pp_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage
        
        print(f"[Rank {dist_ctx.rank}] Initializing Pipeline Stage {dist_ctx.pp_rank}, Layers [{self.start_layer}, {self.end_layer})")

        # Embedding 只在 Stage 0
        self.embed_tokens = None
        if dist_ctx.pp_rank == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device="cuda")
            # SP 模式下 Embedding 输出是复制的，需要同步梯度
            if dist_ctx.sequence_parallel and dist.get_world_size(group=dist_ctx.tp_group) > 1:
                def _emb_hook(grad):
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dist_ctx.tp_group)
                    return grad
                self.embed_tokens.weight.register_hook(_emb_hook)

        # Transformer Layers
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, dist_ctx) for _ in range(layers_per_stage)])

        # Final Norm & Head 只在 Last Stage
        self.norm = None
        self.lm_head = None
        if dist_ctx.pp_rank == dist_ctx.pp_size - 1:
            self.norm = Qwen3RMSNorm(
                config.hidden_size, tp_group=dist_ctx.tp_group, sequence_parallel=dist_ctx.sequence_parallel
            )
            # Head 通常是 Column Parallel (输出 Logits 是切分的)
            self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, tp_group=dist_ctx.tp_group, seed=3001)

        # 预计算 RoPE (简化版，未考虑缓存)
        head_dim = config.hidden_size // config.num_attention_heads
        self.register_buffer(
            "inv_freq",
            1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)),
            persistent=False,
        )

    def _get_rope_emb(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, x):
        # 1. 如果是 Stage 0, 处理 Input IDs -> Embedding
        if self.dist_ctx.pp_rank == 0:
            x = self.embed_tokens(x)  # [B, S, H]
            if self.dist_ctx.sequence_parallel:
                x = scatter_to_sequence_parallel(x, self.dist_ctx) # -> [B, S/TP, H]

        # 2. 准备 RoPE (需要知道完整序列长度)
        b = x.shape[0]
        s_local = x.shape[1]
        s_full = s_local * self.dist_ctx.tp_size if self.dist_ctx.sequence_parallel else s_local
        
        rope_emb = self._get_rope_emb(s_full, x.device)

        # 3. 运行 Layers
        for layer in self.layers:
            x = layer(x, rope_emb)

        # 4. 如果是 Last Stage, 处理 Norm -> Head
        if self.dist_ctx.pp_rank == self.dist_ctx.pp_size - 1:
            x = self.norm(x)
            x = self.lm_head(x) # 输出 Logits [B, S/TP, Vocab/TP]
            
        return x


# =============================================================================
# 7. 训练流程：流水线步进 (Pipeline Step)
# =============================================================================

def run_pipeline_step(model, input_ids_batch, dist_ctx, config, batch_size=2, seq_len=32):
    """
    模拟一个 Pipeline Step (Forward + Loss + Backward)
    简单实现：F-F-B-B (1F1B 需要更复杂的调度)
    """
    device = torch.device(f"cuda:{dist_ctx.rank}")
    
    # SP 模式下，SeqLen 被切分
    seq_len_local = seq_len // dist_ctx.tp_size if dist_ctx.sequence_parallel else seq_len

    # --- 1. 数据广播 (Stage 0, TP Rank 0 拥有数据，广播给 TP 组内其他人) ---
    if dist_ctx.pp_rank == 0:
        if dist_ctx.tp_rank == 0:
            input_ids = input_ids_batch.to(device)
        else:
            input_ids = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
        dist.broadcast(input_ids, src=dist_ctx.pp_rank * dist_ctx.tp_size, group=dist_ctx.tp_group)

    # --- 2. 接收上游数据 (Recv from Prev Stage) ---
    if dist_ctx.pp_rank == 0:
        # Stage 0 直接使用 input_ids
        curr_input = input_ids
        # Label 也就是 Input (Shifted later)
        labels_full = input_ids
        
        # SP: 需要把 Labels 也切分，以便最后算 Loss
        if dist_ctx.sequence_parallel:
            labels = scatter_to_sequence_parallel(labels_full, dist_ctx)
        else:
            labels = labels_full
    else:
        # 中间/最后 Stage: 接收 Activation 和 Labels
        recv_shape = (batch_size, seq_len_local, config.hidden_size)
        curr_input = torch.empty(recv_shape, device=device, dtype=torch.float32)
        labels = torch.empty((batch_size, seq_len_local), device=device, dtype=torch.long) # 简化：Label也传递

        src_rank = dist_ctx.get_prev_stage_rank()
        dist.recv(curr_input, src=src_rank)
        dist.recv(labels, src=src_rank)
        
        curr_input.requires_grad_(True) # 需要对输入求导以传给上游

    # --- 3. 前向计算 (Forward) ---
    if dist_ctx.pp_rank != dist_ctx.pp_size - 1:
        # 非 Last Stage
        output = model(curr_input)
        
        dst_rank = dist_ctx.get_next_stage_rank()
        # 发送 Activation 给下游
        dist.send(output.detach().contiguous(), dst=dst_rank)
        # 传递 Labels (简化处理，实际训练可能只在最后加载 Label)
        dist.send(labels.contiguous(), dst=dst_rank)

        # 等待下游回传梯度 (Backward)
        grad_output = torch.empty_like(output, device=device)
        dist.recv(grad_output, src=dst_rank)
        
        # 反向传播
        torch.autograd.backward(output, grad_output)
        
        # 如果不是 Stage 0，继续回传梯度给上游
        if dist_ctx.pp_rank != 0:
            prev_rank = dist_ctx.get_prev_stage_rank()
            dist.send(curr_input.grad.contiguous(), dst=prev_rank)
            
        return None

    else:
        # Last Stage: 计算 Loss
        logits_part = model(curr_input) # [B, S_local, V_part]

        # 为了计算 Loss，我们需要对齐 Logits 和 Labels
        # 如果是 SP，Gather Logits 和 Labels 回全量 Seq 维度
        # (注：这里可以选择直接在分片上算，但标准 CrossEntropy 需要对齐)
        # 这里演示更高效的 VocabParallelCrossEntropy
        
        if dist_ctx.sequence_parallel:
            # 需要先把 Seq 维度聚合，因为 VocabParallelCrossEntropy 通常接受 [N, V]
            # 这里为了演示，我们先 Gather Seq，保留 Vocab 切分
            logits_part_full_seq = gather_from_sequence_parallel(logits_part, dist_ctx) # [B, S, V_part]
            labels_full_seq = gather_from_sequence_parallel(labels, dist_ctx)           # [B, S]
        else:
            logits_part_full_seq = logits_part
            labels_full_seq = labels

        # Shift Logits/Labels (Next Token Prediction)
        shift_logits = logits_part_full_seq[:, :-1, :].contiguous()
        shift_labels = labels_full_seq[:, 1:].contiguous()
        
        N = shift_labels.numel()
        loss = vocab_parallel_cross_entropy(
            shift_logits.view(N, -1),
            shift_labels.view(N),
            dist_ctx
        )

        # 反向传播
        loss.backward()

        # 将 Input 的梯度回传给上游 Stage
        prev_rank = dist_ctx.get_prev_stage_rank()
        dist.send(curr_input.grad.contiguous(), dst=prev_rank)

        return loss.detach()


def worker_process(rank, world_size, tp_size, pp_size, sequence_parallel=True):
    """工作进程入口"""
    # 简单的单机多卡环境设置
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 1. 初始化上下文
    dist_ctx = DistributedContext(rank, world_size, tp_size, pp_size, sequence_parallel)
    config = Qwen3Config()
    
    # 2. 初始化模型 (只初始化当前 Stage 的层)
    model = Qwen3PipelineStage(config, dist_ctx).to(f"cuda:{rank}")
    model.train()

    # 3. 准备数据 (只有 Stage 0 的 TP Rank 0 生成数据)
    batch, seq = 2, 32
    if dist_ctx.pp_rank == 0 and dist_ctx.tp_rank == 0:
        input_ids = torch.randint(0, config.vocab_size, (batch, seq), device=f"cuda:{rank}")
    else:
        input_ids = None

    # 4. 运行一步
    loss = run_pipeline_step(model, input_ids, dist_ctx, config, batch_size=batch, seq_len=seq)

    # 5. 打印结果
    if dist_ctx.pp_rank == dist_ctx.pp_size - 1 and dist_ctx.tp_rank == 0:
        print(f"✅ [Rank {rank}] Training Step Complete. Loss: {loss.item():.6f}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    TP_SIZE = 2
    PP_SIZE = 2
    WORLD_SIZE = TP_SIZE * PP_SIZE
    SEQUENCE_PARALLEL = True 

    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"❌ 错误: 需要至少 {WORLD_SIZE} 个 GPU, 但只发现了 {torch.cuda.device_count()} 个。")
        return

    print(f"🚀 启动训练: TP={TP_SIZE}, PP={PP_SIZE}, SP={SEQUENCE_PARALLEL}, Total GPUs={WORLD_SIZE}")
    mp.spawn(
        worker_process,
        args=(WORLD_SIZE, TP_SIZE, PP_SIZE, SEQUENCE_PARALLEL),
        nprocs=WORLD_SIZE,
        join=True
    )


if __name__ == "__main__":
    main()