import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

# =============================================================================
# 1. 配置与上下文管理
# =============================================================================

class Qwen3Config:
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

class ParallelContext:
    def __init__(self, rank, world_size, tp_size, pp_size):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size

        # Rank 0,1 -> Stage 0 | Rank 2,3 -> Stage 1
        self.pp_rank = rank // tp_size
        self.tp_rank = rank % tp_size

        self.tp_group = None
        self.init_groups()

    def init_groups(self):
        for i in range(self.pp_size):
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group

    def get_prev_stage_rank(self):
        if self.pp_rank == 0:
            return None
        return self.rank - self.tp_size

    def get_next_stage_rank(self):
        if self.pp_rank == self.pp_size - 1:
            return None
        return self.rank + self.tp_size

# =============================================================================
# 2. TP 通信正确的 Linear (Megatron-style)
#    - RowParallelLinear: forward all_reduce(output) 必须参与 autograd
#    - ColumnParallelLinear: backward all_reduce(dX) 必须做
# =============================================================================

class _RowParallelLinearNoBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, tp_group):
        # x: [*, in_part], weight: [out, in_part]
        ctx.tp_group = tp_group
        ctx.save_for_backward(x, weight)
        out = x.matmul(weight.t())
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)  # ✅ forward sum
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        # grad_out 已经是 full（因为 forward all_reduce 之后各 rank 一致）
        grad_x = grad_out.matmul(weight)  # [*, in_part]
        grad_w = grad_out.reshape(-1, grad_out.shape[-1]).t().matmul(
            x.reshape(-1, x.shape[-1])
        )
        return grad_x, grad_w, None

class _RowParallelLinearBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, tp_group):
        ctx.tp_group = tp_group
        ctx.save_for_backward(x, weight, bias)
        out = x.matmul(weight.t())
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)  # ✅ forward sum
        out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_out.matmul(weight)
        grad_w = grad_out.reshape(-1, grad_out.shape[-1]).t().matmul(
            x.reshape(-1, x.shape[-1])
        )
        grad_b = grad_out.sum(dim=tuple(range(grad_out.dim() - 1)))
        return grad_x, grad_w, grad_b, None

class _ColumnParallelLinearNoBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, tp_group):
        # x: [*, in_full], weight: [out_part, in_full]
        ctx.tp_group = tp_group
        ctx.save_for_backward(x, weight)
        out = x.matmul(weight.t())
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        # 每个 rank 只拥有部分 out，因此 dX 需要 sum across ranks
        grad_x = grad_out.matmul(weight)  # [*, in_full] partial contribution
        dist.all_reduce(grad_x, op=dist.ReduceOp.SUM, group=ctx.tp_group)  # ✅ backward sum(dX)

        grad_w = grad_out.reshape(-1, grad_out.shape[-1]).t().matmul(
            x.reshape(-1, x.shape[-1])
        )
        return grad_x, grad_w, None

class _ColumnParallelLinearBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, tp_group):
        ctx.tp_group = tp_group
        ctx.save_for_backward(x, weight, bias)
        out = x.matmul(weight.t()) + bias
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_out.matmul(weight)
        dist.all_reduce(grad_x, op=dist.ReduceOp.SUM, group=ctx.tp_group)  # ✅ backward sum(dX)

        grad_w = grad_out.reshape(-1, grad_out.shape[-1]).t().matmul(
            x.reshape(-1, x.shape[-1])
        )
        grad_b = grad_out.sum(dim=tuple(range(grad_out.dim() - 1)))
        return grad_x, grad_w, grad_b, None

def _init_full_then_slice(weight_part, full_shape, slice_dim, start, end, seed=1234, init="xavier"):
    """
    为了让 TP 切分真正等价于一个“全矩阵再切片”，这里用相同 seed 生成 full_weight，然后按 slice 取子块。
    slice_dim=0 表示切 rows（ColumnParallel），slice_dim=1 表示切 cols（RowParallel）。
    """
    device = weight_part.device
    dtype = weight_part.dtype
    with torch.no_grad():
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        full = torch.empty(full_shape, device=device, dtype=dtype)
        if init == "xavier":
            nn.init.xavier_normal_(full, generator=g)
        else:
            nn.init.normal_(full, generator=g)

        if slice_dim == 0:
            weight_part.copy_(full[start:end, :])
        else:
            weight_part.copy_(full[:, start:end])

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tp_group=None, seed=1234):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)
        self.tp_rank = dist.get_rank(group=tp_group)

        assert out_features % self.tp_size == 0
        self.out_per_partition = out_features // self.tp_size
        self.out_start = self.tp_rank * self.out_per_partition
        self.out_end = self.out_start + self.out_per_partition

        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features, device="cuda"))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_partition, device="cuda"))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(seed=seed)

    def reset_parameters(self, seed=1234):
        _init_full_then_slice(
            self.weight,
            full_shape=(self.out_per_partition * self.tp_size, self.weight.shape[1]),
            slice_dim=0,
            start=self.out_start,
            end=self.out_end,
            seed=seed,
            init="xavier"
        )
        if self.bias is not None:
            # bias 同样按 out 维切分
            with torch.no_grad():
                g = torch.Generator(device=self.bias.device)
                g.manual_seed(seed + 999)
                full_b = torch.zeros(self.out_per_partition * self.tp_size, device=self.bias.device, dtype=self.bias.dtype)
                self.bias.copy_(full_b[self.out_start:self.out_end])

    def forward(self, x):
        if self.bias is None:
            return _ColumnParallelLinearNoBiasFn.apply(x, self.weight, self.tp_group)
        return _ColumnParallelLinearBiasFn.apply(x, self.weight, self.bias, self.tp_group)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tp_group=None, seed=1234):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)
        self.tp_rank = dist.get_rank(group=tp_group)

        assert in_features % self.tp_size == 0
        self.in_per_partition = in_features // self.tp_size
        self.in_start = self.tp_rank * self.in_per_partition
        self.in_end = self.in_start + self.in_per_partition

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition, device="cuda"))
        if bias:
            # RowParallel 的 bias 是 replicated（每个 rank 一份 full bias）
            self.bias = nn.Parameter(torch.empty(out_features, device="cuda"))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(seed=seed)

    def reset_parameters(self, seed=1234):
        _init_full_then_slice(
            self.weight,
            full_shape=(self.weight.shape[0], self.in_per_partition * self.tp_size),
            slice_dim=1,
            start=self.in_start,
            end=self.in_end,
            seed=seed,
            init="xavier"
        )
        if self.bias is not None:
            with torch.no_grad():
                g = torch.Generator(device=self.bias.device)
                g.manual_seed(seed + 777)
                nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.bias is None:
            return _RowParallelLinearNoBiasFn.apply(x, self.weight, self.tp_group)
        return _RowParallelLinearBiasFn.apply(x, self.weight, self.bias, self.tp_group)

# =============================================================================
# 3. VocabParallelCrossEntropy（logits vocab 切分）
# =============================================================================

class _VocabParallelCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits_part, target, tp_group, vocab_start, vocab_end, ignore_index):
        """
        logits_part: [N, Vp] (Vp = vocab_size/tp)
        target:      [N]
        """
        assert logits_part.dim() == 2
        N, Vp = logits_part.shape
        device = logits_part.device
        dtype = logits_part.dtype

        # float32 for stability
        logits_f = logits_part.float()

        # 1) global max
        local_max = logits_f.max(dim=-1).values  # [N]
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

        # 2) global sum exp
        logits_shifted = logits_f - global_max.unsqueeze(-1)  # [N, Vp]
        exp_local = logits_shifted.exp()
        sumexp_local = exp_local.sum(dim=-1)  # [N]
        sumexp_global = sumexp_local.clone()
        dist.all_reduce(sumexp_global, op=dist.ReduceOp.SUM, group=tp_group)

        log_denom = sumexp_global.log() + global_max  # [N]

        # 3) pick target logit from the rank that owns it
        if ignore_index is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool, device=device)
        else:
            valid_mask = target.ne(ignore_index)

        in_range = (target >= vocab_start) & (target < vocab_end) & valid_mask
        local_t = (target - vocab_start).clamp(min=0)  # [N]
        gathered = logits_f.gather(dim=-1, index=local_t.unsqueeze(-1)).squeeze(-1)  # [N]
        gathered = gathered.masked_fill(~in_range, 0.0)

        target_logit = gathered.clone()
        dist.all_reduce(target_logit, op=dist.ReduceOp.SUM, group=tp_group)  # [N] now global target logit

        loss_vec = (log_denom - target_logit)  # [N]
        loss_vec = loss_vec.masked_fill(~valid_mask, 0.0)

        denom = valid_mask.sum().clamp(min=1).to(loss_vec.dtype)
        loss = loss_vec.sum() / denom

        # softmax part for backward
        softmax_part = exp_local / sumexp_global.unsqueeze(-1)  # [N, Vp]

        ctx.tp_group = tp_group
        ctx.vocab_start = vocab_start
        ctx.vocab_end = vocab_end
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(softmax_part.to(dtype), target, valid_mask, denom.to(dtype))

        return loss

    @staticmethod
    def backward(ctx, grad_out):
        softmax_part, target, valid_mask, denom = ctx.saved_tensors
        vocab_start = ctx.vocab_start
        vocab_end = ctx.vocab_end
        device = softmax_part.device

        grad_logits = softmax_part  # [N, Vp]
        in_range = (target >= vocab_start) & (target < vocab_end) & valid_mask
        if in_range.any():
            local_t = (target[in_range] - vocab_start).to(torch.long)
            rows = torch.nonzero(in_range, as_tuple=False).squeeze(-1)
            grad_logits[rows, local_t] -= 1.0

        # ignore positions
        grad_logits = grad_logits * valid_mask.to(grad_logits.dtype).unsqueeze(-1)

        # mean reduction
        grad_logits = grad_logits * (grad_out / denom).to(grad_logits.dtype)

        # grads for non-tensor args are None
        return grad_logits, None, None, None, None, None

def vocab_parallel_cross_entropy(logits_part_2d, target_1d, context, ignore_index=None):
    # logits_part_2d: [N, vocab_per_part]
    vocab_per_part = logits_part_2d.shape[-1]
    vocab_start = context.tp_rank * vocab_per_part
    vocab_end = vocab_start + vocab_per_part
    return _VocabParallelCrossEntropyFn.apply(
        logits_part_2d, target_1d, context.tp_group, vocab_start, vocab_end, ignore_index
    )

# =============================================================================
# 4. 模型组件 (RoPE, Norm, MLP, Attention)
# =============================================================================

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device="cuda"))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class Qwen3AttentionTP(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.local_heads = config.num_attention_heads // context.tp_size

        self.q_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=context.tp_group, seed=1001)
        self.k_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=context.tp_group, seed=1002)
        self.v_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, bias=True, tp_group=context.tp_group, seed=1003)
        self.o_proj = RowParallelLinear(config.hidden_size, config.hidden_size, bias=False, tp_group=context.tp_group, seed=1004)

    def forward(self, x, freqs_cis):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)

        cos, sin = freqs_cis
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)  # [b,s,hidden/tp]
        return self.o_proj(out)  # -> all_reduce in forward => full hidden

class Qwen3MLPTP(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, tp_group=context.tp_group, seed=2001)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, tp_group=context.tp_group, seed=2002)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, tp_group=context.tp_group, seed=2003)

    def forward(self, x):
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
# 5. Pipeline Stage
# =============================================================================

class Qwen3PipelineStage(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.config = config
        self.context = context

        layers_per_stage = config.num_hidden_layers // context.pp_size
        self.start_layer = context.pp_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage

        print(f"[Rank {context.rank}] Init Stage {context.pp_rank}, Layers {self.start_layer}-{self.end_layer-1}")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device="cuda") if context.pp_rank == 0 else None

        self.layers = nn.ModuleList([Qwen3DecoderLayerTP(config, context) for _ in range(layers_per_stage)])

        self.norm = Qwen3RMSNorm(config.hidden_size) if context.pp_rank == context.pp_size - 1 else None
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, tp_group=context.tp_group, seed=3001) \
            if context.pp_rank == context.pp_size - 1 else None

        head_dim = config.hidden_size // config.num_attention_heads
        self.register_buffer(
            "inv_freq",
            1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)),
            persistent=False,
        )

    def get_rope(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, x):
        if self.context.pp_rank == 0:
            x = self.embed_tokens(x)

        b, s, _ = x.shape
        freqs_cis = self.get_rope(s, x.device)

        for layer in self.layers:
            x = layer(x, freqs_cis)

        if self.context.pp_rank == self.context.pp_size - 1:
            x = self.norm(x)
            x = self.lm_head(x)  # [b,s,vocab_part]
        return x

# =============================================================================
# 6. 执行：forward + loss + backward（含 PP 显式梯度回传）
# =============================================================================

def run_pipeline_step(model, input_ids, context, config, batch=2, seq=32):
    device = torch.device(f"cuda:{context.rank}")

    # -------- Stage 0: broadcast input_ids within TP group --------
    if context.pp_rank == 0:
        if context.tp_rank == 0:
            input_ids = input_ids.to(device)
        else:
            input_ids = torch.empty((batch, seq), device=device, dtype=torch.long)

        # ✅ 修复你原代码的关键问题：确保同一 stage 的 TP ranks 输入一致
        dist.broadcast(input_ids, src=context.pp_rank * context.tp_size, group=context.tp_group)

    # -------- Forward receive activation --------
    if context.pp_rank == 0:
        curr_input = input_ids  # [b,s]
        labels = input_ids      # causal LM labels
    else:
        recv_shape = (batch, seq, config.hidden_size)
        curr_input = torch.empty(recv_shape, device=device, dtype=torch.float32)
        src_rank = context.get_prev_stage_rank()
        dist.recv(curr_input, src=src_rank)
        labels = torch.empty((batch, seq), device=device, dtype=torch.long)
        dist.recv(labels, src=src_rank)

    # -------- Forward compute --------
    if context.pp_rank == 0:
        act = model(curr_input)  # [b,s,h]
        # send activation + labels to next stage
        dst = context.get_next_stage_rank()
        dist.send(act.detach().contiguous(), dst=dst)
        dist.send(labels.contiguous(), dst=dst)

        # -------- Backward receive grad(act) from next stage --------
        grad_act = torch.empty_like(act, device=device)
        dist.recv(grad_act, src=dst)

        # backward for stage0 params
        torch.autograd.backward(act, grad_act)
        return None

    else:
        # last stage compute logits -> loss
        curr_input.requires_grad_(True)
        logits_part = model(curr_input)  # [b,s,vocab_part]

        # shift for causal LM
        shift_logits = logits_part[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        N = shift_labels.numel()
        loss = vocab_parallel_cross_entropy(
            shift_logits.view(N, -1),
            shift_labels.view(N),
            context,
            ignore_index=None
        )

        loss.backward()

        # send grad w.r.t input activation back to prev stage
        dst = context.get_prev_stage_rank()
        dist.send(curr_input.grad.contiguous(), dst=dst)

        return loss.detach()

def worker_process(rank, world_size, tp_size, pp_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    context = ParallelContext(rank, world_size, tp_size, pp_size)
    config = Qwen3Config()
    model = Qwen3PipelineStage(config, context).to(f"cuda:{rank}")
    model.train()

    batch, seq = 2, 32

    # 只有 Stage0 生成 input_ids（tp_rank0 生成，之后 broadcast）
    if context.pp_rank == 0 and context.tp_rank == 0:
        input_ids = torch.randint(0, config.vocab_size, (batch, seq), device=f"cuda:{rank}")
    else:
        input_ids = None

    loss = run_pipeline_step(model, input_ids, context, config, batch=batch, seq=seq)

    # 打印 loss：只在 last stage 的 tp_rank0 打印
    if context.pp_rank == context.pp_size - 1 and context.tp_rank == 0:
        print(f"✅ [Rank {rank}] Loss: {loss.item():.6f}")

    dist.barrier()
    dist.destroy_process_group()

def main():
    TP_SIZE = 2
    PP_SIZE = 2
    WORLD_SIZE = TP_SIZE * PP_SIZE

    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"❌ Need at least {WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
        return

    print(f"🚀 Start: TP={TP_SIZE}, PP={PP_SIZE}, World={WORLD_SIZE}")
    mp.spawn(worker_process, args=(WORLD_SIZE, TP_SIZE, PP_SIZE), nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()
