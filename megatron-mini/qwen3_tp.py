# coding=utf-8
"""Single-machine tensor parallel (tp=4) demo for Qwen3 attention + MLP.

Run:
  torchrun --nproc_per_node=4 /mnt/code/yehangcheng/megatron-mini/qwen3_tp.py
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional, Tuple

import argparse
import json
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = "/mnt/code/yehangcheng"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.model_code.qwen3.configuration_qwen3 import Qwen3Config
from llm.model_code.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def init_distributed() -> Tuple[int, int, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, tp_rank: int, tp_world: int, gather_output: bool = True):
        super().__init__()
        if linear.out_features % tp_world != 0:
            raise ValueError("out_features must be divisible by tp_world")
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.gather_output = gather_output
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        out_per_rank = self.out_features // tp_world

        self.linear = nn.Linear(
            self.in_features,
            out_per_rank,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.load_from_full_weight(linear.weight, linear.bias)

    def load_from_full_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        out_per_rank = self.out_features // self.tp_world
        start = self.tp_rank * out_per_rank
        end = start + out_per_rank
        self.linear.weight.data.copy_(weight[start:end, :])
        if bias is not None:
            self.linear.bias.data.copy_(bias[start:end])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ColumnParallelLinearFn.apply(
            x, self.linear.weight, self.linear.bias, self.tp_rank, self.tp_world, self.gather_output
        )


class RowParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, tp_rank: int, tp_world: int):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("RowParallelLinear does not support bias (bias should be False in Qwen3).")
        if linear.in_features % tp_world != 0:
            raise ValueError("in_features must be divisible by tp_world")
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        in_per_rank = self.in_features // tp_world

        self.linear = nn.Linear(
            in_per_rank,
            self.out_features,
            bias=False,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.load_from_full_weight(linear.weight, linear.bias)

    def load_from_full_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        in_per_rank = self.in_features // self.tp_world
        start = self.tp_rank * in_per_rank
        end = start + in_per_rank
        self.linear.weight.data.copy_(weight[:, start:end])
        if bias is not None:
            self.linear.bias.data.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RowParallelLinearFn.apply(x, self.linear.weight, None, self.tp_rank, self.tp_world)


class _ColumnParallelLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        tp_rank: int,
        tp_world: int,
        gather_output: bool,
    ):
        ctx.tp_rank = tp_rank
        ctx.tp_world = tp_world
        ctx.gather_output = gather_output
        ctx.save_for_backward(x, weight)
        local_out = F.linear(x, weight, bias)
        if tp_world == 1 or not gather_output:
            return local_out
        out_list = [torch.empty_like(local_out) for _ in range(tp_world)]
        dist.all_gather(out_list, local_out)
        return torch.cat(out_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        tp_rank = ctx.tp_rank
        tp_world = ctx.tp_world
        gather_output = ctx.gather_output

        if tp_world == 1 or not gather_output:
            grad_out_local = grad_output
        else:
            out_per_rank = weight.shape[0]
            start = tp_rank * out_per_rank
            end = start + out_per_rank
            grad_out_local = grad_output[..., start:end]

        grad_input = grad_out_local.matmul(weight)
        if tp_world > 1:
            dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)

        x_2d = x.reshape(-1, x.shape[-1])
        grad_out_2d = grad_out_local.reshape(-1, grad_out_local.shape[-1])
        grad_weight = grad_out_2d.t().matmul(x_2d)

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None


class _RowParallelLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], tp_rank: int, tp_world: int):
        ctx.tp_rank = tp_rank
        ctx.tp_world = tp_world
        in_per_rank = weight.shape[1]
        if tp_world == 1:
            x_local = x
        else:
            start = tp_rank * in_per_rank
            end = start + in_per_rank
            x_local = x[..., start:end]
        ctx.save_for_backward(x_local, weight)
        local_out = F.linear(x_local, weight, bias)
        if tp_world > 1:
            dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_local, weight = ctx.saved_tensors
        tp_rank = ctx.tp_rank
        tp_world = ctx.tp_world

        grad_input_local = grad_output.matmul(weight)
        if tp_world == 1:
            grad_input = grad_input_local
        else:
            chunks = [torch.empty_like(grad_input_local) for _ in range(tp_world)]
            dist.all_gather(chunks, grad_input_local)
            grad_input = torch.cat(chunks, dim=-1)

        x_2d = x_local.reshape(-1, x_local.shape[-1])
        grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_weight = grad_out_2d.t().matmul(x_2d)

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None


def apply_tp(model: Qwen3ForCausalLM, tp_rank: int, tp_world: int) -> None:
    for layer in model.model.layers:
        attn = layer.self_attn
        if attn.num_attention_heads % tp_world != 0 or attn.config.num_key_value_heads % tp_world != 0:
            raise ValueError("num_attention_heads and num_key_value_heads must be divisible by tp_world.")
        attn.num_attention_heads = attn.num_attention_heads // tp_world
        attn.config.num_attention_heads = attn.config.num_attention_heads // tp_world
        attn.config.num_key_value_heads = attn.config.num_key_value_heads // tp_world
        attn.num_key_value_groups = attn.num_attention_heads // attn.config.num_key_value_heads

        attn.q_proj = ColumnParallelLinear(attn.q_proj, tp_rank, tp_world, gather_output=False)
        attn.k_proj = ColumnParallelLinear(attn.k_proj, tp_rank, tp_world, gather_output=False)
        attn.v_proj = ColumnParallelLinear(attn.v_proj, tp_rank, tp_world, gather_output=False)
        attn.o_proj = RowParallelLinear(attn.o_proj, tp_rank, tp_world)

        mlp = layer.mlp
        mlp.gate_proj = ColumnParallelLinear(mlp.gate_proj, tp_rank, tp_world, gather_output=False)
        mlp.up_proj = ColumnParallelLinear(mlp.up_proj, tp_rank, tp_world, gather_output=False)
        mlp.down_proj = RowParallelLinear(mlp.down_proj, tp_rank, tp_world)

    model.lm_head = ColumnParallelLinear(model.lm_head, tp_rank, tp_world, gather_output=False)


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor, targets: torch.Tensor, tp_rank: int, tp_world: int, vocab_size: int
) -> torch.Tensor:
    vocab_per_rank = vocab_size // tp_world
    start = tp_rank * vocab_per_rank
    end = start + vocab_per_rank

    local_max = local_logits.max(dim=-1).values
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

    local_exp = torch.exp(local_logits - local_max.unsqueeze(-1))
    local_sumexp = local_exp.sum(dim=-1)
    dist.all_reduce(local_sumexp, op=dist.ReduceOp.SUM)
    logsumexp = torch.log(local_sumexp) + local_max

    in_range = (targets >= start) & (targets < end)
    safe_targets = (targets - start).clamp(min=0, max=vocab_per_rank - 1)
    local_target = torch.gather(local_logits, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    local_target = torch.where(in_range, local_target, torch.zeros_like(local_target))
    dist.all_reduce(local_target, op=dist.ReduceOp.SUM)

    loss = logsumexp - local_target
    return loss.mean()


def build_demo_config(vocab_size: int, max_position_embeddings: int) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=32,
        max_position_embeddings=max_position_embeddings,
        use_sliding_window=False,
    )


def load_conversations(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    conversations = payload.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("`conversations` must be a list of {role, content} dicts.")
    return conversations


def encode_conversation(
    conversations: list[dict],
    tokenizer,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids: list[int] = []
    labels: list[int] = []

    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "user":
            prefix = "User: "
            suffix = "\nAssistant: "
            text = f"{prefix}{content}\n"
            target_mask = False
        elif role == "assistant":
            text = f"{content}\n"
            target_mask = True
        else:
            text = f"{role}: {content}\n"
            target_mask = False

        tokens = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(tokens)
        if target_mask:
            labels.extend(tokens)
        else:
            labels.extend([-100] * len(tokens))

    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def load_jsonl_conversations(path: str) -> list[list[dict]]:
    conversations_list: list[list[dict]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            conversations = payload.get("conversations", [])
            if isinstance(conversations, list):
                conversations_list.append(conversations)
    return conversations_list


def iter_epochs(data: list[list[dict]], epochs: int, shuffle: bool = False, seed: int = 42) -> Iterable[list[dict]]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    for _ in range(epochs):
        indices = torch.randperm(len(data), generator=rng).tolist() if shuffle else range(len(data))
        for idx in indices:
            yield data[idx]


def collate_batch(
    items: list[list[dict]],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [encode_conversation(item, tokenizer, max_length=max_length) for item in items]
    lengths = [ids.numel() for ids, _ in encoded]
    max_len = min(max(lengths), max_length)

    batch_size = len(encoded)
    input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, (ids, labs) in enumerate(encoded):
        ids = ids[-max_len:]
        labs = labs[-max_len:]
        input_ids[i, : ids.numel()] = ids
        labels[i, : labs.numel()] = labs
        attention_mask[i, : ids.numel()] = 1

    return input_ids, labels, attention_mask


def iter_batches(
    data: list[list[dict]],
    epochs: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterable[list[list[dict]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    rng = torch.Generator()
    rng.manual_seed(seed)
    for _ in range(epochs):
        indices = torch.randperm(len(data), generator=rng).tolist() if shuffle else list(range(len(data)))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            if len(batch_idx) < batch_size:
                break  # drop last incomplete batch to keep shapes consistent for broadcast
            yield [data[j] for j in batch_idx]


def main() -> None:
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="/mnt/code/yehangcheng/megatron-mini/ckpt")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    else:
        tokenizer = None

    vocab_size = tokenizer.vocab_size if rank == 0 else 0
    vocab_size_tensor = torch.tensor([vocab_size], device=device, dtype=torch.long)
    dist.broadcast(vocab_size_tensor, src=0)
    vocab_size = int(vocab_size_tensor.item())

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = build_demo_config(vocab_size=vocab_size, max_position_embeddings=max(args.max_length, 128))
    model = Qwen3ForCausalLM(config).to(device)
    model.eval()

    ref_model = None
    if rank == 0 and args.data_path is None:
        ref_model = Qwen3ForCausalLM(config)
        ref_model.load_state_dict(model.state_dict())
        ref_model.to(device)
        ref_model.eval()

    apply_tp(model, rank, world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.data_path:
        if rank == 0:
            dataset = load_jsonl_conversations(args.data_path)
            if not dataset:
                raise ValueError("No conversations found in the dataset.")
        else:
            dataset = []

        data_count = torch.tensor([len(dataset)], device=device, dtype=torch.long)
        dist.broadcast(data_count, src=0)
        data_count = int(data_count.item())
        if data_count == 0:
            raise ValueError("Empty dataset.")

        global_step = 0
        micro_step = 0

        if rank == 0 and args.save_steps > 0:
            os.makedirs(args.output_dir, exist_ok=True)
        dist.barrier()

        for batch in iter_batches(dataset, args.epochs, args.batch_size, shuffle=args.shuffle):
            if rank == 0:
                input_ids, labels, attention_mask = collate_batch(
                    batch, tokenizer, args.max_length, device=device
                )
            else:
                input_ids = torch.empty((args.batch_size, args.max_length), device=device, dtype=torch.long)
                labels = torch.empty((args.batch_size, args.max_length), device=device, dtype=torch.long)
                attention_mask = torch.empty((args.batch_size, args.max_length), device=device, dtype=torch.long)

            dist.broadcast(input_ids, src=0)
            dist.broadcast(labels, src=0)
            dist.broadcast(attention_mask, src=0)

            tp_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            tp_logits = tp_outputs.logits
            tp_loss = vocab_parallel_cross_entropy(tp_logits, labels, rank, world_size, config.vocab_size)
            (tp_loss / args.grad_accum).backward()
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if rank == 0:
                    print(f"step={global_step} loss={tp_loss.item():.4f}")

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"step_{global_step}")
                    if rank == 0:
                        os.makedirs(save_dir, exist_ok=True)
                    dist.barrier()
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": global_step,
                        },
                        os.path.join(save_dir, f"rank_{rank}.pt"),
                    )
                    dist.barrier()

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.max_steps and global_step >= args.max_steps:
                break
    else:
        if rank == 0:
            input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            input_ids = torch.empty((2, 16), device=device, dtype=torch.long)
            labels = torch.empty((2, 16), device=device, dtype=torch.long)
            attention_mask = torch.empty((2, 16), device=device, dtype=torch.long)
        dist.broadcast(input_ids, src=0)
        dist.broadcast(labels, src=0)
        dist.broadcast(attention_mask, src=0)

        tp_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        tp_logits = tp_outputs.logits
        tp_loss = vocab_parallel_cross_entropy(tp_logits, labels, rank, world_size, config.vocab_size)
        tp_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if rank == 0 and ref_model is not None:
            with torch.no_grad():
                ref_logits = ref_model(input_ids=input_ids).logits
                torch.testing.assert_close(tp_logits, ref_logits, rtol=1e-4, atol=1e-4)

                shift_logits = ref_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, :-1].contiguous()
                ref_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                torch.testing.assert_close(tp_loss, ref_loss, rtol=1e-4, atol=1e-4)
                print("TP logits and loss match reference, backward+step done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
