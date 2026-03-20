# coding=utf-8
"""Convert TP-sharded checkpoints into HuggingFace format for Qwen3 demo.

Example:
  python /mnt/code/yehangcheng/megatron-mini/convert_tp_to_hf.py \
    --checkpoint_dir /mnt/code/yehangcheng/megatron-mini/ckpt/step_100 \
    --output_dir /mnt/code/yehangcheng/megatron-mini/hf_ckpt \
    --tokenizer_path Qwen/Qwen3-8B \
    --max_length 1024
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = "/mnt/code/yehangcheng"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.model_code.qwen3.configuration_qwen3 import Qwen3Config
from llm.model_code.qwen3.modeling_qwen3 import Qwen3ForCausalLM


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


def load_tp_shards(checkpoint_dir: str, tp_world: int | None = None) -> List[Dict[str, torch.Tensor]]:
    if tp_world is None:
        rank_files = sorted(glob.glob(os.path.join(checkpoint_dir, "rank_*.pt")))
        if not rank_files:
            raise FileNotFoundError(f"No rank_*.pt files found in {checkpoint_dir}")
    else:
        rank_files = [os.path.join(checkpoint_dir, f"rank_{r}.pt") for r in range(tp_world)]

    def extract_rank(path: str) -> int:
        match = re.search(r"rank_(\d+)\.pt$", path)
        return int(match.group(1)) if match else -1

    rank_files = sorted(rank_files, key=extract_rank)
    shards: List[Dict[str, torch.Tensor]] = []
    for path in rank_files:
        ckpt = torch.load(path, map_location="cpu")
        if "model" not in ckpt:
            raise KeyError(f"Missing 'model' in {path}")
        shards.append(ckpt["model"])
    return shards


def merge_tp_state_dicts(
    shards: List[Dict[str, torch.Tensor]], full_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    tp_world = len(shards)

    def col_merge(full_key: str) -> torch.Tensor:
        tp_key = full_key.replace(".weight", ".linear.weight")
        parts = [shards[r][tp_key] for r in range(tp_world)]
        return torch.cat(parts, dim=0)

    def row_merge(full_key: str) -> torch.Tensor:
        tp_key = full_key.replace(".weight", ".linear.weight")
        parts = [shards[r][tp_key] for r in range(tp_world)]
        return torch.cat(parts, dim=1)

    col_keys = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "lm_head.weight",
    }
    row_keys = {
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
    }

    for key in full_state.keys():
        if any(key.endswith(suffix) for suffix in col_keys):
            merged[key] = col_merge(key)
        elif any(key.endswith(suffix) for suffix in row_keys):
            merged[key] = row_merge(key)
        else:
            if key in shards[0]:
                merged[key] = shards[0][key]
            else:
                raise KeyError(f"Key {key} not found in TP shard 0")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tp_world", type=int, default=None)
    parser.add_argument("--save_tokenizer", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    vocab_size = tokenizer.vocab_size

    config = build_demo_config(vocab_size=vocab_size, max_position_embeddings=max(args.max_length, 128))
    model = Qwen3ForCausalLM(config)
    full_state = model.state_dict()

    shards = load_tp_shards(args.checkpoint_dir, args.tp_world)
    merged_state = merge_tp_state_dicts(shards, full_state)
    model.load_state_dict(merged_state, strict=True)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    if args.save_tokenizer:
        tokenizer.save_pretrained(args.output_dir)

    print(f"Saved HF checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
