# Pure GRPO-related tensor math extracted from grpo_trainer.py.
# Assumes: vLLM/tokenizer/dataloader/models are provided by the caller.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from trl.core import masked_mean, masked_whiten


INVALID_LOGPROB = 1.0


class GRPOAlgorithm:
    """
    One-class bundle of the trainer's non-I/O algorithm:
    group score normalization, optional subsample, sparse terminal rewards,
    MC advantages over the response, clipped policy loss + low-var KL penalty.
    """

    invalid_logprob: float = INVALID_LOGPROB

    @staticmethod
    def group_normalize_scores(scores_flat: torch.Tensor, grpo_sample_n: int) -> torch.Tensor:
        """
        Flat scores in prompt-major order: (B * N,). Returns (B, N) within-group z-score.
        Same as reshaped_scores = scores.view(-1, N) then (x - mean) / std per row.
        """
        b = scores_flat.numel() // grpo_sample_n
        x = scores_flat.view(b, grpo_sample_n)
        return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

    @staticmethod
    def subsample_one_completion_per_prompt(
        group_scores_bn: torch.Tensor,
        responses_bnt: torch.Tensor,
        *,
        random_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        group_scores_bn: (B, N) advantages after group normalize.
        responses_bnt:   (B, N, T).
        Returns (scores_b,), (responses_bt,), (indices[B],) with NaNs zeroed on scores_b.
        """
        b, n, _ = responses_bnt.shape
        device = group_scores_bn.device
        if random_indices is None:
            random_indices = torch.randint(0, n, (b,), device=device)
        row = torch.arange(b, device=device)
        scores_b = group_scores_bn[row, random_indices].contiguous()
        scores_b = scores_b.clone()
        scores_b[torch.isnan(scores_b)] = 0
        responses_bt = responses_bnt[row, random_indices].contiguous()
        return scores_b, responses_bt, random_indices

    @staticmethod
    def response_padding_masks(
        *,
        num_rows: int,
        response_width: int,
        sequence_lengths: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mirrors trainer masks on the response piece. sequence_lengths: (B,) last token index per row.
        Returns (response_idxs, padding_mask for logprob positions, padding_mask_p1 for reward/whiten).
        """
        response_idxs = torch.arange(response_width, device=device).repeat(num_rows, 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        sequence_lengths_p1 = sequence_lengths + 1
        padding_mask_p1 = response_idxs > sequence_lengths_p1.unsqueeze(1)
        return response_idxs, padding_mask, padding_mask_p1

    @staticmethod
    def apply_invalid_logprob_to_pairs(
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ip = GRPOAlgorithm.invalid_logprob
        return (
            torch.masked_fill(logprobs, padding_mask, ip),
            torch.masked_fill(ref_logprobs, padding_mask, ip),
        )

    @staticmethod
    def scatter_terminal_rewards(
        *,
        ref_logprobs_like: torch.Tensor,
        scores_per_sequence: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse reward on the timestep used in grpo_trainer (actual_end indexing)."""
        rewards = torch.zeros_like(ref_logprobs_like)
        actual_start = torch.arange(rewards.size(0), device=rewards.device)
        sequence_lengths_p1 = sequence_lengths + 1
        actual_end = torch.where(
            sequence_lengths_p1 < rewards.size(1),
            sequence_lengths_p1,
            sequence_lengths,
        )
        rewards[[actual_start, actual_end]] += scores_per_sequence
        return rewards

    @staticmethod
    def monte_carlo_advantages(
        rewards: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
        advantage_whiten: bool,
    ) -> torch.Tensor:
        """Backwards accumulation over response timesteps, then optional masked whitening."""
        gen_length = rewards.size(1)
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(gen_length)):
            lastgaelam = rewards[:, t] + lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        if advantage_whiten:
            advantages = masked_whiten(advantages, ~padding_mask, shift_mean=True)
        advantages = torch.masked_fill(advantages, padding_mask, 0)
        return advantages

    @staticmethod
    def build_advantages_from_terminal_scores(
        *,
        ref_logprobs_like: torch.Tensor,
        scores_per_sequence: torch.Tensor,
        sequence_lengths: torch.Tensor,
        padding_mask: torch.Tensor,
        padding_mask_p1: torch.Tensor,
        whiten_rewards: bool,
        advantage_whiten: bool,
    ) -> torch.Tensor:
        """
        End-to-end: sparse terminal rewards → optional per-token reward whitening → MC advantages.
        Matches grpo_trainer order (whiten rewards before the reverse cumulative sum).
        """
        rewards = GRPOAlgorithm.scatter_terminal_rewards(
            ref_logprobs_like=ref_logprobs_like,
            scores_per_sequence=scores_per_sequence,
            sequence_lengths=sequence_lengths,
        )
        rewards = GRPOAlgorithm.maybe_whiten_rewards(
            rewards, padding_mask_p1=padding_mask_p1, whiten_rewards=whiten_rewards
        )
        return GRPOAlgorithm.monte_carlo_advantages(
            rewards, padding_mask=padding_mask, advantage_whiten=advantage_whiten
        )

    @staticmethod
    def maybe_whiten_rewards(
        rewards: torch.Tensor,
        *,
        padding_mask_p1: torch.Tensor,
        whiten_rewards: bool,
    ) -> torch.Tensor:
        if not whiten_rewards:
            return rewards
        out = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=True)
        return torch.masked_fill(out, padding_mask_p1, 0)

    @staticmethod
    def maybe_penalize_missing_eos(
        scores_1d: torch.Tensor,
        postprocessed_responses: torch.Tensor,
        eos_token_id: int,
        missing_eos_penalty: Optional[float],
    ) -> torch.Tensor:
        if missing_eos_penalty is None:
            return scores_1d
        contain_eos = torch.any(postprocessed_responses == eos_token_id, dim=-1)
        s = scores_1d.clone()
        s[~contain_eos] = s[~contain_eos] - missing_eos_penalty
        return s

    @staticmethod
    def grpo_policy_loss(
        *,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        padding_mask: torch.Tensor,
        cliprange: float,
        kl_coef: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (loss, kl_pairwise) where kl_pairwise = new - ref (detached stats in trainer).
        """
        new_logprobs = torch.masked_fill(new_logprobs, padding_mask, GRPOAlgorithm.invalid_logprob)
        logprobs_diff = new_logprobs - old_logprobs
        ratio = torch.exp(logprobs_diff)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        kl = new_logprobs - ref_logprobs
        kl_penalty = kl_coef * (torch.exp(-kl) + kl - 1)
        pg_loss_max = torch.max(pg_losses, pg_losses2) + kl_penalty
        loss = masked_mean(pg_loss_max, ~padding_mask)
        return loss, kl

    @staticmethod
    def forward_logits_to_chosen_logprobs(
        logits: torch.Tensor,
        response_tokens: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """logits: (B, T, V) aligned with response positions; response_tokens: (B, T)."""
        logits = logits / temperature
        all_lp = F.log_softmax(logits, dim=-1)
        return torch.gather(all_lp, 2, response_tokens.unsqueeze(-1)).squeeze(-1)
