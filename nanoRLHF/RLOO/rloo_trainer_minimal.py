from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RLOOConfig:
    group_size: int = 4
    response_length: int = 128
    temperature: float = 1.0
    clip_range: float = 0.2
    kl_coef: float = 0.02
    ppo_epochs: int = 1
    mini_batch_size: int = 8
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    advantage_eps: float = 1e-6


class MinimalRLOOTrainer:
    """
    Minimal RLOO trainer:
    - sample K responses per prompt
    - sequence reward = reward_model_score + per-token KL shaping
    - leave-one-out baseline within each prompt's K samples
    - PPO clipped objective on sequence logprob ratios
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: RLOOConfig,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy.eval()
        self.optimizer = optimizer
        self.cfg = cfg
        self.reward_fn = reward_fn

    @staticmethod
    def _gather_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def _policy_logprobs(self, model: nn.Module, query: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query, response], dim=1)
        logits = model(x).logits
        q_len = query.size(1)
        resp_logits = logits[:, q_len - 1 : -1, :] / self.cfg.temperature
        return self._gather_token_logprobs(resp_logits, response)

    def _make_mask(self, response: torch.Tensor) -> torch.Tensor:
        pad_mask = response.eq(self.cfg.pad_token_id)
        if self.cfg.eos_token_id is None:
            return pad_mask
        after_eos = response.eq(self.cfg.eos_token_id).cumsum(dim=1) > 1
        return pad_mask | after_eos

    def train_step(
        self,
        queries: torch.Tensor,
        sample_fn: Callable[[nn.Module, torch.Tensor, int, int], torch.Tensor],
    ) -> Dict[str, float]:
        """
        sample_fn(policy, queries, group_size, response_length) -> responses [B*K, R]
        """
        device = next(self.policy.parameters()).device
        queries = queries.to(device)
        bsz = queries.size(0)
        k = self.cfg.group_size

        with torch.no_grad():
            responses = sample_fn(self.policy, queries, k, self.cfg.response_length).to(device)  # [B*K, R]
            expanded_queries = queries.repeat_interleave(k, dim=0)
            scores = self.reward_fn(expanded_queries, responses).to(device)  # [B*K]

            old_logprobs = self._policy_logprobs(self.policy, expanded_queries, responses)      # [B*K, R]
            ref_logprobs = self._policy_logprobs(self.ref_policy, expanded_queries, responses)  # [B*K, R]
            token_invalid = self._make_mask(responses)

            kl = old_logprobs - ref_logprobs
            non_score = -self.cfg.kl_coef * kl
            seq_reward = non_score.masked_fill(token_invalid, 0.0).sum(dim=1) + scores  # [B*K]

            # Leave-one-out baseline inside each prompt group.
            group_reward = seq_reward.view(bsz, k)
            baseline = (group_reward.sum(dim=1, keepdim=True) - group_reward) / (k - 1)
            advantages = (group_reward - baseline).reshape(-1)  # [B*K]

            # Match your source strategy: randomly keep one sample per prompt.
            pick = torch.randint(0, k, (bsz,), device=device)
            row = torch.arange(bsz, device=device)
            flat_pick = row * k + pick

            sel_queries = expanded_queries[flat_pick]
            sel_responses = responses[flat_pick]
            sel_old_logprobs = old_logprobs[flat_pick]
            sel_ref_logprobs = ref_logprobs[flat_pick]
            sel_mask = token_invalid[flat_pick]
            sel_adv = advantages[flat_pick]

            # Optional whitening for scalar advantages.
            if sel_adv.numel() > 1:
                sel_adv = (sel_adv - sel_adv.mean()) / sel_adv.std(unbiased=False).clamp_min(self.cfg.advantage_eps)

        n = sel_responses.size(0)
        idx = torch.randperm(n, device=device)
        total_loss = total_kl = total_ratio = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for s in range(0, n, self.cfg.mini_batch_size):
                mb = idx[s : s + self.cfg.mini_batch_size]
                mb_q = sel_queries[mb]
                mb_r = sel_responses[mb]
                mb_old_lp = sel_old_logprobs[mb]
                mb_ref_lp = sel_ref_logprobs[mb]
                mb_mask = sel_mask[mb]
                mb_adv = sel_adv[mb]  # [M]

                new_logprobs = self._policy_logprobs(self.policy, mb_q, mb_r).masked_fill(mb_mask, 0.0)
                old_logprobs = mb_old_lp.masked_fill(mb_mask, 0.0)
                new_seq_logp = new_logprobs.sum(dim=1)
                old_seq_logp = old_logprobs.sum(dim=1)

                ratio = torch.exp(new_seq_logp - old_seq_logp)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
                loss = torch.max(pg1, pg2).mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item()
                    total_kl += (
                        (mb_old_lp - mb_ref_lp).masked_fill(mb_mask, 0.0).sum(dim=1).mean().item()
                    )
                    total_ratio += ratio.mean().item()
                    num_updates += 1

        return {
            "loss": total_loss / max(1, num_updates),
            "score": scores.mean().item(),
            "seq_reward": seq_reward.mean().item(),
            "advantage": sel_adv.mean().item(),
            "kl": total_kl / max(1, num_updates),
            "ratio": total_ratio / max(1, num_updates),
        }

    def train(
        self,
        dataloader,
        sample_fn: Callable[[nn.Module, torch.Tensor, int, int], torch.Tensor],
        max_steps: int,
    ) -> None:
        self.policy.train()
        step = 0
        while step < max_steps:
            for batch in dataloader:
                metrics = self.train_step(batch["input_ids"], sample_fn)
                step += 1
                print(f"[step {step}] {metrics}")
                if step >= max_steps:
                    break
