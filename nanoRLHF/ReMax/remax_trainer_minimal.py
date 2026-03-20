from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReMaxConfig:
    response_length: int = 128
    temperature: float = 1.0
    clip_range: float = 0.2
    kl_coef: float = 0.02
    ppo_epochs: int = 1
    mini_batch_size: int = 8
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    advantage_eps: float = 1e-6


class MinimalReMaxTrainer:
    """
    Minimal ReMax trainer:
    - sample one stochastic response and one greedy baseline response
    - scalar advantage = reward(sample) - reward(greedy_baseline)
    - token reward includes KL penalty and terminal scalar advantage
    - PPO clipped policy update (no value model)
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: ReMaxConfig,
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

    @staticmethod
    def _reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
        # cumulative sum from right to left, same as returns with gamma=1
        return torch.flip(torch.cumsum(torch.flip(x, dims=[1]), dim=1), dims=[1])

    def train_step(
        self,
        queries: torch.Tensor,
        sample_fn: Callable[[nn.Module, torch.Tensor, int, bool], torch.Tensor],
    ) -> Dict[str, float]:
        """
        sample_fn(policy, queries, response_length, greedy) -> responses [B, R]
        """
        device = next(self.policy.parameters()).device
        queries = queries.to(device)

        with torch.no_grad():
            responses = sample_fn(self.policy, queries, self.cfg.response_length, False).to(device)
            baseline_responses = sample_fn(self.policy, queries, self.cfg.response_length, True).to(device)

            reward_sample = self.reward_fn(queries, responses).to(device)
            reward_baseline = self.reward_fn(queries, baseline_responses).to(device)
            advantage_scalar = reward_sample - reward_baseline  # ReMax core

            old_logprobs = self._policy_logprobs(self.policy, queries, responses)
            ref_logprobs = self._policy_logprobs(self.ref_policy, queries, responses)
            token_invalid = self._make_mask(responses)

            kl = old_logprobs - ref_logprobs
            rewards = -self.cfg.kl_coef * kl
            end_idx = (~token_invalid).sum(dim=1).clamp_min(1) - 1
            rewards[torch.arange(rewards.size(0), device=device), end_idx] += advantage_scalar

            advantages = self._reverse_cumsum(rewards)
            advantages = advantages.masked_fill(token_invalid, 0.0)
            valid = advantages[~token_invalid]
            if valid.numel() > 1:
                mean = valid.mean()
                std = valid.std(unbiased=False).clamp_min(self.cfg.advantage_eps)
                advantages = (advantages - mean) / std
                advantages = advantages.masked_fill(token_invalid, 0.0)

        bsz = responses.size(0)
        idx = torch.randperm(bsz, device=device)
        total_loss = total_kl = total_ratio = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for s in range(0, bsz, self.cfg.mini_batch_size):
                mb = idx[s : s + self.cfg.mini_batch_size]
                mb_q = queries[mb]
                mb_r = responses[mb]
                mb_old = old_logprobs[mb]
                mb_adv = advantages[mb]
                mb_invalid = token_invalid[mb]

                new_logprobs = self._policy_logprobs(self.policy, mb_q, mb_r)
                ratio = torch.exp(new_logprobs - mb_old)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
                loss = torch.max(pg1, pg2)[~mb_invalid].mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item()
                    total_kl += (mb_old - ref_logprobs[mb])[~mb_invalid].mean().item()
                    total_ratio += ratio[~mb_invalid].mean().item()
                    num_updates += 1

        return {
            "loss": total_loss / max(1, num_updates),
            "score": reward_sample.mean().item(),
            "baseline_score": reward_baseline.mean().item(),
            "advantage_scalar": advantage_scalar.mean().item(),
            "kl": total_kl / max(1, num_updates),
            "ratio": total_ratio / max(1, num_updates),
        }

    def train(
        self,
        dataloader,
        sample_fn: Callable[[nn.Module, torch.Tensor, int, bool], torch.Tensor],
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
