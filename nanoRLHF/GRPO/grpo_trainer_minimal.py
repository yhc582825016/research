from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOConfig:
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


class MinimalGRPOTrainer:
    """
    A minimal GRPO trainer focused on the core algorithm:
    1) Sample K responses per prompt
    2) Compute group-normalized advantages from rewards
    3) Optimize with PPO clipping + KL regularization to reference policy
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: GRPOConfig,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy.eval()
        self.optimizer = optimizer
        self.cfg = cfg
        self.reward_fn = reward_fn

    @staticmethod
    def _gather_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V], labels: [B, T]
        logp = F.log_softmax(logits, dim=-1)
        return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def _forward_logprobs(self, model: nn.Module, query: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """
        Compute token logprobs of response tokens conditioned on query.
        query: [B, Q], response: [B, R]
        return: [B, R]
        """
        x = torch.cat([query, response], dim=1)  # [B, Q+R]
        logits = model(x).logits  # [B, Q+R, V]
        q_len = query.size(1)
        # Predict each response token from previous token:
        # logits positions [q_len-1, ..., q_len+R-2] -> labels response[0:R]
        resp_logits = logits[:, q_len - 1 : -1, :] / self.cfg.temperature
        return self._gather_token_logprobs(resp_logits, response)

    def _make_padding_mask(self, response: torch.Tensor) -> torch.Tensor:
        """
        True means invalid/padded token.
        """
        pad_mask = response.eq(self.cfg.pad_token_id)
        if self.cfg.eos_token_id is None:
            return pad_mask
        eos_hit = response.eq(self.cfg.eos_token_id).cumsum(dim=1) > 1
        return pad_mask | eos_hit

    def _group_normalized_advantage(self, rewards: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        rewards: [B*K]
        return:   [B*K]
        """
        r = rewards.view(batch_size, self.cfg.group_size)
        r_mean = r.mean(dim=1, keepdim=True)
        r_std = r.std(dim=1, keepdim=True)
        adv = (r - r_mean) / (r_std + self.cfg.advantage_eps)
        return adv.reshape(-1)

    def train_step(
        self,
        queries: torch.Tensor,
        sample_fn: Callable[[nn.Module, torch.Tensor, int, int], torch.Tensor],
    ) -> Dict[str, float]:
        """
        queries: [B, Q]
        sample_fn should return sampled responses [B*K, R]
        """
        device = next(self.policy.parameters()).device
        queries = queries.to(device)
        batch_size = queries.size(0)
        k = self.cfg.group_size

        # 1) Sample K completions per prompt.
        with torch.no_grad():
            expanded_queries = queries.repeat_interleave(k, dim=0)  # [B*K, Q]
            responses = sample_fn(self.policy, queries, k, self.cfg.response_length).to(device)  # [B*K, R]

            rewards = self.reward_fn(expanded_queries, responses).to(device)  # [B*K]
            advantages = self._group_normalized_advantage(rewards, batch_size)  # [B*K]
            advantages = advantages.unsqueeze(1)  # [B*K, 1], broadcast to token dim

            old_logprobs = self._forward_logprobs(self.policy, expanded_queries, responses)  # [B*K, R]
            ref_logprobs = self._forward_logprobs(self.ref_policy, expanded_queries, responses)  # [B*K, R]
            invalid_mask = self._make_padding_mask(responses)  # [B*K, R]

        # 2) PPO-style update with KL regularization.
        n = responses.size(0)
        idx = torch.randperm(n, device=device)

        total_loss = 0.0
        total_ratio = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for s in range(0, n, self.cfg.mini_batch_size):
                mb = idx[s : s + self.cfg.mini_batch_size]
                mb_q = expanded_queries[mb]
                mb_r = responses[mb]
                mb_adv = advantages[mb]
                mb_old = old_logprobs[mb]
                mb_ref = ref_logprobs[mb]
                mb_invalid = invalid_mask[mb]

                new_logprobs = self._forward_logprobs(self.policy, mb_q, mb_r)
                ratio = torch.exp(new_logprobs - mb_old)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range)
                pg = torch.max(pg1, pg2)

                kl = new_logprobs - mb_ref
                # Same non-negative KL penalty style used in your source file.
                kl_penalty = self.cfg.kl_coef * (torch.exp(-kl) + kl - 1.0)

                loss_token = pg + kl_penalty
                valid = ~mb_invalid
                loss = loss_token[valid].mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item()
                    total_ratio += ratio[valid].mean().item()
                    total_kl += kl[valid].mean().item()
                    total_clipfrac += (pg2 > pg1)[valid].float().mean().item()
                    num_updates += 1

        return {
            "loss": total_loss / max(1, num_updates),
            "reward": rewards.mean().item(),
            "ratio": total_ratio / max(1, num_updates),
            "kl": total_kl / max(1, num_updates),
            "clipfrac": total_clipfrac / max(1, num_updates),
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
                queries = batch["input_ids"]  # assume tokenized prompts already exist
                metrics = self.train_step(queries, sample_fn)
                step += 1
                print(f"[step {step}] {metrics}")
                if step >= max_steps:
                    break
