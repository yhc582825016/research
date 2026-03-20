from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    response_length: int = 128
    temperature: float = 1.0
    gamma: float = 1.0
    lam: float = 0.95
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.5
    kl_coef: float = 0.02
    ppo_epochs: int = 1
    mini_batch_size: int = 8
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    advantage_eps: float = 1e-6


class MinimalPPOTrainer:
    """
    Core PPO trainer for RLHF-like token-level optimization:
    1) rollout: sample response + collect old logprobs/ref logprobs/values
    2) reward shaping: reward model score + KL penalty to reference policy
    3) GAE + returns
    4) PPO update: clipped policy loss + clipped value loss
    """

    def __init__(
        self,
        policy: nn.Module,
        value_model: nn.Module,
        ref_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: PPOConfig,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.policy = policy
        self.value_model = value_model
        self.ref_policy = ref_policy.eval()
        self.optimizer = optimizer
        self.cfg = cfg
        self.reward_fn = reward_fn

    @staticmethod
    def _gather_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def _policy_logprobs(self, model: nn.Module, query: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query, response], dim=1)  # [B, Q+R]
        logits = model(x).logits
        q_len = query.size(1)
        resp_logits = logits[:, q_len - 1 : -1, :] / self.cfg.temperature
        return self._gather_token_logprobs(resp_logits, response)  # [B, R]

    def _values(self, query: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """
        Return token-level state values aligned to response tokens: [B, R].
        Assume value_model(x).values has shape [B, Q+R] or [B, Q+R, 1].
        """
        x = torch.cat([query, response], dim=1)
        out = self.value_model(x).values
        if out.dim() == 3:
            out = out.squeeze(-1)
        q_len = query.size(1)
        return out[:, q_len - 1 : -1]

    def _make_masks(self, response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            padding_mask:    True where token is invalid for policy loss [B, R]
            padding_mask_p1: True where token/state is invalid for value/reward [B, R]
        """
        pad_mask = response.eq(self.cfg.pad_token_id)
        if self.cfg.eos_token_id is None:
            return pad_mask, pad_mask
        # tokens after first eos are invalid
        after_eos = response.eq(self.cfg.eos_token_id).cumsum(dim=1) > 1
        token_invalid = pad_mask | after_eos
        return token_invalid, token_invalid

    def _compute_gae_and_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        token_invalid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewards/values: [B, R]
        """
        bsz, t_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(bsz, device=rewards.device)

        for t in reversed(range(t_len)):
            next_v = values[:, t + 1] if t < t_len - 1 else 0.0
            delta = rewards[:, t] + self.cfg.gamma * next_v - values[:, t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.lam * lastgaelam
            advantages[:, t] = lastgaelam

        advantages = advantages.masked_fill(token_invalid, 0.0)
        returns = advantages + values

        # Optional global whitening for stability in practice.
        valid_adv = advantages[~token_invalid]
        if valid_adv.numel() > 1:
            mean = valid_adv.mean()
            std = valid_adv.std(unbiased=False).clamp_min(self.cfg.advantage_eps)
            advantages = (advantages - mean) / std
            advantages = advantages.masked_fill(token_invalid, 0.0)

        return advantages, returns

    def train_step(
        self,
        queries: torch.Tensor,
        sample_fn: Callable[[nn.Module, torch.Tensor, int], torch.Tensor],
    ) -> Dict[str, float]:
        """
        queries: [B, Q]
        sample_fn should return sampled responses [B, R]
        """
        device = next(self.policy.parameters()).device
        queries = queries.to(device)

        # 1) Rollout
        with torch.no_grad():
            responses = sample_fn(self.policy, queries, self.cfg.response_length).to(device)
            rewards_scalar = self.reward_fn(queries, responses).to(device)  # [B]

            old_logprobs = self._policy_logprobs(self.policy, queries, responses)      # [B, R]
            ref_logprobs = self._policy_logprobs(self.ref_policy, queries, responses)  # [B, R]
            old_values = self._values(queries, responses)                                # [B, R]

            token_invalid, value_invalid = self._make_masks(responses)
            old_values = old_values.masked_fill(value_invalid, 0.0)

            # 2) Reward shaping: -kl_coef * KL(token) + sparse terminal reward.
            kl = old_logprobs - ref_logprobs
            rewards = -self.cfg.kl_coef * kl
            end_idx = (~value_invalid).sum(dim=1).clamp_min(1) - 1
            rewards[torch.arange(rewards.size(0), device=device), end_idx] += rewards_scalar

            # 3) GAE and returns
            advantages, returns = self._compute_gae_and_returns(rewards, old_values, token_invalid)

        # 4) PPO updates
        bsz = responses.size(0)
        idx = torch.randperm(bsz, device=device)

        total_loss = 0.0
        total_pg = 0.0
        total_vf = 0.0
        total_kl = 0.0
        total_ratio = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for s in range(0, bsz, self.cfg.mini_batch_size):
                mb = idx[s : s + self.cfg.mini_batch_size]
                mb_q = queries[mb]
                mb_r = responses[mb]
                mb_old_lp = old_logprobs[mb]
                mb_old_v = old_values[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]
                mb_token_invalid = token_invalid[mb]
                mb_value_invalid = value_invalid[mb]

                new_logprobs = self._policy_logprobs(self.policy, mb_q, mb_r)
                new_values = self._values(mb_q, mb_r).masked_fill(mb_value_invalid, 0.0)

                log_ratio = new_logprobs - mb_old_lp
                ratio = torch.exp(log_ratio)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
                pg_loss = torch.max(pg1, pg2)[~mb_token_invalid].mean()

                v_clipped = torch.clamp(
                    new_values,
                    mb_old_v - self.cfg.clip_range_value,
                    mb_old_v + self.cfg.clip_range_value,
                )
                vf1 = (new_values - mb_ret) ** 2
                vf2 = (v_clipped - mb_ret) ** 2
                vf_loss = 0.5 * torch.max(vf1, vf2)[~mb_value_invalid].mean()

                loss = pg_loss + self.cfg.vf_coef * vf_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item()
                    total_pg += pg_loss.item()
                    total_vf += vf_loss.item()
                    total_kl += (mb_old_lp - ref_logprobs[mb])[~mb_token_invalid].mean().item()
                    total_ratio += ratio[~mb_token_invalid].mean().item()
                    num_updates += 1

        return {
            "loss": total_loss / max(1, num_updates),
            "pg_loss": total_pg / max(1, num_updates),
            "vf_loss": total_vf / max(1, num_updates),
            "reward": rewards_scalar.mean().item(),
            "kl": total_kl / max(1, num_updates),
            "ratio": total_ratio / max(1, num_updates),
        }

    def train(
        self,
        dataloader,
        sample_fn: Callable[[nn.Module, torch.Tensor, int], torch.Tensor],
        max_steps: int,
    ) -> None:
        self.policy.train()
        self.value_model.train()
        step = 0
        while step < max_steps:
            for batch in dataloader:
                queries = batch["input_ids"]
                metrics = self.train_step(queries, sample_fn)
                step += 1
                print(f"[step {step}] {metrics}")
                if step >= max_steps:
                    break
