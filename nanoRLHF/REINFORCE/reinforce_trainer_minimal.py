from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReinforceConfig:
    response_length: int = 128
    temperature: float = 1.0
    gamma: float = 1.0
    clip_range: float = 0.2
    kl_coef: float = 0.02
    ppo_epochs: int = 1
    mini_batch_size: int = 8
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    advantage_eps: float = 1e-6


class MinimalReinforceTrainer:
    """
    Minimal REINFORCE-style trainer matching your script's core idea:
    - reward shaping: task reward + KL penalty to reference policy
    - no value network
    - cumulative return as token-level advantage
    - PPO clipped objective for stable/off-policy updates
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: ReinforceConfig,
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

    def _make_masks(self, response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_mask = response.eq(self.cfg.pad_token_id)
        if self.cfg.eos_token_id is None:
            return pad_mask, pad_mask
        after_eos = response.eq(self.cfg.eos_token_id).cumsum(dim=1) > 1
        token_invalid = pad_mask | after_eos
        return token_invalid, token_invalid

    def _discounted_returns(self, rewards: torch.Tensor, token_invalid: torch.Tensor) -> torch.Tensor:
        bsz, t_len = rewards.shape
        ret = torch.zeros_like(rewards)
        running = torch.zeros(bsz, device=rewards.device)
        for t in reversed(range(t_len)):
            running = rewards[:, t] + self.cfg.gamma * running
            ret[:, t] = running
        ret = ret.masked_fill(token_invalid, 0.0)
        valid = ret[~token_invalid]
        if valid.numel() > 1:
            mean = valid.mean()
            std = valid.std(unbiased=False).clamp_min(self.cfg.advantage_eps)
            ret = (ret - mean) / std
            ret = ret.masked_fill(token_invalid, 0.0)
        return ret

    def train_step(
        self,
        queries: torch.Tensor,
        sample_fn: Callable[[nn.Module, torch.Tensor, int], torch.Tensor],
    ) -> Dict[str, float]:
        device = next(self.policy.parameters()).device
        queries = queries.to(device)

        with torch.no_grad():
            responses = sample_fn(self.policy, queries, self.cfg.response_length).to(device)  # [B, R]
            scores = self.reward_fn(queries, responses).to(device)  # [B]

            old_logprobs = self._policy_logprobs(self.policy, queries, responses)
            ref_logprobs = self._policy_logprobs(self.ref_policy, queries, responses)

            token_invalid, reward_invalid = self._make_masks(responses)
            kl = old_logprobs - ref_logprobs
            rewards = -self.cfg.kl_coef * kl

            end_idx = (~reward_invalid).sum(dim=1).clamp_min(1) - 1
            rewards[torch.arange(rewards.size(0), device=device), end_idx] += scores
            advantages = self._discounted_returns(rewards, token_invalid)

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
            "score": scores.mean().item(),
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
        step = 0
        while step < max_steps:
            for batch in dataloader:
                metrics = self.train_step(batch["input_ids"], sample_fn)
                step += 1
                print(f"[step {step}] {metrics}")
                if step >= max_steps:
                    break
