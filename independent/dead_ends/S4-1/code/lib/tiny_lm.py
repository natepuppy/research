"""A tiny transformer with a single-token verifiable answer, and exact GRPO on it.

Used by T2 (fixation vs drift) and T3 (what does a reset reset).

Why single-token answers: with one answer token the per-prompt success
probability p_i = softmax(logits_i)[correct_i] is *exact* from one forward
pass, no sampling. Group sampling for GRPO is a multinomial draw from the
same softmax, so a GRPO step costs one forward/backward on the prompt batch.
That is what makes 10^4-step runs with 20 seeds affordable on a laptop, and
what lets us measure fixation/extinction and interference without estimation
noise.

Task families: a family f is a random linear hash
    answer(x) = (sum_j w_fj x_j + b_f) mod n_answers.
Pretraining on a family samples answers from a per-prompt target distribution
whose mass on the correct answer is p_i^target, so the *base* model's success
probabilities are spread over (0, 1) by design, which is what the fixation
theory needs. Pretraining on a family with p^target = 1 just teaches the rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Family:
    weights: torch.Tensor  # (prompt_len,)
    bias: int
    n_answers: int

    def answers(self, prompts: torch.Tensor) -> torch.Tensor:
        return ((prompts.long() * self.weights.long()).sum(-1) + self.bias) % self.n_answers


def make_family(prompt_len: int, n_answers: int, vocab: int, seed: int) -> Family:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randint(1, vocab, (prompt_len,), generator=generator)
    bias = int(torch.randint(0, n_answers, (1,), generator=generator))
    return Family(weights, bias, n_answers)


def make_prompts(n: int, prompt_len: int, vocab: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n, prompt_len), generator=generator)


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, prompt_len: int = 8, n_answers: int = 16, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.vocab, self.prompt_len, self.n_answers = vocab, prompt_len, n_answers
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(prompt_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_answers)

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        h = self.tok(prompts) + self.pos[None, : prompts.shape[1]]
        h = self.blocks(h)
        return self.head(self.norm(h.mean(dim=1)))

    def features(self, prompts: torch.Tensor) -> torch.Tensor:
        h = self.tok(prompts) + self.pos[None, : prompts.shape[1]]
        return self.norm(self.blocks(h).mean(dim=1))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def success_probs(model: nn.Module, prompts: torch.Tensor, answers: torch.Tensor, batch: int = 2048) -> torch.Tensor:
    out = []
    for start in range(0, prompts.shape[0], batch):
        logits = model(prompts[start : start + batch])
        probs = logits.softmax(-1)
        out.append(probs.gather(1, answers[start : start + batch, None]).squeeze(1))
    return torch.cat(out)


@torch.no_grad()
def entropy(model: nn.Module, prompts: torch.Tensor, batch: int = 2048) -> float:
    total = 0.0
    for start in range(0, prompts.shape[0], batch):
        logp = model(prompts[start : start + batch]).log_softmax(-1)
        total += float(-(logp.exp() * logp).sum(-1).sum())
    return total / prompts.shape[0]


def pretrain(model: nn.Module, prompts: torch.Tensor, answers: torch.Tensor, target_p: torch.Tensor, steps: int, lr: float = 1e-3, batch: int = 256, seed: int = 0) -> float:
    """Cross-entropy on soft targets: mass target_p on the correct answer, rest uniform.

    Returns the final training loss.
    """
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, A = prompts.shape[0], model.n_answers
    loss_value = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=generator)
        p = target_p[idx]
        target = torch.full((batch, A), 0.0)
        target += ((1.0 - p) / (A - 1))[:, None]
        target[torch.arange(batch), answers[idx]] = p
        logp = model(prompts[idx]).log_softmax(-1)
        loss = -(target * logp).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach())
    return loss_value


def grpo_loss(
    model: nn.Module,
    prompts: torch.Tensor,
    answers: torch.Tensor,
    group_sizes: torch.Tensor,
    ref_logits: torch.Tensor | None = None,
    beta: float = 0.0,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """One exact GRPO step on a batch of prompts with (possibly per-prompt) group sizes.

    Sampling and the policy gradient use the same forward pass, so this is
    strictly on-policy (mu = 1 in GRPO's notation); the clipped ratio is 1 and
    is omitted. Group-normalised advantages: (r - mean) / std, zero when the
    group is all-pass or all-fail. The KL term, if any, is the exact
    categorical KL(pi || pi_ref) on the prompt batch.
    """
    logits = model(prompts)
    logp = (logits / temperature).log_softmax(-1)
    probs = logp.exp().detach()
    B = prompts.shape[0]
    rows = torch.repeat_interleave(torch.arange(B), group_sizes)
    samples = torch.multinomial(probs[rows], 1, generator=generator).squeeze(1)
    reward = (samples == answers[rows]).float()
    # segment mean/std
    group_sum = torch.zeros(B).index_add_(0, rows, reward)
    group_sq = torch.zeros(B).index_add_(0, rows, reward.square())
    sizes = group_sizes.float()
    mean = group_sum / sizes
    var = (group_sq / sizes - mean.square()).clamp_min(0.0)
    std = var.sqrt()
    advantage = torch.where(std[rows] > 1e-8, (reward - mean[rows]) / std[rows].clamp_min(1e-8), torch.zeros_like(reward))
    chosen_logp = (logits.log_softmax(-1))[rows].gather(1, samples[:, None]).squeeze(1)
    # standard GRPO: mean over the group, then mean over prompts (so unequal group sizes get equal weight)
    pg = -(advantage * chosen_logp / sizes[rows]).sum() / B
    kl = torch.zeros(())
    if ref_logits is not None and beta > 0.0:
        logp_now = logits.log_softmax(-1)
        logp_ref = ref_logits.log_softmax(-1)
        kl = (logp_now.exp() * (logp_now - logp_ref)).sum(-1).mean()
    loss = pg + beta * kl
    info = {
        "mixed_frac": float((std > 1e-8).float().mean()),
        "batch_acc": float(reward.mean()),
        "kl": float(kl.detach()),
        "group_correct": group_sum.detach().long(),  # k per prompt
    }
    return loss, info
