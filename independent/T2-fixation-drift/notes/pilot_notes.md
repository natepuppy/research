# T2 pilot: fixation vs extinction of rare correct modes under finite-group GRPO

Run 14 Oct 2026 on an M1 Pro CPU, default grid: 3 seeds × (5 group sizes × 2 learning rates + 1 adaptive-G run), 1500 exact GRPO steps each on a 104k-parameter transformer with 2048 trained prompts and 512 held-out prompts. Rows in `results/cells.csv` (per p₀-bin × config), `results/budget.csv` (per run), `results/windows.csv` (calibration), `results/prompts_seed0.csv` (every prompt's p₀ → p_T for seed 0). `results/v0_pooled_drift/` is the first run, before the two fixes described under "What changed".

## Verdict

`INCONCLUSIVE` by the pre-registered rule (chain MAE 0.122, between the 0.10 support and 0.20 kill thresholds). The two qualitative predictions are strongly supported; the quantitative calibration is good at one learning rate and biased at the other, for a reason that is now identifiable.

## Results against the predictions

**P1, bimodal kernel at the boundary: supported.** For prompts with p₀ ∈ [0.01, 0.05), the final state is bimodal in 87% of (seed, G, lr) cells: a fraction fixes (p_T > 0.88), the rest *declines* by more than one log-odds unit, and almost nothing stays or lands in between (1–5% "intermediate" mass). Prompts with equal base success end at opposite extremes, repeatably. This is the shape "RLVR is a kernel, not a function" reported and did not explain.

**P2, fixation rises with G with diminishing returns: supported.** Boundary fixation at lr 3e-4: G = 2 → 0.15, 4 → 0.22, 8 → 0.30, 16 → 0.36, 32 → 0.41 (ρ = +0.87 across lr); roughly linear in log G, concave, no interior optimum inside the range tested. At fixed step count, more rollouts per prompt keep helping; the IsoCompute saturation would need a fixed total rollout budget (B·G constant), which is the `--full` axis.

**P4, adaptive group size at matched budget: supported, modestly.** G_i ∝ 1/p̂_i from the running reward history (clipped to [2, 32]) with the same 512 rollouts per step as uniform G = 8 gives boundary fixation 0.464 ± 0.032 vs 0.421 ± 0.008 (+0.043), at a cost of slightly lower average success (0.74 vs 0.76) and worse held-out generalisation (0.17 vs 0.21). In the first run (group-sum loss, which overweighted large groups) adaptive G *hurt* by 0.11; the gain only appeared with the standard group-mean loss. Any adaptive-G paper must state its loss normalisation.

**Calibrated chain, quantitative:**

| | all 210 cells, MAE | signed bias | boundary cells (p₀ ∈ [0.01, 0.05)), MAE | boundary bias |
|---|---|---|---|---|
| pooled neutral drift (v0 model) | 0.144 | −0.096 | 0.289 | −0.289 |
| p-dependent interference (current) | 0.122 | +0.075 | 0.150 | +0.045 |
| current, lr 1e-3 only | 0.121 | +0.028 | | |
| current, lr 3e-4 only | 0.123 | +0.122 | | |

The chain has no parameter fitted to fixation outcomes: η_eff comes from regressing observed per-step log-odds changes on √(k(G−k))/G for in-batch prompts, and (μ_int, σ_int) come from held-out prompts, both per 50-step window.

## The finding that was not in the idea file

**Interference is not neutral drift. It is selection against rare modes.** Held-out prompts, which receive no gradient of their own, drift at −0.003 to −0.008 log-odds per step when their p is below 0.1 and at +0.002 when p is above 0.7 (seed 0, G = 8). Pooled over all held-out prompts this averages to ~0, which is why the v0 chain over-predicted boundary fixation by 0.29 while looking well calibrated elsewhere. Over 1500 steps the rare-mode drift is −5 to −10 log-odds: updates on *other* prompts are what extinguishes a rare correct mode, not sampling noise. In population-genetics terms, hitchhiking here is systematically deleterious for rare alleles, because every update sharpens the shared representation toward whatever is already common. This is the "entropy mechanism" (Cov(log π, A) < 0 for rare tokens) seen from the per-prompt side, and it gives P3 a mechanism: anything that reduces cross-prompt sharpening (KL anchor, reference reset, entropy bonus) should reduce extinction. That is testable in `--full`.

## Why lr 3e-4 is still biased

With p-dependent drift the chain *under*-predicts fixation at the low learning rate (it predicts ~0 for p₀ < 0.02, observed 0.14–0.25). The likely cause is Adam momentum: a prompt's own update is spread over the next ~10 steps (β₁ = 0.9), so the same-step regression captures only part of it and the rest is misattributed to interference noise. At lr 1e-3 interference dominates and the misattribution matters less (bias +0.03). The fix is to regress Δℓ over a 10-step window after each own update, or to run the calibration with SGD as a check. Either is cheap.

## What changed between v0 and the current run

1. Interference calibrated per current-p bin instead of pooled (theory refinement, above).
2. `grpo_loss` switched from group-sum to the standard group-mean normalisation, so unequal group sizes get equal weight. The own-update term is now η√(k(G−k))/G (checked numerically to 4e-8 in `tabular_step_check`).

## Next steps that would make this a paper result

1. Momentum-aware η_eff (or SGD control) to remove the lr 3e-4 bias; target MAE < 0.08.
2. Fixed total rollout budget sweep (B·G = const) for the IsoCompute saturation prediction.
3. The P3 test: KL anchor / reference reset / entropy bonus vs measured rare-mode drift.
4. Closed form: the diffusion limit of the chain with p-dependent drift μ_int(p) gives a fixation probability u(p₀) via the backward Kolmogorov equation; the measured μ_int(p) is roughly −c·(p̄ − p), which is solvable.
5. Then Qwen2.5-0.5B on Countdown (E2) and the public base/RL checkpoint re-analysis (E3) on the 3080.

## Re-running

```bash
python code/t2_fixation.py                 # default grid, ~50 min on an M1 Pro
python code/t2_fixation.py --full --device cuda   # 4096 prompts, 800 steps... 10 seeds
```
