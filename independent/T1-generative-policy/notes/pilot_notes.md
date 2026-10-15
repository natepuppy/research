# T1 pilot: policy-head decomposition on point-mass tasks

Run 14 Oct 2026 on an M1 Pro CPU, default grid: 8 heads × 3 tasks × 3 seeds, 80 PPO iterations × 256 envs × 64 steps (~1.3M env steps per run). Raw rows in `results/runs.csv`, per-task summary in `results/summary.csv`, verdict in `results/verdict.txt`.

## Verdict

`INCONCLUSIVE` by the pre-registered rule: flow8 did not beat gdiag *on average* beyond seed spread on any task, so these tasks do not reproduce the humanoid gap and the pilot cannot attribute it. What it did show is below.

## Deterministic-evaluation return (mean ± std over 3 seeds; worst seed in brackets)

| head | reach | fork | corridor |
|---|---|---|---|
| gdiag | 31.3 ± 1.4 [29.4] | 17.7 ± 18.7 [−8.7] | 49.2 ± 1.1 |
| glowrank (Σ = D + UUᵀ, r = 4) | 31.0 ± 0.8 | 26.9 ± 3.1 | 47.6 ± 1.2 |
| gou (ρ = 0.8) | 32.5 ± 0.1 | 22.4 ± 6.1 | 43.5 ± 1.5 |
| gmip (two-step residual) | 32.7 ± 0.6 | 17.4 ± 19.4 [−10] | 49.6 ± 1.1 |
| ggmm (K = 4) | 29.6 ± 3.7 | **30.6 ± 0.3** | 29.2 ± 29.4 [−12] |
| flow1 (FPO, 1 Euler step) | 32.0 ± 1.0 | 26.8 ± 4.4 | −25.0 ± 4.6 |
| flow8 (FPO, 8 steps) | 31.6 ± 1.1 | **29.9 ± 0.6** [29.2] | −9.8 ± 1.5 |
| flowfrozen | diverged | diverged | diverged |

## What the three tasks said

- **Reach (control).** All heads within noise. Good: the trainer is not biased toward any head.
- **Fork (bimodal start).** The effect is in *reliability*, not mean. gdiag and gmip each lost one seed of three (stuck pushing into the wall); flow8 and ggmm never did. Among Gaussian heads, the mixture closes the gap completely (30.6 vs 29.9) and the low-rank covariance closes 76% of it. Temporal correlation alone (gou) closes 38%; iterative computation alone (gmip) closes nothing. Sampled-action diagnostics show *no* bimodality at visited states for any head after training (every policy commits to one side), so the mechanism is exploration-time symmetry breaking, not multimodal control at convergence. This is consistent with the hypothesis that correlated/multimodal exploration, not expressivity at decision time, is what flows buy.
- **Corridor (null-space penalty).** Flows lose badly. The FPO surrogate gives the flow no cheap way to shrink its exploration noise, so unit-variance noise in six penalised directions costs it ~60 return per episode. glowrank did *not* beat gdiag here (47.6 vs 49.2): with PPO's entropy bonus the diagonal Gaussian already learns small σ in the penalised directions, so a learned covariance was not needed. The corridor as built is not a test of correlated exploration; it is a test of noise-scale control, and Gaussians win it.

## Caveats

- 3 seeds, one PPO configuration, no per-head tuning. The Fork reliability result needs 10+ seeds to be more than suggestive.
- `flowfrozen` (flow centre + Gaussian noise trained with the FPO surrogate on the perturbed action) diverged on every task. The construction is wrong as a head: the surrogate pulls the flow toward *noised* actions with no mechanism to separate centre from noise. A correct version needs a likelihood-tractable centre (train the flow on advantage-weighted *mean* actions, or use the GoRL-style entropy knob). Drop or rebuild before the 3080 run.
- Point-mass tasks have 2–8 action dims and no contact; the humanoid gap lives at 20+ dims with contact. The pilot tests the *mechanism isolation design*, not the gap itself.

## What this means for the paper

The design works: the heads train under one trainer and the diagnostics discriminate (cross-correlation 0.1 for diagonal heads vs 0.3–0.5 for glowrank/flows; effective rank 8 → 4–5 for flows on corridor). Two things to carry to MJX: (1) measure reliability (IQM and worst-seed), since that is where flows separated here; (2) add a noise-scale-control axis, since flows under FPO cannot shrink exploration and that is a real disadvantage the FPO/FFM papers do not report.

## Re-running

```bash
python code/t1_policy_heads.py                 # default grid, ~1 h on an M1 Pro
python code/t1_policy_heads.py --resume        # skip runs already in results/runs.csv
python code/t1_policy_heads.py --summarize     # recompute summary/verdict only
python code/t1_policy_heads.py --full --device cuda   # 10 seeds, 150 iterations, 512 envs
```
