# R13 pilot notes

One run. Exact-gradient softmax PG (η = 2, cap 50k iterations) on four 10×10 layouts × nine training-start distributions = 36 configurations, deterministic. CPU, ~13 min. Log: `logs/pilot.log`. Raw: `results/runs.csv`. Post hoc: `results/posthoc.txt`.

## Verdict

**KILLED.** K1: Spearman(log D_μ, log t) = **−0.67** over the 23 configs with finite D_μ (needed ≥ +0.7). K2: Spearman(log D_0, log t) = **−0.52** over 31 converged configs (needed ≥ +0.7); 5 of 36 configs hit the cap, so "every config converges" is also false. K3 within-layout D_0 correlations: open −0.63, wall −0.40, snake +0.70, rooms −0.05.

The sign is wrong. On the easy layouts a *larger* mismatch coefficient goes with *faster* learning.

## The table that matters (iterations to 90% of V*(ρ))

| layout | δ_start (β=0) | β=0.1 | β=0.3 | μ = d^{π*} | uniform (β=1) | far | far_only |
|---|---|---|---|---|---|---|---|
| open  | 964  | 892  | 902  | 1393  | 3470  | 1591 | 41365 |
| wall  | 1051 | 855  | 797  | 1306  | 2901  | 2419 | 42205 |
| rooms | 1446 | 1037 | 904  | 1216  | 2445  | 1625 | cap |
| snake | cap  | 19377| 11004| 10339 | 21530 | cap  | cap |

D_μ for δ_start is ∞ (μ misses every state but one), for d^{π*} it is exactly 1, for uniform it is |S|·max d^{π*} ≈ 30–35. The theory's best case (μ = d^{π*}, D_μ = 1) is 20–45% *slower* than the theory's worst case (single start, D_μ = ∞) on open and wall, and uniform is 2–4× slower still.

## What happened

1. **Coverage dilutes the gradient.** With exact gradients the per-iteration update at state s is proportional to d^{π}_μ(s). Mass placed on states the evaluation start never reaches (ρ = δ_start) is mass taken away from the states that matter for V^π(ρ). The uniform start spends ~99% of its gradient budget on the wrong states; the single start spends all of it near the evaluation start and the goal path. The theory's D_μ term is an upper bound on how badly things can go, and on an easy layout the bound is loose by orders of magnitude and the sign of the real effect is the opposite.
2. **Small β is a free win.** β = 0.1–0.3 beats β = 0 on every layout (open 892 vs 964; wall 797 vs 1051; rooms 904 vs 1446), because a little mass on states away from the start breaks the "softmax plateau" (Mei et al. 2020) at those states without starving the start. The curve of t against β is U-shaped, with the minimum at β ≈ 0.1–0.3 for all three easy layouts.
3. **Snake is the regime the theory describes.** The snake layout (V* = 0.045 from the start, a 60-step corridor) does not converge from a single start within 50k iterations; the gradient at the start is exponentially small because the initial policy almost never reaches the goal. Here coverage is the thing that helps: the within-snake D_0 correlation is +0.70 (n = 5), the theory's sign. The best start distribution is d^{π*} (10339), then β = 0.6 (9387), then β = 0.3 (11004). Uniform (21530) is again worse than d^{π*}, by 2×, because D_μ = 3.7 vs 1.
4. **far_only never works.** Starting only on states the optimal policy visits least (D_μ = ∞ because μ puts no mass on the optimal path from ρ) takes 41–42k iterations on the easy layouts and hits the cap on rooms and snake. This is the one case where the ∞ actually means something: those starts never see the evaluation region at all.
5. **Where μ puts its mass on the start state is the best single predictor on easy layouts.** Post hoc, Spearman(log t, μ(start)) = −0.65 on open/wall/rooms (more mass on the evaluation start → faster), and −0.55 overall. The overlap and total-mass-on-optimal-path proxies predict nothing (+0.07, −0.05).

## What this says

The distribution-mismatch coefficient is a worst-case quantity for a hard-exploration regime. In the regime where the initial policy already reaches reward with non-negligible probability, the dominant effect is the *opposite* one: gradient mass spent off the evaluation distribution is wasted, and concentrated starts win. The practical rule that comes out of the pilot is two-regime: if the initial policy reaches the goal from ρ at all, start from ρ with a small (10–30%) exploratory mixture; if it does not, start as close to d^{π*} as you can and avoid uniform. Neither D_μ nor D_0 tells you which regime you are in; V^{π_0}(ρ) relative to V* does (open/wall/rooms: V^{π_0}(ρ)/V* ≈ 0.03–0.05 and learning is fast; snake: ≈ 10⁻⁵ and learning stalls).

## Caveats

- Exact gradients, one step size, one initial policy (uniform softmax), four layouts, 36 points. The U-shape in β is consistent across three easy layouts, but n = 1 per cell (the process is deterministic).
- Iterations, not samples. With sampled gradients the variance of the estimate depends on μ too, and broader μ may reduce variance at states near ρ; this could move the optimum β upward. That is the first thing the 3080 version should measure.
- The 90%-of-V* criterion; a 99% criterion would weight the tail of convergence, where plateau escape at far states matters more.
- `snake` has |S| = 64 reachable states and a tiny V*, so its 5 converged points are a thin basis for the +0.70.
