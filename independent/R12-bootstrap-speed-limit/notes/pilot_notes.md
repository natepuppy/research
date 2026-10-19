# R12 pilot notes

Two runs. **Pilot** (grid in README, 256 configs × 3 seeds, 400k-update cap, every cell converged; CPU, 4 min). **Confirmation** (fresh grid sharing no values with the pilot, 192 configs × 3 seeds; CPU, 1 min). Logs: `logs/pilot.log`, `logs/confirm.log`. Raw: `results/runs.csv`, `results/confirm/runs.csv`.

## Verdicts

- Pilot, pre-registered law `hops · max(K, c·L/B)`: **KILLED** by the rule. SGD R² = 0.883 (needed 0.9); Adam R² = 0.902 but its K2 could not be evaluated because the grid had a single K value in the replay-limited regime (a test-design flaw, not a data result). K2 for SGD passed (slope on K: 0.91 target-limited, 0.03 replay-limited). K3 failed for both (n-slope −1.9 / −1.5 instead of −1), because the grid included n ≥ L cells where no bootstrapping happens at all.
- Post hoc (`results/posthoc.txt`): the target-limited regime is *exactly* `(⌈L/n⌉ − 1)·K` (median ratio 1.002 over 60 cells with K ≥ 100), the replay-limited regime scales as `L²/(nB)` (SGD slope 0.99, R² 0.99), and the two add rather than compete. Additive law `(⌈L/n⌉ − 1)·K + c·⌈L/n⌉·L/B`: identity-fit R² 0.996 (SGD), 0.924 (Adam).
- Confirmation of the additive law, c fixed from the pilot, out of sample: **SUPPORTED**. R² = 0.997 (SGD, threshold 0.95), 0.971 (Adam, threshold 0.90). Median t/pred 0.97 and 0.96.

## Pilot table (n = 1, geometric mean over 3 seeds; `pred` is the additive law with the pilot-fitted c)

| opt | L | K | B | t_prop | (hops−1)·K | t_prop / additive pred |
|---|---|---|---|---|---|---|
| sgd | 64 | 1 | 32 | 422 | 63 | ~1.0 |
| sgd | 64 | 10 | 32 | 790 | 630 | ~1.0 |
| sgd | 64 | 100 | 32 | 6303 | 6300 | 1.00 |
| sgd | 64 | 1000 | 32 | 63003 | 63000 | 1.00 |
| sgd | 64 | 1 | 4 | 3422 | 63 | ~1.0 |
| sgd | 64 | 100 | 4 | 7445 | 6300 | ~1.0 |
| sgd | 16 | 1 | 32 | 26 | 15 | ~1.0 |
| sgd | 16 | 1000 | 32 | 15001 | 15000 | 1.00 |
| adam | 64 | 1 | 32 | 186 | 63 | ~0.9 |
| adam | 64 | 1000 | 32 | 63005 | 63000 | 1.00 |

Full tables in `results/summary.csv` and `results/confirm/summary.csv`.

## What this says

1. **Target networks impose a hard speed limit.** With a hard target network refreshed every K updates, the value of a state L steps from the reward cannot move before (⌈L/n⌉ − 1) refreshes, and in the pilot it moves at exactly that time: t_prop = (⌈L/n⌉ − 1)·K + O(1). No step size, batch size or optimiser changes this. It is a counting argument, and the data honour it to three or four significant figures.
2. **Replay imposes a second, additive cost.** With no target network (K = 1), propagation takes ∝ L²/(nB) updates: L/n hops, each waiting ~L/B updates for the frontier transition to be sampled. The two costs add: `t ≈ (⌈L/n⌉−1)·K + c·⌈L/n⌉·L/B`. The pre-registered max() form was wrong in the crossover region, which is where real DQN settings sit.
3. **n-step is a factor n in both terms.** Within cells where bootstrapping actually happens (n < L) the slope on n is −1.2 (SGD) / −1.2 (Adam). When n ≥ L the problem is Monte-Carlo and the law degenerates, as it should.
4. **The constant depends on the optimiser.** c_SGD = 2.63 (per-hit step 0.3), c_Adam = 4.21 (lr 0.05). Adam is also noisier in the replay-limited regime (10–90% ratio range 0.47–1.00 in the confirmation), because its per-state normalisation makes the first few hits on a frontier state take large steps.

## Caveats

- Tabular, deterministic chain, one reward at the end, uniform replay with a stationary buffer. The "frontier" is a single state. In stochastic environments it is a distribution and the hop count becomes an expectation; with function approximation a hop can cover more than n states because of generalisation (an effective n_eff > n), which is the main thing the deep version must measure.
- The convergence criterion (V(start) ≥ half its true value) measures arrival of the signal, not accuracy.
- Polyak-averaged targets were not tested; K_eff ≈ 1/τ is the natural guess and needs its own check.
- Prioritised replay should remove the replay term (the frontier has the largest TD error, so it is sampled immediately) and leave the target term untouched. Not tested.

## Why this matters for practice

Nature-DQN: K = 10k env steps / 4 = 2.5k updates, B = 32, n = 1. For an episode of L = 100 steps the target term is 99 × 2.5k ≈ 250k updates = 1M env steps before a terminal reward *first* reaches the start state; the replay term is c·100·100/32 ≈ 800 updates, negligible. Rainbow's n = 3 cuts the first term to ~330k env steps; R2D2's n = 5 with K = 2.5k to 200k. Deep nets generalise and real tasks have dense rewards, so this is a lower bound on a stylised quantity, but it says why target period and n-step interact, and why a very long target period is expensive in exactly the sparse-reward tasks where people reach for it.

## 3080 plan

1. Stochastic chains and gridworlds (frontier as a distribution): does the law hold in expectation?
2. Linear and deep FA on the chain: measure n_eff from the propagation time and relate it to the feature similarity between adjacent states.
3. DQN on MinAtar with rewards sparsified (reward only on episode success): propagation time vs (K, n), the additive law with a fitted n_eff, and the design rule K·(L/n_eff) ≤ budget.
4. Polyak targets (K_eff = 1/τ?) and PER (replay term removed?). Each is a one-figure experiment.
Compute: everything but (3) is CPU; (3) is a few GPU-hours.
