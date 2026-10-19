# R7 pilot notes

Run: 16 parents × 30k steps, batched DQN on CartPole (CPU, M1 Pro, 34 min). Forks at 0, 2.5k, 5k, 10k, 15k, 20k, 25k; 3 children per (parent, fork, condition); conditions `all`, `explore`, `batch`. Final score = mean of the last three greedy evaluations (4 episodes each). Log: `logs/pilot.log`. Raw: `results/runs.csv`, `results/parents.json`.

## Verdict

**SUPPORTED** (borderline). K1 true at the T/2 boundary, K2 true by the letter of the rule but not in spirit, K3 strongly true.

Note on the summariser: the first run of `--summarize` took T to be the last fork time (25k) instead of the training length (30k), which excluded the 15k fork from "≤ T/2" and printed KILLED. The README defines T as the training length, so the code was corrected and re-run on the saved `runs.csv`; nothing else changed. Both lines are in `logs/pilot.log`.

## Lock-in curve (ICC of children's final score, grouped by parent)

| fork | all | explore | batch | R² on parent's visible score (all) |
|---|---|---|---|---|
| 0 | 0.14 [−0.12, 0.44] | −0.11 | 0.36 | — |
| 2.5k | 0.18 [−0.02, 0.32] | 0.33 | 0.35 | 0.04 |
| 5k | 0.43 [0.09, 0.66] | 0.23 | 0.51 | 0.06 |
| 10k | 0.45 [0.17, 0.62] | 0.53 | 0.32 | 0.00 |
| 15k | **0.63 [0.35, 0.75]** | 0.45 | 0.68 | **0.00** |
| 20k | 0.39 [0.02, 0.62] | 0.70 | 0.25 | 0.31 |
| 25k | 0.76 [0.53, 0.85] | 0.84 | 0.63 | 0.68 |

Brackets: bootstrap 90% CI over parents.

## What this says

- **K1 (lock-in by T/2).** ICC_all rises from ~0.15 at fork 0 (a sanity check: children of the same init that share nothing else should be nearly uncorrelated, and are) to 0.63 at 15k. So by half-way, about 60% of the variance in where a run ends is already determined by the state of that run. Caveat: it passes exactly at the boundary, and the curve is not monotone (0.39 at 20k). With 16 parents the CIs are wide; the 20k dip is within noise of the 15k value.
- **K3 (fate is not visible).** At 15k the parent's own evaluation score explains 0% of the children's final score, while the parent's identity explains 63%. Whatever is locked in at 15k is not the training curve. This is the result worth building on: it says early-stopping or pruning rules based on the visible score are looking at the wrong thing, and that a fork-based "fate" measurement would be a different kind of predictor. Visible score only starts to predict at 20k–25k, when there is little training left.
- **K2 (which noise source).** The pre-registered rule asks for |ICC_explore − ICC_batch| ≥ 0.15 at two or more forks and gets it at five. But the sign flips fork to fork (batch higher at 5k and 15k, explore higher at 10k, 20k, 25k), so this is not evidence that one source matters more than the other. It is evidence that 3 children × 16 parents cannot resolve the difference. Treat K2 as not answered.

## Caveats

- CartPole is a bad environment for this: scores saturate at 200 and DQN on CartPole is known to collapse and recover, so "final score" is noisy even with 3 evaluations averaged. Two parents hit 200 exactly.
- 16 parents and 3 children is the minimum that gives an ICC at all. CIs of ±0.2 are the result.
- One architecture, one algorithm, one set of hyperparameters.
- The ICC at fork 0 (same init, everything else fresh) is 0.14 with CI covering 0, so initialisation alone contributes little here. That is consistent with Frankle et al. on supervised nets but is a one-environment observation.

## 3080 plan

- MinAtar (5 games, DQN) and DMC (3 tasks, SAC or TD3): 32 parents, 5 children per condition, 8 fork times. The batched implementation puts all children of one fork time in one process; on a 3080 a MinAtar fork sweep is roughly 1 GPU-hour per game.
- Add a fourth condition: `all` plus a fresh replay buffer, to separate "state of the network" from "state of the buffer".
- Report lock-in time t* = first fork with ICC ≥ 0.5 per game, and compare t*/T across games; test whether t* correlates with the step at which the visible score becomes predictive.
- Use the fork machinery as the ground truth for an "is this run doomed" classifier trained on cheap features (gradient norm, dormant fraction, TD-error distribution), which is the practical payoff.
