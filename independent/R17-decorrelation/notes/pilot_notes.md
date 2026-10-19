# R17 pilot notes

One run. Batched DQN on CartPole (200-cap), 4 minibatch samplers × 2 target periods × 16 seeds = 128 runs, 30k steps, greedy evaluation every 1k, gradient cosine to a fixed 1024-transition buffer gradient every 1k. CPU, 14 min. Log: `logs/pilot.log`. Raw: `results/runs.csv`, `results/per_run.csv`. Post hoc: `results/posthoc.txt`.

## Verdict

**KILLED.** K1 (correlated batches hurt at K = 100): AUC(iid) − AUC(win1) = **+8.3** (needed ≥ 20), 12/16 wins (p = 0.08). K2 (the gap halves at K = 2500): gap = +8.1, not ≤ 4.2. K3 (reuse control, win1 − recent ≥ 20 at both K): **−48.1** at K = 100, +23.9 at K = 2500.

The hypothesis as posed (correlation hurts, through the target) is dead: correlation barely matters at either target period. The reuse control came out backwards at the fast target, which is the finding.

## Cell table (AUC = mean of 30 evaluations; final = mean of last 3)

| K | sampler | final (sd) | AUC (sd) | grad cos |
|---|---|---|---|---|
| 100 | iid | 113.7 (38.1) | 119.1 (19.1) | 0.66 |
| 100 | win4 | 125.3 (38.5) | 126.6 (23.9) | 0.62 |
| 100 | win1 | 101.2 (50.8) | 110.8 (22.8) | 0.59 |
| 100 | **recent** | **183.0 (22.6)** | **158.9 (14.2)** | 0.10 |
| 2500 | iid | 146.2 (30.6) | 122.4 (16.0) | 0.51 |
| 2500 | win4 | 133.2 (17.3) | 111.9 (12.6) | 0.50 |
| 2500 | win1 | 139.2 (23.0) | 114.3 (15.1) | 0.45 |
| 2500 | recent | 139.5 (35.5) | 90.4 (12.8) | 0.15 |

Learning curves (mean score at 5k / 10k / 20k / 30k): K = 100 iid 170 / 113 / 91 / 117, win1 166 / 107 / 79 / 102, recent 143 / 171 / 164 / 181; K = 2500 iid 18 / 107 / 159 / 151, recent 15 / 48 / 133 / 127.

## What happened

1. **Within-batch correlation costs almost nothing at fixed data.** One contiguous window of 32 (win1) versus 32 independent draws: −8 AUC points at both K, 4/16 win1 wins at each (p = 0.08). Four windows of 8 (win4) is indistinguishable from iid (+7 at K = 100, −10 at K = 2500, both n.s.). The gradient cosine to the buffer gradient does drop with correlation (0.66 → 0.59 at K = 100), and it does not matter: per-run Spearman(grad cos, AUC) = −0.02.
2. **No replay beats replay with a fast target.** Training on the most recent 32 transitions (maximal correlation, zero reuse) wins against uniform replay at K = 100 by 40 AUC points and 69 final-return points, 15/16 and 14/16 paired wins (p = 0.001, 0.004). Its gradient is nearly orthogonal to the buffer gradient (cosine 0.10); it is optimising a different, on-policy objective, and doing better at the task.
3. **With a slow target the ordering flips on AUC and vanishes at the end.** At K = 2500 nothing learns before 10k steps (the target-period speed limit, R12), replay then climbs to 150–160 and recent-only to 130; AUC favours replay by 32 (1/16 recent wins) but the final returns are equal (7/16).
4. **The replay learners at K = 100 collapse.** Every replay sampler peaks at ~165 at 5k steps and declines to 80–120 by 20k. The recent-only learner does not: 143 → 171 → 164 → 181. This is the reach-ceiling-then-collapse pattern of R11, and the recent-only sampler is immune to it by construction: every failure the agent makes is in its next few minibatches, so it cannot forget how to fail.

## What this says

The textbook's decorrelation half of the replay story does not survive a controlled test on this task: the effect is under 10 AUC points and not significant. Replay's value here is reuse, and it is only visible when the target is slow; with a fast target, replay *hurts* relative to on-policy data, through a collapse that the on-policy learner does not suffer. Two explanations compete and are not separated by this pilot: (a) stale replay targets (actions and next-states from older, worse policies) feed the collapse; (b) the on-policy learner keeps its failure transitions (R11's mechanism). (b) is directly testable with R11's reservoir: if `iid + terminal reservoir` at K = 100 matches `recent`, it was the terminals.

## Caveats

- CartPole only, 30k steps, one learning rate and network; `recent` with 32 transitions per update at one update per step is a 32× data-reuse ratio of zero, and its advantage may not survive tasks where the current policy's data is not informative enough on its own.
- Two target periods. K = 100 vs 2500 is a coarse dial; the crossover K is unknown.
- AUC over 30k steps penalises K = 2500 for its slow start, which is a propagation effect (R12) rather than anything about sampling.
- Gradient cosine was measured against a uniform buffer sample, which is the replay objective's gradient; for `recent` the low cosine is a statement about objectives, not estimate quality.
