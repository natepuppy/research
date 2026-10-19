# R14 pilot notes

One run. Batched DQN on CartPole (200-step cap), 4 target periods K × 3 constant ε × 8 seeds = 96 runs, 30k steps each, greedy evaluation every 1k, churn measured on 256 probe states after every update. CPU, 11 min. Log: `logs/pilot.log`. Raw: `results/runs.csv` (per eval), `results/per_run.csv`. Post hoc: `results/posthoc.txt`.

## Verdict

**KILLED.** K1 (K controls churn, negatively): Spearman(churn, log K) over the 12 cells = **+0.88**, the opposite sign (needed ≤ −0.8). K2 (substitution: ε's AUC gain rises with K): gains were +0, +79, +65, +73 for K = 20, 100, 500, 2500; Spearman +0.40 (needed ≥ 0.8). K3 (at ε = 0, early churn predicts AUC): **+0.72** (needed ≥ 0.4), passed. The decision rule needs K1, so the verdict is KILLED.

## Cell table

| K | ε | final (sd) | AUC | churn % | early churn % |
|---|---|---|---|---|---|
| 20 | 0 / 0.05 / 0.2 | 9.5 / 8.9 / 8.9 | 9.5 / 9.9 / 9.5 | 0.00 / 0.07 / 0.02 | 0.01 / 0.20 / 0.07 |
| 100 | 0 | 78.0 (86.8) | 58.8 | 0.34 | 0.25 |
| 100 | 0.05 | 68.9 (69.4) | 83.7 | 0.74 | 0.87 |
| 100 | 0.2 | 157.1 (49.7) | 138.1 | 1.83 | 1.52 |
| 500 | 0 | 77.8 (73.6) | 66.7 | 1.11 | 0.46 |
| 500 | 0.05 | 101.3 (24.8) | 114.4 | 2.21 | 1.50 |
| 500 | 0.2 | 113.2 (20.0) | 132.0 | 3.40 | 1.71 |
| 2500 | 0 | 63.7 (71.8) | 51.2 | 0.81 | 1.03 |
| 2500 | 0.05 | 159.7 (20.3) | 132.6 | 2.28 | 3.14 |
| 2500 | 0.2 | 130.4 (11.1) | 123.8 | 3.30 | 5.3 |

## What happened

1. **The premise was backwards.** Churn *rises* with target period. With K = 100 the online network is re-anchored to a fresh target every 100 updates and its greedy action on the probe set changes in 0.3–1.8% of states per update; with K = 2500 it drifts for 2500 updates against a stale target and changes 0.8–3.3%. Per-run, Spearman(early churn, log K) = +0.53. A frequently refreshed target does not damp churn; it damps the *accumulated* drift, and churn as Schaul et al. define it (per-update greedy flips) comes from the fit to whatever the target currently is.
2. **K = 20 is degenerate.** All 24 runs collapsed to a constant action within the first evaluation interval (final return 8.8–10.2, churn ≈ 0), at every ε. With a near-on-policy target on CartPole the max-bias feedback runs away before any learning signal arrives. These cells carry no information about churn and are excluded from the post-hoc analysis.
3. **ε still matters more as K grows, but not monotonically.** ε = 0.2 vs ε = 0 gains 79, 65, 73 AUC points for K = 100, 500, 2500 (K = 20 is zero because nothing learns). Churn-only exploration (ε = 0) is bimodal: within each K, some seeds solve the task and others never leave ~10 (sd 72–87 at ε = 0 vs 11–50 at ε = 0.2).
4. **Churn does predict learning at ε = 0, within every K.** Post hoc: Spearman(AUC, early churn) among ε = 0 runs is +0.78 (K = 100), +0.70 (K = 500), +0.83 (K = 2500), pooled +0.80 (n = 24, p < 0.001). The correlation falls to +0.67 at ε = 0.05 and to **−0.02** at ε = 0.2. That is the substitution pattern the hypothesis predicted, but on the *seed* axis rather than the K axis: when ε supplies exploration, churn no longer matters; when it does not, the seeds whose networks churn are the ones that learn.

## What this says

- The target period is the wrong knob for churn. It controls drift between syncs, and higher K means more churn per update, not less. The original K2 test (substitution along K) therefore could not have passed in the direction written.
- The mediation result (K3, and the post-hoc ε-gradient of the churn–AUC correlation) is the part of Schaul et al.'s conjecture that survives: churn correlates with learning only when there is no other exploration. The causal direction is open. Networks that learn change their greedy actions because their values change; the correlation is consistent with churn-as-exploration and with learning-causes-churn. Separating them needs a direct churn manipulation at fixed learning progress (a churn regulariser such as CHAIN, Tang & Berseth 2024, crossed with ε), which is the 3080 design.
- Honest reading: the pilot killed the specific mechanism (K → churn → exploration), and left a cheaper, correlational version of the conjecture standing with the causal question open.

## Caveats

- CartPole, one architecture, one learning rate, 30k steps, constant ε only (no annealing). K = 20 collapse may be specific to the learning rate.
- Churn measured as greedy flips on 256 fixed probe states drawn early in training; probe coverage of the late-training state distribution is not controlled.
- 8 seeds per cell. The ε = 0 cells have bimodal outcomes, so means and sds are rough summaries.
- Early churn and AUC are both measured over the same run; the +0.80 is within-run and cannot by itself distinguish cause from effect.
