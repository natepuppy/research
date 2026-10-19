# R3 pilot: does every lossy preprocessing step make the pipeline over-rate its own policy?

Run 15 Oct 2026 on an M1 Pro CPU, ~12 min. 5 dataset seeds × 18 pipelines. Each pipeline: learn a greedy policy on φ(3000 logged patients), FQE it on φ(3000 held-out patients), deploy it on 3000 fresh patients for the truth, and FQE it again on the *raw* held-out log with the timing-aware (dose, wait) estimator from R1.

## Verdict

`SUPPORTED`: K1 (binning inflates: +17.8 over raw at fine state/action), K2 (4 of 5 coarser state/action variants add optimism beyond 2 seed-SDs), K3 (raw-log FQE closer to the truth than in-φ FQE in 77% of binned runs).

## Optimism = in-φ FQE − deployed value (mean over 5 seeds, sd in brackets); behaviour's true value ≈ 6.2

| time | state | action | deployed V | optimism | raw-log FQE error |
|---|---|---|---|---|---|
| raw | fine | fine | 13.1 | **−0.4 (1.1)** | −0.4 |
| raw | fine | coarse | 13.1 | −1.3 (0.3) | −1.3 |
| raw | coarse | fine | 4.7 | **+7.1 (4.2)** | −5.1 |
| raw | coarse | coarse | 8.5 | +4.6 (1.6) | −1.4 |
| raw | binary | fine | −19.3 | **+29.1 (3.9)** | −12.1 |
| raw | binary | coarse | −12.0 | +22.9 (4.4) | −26.6 |
| bin1h | fine | fine | −4.4 | +20.5 (2.3) | +6.2 |
| bin1h | fine | coarse | 0.6 | +11.5 (1.6) | −3.6 |
| bin1h | binary | fine | −34.5 | +48.9 (2.0) | +12.0 |
| bin4h | fine | fine | 1.9 | +14.4 (0.7) | −5.5 |
| bin4h | binary | coarse | 0.3 | +15.8 (0.3) | −14.2 |

(full table in `results/summary.csv`)

## What it says

1. **With no lossy preprocessing the pipeline is honest.** Raw timing, 6 glucose bins, 5 dose levels: the learned policy beats the clinician (13.1 vs 6.2) and FQE on held-out φ-data says so within 0.4. So the optimism elsewhere is not "offline RL overfits"; it is the map φ.
2. **State coarsening alone does what binning does.** Keep raw timing, merge glucose into 3 bins: the learned policy is now *worse* than the clinician (4.7) and the pipeline reports 11.8. Merge into 2 bins: deployed −19, reported +10. The policy learned in the fictional MDP exploits actions that look good in the merged state and are harmful in the real one, and FQE in the same fictional MDP cannot see it. Nobody has reported this for clinical RL pipelines; the AI Clinician's 750 k-means states sit somewhere on this axis.
3. **Action coarsening alone is harmless here** (−1.3). Restricting the policy class loses a little value but does not create a fictional MDP, because the map is applied to the agent's choice, not to the data's state or clock. That is consistent with the φ-invariance idea: coarsening *what the agent can do* preserves the evaluation; coarsening *what the agent sees or when it sees it* does not.
4. **Binning is the worst single map** (+11 to +49), and every binned policy deployed worse than the clinician while being reported as 2–3x better. Frost's 1.5–3x reproduces on a toy.
5. **Raw-log evaluation is a detector, not a fix.** It moved the estimate toward the truth in most binned runs (77%) and always said the binned policies were bad, but its own errors are 4–24 because the deployed policies (check every hour, dose on a 2-bin state) are far outside the logger's support, and tabular FQE with neutral fallback cannot evaluate them. The honest protocol is: evaluate on raw trajectories *and* report the support flag; when the flag is high, the estimate is not a number, it is a warning.

## Caveats

- One synthetic simulator; magnitudes are not transferable, the ordering probably is.
- Model-based tabular Q-iteration with a count ≥ 30 constraint. A pessimistic learner (CQL-style) would deploy less badly but the in-φ evaluation would still be optimistic.
- Held-out φ-FQE already removes same-data overfitting, which makes point 1 cleaner than most published pipelines manage.
- No imputation or weight-clipping maps yet.

## What this means for the paper

This is the strongest of the three pilots and the one to build the paper on. Shape: (i) a φ-invariance condition (OPE on φ(logs) equals real value iff φ preserves the target's induced reward and transition kernels), with the observation that action maps satisfy it and state/time maps generally do not; (ii) the shared-pipeline optimism result (training and evaluating inside the same φ lets the optimiser exploit φ's artefacts and hides it from the evaluator); (iii) the same grid on Frost's UVA/Padova simulator, then an AI-Clinician-style MIMIC pipeline with each step swapped. Merge R1's binning sign-flip result into (i).

## Re-running

```bash
python code/r3_fictional_mdp.py             # ~12 min
python code/r3_fictional_mdp.py --summarize
python code/r3_fictional_mdp.py --full      # 20 seeds × 10000 patients
```
