# Lossy Preprocessing Builds a Fictional MDP That Offline Evaluation Cannot See

*Conference-length version (RLC format). Every number in this paper comes from this folder's results.*

## Abstract

Frost et al. (arXiv:2602.06603) showed that binning ICU logs onto a fixed grid makes fitted Q evaluation over-rate the policy learned on the same binned data by 1.5–3×. Clinical pipelines also cluster states, coarsen doses, and carry observations forward. Each step is a map \(\varphi\) from real trajectories to fictional ones, and the usual pipeline both learns and evaluates inside \(\varphi\). This pilot asks whether the self-deception is special to time binning. On a synthetic continuous-time glucose simulator, 5 dataset seeds and 18 pipelines (time \(\times\) state \(\times\) action), the same model-based Q-iteration and the same in-\(\varphi\) FQE are applied throughout, and the greedy table is deployed for a true value. With no lossy map the pipeline is honest: deployed value 13.13 against a behaviour value of about 6.2, optimism \(-0.37\) (sd 1.12). Merging glucose into 3 bins gives optimism \(+4.62\) to \(+7.09\); merging into 2 bins gives \(+22.85\) to \(+29.09\); time binning gives \(+11.46\) to \(+48.89\). Action coarsening alone is \(-1.27\). The kill test is **SUPPORTED**: at fine state and action, binned pipelines exceed the raw pipeline's optimism by \(+17.82\); 4 of 5 coarser raw-time variants add absolute optimism beyond two seed standard deviations; and on binned pipelines a timing-aware FQE on the raw log is closer to the deployed value in 77% of runs. Every binned or binary-state policy deploys worse than the behaviour policy while being reported as better. Raw-log evaluation flags the failure. Off the logger's support it does not repair the number.

## 1 Introduction

An offline pipeline chooses a representation, learns a policy in it, and estimates that policy's value in it. If the representation is a lossy map \(\varphi\), both the learner and the evaluator see \(\varphi(\mathrm{logs})\). They can agree with each other and both be wrong about the process that generated the logs. Frost et al. documented one case: fixed-grid time binning, FQE over-rating the binned policy by 1.5–3×. The AI Clinician pipeline also clusters states (the notes place its 750 k-means states on the same axis), discretises doses, and carries observations forward. The question is whether every lossy \(\varphi\) does this, and whether evaluating the learned policy on the raw trajectories fixes the estimate or only detects it.

The pilot's reading, stated before these numbers and kept because the numbers split the same way, is a \(\varphi\)-invariance condition. Off-policy evaluation on \(\varphi(\mathrm{logs})\) equals the real value of the deployed policy when \(\varphi\) preserves the reward and transition kernels induced by the target. A map applied to the action the agent is allowed to take can preserve those kernels. A map applied to the state the agent sees, or to the clock, generally does not: the optimiser exploits an artefact of \(\varphi\), and an evaluator who lives in the same \(\varphi\) scores the artefact as if it were the process.

**Contributions.** (1) One grid, 18 maps, one learner, one in-map FQE, one deployment, and one raw-log FQE, on a simulator whose decisions happen at irregular times. (2) The honest cell. Raw timing, 6 glucose bins, 5 dose levels: optimism \(-0.37\), and the learned policy beats the behaviour policy (13.13 against about 6.2). The optimism elsewhere is the map. (3) State coarsening reproduces binning. Three glucose bins: deployed 4.73, optimism \(+7.09\). Two bins: deployed \(-19.30\), optimism \(+29.09\). Action coarsening alone does not: optimism \(-1.27\). (4) Raw-log FQE is closer to the truth on 77% of binned runs and still carries absolute errors from a few units to the tens, because the deployed policies leave the logger's support. The verdict is SUPPORTED on the two tests the rule requires, and the binning sanity check passes as well.

## 2 Related work

Frost et al. (arXiv:2602.06603) are the binning result this grid replicates on a toy: in-pipeline FQE over-rates, here by \(+11.46\) to \(+48.89\) in optimism, which the notes describe as the binned policies being reported 2–3× better while deploying worse. The simulator is built around the three properties that paper, and Insulin4RL (arXiv:2606.19481), say matter and that fixed-interval binning destroys: decisions at irregular, policy-chosen times; observations only when the policy checks; a cost to checking. The AI Clinician is the clinical pipeline whose clustered states sit on the coarsening axis studied here; this run does not touch MIMIC. A pessimistic penalty of CQL type is named in the notes as a learner that would deploy less badly while leaving the in-\(\varphi\) evaluation optimistic. It was not run. A companion pilot's binning sign-flip is a related lab result and is not folded into these numbers.

## 3 The simulator, the maps, and the tests

The process is a synthetic continuous-time glucose model, not a physiological one. A stay lasts 48 hours, discounted at 0.98 per hour. Glucose moves with insulin on board and a disturbance. The reward rate is \(+1\) inside \([70, 180]\) mg/dL, \(-4\) below 70, \(-0.5\) on \((180, 250]\), and \(-2\) above 250. Each check costs 0.25. Actions are a dose and a wait. Fine doses are a hypoglycaemia rescue, then 0, 2, 4, and 6 units. Waits are 1, 2, and 4 hours. The behaviour policy generates the logs.

A pipeline is \(\varphi = (\mathrm{time}, \mathrm{state}, \mathrm{action})\).

- Time: `raw` (decision points are the logged checks, and the action includes the wait), `bin1h`, or `bin4h`.
- State: `fine` (6 glucose bins, edges 70, 110, 150, 180, 250), `coarse` (3 bins: below 70, 70–180, above 180), or `binary` (2 bins, split at 180).
- Action: `fine` (5 dose levels) or `coarse` (rescue, none, or 4 units).

That is 18 pipelines. For each of 5 seeds the learner sees \(\varphi\) of 3{,}000 behaviour episodes, FQE sees \(\varphi\) of 3{,}000 held-out behaviour episodes, deployment is 3{,}000 fresh episodes in the simulator, and a second FQE reads the raw held-out log with a timing-aware estimator. The learner is model-based tabular Q-iteration, 80 iterations. An action seen fewer than 30 times in a state is forbidden; a state with no allowed action falls back to its most frequent logged action. Optimism is in-\(\varphi\) FQE minus the deployed value. The raw-log error is raw FQE minus the deployed value. The behaviour policy's true value on this simulator is about 6.2. The run took about 12 minutes on an M1 Pro.

- **K1**, a sanity check in the spirit of Frost. At fine state and fine action, the mean optimism of the binned pipelines exceeds the raw pipeline's by more than 2.
- **K2.** Among the five raw-time pipelines that coarsen state or action, at least two have mean absolute optimism more than two seed standard deviations above the fine/fine cell.
- **K3.** On binned pipelines, the raw-log FQE is closer to the deployed value than the in-\(\varphi\) FQE in more than half the runs.

SUPPORTED if K2 and K3. INCONCLUSIVE if exactly one. KILLED if neither. K1 is reported and is not required.

## 4 The grid

| Time | State | Action | Deployed \(V\) | Optimism (sd) | Raw-log error | Fraction raw closer |
|---|---|---|---:|---:|---:|---:|
| raw | fine | fine | 13.13 | \(-0.37\) (1.12) | \(-0.37\) | 0.00 |
| raw | fine | coarse | 13.08 | \(-1.27\) (0.28) | \(-1.26\) | 0.40 |
| raw | coarse | fine | 4.73 | \(+7.09\) (4.20) | \(-5.07\) | 0.20 |
| raw | coarse | coarse | 8.51 | \(+4.62\) (1.55) | \(-1.40\) | 0.60 |
| raw | binary | fine | \(-19.30\) | \(+29.09\) (3.90) | \(-12.06\) | 0.80 |
| raw | binary | coarse | \(-11.95\) | \(+22.85\) (4.35) | \(-26.61\) | 0.40 |
| bin1h | fine | fine | \(-4.44\) | \(+20.48\) (2.33) | \(+6.16\) | 1.00 |
| bin1h | fine | coarse | 0.57 | \(+11.46\) (1.59) | \(-3.60\) | 0.80 |
| bin1h | coarse | fine | \(-8.01\) | \(+22.37\) (3.15) | \(+5.71\) | 0.60 |
| bin1h | coarse | coarse | \(-0.85\) | \(+11.91\) (2.04) | \(+2.19\) | 1.00 |
| bin1h | binary | fine | \(-34.48\) | \(+48.89\) (2.00) | \(+12.04\) | 1.00 |
| bin1h | binary | coarse | \(-4.53\) | \(+15.03\) (1.21) | \(+3.17\) | 1.00 |
| bin4h | fine | fine | 1.88 | \(+14.42\) (0.70) | \(-5.46\) | 0.80 |
| bin4h | fine | coarse | 2.25 | \(+13.94\) (0.67) | \(-9.18\) | 0.80 |
| bin4h | coarse | fine | 1.29 | \(+15.44\) (1.06) | \(-8.12\) | 0.80 |
| bin4h | coarse | coarse | 1.56 | \(+14.90\) (0.78) | \(-9.35\) | 0.60 |
| bin4h | binary | fine | \(-1.08\) | \(+17.37\) (0.91) | \(-23.32\) | 0.20 |
| bin4h | binary | coarse | 0.30 | \(+15.82\) (0.31) | \(-14.23\) | 0.60 |

*Means over 5 seeds. Optimism is in-\(\varphi\) FQE minus deployed value. The last column is the fraction of seeds on which the raw-log error is smaller in absolute value.*

**The honest cell.** Raw timing, fine state, fine action: the learned policy deploys at 13.13, above the behaviour value of about 6.2, and held-out FQE inside the same \(\varphi\) says so within 0.37. Same-data overfitting is already removed, because FQE uses a held-out log. The agreement is the map being fine enough, not a general property of offline RL.

**State coarsening, raw clock.** Three bins and fine actions: deployed value falls to 4.73, below the behaviour policy, and in-\(\varphi\) FQE reports a value 7.09 higher. The notes' arithmetic on the rounded cell is that the pipeline reports 11.8. Two bins and fine actions: deployed \(-19.30\), optimism \(+29.09\); the notes summarise this as deployed \(-19\), reported \(+10\). The policy learned in the merged state exploits an action that looks good after the merge and is harmful in the unmerged process. FQE on the merged transitions assigns that action the merged return. Coarsening the action as well does not remove the optimism (\(+4.62\) at 3 bins, \(+22.85\) at 2 bins). At 3 bins with coarse actions the deployed value is 8.51, which is above the behaviour value, and the pipeline still over-rates it by 4.62.

**Action coarsening alone.** Raw timing, fine state, coarse action: deployed 13.08, optimism \(-1.27\) (sd 0.28). Restricting the dose set costs a little and does not create a fictional state. The map is applied to the agent's choice. The state and the clock that the evaluator sees are the state and the clock of the log.

**Time binning.** At fine state and fine action, hourly binning deploys at \(-4.44\) with optimism \(+20.48\), and four-hour binning deploys at 1.88 with optimism \(+14.42\). Across the twelve binned cells, optimism runs from \(+11.46\) to \(+48.89\), and every deployed value in that block is below the behaviour value of about 6.2. The worst single cell is hourly binning with a binary state and fine actions: deployed \(-34.48\), optimism \(+48.89\). K1 is the fine/fine contrast: the mean of the two binned optimisms, minus the raw optimism, is \(+17.82\), against a bar of 2.

**K2.** The five raw-time variants coarser than fine/fine are fine/coarse, coarse/fine, coarse/coarse, binary/fine, and binary/coarse. Absolute optimism exceeds the fine/fine baseline by more than two seed standard deviations in four of them: coarse/fine, coarse/coarse, binary/fine, binary/coarse. The miss is fine/coarse, the action-only map, whose absolute optimism is 1.27 against a fine/fine absolute optimism of 1.04. The rule asked for two of five. Four hit.

**K3.** The mean, over binned pipelines, of the per-pipeline fraction of seeds on which raw-log FQE is closer to the deployed value is 77%. Several hourly-binned cells have fraction 1.00. The four-hour binary/fine cell has fraction 0.20: raw-log error \(-23.32\) against optimism \(+17.37\). Closer on most runs is the claim that was registered. Uniformly closer is not what happened.

## 5 Detector, not a fix

On the binned block the raw-log column has the right sign often enough to say the binned policies are bad, which the in-\(\varphi\) column never does: every binned optimism is positive. The raw-log errors are still large. The notes put them at 4–24 and name the reason. A policy that checks every hour, or that doses from a 2-bin state, is far outside the support of a behaviour log whose checks and doses follow a different rule. Tabular FQE with a neutral fallback cannot evaluate that policy; the estimate is then a warning that the support flag has fired, not a value. The protocol the notes recommend is to evaluate on raw trajectories and to report the support flag. This paper's table is the first half of that protocol. The per-run support fractions were logged and are not a summary statistic here.

The split between maps matches the invariance reading. Action coarsening is the only lossy map with optimism near zero and a deployed value still near 13. State merges and time bins move the deployed value down and the reported value up together. That is the shared-pipeline failure: training and evaluating inside one \(\varphi\) lets the optimiser use \(\varphi\)'s artefacts and hides them from the evaluator.

## 6 Limitations

One synthetic simulator. The notes' judgment is that the magnitudes will not transfer and the ordering probably will. The learner is model-based tabular Q-iteration with a count of at least 30, not a neural offline algorithm and not a pessimistic one. Held-out FQE already removes same-data overfitting, so the honest cell is cleaner than many published pipelines. Imputation and weight-clipping maps were not included. The same grid on Frost's UVA/Padova simulator was not run, and neither was an AI-Clinician-style MIMIC pipeline with each preprocessing step swapped. The \(\varphi\)-invariance condition is the reading that organises the grid. It is not a proved theorem in this folder.

## 7 Conclusion

On 18 pipelines and 5 seeds, offline RL over-rates its own policy when the preprocessing map merges states or bins time, and it does not when the only lossy step coarsens the action. The untouched pipeline is honest to within 0.37 and beats the behaviour policy. Binary states and time bins produce policies that deploy below the behaviour value while in-map FQE reports them above it, by optimisms from \(+11.46\) to \(+48.89\). Raw-log FQE is the closer estimate on 77% of binned runs and remains a detector once the deployed policy leaves the logger's support. The kill test is SUPPORTED.

## References

Frost et al. Binning of ICU logs and in-pipeline FQE. arXiv:2602.06603. · Insulin4RL. arXiv:2606.19481. · The AI Clinician pipeline, cited in the pilot notes for its 750 k-means states. · CQL-style pessimistic learning, named in the notes as an unrun control.
