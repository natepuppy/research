# Three more NeurIPS-shaped ideas, round 7, with pilots run and paper drafts written (17 Oct 2026)

Same scope: NeurIPS main track, reinforcement learning, RTX 3080 as the ceiling, no overlap with the stub list, T1–T3, R1–R15 or the lab's areas. Theme this round: *the defaults nobody audits* — the loss function, the minibatch sampler, and the accuracy statistic that value-based RL reports. Each has a textbook justification; each was tested where the test is exact. One idea dropped before running because it matched a stub (a γ–learning-rate scaling law; N1-9). Adjacent stubs named in the READMEs: none for R16; S3-4 and S3-12 (batch size, residual autocorrelation) for R17; none for R18.

Folders `R16-huber-risk`, `R17-decorrelation`, `R18-value-error-yardstick` under `research/independent/`, standard layout, pre-registered kill tests, `paper/main.md` drafts.

| | idea | pilot verdict | honest odds (NeurIPS main) |
|---|---|---|---|
| R16 | The Huber loss is a risk preference | pilot KILLED by a design flaw; pre-registered follow-up INCONCLUSIVE by rule, every substantive prediction confirmed (R² = 1.000, 60/60 policy flips) | 30–35% |
| R17 | Does decorrelation do anything? | KILLED; no-replay beats replay with a fast target | 15–20% |
| R18 | Which value-error statistic predicts the greedy policy? | KILLED; norms work on diffuse errors; overestimation has the wrong sign at fixed time | 5–10% |

---

## R16. The Huber loss is a risk preference

**Gap.** Huber on the TD error (κ = 1) is the DQN default, justified as stability. A Huber M-estimator does not estimate the mean of a skewed distribution, and TD targets are skewed wherever a rare outcome carries a large value difference. Nobody had written down the fixed point or asked what it does to the policy.

**Pilot.** Fork MDPs with a safe arm (sure 1) and a risky arm (lottery: M w.p. p, mean 1 + g; or hazard: rare −M, mean 1 − g), tabular Q-learning with clipped updates, exact Huber fixed points by bisection. First grid: the closed form predicted learned values at R² = 0.986, but two sign/policy tests failed because chain states had duplicate actions and the squared-loss cells carried a +0.09–0.31 max-bias on the noisy arm. Pre-registered single-action follow-up: **R² = 1.000**, greedy policy wrong exactly where |predicted bias| > gap (**60/60** cells), never wrong under the squared loss, bias monotone in κ everywhere; the sign clause failed in four cells where the prediction is ≈ 0 (clip inactive), so INCONCLUSIVE by my own wording. At κ = 1 a 5%-probability payoff of 22 is valued at 0.2 instead of 1.1; the agent takes the wrong fork in all 24 (skew, g, p) cells, lotteries and hazards alike. DQN check: on CartPole, 98% of TD errors are below κ; the terminal ones are not, and Huber inflates the learned value of terminating actions by 21% relative to the squared loss (13/16 paired seeds).

**Reading.** The clip replaces the mean of the target by its Huber location, which drops the rare side's contribution. For an agent this is a risk preference: averse to lotteries, blind to hazards, size set by κ relative to the value jump. In reward-clipped training it acts on *value* jumps: rare terminations and rare large bootstrapped values, exactly what safety cares about. Bonus finding from the flawed first grid: the squared loss is itself risk-seeking toward noisy arms through the max, so neither default is neutral.

**What the paper needs.** MinAtar with injected 2% lotteries and hazards, κ ∈ {0.3, 1, 3, ∞} × double-Q, exact-MC values of the injected outcomes, with the prediction that take-rates move by the computed Huber bias; the terminal-value accounting against R11's reservoir (loss vs buffer); κ as a risk dial against QR-DQN.

**Why 30–35%.** A closed-form, exactly confirmed statement about a default used in thousands of papers, with a safety angle and a cheap fix (κ scaled to the value range). Risks: "obvious once said" (the fixed point is a one-line calculation) and "tabular". The deep lottery/hazard experiment is what makes it a paper. Scoop risk moderate.

---

## R17. Does decorrelation do anything?

**Gap.** Replay is justified by reuse *and* decorrelation (Mnih 2015). The two had never been separated: every replay-vs-none comparison changes both.

**Pilot.** Batched DQN on CartPole, buffer and update count fixed, minibatch sampler varied: 32 iid draws, 4 windows of 8, 1 window of 32, or the last 32 transitions (no reuse); × target period K ∈ {100, 2500} × 16 seeds. Fully contiguous vs iid: −8 AUC points at either K, 4/16 wins (p = 0.08), despite the gradient's cosine to the buffer gradient falling 0.66 → 0.59. Decorrelation is a < 10-point, non-significant effect. The no-replay sampler, which the textbook says should be worst, is **best at K = 100**: +40 AUC, +69 final return, 15/16 and 14/16 wins. The replay learners peak at 5k steps (~165) and collapse to 80–120; the on-policy learner climbs to 181. At K = 2500 nothing learns before 10k (R12's speed limit), replay then wins on AUC by reuse and the two finish equal.

**Reading.** Replay's value on this task is reuse, and it is contingent on a slow target. With a fast target, uniform replay *hurts* relative to on-policy minibatches, through a collapse the on-policy learner cannot suffer: every failure it makes is in its next minibatches (R11's mechanism) — or, alternatively, through stale off-policy data against fast-moving targets. The pilot does not separate these two; `iid + terminal reservoir` at K = 100 is the decisive cheap test.

**What the paper needs.** MinAtar with the same samplers × K ∈ {100, 500, 2500, 10k}; the mechanism split (terminal reservoir; policy-age cap on sampled transitions); Polyak targets; replay ratio × sampler.

**Why 15–20%.** A clean negative on a textbook claim plus a surprising positive (no-replay wins with a fast target). Risks: CartPole-specific; the collapse is a known phenomenon (R11) and the `recent` win may be entirely that; reviewers may see "replay ratio paper". If the MinAtar version holds, the reframing of replay as reuse-only with a target-period condition is a real contribution.

---

## R18. Which value-error statistic predicts the greedy policy?

**Gap.** Theory bounds performance by the sup-norm of value error; practice plots overestimation and argues less is better. Fujimoto 2022 showed Bellman error ≠ value error. Does any accuracy statistic predict performance, at fixed task and time?

**Pilot.** 24 Garnet MDPs × 24 Q-learning configurations × 10 checkpoints = 5760 Q̂s with exact Q*, V^{π̂}, visitations; eleven statistics; Spearman with exact suboptimality within (MDP, time) groups. Hypothesis (norms fail, gap-crossing works) KILLED: norms predict at 0.72–0.78, the gap-crossing statistic at 0.93, visitation weighting adds 0.02. The corpus's errors are diffuse (sup/RMS = 2.9, the i.i.d. value), so all norms are proxies for progress; the localised-error regime the hypothesis was about was not produced. Two results stand: the Bellman error is the weakest predictor (0.58), and **overestimation has the wrong sign at fixed time** (ρ = −0.65): zero-initialised learners that are behind under-estimate more and perform worse, so "more overestimation" accompanies better policies.

**Reading.** A pilot that tested the wrong regime, honestly reported. The transferable piece is the sign result: overestimation compared across methods at matched steps measures progress, not bias, and should be compared at matched return or at convergence. That is a methodological caution relevant to the Double-DQN/TD3 style of plot, and to R15/R16's own use of overestimation (there measured at convergence).

**What the paper needs.** A localised-error corpus (narrow behaviour, no restarts, linear FA with aliasing, DQN with exact-MC truth) where the prediction is that the sup-norm drops below 0.5 and the d^{π̂}-weighted crossing stays above 0.8; and the re-plot of Double DQN vs DQN overestimation on MinAtar at matched return.

**Why 5–10%.** As it stands this is a workshop note. The overestimation-sign result could become a short methodological paper if the matched-return re-plot changes a published conclusion.

---

## Recommendation

Across rounds 4–7 the live main-track candidates are **R8** (mixing barrier), **R11** (forgetting how to fail), **R12** (speed limit), **R15** (two biases), **R16** (Huber as risk preference) and **R3** (fictional MDP), with **R13** and **R17** as short papers and **R7+R9** merged. R16 is the strongest new entry: closed form, exact confirmation, a safety angle, and a one-afternoon deep experiment. R11, R16 and R17 now share a mechanism question (what inflates terminal values: the buffer, the loss, or stale targets) and their MinAtar DQN runs should be one harness with three factors: loss (Huber/squared), sampler (iid/recent), reservoir (on/off), at K ∈ {100, 2500}. That single 3080 week would settle R11's mechanism split, R16's terminal accounting and R17's collapse explanation together.
