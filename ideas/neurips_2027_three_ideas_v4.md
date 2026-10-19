# Three more NeurIPS-shaped ideas, round 4, with pilots run (15 Oct 2026)

Scope set by the request: NeurIPS main track, reinforcement learning, RTX 3080 as the ceiling. All three stay away from the stub list, T1–T3, R1–R6 and the lab's areas (no OPE, no quantization, no delayed reward, no offline-RL data floor, no experience borrowing). They share one theme: treat a single RL run as the object of study and ask causal questions about it with exact forks, instead of reporting means over seeds. The code for this (a batched DQN where every run has its own generators and can be snapshotted and forked bit-exactly) is in `research/independent/R7-fork-test/code/lib/dqn_batched.py` and is reused by R9.

Pilots ran on the M1 Pro (CPU) today; folders `R7-fork-test`, `R8-mixing-barrier`, `R9-rescue-window` under `research/independent/` with the standard layout and pre-registered kill tests.

| | idea | pilot verdict | honest odds (NeurIPS main) |
|---|---|---|---|
| R7 | The fork test: when is a run's fate decided, and by what? | SUPPORTED (borderline) | 25–30% |
| R8 | The mixing barrier: what reward centering cannot fix | SUPPORTED | 30–35% |
| R9 | The rescue window: unlucky or doomed? | KILLED (test under-powered; signal in means) | 15% alone, 30% merged into R7 |

---

## R7. The fork test: when is a run's fate decided, and by what?

**Gap.** Seed variance in deep RL is documented endlessly (Henderson 2018, Agarwal 2021, Colas 2018, JumpStart's 160k runs) and never dissected. Frankle et al. 2020 did this for supervised nets with "instability analysis": train to step t, fork two children with different SGD noise, measure whether they end in the same basin. Nobody has done the RL version, where there are *two* noise sources (exploration and minibatch sampling) and where the thing that "locks in" might be the network, the buffer, or the policy's data distribution.

**Claim.** There is a lock-in time t* well before the end of training at which the identity of the run (its state at t*) explains most of the variance in its final score, and at t* the run's visible training curve explains almost none of it. Fate is set before it can be seen.

**Design.** Parents × forks × conditions. Children share the parent's full state (weights, optimiser, buffer, ε schedule, generator states) at the fork and differ only in which random streams are reseeded: `all`, `explore` only, `batch` only. ICC of children's final score grouped by parent is the lock-in curve. R² of child final on the parent's visible score at the fork is the "could you have known" curve. Fork 0 (same init, nothing else shared) is the control for the init-matters hypothesis.

**Pilot (CartPole DQN, 16 parents, 3 children, 7 forks).** ICC_all: 0.14 at fork 0, 0.43 at 5k, 0.63 at 15k (= T/2), 0.76 at 25k. R² of child final on parent's visible score at 15k: 0.00. So at half-way the run's state explains ~60% of the outcome and its curve explains none. Init alone explains ~nothing (fork 0 ICC CI covers 0). The explore-vs-batch comparison flips sign fork to fork: unresolved at this sample size. Full table in `R7-fork-test/notes/pilot_notes.md`. One summariser bug (T read as the last fork rather than the training length) was corrected after the run; it only affected the ≤ T/2 bookkeeping, and the fix is logged.

**Kill test (pre-registered, in README).** K1 ICC_all ≥ 0.5 with CI lower > 0.2 at a fork ≤ T/2 (passed at exactly T/2). K2 explore vs batch differ by ≥ 0.15 at ≥ 2 forks (passed by rule, not in spirit). K3 R² visible ≤ ICC − 0.2 at the lock-in fork (passed, 0.00 vs 0.63).

**3080 plan (the paper).** MinAtar ×5 (DQN) and DMC ×3 (SAC), 32 parents × 5 children × 8 forks × 4 conditions (add `all + fresh buffer`). Each fork sweep is ~1 GPU-hour per MinAtar game because all children of a fork time run in one batched process. Deliverables: lock-in curves per game; t*/T distribution; the gap between "fate set" and "fate visible"; a doomed-run classifier trained on cheap per-step features with fork-ICC as the label, evaluated as an early-stopping rule against the visible-score baseline. The classifier is what makes it a methods paper and not a measurement paper.

**Scoop risk.** Moderate. The plasticity-loss groups (Nikishin, Lyle, Sokar) have the tooling and could do this in a month if they thought of it. The JumpStart authors have the data but not forks. Put the CartPole + one MinAtar version on arXiv early.

**Why 25–30% and not more.** Measurement papers need a surprising number or a tool people adopt. The surprising number is "visible score explains 0% when fate is 60% set"; if that survives MinAtar it is a strong figure. Without the classifier it is a workshop paper.

---

## R8. The mixing barrier: what reward centering cannot fix

**Gap.** Reward centering (Naik, Wan, Tomar, Sutton 2024) and the differential/average-reward line of work show that subtracting the average reward lets TD use γ → 1 without the 1/(1−γ) blow-up in the value scale. The follow-ups treat this as "γ is now free." But the convergence rate of TD on a chain depends on the spectral gap of the *discounted* operator γP, whose second eigenvalue is γλ₂. Centering removes the constant eigenvector (the 1/(1−γ) term); it leaves the γλ₂ term. On a slowly mixing chain, γ → 1 should still be slow, by a factor the theory names exactly.

**Claim.** After centering, t_conv ∝ 1/(1−γλ₂). On fast-mixing tasks γ is free; on slow-mixing tasks it is not, and the barrier is the mixing time, not the horizon. Target scaling (Pop-Art style) does not remove it.

**Pilot (tabular TD(0), 40-state chains with controlled λ₂, γ up to 0.9999, 3M steps).** On the fast chain (λ₂ = 0.23) centered TD converges in 5k steps at every γ; plain TD never converges past γ = 0.99. On the slowest chain (λ₂ = 0.99) centered TD at γ = 0.9999 never converges in 3M steps while γ = 0.99 takes 525k (ratio ≥ 5.7). Across all 20 centered cells, log t_conv on log 1/(1−γλ₂): R² = 0.98 (0.99 on the 14 uncensored cells, slope 1.23); on log 1/(1−γ): R² = 0.08. Target scaling never converged on the slow chain and was slower than centering alone on mid-speed chains. Table in `R8-mixing-barrier/notes/pilot_notes.md`.

**Kill test.** K1 (γ-ratio < 1.5 fast, > 3 slow) passed. K3 (R² > 0.8 and beats 1/(1−γ) by 0.1) passed. K2 (scaling leaves ratio > 2) uninformative because scaling never converged; the honest statement is "scaling hurt", which is more than K2 asked.

**3080 plan (the paper).**
1. Theory: finite-time bound for centered TD(0) with the γλ₂ dependence made explicit (the pieces exist in Bhandari et al. 2018 and the differential-TD convergence proofs; the contribution is putting the mixing term in front and showing it is tight on the chain family).
2. Linear features on the same chains: does the feature-space operator's second eigenvalue take over?
3. Deep continuing control: MinAtar with resets removed, MuJoCo swimmer/hopper as continuing tasks; DQN/SAC with centering across γ ∈ {0.99, 0.999, 0.9999}; λ₂ estimated from an empirical transition matrix on a learned discretisation and from the critic's feature covariance. Prediction: the γ at which learning stalls is the one where 1/(1−γλ̂₂) crosses the training budget.
4. Intervention: anything that cuts mixing time (count-based bonus, options) should move the barrier. One clean experiment on the slow chain plus one deep task.
Compute: the full deep grid (3 tasks × 3 γ × 2 methods × 5 seeds) is under a GPU-day.

**Scoop risk.** Low-moderate. Sutton's group and the average-reward people (Wan, Naik, Zhang) are the obvious ones; the point is simple enough that someone may already have a note. Check arXiv for "reward centering mixing time" monthly.

**Why 30–35%.** Clean theory + clean synthetic + one deep demonstration is the recipe that gets RL theory-adjacent papers in. The risk is a reviewer saying "obvious from the spectral picture." The defence is the 1.00-vs-≥5.7 ratio and the R² 0.98 vs 0.08 figure, which turn "obvious" into "quantified and predictive."

---

## R9. The rescue window: unlucky or doomed?

**Gap.** Reset methods (Nikishin 2022, D'Oro 2023, Sokar 2023) are evaluated on average over seeds and schedules. The causal question — take a run that *will* end badly, intervene once at t, can you still save it, and until when — has not been asked, because without forks you cannot know a run's fate at the time you intervene. With R7's forks you can.

**Claim.** There is a window: before some t_r, a single cheap intervention (full reset keeping the buffer, in particular) moves a doomed run's outcome to the good runs' level; after t_r it does not. The window closes before the end of training.

**Pilot (same 16 CartPole parents; forks 5k–25k; control / reset_last / reset_all / lr_drop).** KILLED by the pre-registered rule: no intervention lifted P(rescue | bad parent) by ≥ 0.2 at any fork ≤ 0.6T. The reason is a design error: rescue was defined as "reach the good parents' median", which the good parents' own re-forked children miss 78–100% of the time, so the measure sat at the floor. The means tell the story the test could not: `reset_all` lifts bad-parent children from ~110–118 to 155–169 at 15k–20k (good-parent children: 158–165) and the lift is gone at 25k (102). Last-layer reset and LR halving did nothing on this tiny network. Table and rerun design in `R9-rescue-window/notes/pilot_notes.md`.

**What the rerun changes.** Rescue relative to the same parent's control children (> 1 pooled SD above), score = mean of last 20% of evaluations, forks at 0.5T–0.9T, MinAtar instead of CartPole, add shrink-and-perturb and buffer reweighting. This is a CPU/laptop rerun again before any 3080 time. If it still finds nothing, the idea is dead.

**Odds.** 15% standing alone: an intervention paper on top of a measurement paper, with a killed pilot. 30% merged into R7 as "forks establish when fate is set, resets establish until when it can be changed" — one paper, one tool, one practical rule (evaluate at t*, reset or prune, do not wait).

---

## Recommendation

Build the paper around **R8** (strongest pilot, cleanest theory story, cheapest compute) and run **R7 + R9 merged** as the second paper, with the R9 rerun as the gate. Together with R3 from round 2 that is three live candidates. R1, R2, R4–R6 remain documented and stopped or queued.
