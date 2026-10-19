# Three more NeurIPS-shaped ideas, round 3, refined (15 Oct 2026)

Built on today's pilots (R1–R3 in `research/independent/`) and papers from the last few weeks. Refined for two things: what a NeurIPS reviewer needs to see (a theorem or a tool, a real dataset, a result that changes what practitioners do) and what one person with an M1 Pro, a 3080, and free PhysioNet credentialing can finish by May.

Honest odds after refinement: 25–35% each if executed well. The single biggest odds boost is at the bottom: fold R4 into R3 as one paper with one theorem.

---

## R4. Leaky states: information that arrives late, and the pipelines that pretend it didn't

**Domain:** clinical offline RL / OPE. **Odds:** ~30% alone, higher merged with R3. **Compute:** CPU for everything but the MIMIC critic.

### What changed in the refinement
- **One theorem covers lossy and leaky maps.** OPE on φ(logs) equals the real value iff φ is *adapted* to the true information filtration (no event enters the state before it existed) and *kernel-preserving* (R3's condition). R3's maps break the second clause; result-time leakage breaks the first. Binning breaks both. That is one clean statement with three named corollaries, each with its own experiment. Reviewers like one theorem with three consequences more than three observations.
- **Rewards leak too, not just states.** SOFA-based intermediate rewards (AI Clinician, most sepsis work) use creatinine, bilirubin, platelets: all delayed labs. A pipeline that credits the policy at the draw time pays it for an outcome it could not yet see. This is a second, independent leak with its own audit number.
- **The audit statistic is computable without credentialing today.** The MIMIC-IV *demo* (100 patients, open) has `charttime` and `storetime`; eICU's `lab` table has `labresultoffset` and `labresultrevisedoffset`. Use the demo to measure the delay distribution per lab and *calibrate the simulator's delays to it*. The full MIMIC-IV and eICU runs then give two datasets, not one.
- **It ends with a method, not a complaint.** "Pending-aware" state construction: results enter the state at result time, and a `pending` indicator says a draw is outstanding. Show it recovers most of the value of the leaky pipeline while being deployable. A fix is what turns an audit paper into a methods paper.

### Design
1. **Simulator (today):** add a delayed lab channel to `ctsim.py` with delays drawn from the demo's empirical distribution; logger acts only on available results. Pipelines: draw-time state (leaky), result-time state (adapted), pending-aware. Learn → in-φ FQE on held-out → deploy. Report optimism vs median delay / decision interval, and the fraction of leaky-policy decisions that used not-yet-available values.
2. **Theory:** the adapted + kernel-preserving condition; in-φ value under a non-adapted map upper-bounds the value of any adapted policy; the gap bounded by the information the leak adds about the next state.
3. **Real data:** MIMIC-IV and eICU sepsis cohorts, AI-Clinician-style pipeline built three ways; headline numbers: % of decisions conditioning on absent results, % of intermediate rewards computed from absent results, change in recommendations, change in OPE value.

### Kill test
Optimism must grow with the delay/decision-interval ratio in the simulator and the adapted pipeline must be calibrated. If it holds in simulation but the MIMIC policy's recommendations change in < 5% of states, it is a data note (ML4H) and goes into R3 as a paragraph.

### Scoop risk
The 2610.13350 authors (posted this week) and the Off-by-a-Beat group. Put the simulator result and the theorem on arXiv within a month.

---

## R5. Where support comes from: protocol diversity as a data-collection principle for offline RL

**Domain:** offline RL identification; clinical and robotics. **Odds:** ~25–30%. **Compute:** CPU; eICU later; one D4RL/Minari replication on the 3080.

### What changed in the refinement
- **The real-data part no longer needs OPE.** The quantity is *support coverage*: the fraction of a target policy's state-actions whose estimated behaviour propensity exceeds τ. That needs only behaviour-policy estimation, which is cheap and robust, and it can be computed per hospital and pooled. OPE calibration stays in the simulator where there is ground truth.
- **The claim is about pessimistic learners, not just evaluation.** CQL/IQL/BCQ constrain to the support. If pooling hospitals with different protocols expands support, pessimistic learners can *find* better policies, not just evaluate them. That turns "identification" into "better offline RL from the same patients," which is what the offline-RL crowd cares about.
- **A non-clinical replication makes it a general paper.** Robotics datasets with several operators or several scripted policies (Minari "mixed," D4RL "medium-replay," JumpStart datasets with known sources) are the same situation: each source is a protocol. Show the same support-gain and policy-gain pattern there, on the 3080, and the paper is no longer "clinical RL."
- **Confounding gets a test and a bound, not a caveat.** Hospital fixed effects for within-site estimation, Rosenbaum Γ sensitivity for between-site pooling, and R2's negative-control policy as a detector. State the result as "pooling gains hold up to Γ = x."

### Design
1. **Simulator (today):** K loggers on `ctsim.py` with shifted thresholds and wait rules; target deviates from every single protocol. Compare one hospital × N, one hospital × KN, K hospitals × N on support coverage, FQE calibration, and the value of the best policy a count-constrained learner can reach. Add a hospital-level confounder; show the bias and the `ncp` flag.
2. **Theory:** identifiable region = union of per-source supports; the value of the best support-constrained policy is monotone in that union; pooled-estimator variance vs single-site; bias under source-level confounding with Γ.
3. **Real data:** eICU: per-hospital behaviour for vasopressors or insulin; support maps; coverage of an AI-Clinician-style learned policy in its own hospital vs the pool; a CQL/IQL policy trained pooled vs single-site with coverage reported. One Minari/D4RL mixed-source dataset for the replication.

### Kill test
If one hospital × KN matches K hospitals × N on support coverage and best-reachable value in the simulator, support is about sample size and the idea dies. If pooling wins only without the confounder and `ncp` does not flag it, the paper is a caution.

### Scoop risk
Low. The framing (protocols as the source of support) is not in the literature.

---

## R6. Offline policy selection at scale: what survives of JumpStart without the simulator?

**Domain:** offline RL methodology / benchmarks. **Odds:** ~25–30% (D&B track is the safer target). **Compute:** the 3080, three to four weeks if scoped as below.

### What changed in the refinement
- **Scope to the realistic question: within-algorithm hyperparameter selection.** JumpStart's own headline is that hyperparameters dominate seeds (92% of cells). The practitioner's problem is picking a configuration of *one* algorithm for *one* dataset offline. That cuts the policies per cell to tens, not thousands, and matches how people actually use offline RL.
- **Cut the compute with one shared critic recipe per dataset and two cheap scores.** Per dataset: (i) one behaviour critic and one dynamics model trained once; (ii) FQE per policy with a small critic and a fixed budget; (iii) two cheap scores needing no training per policy: behaviour divergence of the policy's actions on dataset states, and short model-based rollouts under the shared model. Report which cheap score tracks FQE, because the cheap one is what people will use.
- **Make the deliverable a number per dataset, not a method.** "Offline-selectable gain": the fraction of the online-tuned improvement over library defaults that an offline rule recovers. Add it to JumpStart's leaderboard. A benchmark-side contribution is the most durable thing a single-person team can produce here, and D&B reviewers reward it.
- **Use R2 and R3 as predictors.** The behaviour self-evaluation test and a support-coverage statistic (R5) should predict which datasets are selectable. If they do, the paper has a mechanism, not just a table.
- **Skip pixels; say so.** Locomotion, antmaze, adroit, and a few Minari tasks. Atari would eat the GPU budget for one row of a table.

### Design
1. **Pilot (1–2 days):** confirm the corpus format; pull all configurations of IQL and CQL for two datasets; compute the three scores; regret of top-1 and top-3 vs online best; Spearman vs online score.
2. **Full:** ~1,500 policies: 5 algorithms × 15 datasets × ~20 configurations; three selection rules plus agreement; regret curves; selectability vs the R2/R5 statistics.

### Kill test
If FQE alone recovers > 90% of the tuned gain everywhere, the problem is solved and this is a confirmation. If every rule is near-random everywhere, it is a negative result for a workshop.

### Scoop risk
High: the JumpStart authors are the natural authors. Start the pilot this week or drop it.

---

## The structural recommendation

Merge **R3 + R4** into one paper: *"Fictional MDPs: when off-policy evaluation on preprocessed logs is valid, and three ways clinical pipelines break it."* One theorem (adapted + kernel-preserving), three corollaries (time binning, state coarsening, result-time leakage), one calibrated simulator, two public datasets, one deployable fix (raw-time evaluation with support flags; pending-aware states). That is the paper with the best odds on this list, ~30–35%, and everything it needs is either done or startable today. R5 becomes the follow-up with the general offline-RL audience; R6 is the hedge if you want something non-clinical and can start it immediately.
