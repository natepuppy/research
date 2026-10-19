# Three NeurIPS-shaped ideas, round 2 (research as of 15 Oct 2026)

Target: NeurIPS 2027 main track (deadline ~mid May 2027). Fallbacks: ML4H / CHIL (clinical), AISTATS, RLC, and the NeurIPS workshops.

**Pilot status (15 Oct 2026).** All three were piloted on a laptop the same day; see `research/independent/R1-timing-sensing-ope`, `R2-falsify-ope`, `R3-fictional-mdp`. R1: KILLED (the masking identification trick does not help; the SMDP (dose, wait) FQE and the binning sign-flip survive). R2: KILLED by the pre-registered rule, but the negative-control policy test catches confounding with power 0.99 vs 0.35 for agreement. R3: SUPPORTED (state coarsening alone produces the same self-deception as time binning). Build the paper on R3, fold R1's binning result into it, keep R2's two working tests as a second paper.

## Honest framing

Nothing is "very likely" to get into NeurIPS. NeurIPS 2026 took about a quarter of submissions and reviewer noise is real. What I can do is pick problems where (1) two or more 2025–26 papers explicitly say the problem is open, (2) you have something most submitters do not (MIMIC/eICU access, a clinical lab, cheap compute is enough), (3) the paper has a theory piece plus a real-data piece, which is the shape that gets through, and (4) the result is useful whichever way it comes out.

All three below are about one theme: **offline RL on real logs lies to itself, and how to catch it.** That is deliberate. It is where your lab has an edge, where NeurIPS 2026 accepted several OPE papers (Q-MMR, OTROPE, Budgeted Multi-Source Counterfactual Annotation, Variational IPS for multi-logger), and where none of your 250 stubs or T1–T3 sit. The downside is concentration: if a reviewer pool is cold on OPE, all three suffer. Two non-clinical alternates are at the bottom.

Odds below assume competent execution and are not inflated. Realistic range for any of them is 25–35%. You cannot run all three before May. Pick one of Idea 1 or Idea 3 as the main paper and run Idea 2 as the cheap second, or merge Ideas 1 and 3 into one bigger paper.

---

## Idea 1. Look less, act later: offline evaluation and learning for policies that choose *when to measure* and *when to act*

**Domain:** clinical offline RL / OPE in irregular time. **Odds if executed well:** ~25–35%.

### Where it sits
- Frost et al. 2026 (arXiv 2602.06603, "Hidden risks of temporal resampling") show binning ICU logs to fixed intervals makes FQE overestimate deployed value by 1.5–3x, and agents trained on binned data lose up to 60% at deployment. Their fix is "train on unprocessed trajectories." They do not touch the question of a policy that *chooses* its own timing.
- Insulin4RL (arXiv 2606.19481, 12k patients, 375k decisions, PhysioNet) formalizes ICU insulin dosing as a partially observed semi-MDP with γ^η discounting. Timing is still exogenous: the agent acts whenever the clinician happened to act.
- EDQ (ICLR 2025, arXiv 2503.15890, "Time After Time") is the closest method: a deep Q-evaluation for *when and what* to treat in continuous time. Simulation only (survival, tumor growth). They state explicitly that intervening on the *observation* process is out of scope.
- Zhou et al. (AISTATS 2024, "Timing as an Action") learn when to act, but model-based, online, small state spaces.
- Statistics/epi already has the identification result for the measurement part: Robins, Orellana & Rotnitzky (2008) and Caniglia/Hernán et al. (2019) show that under a "no direct effect of monitoring" assumption you can evaluate a *less* frequent monitoring schedule from logs by masking the extra measurements, and you generally cannot evaluate a *more* frequent one. Nobody has brought this into deep offline RL, and no RL paper combines it with treatment timing.

### The question
Can we evaluate, and then learn, a policy that decides (treatment, next-check time) jointly, from ICU logs where clinicians chose both, without binning?

### Claims the paper would make
1. **Identification.** A joint sensing-and-treatment policy is identifiable from logs under: overlap on the (action, sojourn) pair, and no direct effect of measurement on the patient (true for fingerstick glucose, false for arterial blood draws, which is itself a nice point). You can evaluate policies that look less often than the logger; looking more often is not identified. Fixed-interval binning is shown to be a *degenerate timing policy* that the logger never followed, so standard FQE on binned data is evaluating a policy with zero support. That gives a closed-form account of Frost's 1.5–3x.
2. **Estimator.** An FQE / EDQ-style recursion over event histories where the target policy outputs (dose, wait-time), the masked-observation trick handles sparser sensing, and the wait-time gets a propensity so IS/DR versions exist too. Add measurement cost to the reward so "how much less often could we check glucose without hurting control" is a real estimand.
3. **Learning.** Timing-aware IQL/CQL where the action is (dose, wait). Compare to the same algorithms on binned data and on raw data with exogenous timing.

### Design
- **Ground truth:** Frost's irregular UVA/Padova simulator (MIT, `tdgfrost/temporal-resampling-experiments`). Add a glucose-measurement cost and let policies choose check intervals. You can read off true deployed value, so estimator calibration plots are exact.
- **Real data:** Insulin4RL (same group, code public) for the main real-data result; then the same pipeline on MIMIC-IV / eICU through the lab for a second cohort (vasopressor titration or heparin are other natural fits).
- **Comparisons:** FQE on 1h/2h/4h bins; FQE on raw with exogenous timing (Insulin4RL protocol); EDQ as published; your estimator. Report calibration vs ground truth in sim and ESS / support diagnostics on real data.
- **Clinical readout:** value vs measurement frequency curves. If the curve is flat down to half the current check rate, that is the headline a clinician will care about.

### Kill test
If in the simulator the timing-aware estimator is no better calibrated than EDQ or raw-time FQE, and policies that choose wait-times do no better than exogenous-timing policies, the method part dies. The identification result and the "binning is a zero-support policy" explanation still stand as a short paper (ML4H / AISTATS).

### Compute
All CPU-feasible on the M1 Pro; the 3080 helps if you use a transformer over event histories for the Q-function on Insulin4RL. Simulator sweeps are hours, not days.

### Risks
Frost/Harris (UCL) are the natural authors of this follow-up; post to arXiv early. Reviewers may say "this is Robins 2008 with a neural net" — the answer is the joint treatment-plus-sensing timing, the degenerate-policy view of binning, the estimator, and real ICU results, none of which exist. You must cite and beat EDQ, not ignore it.

### Reading list
2602.06603, 2606.19481, 2503.15890 (EDQ), Zhou et al. AISTATS 2024, 2409.09236 (OPE with outcome-dependent observation times), Fatemi et al. CHIL 2022 (SMDP offline RL), Robins–Orellana–Rotnitzky 2008, Caniglia et al. 2019, Kreif et al. 2021 (joint monitoring + treatment strategies), "Off by a Beat" (OpenReview E0bdq7bwQH, observation/intervention misalignment).

---

## Idea 2. Falsification tests for off-policy evaluation

**Domain:** OPE methodology, general (clinical demo). **Odds if executed well:** ~25–30%.

### Where it sits
- Every applied OPE paper ends with "FQE, WIS and DR agree, so we trust the number." Nobody has shown that agreement means anything when the estimators share the same bias (same logs, same behavior-policy model, same preprocessing).
- 2608.12489 (top-k allocation OPE benchmark) measured common diagnostics (ESS, weight tail, agreement) against true error and found correlations around 0.4. The Evidence Ladder note (2609.23374) says "agreement is a diagnostic, not permission." Both are observations, not a method.
- Negative controls are standard in epidemiology (Lipsitch 2010; Shi, Miao & Tchetgen Tchetgen 2020) and are used *for identification* in proximal RL (Bennett & Kallus). They are not used *as tests* in sequential OPE. "Falsifying Predictive Algorithms" (2601.17146) does this for prediction models, not for policy values. IV-falsification (2312.15624) is one-shot.
- Estimator-selection work (AutoOPE, PAS-IF, OCV) picks the lowest-error estimator; it does not tell you whether *all* of them are wrong.

### The question
Given only the logs, what tests have real power to reject an OPE estimate, against which failure modes, and can we prove which common "sanity checks" have none?

### Claims
1. **Agreement is uninformative.** Prove that for the standard family (FQE, IS, WIS, DR, MAGIC) on the same logs with the same behavior model, there are failure modes (unmeasured confounding, timing shift as in Idea 1, behavior-policy miscalibration) under which all estimators converge to the same wrong number. Agreement has zero power against them.
2. **A test battery with known power.** Tests computable from logs alone: (a) behavior-policy self-evaluation (must recover the empirical return; detects behavior-model miscalibration and preprocessing); (b) placebo rewards (shuffled or pre-treatment outcomes; detects estimators that find effects where none exist); (c) negative-control policies (perturbations of the behavior policy that cannot matter, e.g. changing actions in states where the action is clinically inert; detects confounding); (d) "future-blind" policies (same policy with future-leaking features removed); (e) split-logger checks when multiple hospitals or units exist. For each test, state the failure class it is sensitive to and give finite-sample power under a sequential model.
3. **Empirical.** Thousands of (environment, logger, target, estimator, preprocessing) configurations with ground truth: sepsis simulator, Frost's UVA/Padova, a couple of D4RL/DOPE tasks. Measure which tests fire when the estimate is actually wrong. Then apply the battery to a MIMIC-IV sepsis pipeline in the AI Clinician style and show which published-style setups fail which tests.

### Design
Treat each test as a hypothesis test with a null that must hold if OPE is valid. Report power vs. each injected failure. Deliverable is a `falsify-ope` package: pass the logs and the estimators, get a report. Your `BettingCS` tool gives anytime-valid versions for free.

### Kill test
If the battery's power against injected failures is no better than ESS thresholds and estimator agreement across the board, the method is not useful; the agreement-is-uninformative theorem and the empirical audit still make a workshop or AISTATS paper.

### Compute
CPU. Tabular and small-net FQE; the big cost is the number of configurations, which parallelizes on the M1 Pro cores.

### Risks
Reviewers may call it "a list of sanity checks." The defense is the theorem, the power analysis, and the real-data audit showing that pipelines in published papers fail. Make sure at least one test has a formal guarantee, not just empirics.

### Reading list
2608.12489, 2609.23374, 2601.17146, 2312.15624, Lipsitch 2010, Shi–Miao–Tchetgen Tchetgen 2020, Bennett & Kallus (proximal RL), Namkoong et al. 2020 (confounding sensitivity in OPE), Raghu et al. 2018 (behavior policy calibration), Doroudi et al. 2017 (IS favors short policies), DOPE (Fu et al. 2021).

---

## Idea 3. Preprocessing builds a fictional MDP: when offline RL's own evaluation fools it

**Domain:** offline RL / OPE theory with clinical and benchmark experiments. **Odds if executed well:** ~25–30%.

### Where it sits
- Frost 2026 showed one preprocessing step (temporal binning) inflates FQE 1.5–3x. The AI Clinician pipeline (Komorowski 2018) also uses k-means into 750 states, 25 quantile-binned actions, carry-forward imputation, and outlier removal. Each step changes the MDP that gets evaluated.
- "OPE with deeply-abstracted states" (2406.19531) gives a bias-variance decomposition for *state* abstractions and shows non-Markov abstractions bias OPE. "Off by a Beat" shows observation/intervention alignment inside a window changes the learned policy while OPE barely moves. The recent MIMIC time-step paper (OpenReview swaYG5XI6G) found ESS "improvements" from finer steps were an artifact of weight clipping.
- Nobody has put these together: one framework where a preprocessing map φ turns real trajectories into fictional ones, and the same φ is used for training and evaluation, so the evaluation is of the policy in the fictional MDP, not reality.

### The question
For which preprocessing maps is OPE on processed data an unbiased (or rank-preserving) estimate of value in the real process, and how large is the bias for the maps people actually use?

### Claims
1. **A φ-invariance condition.** Policy value in the real process equals FQE/IS value on φ(logs) iff φ preserves the policy's induced reward and transition kernels. Give bias bounds for: fixed-interval binning (via the degenerate timing-policy view from Idea 1), state aggregation (bisimulation error; connects to Li–Walsh–Littman 2006 and 2406.19531), action discretization (policy-class restriction plus propensity misspecification), carry-forward imputation (informative observation times), and weight clipping (ESS inflation).
2. **The shared-pipeline problem.** Show formally that applying φ to both training and evaluation data produces optimism that grows with how much the learned policy exploits artifacts of φ, which is exactly what offline RL optimizers are good at.
3. **A protocol.** Always evaluate on raw trajectories: map the policy's output back to raw actions and times, and run OPE there. Show in simulators that this closes the gap, and on MIMIC/eICU that it changes which published-style pipeline looks best.

### Design
Simulators with ground truth (sepsis sim, Frost UVA/Padova, a MuJoCo task with a continuous-time variant). Apply each φ alone and in combination; measure OPE bias vs truth. Then the AI Clinician pipeline on MIMIC-IV end-to-end, swapping each step, and a cross-hospital check on eICU (208 hospitals) to show the ranking of pipelines is not stable under φ.

### Kill test
If only binning matters and state/action/imputation choices move OPE by less than seed noise, the paper is "Frost was right and nothing else matters," which is a note, not a NeurIPS paper. The φ-invariance theory would still be a clean AISTATS submission.

### Compute
CPU for simulators and tabular pipelines; the 3080 for the neural FQE/CQL variants on MIMIC-sized data.

### Risks
Overlaps Idea 1 on binning; if you run both, give binning theory to Idea 1 and keep Idea 3 about the general map φ. Reviewers may see it as "known pitfalls." The defense is the unified condition, the shared-pipeline optimism result, and real data.

### Reading list
2602.06603, 2406.19531, "Off by a Beat" (E0bdq7bwQH), OpenReview swaYG5XI6G, Komorowski 2018, 2608.16482, Gottesman et al. 2019 (guidelines for OPE in healthcare), Li–Walsh–Littman 2006, Abel et al. 2016 (approximate state abstraction).

---

## Alternates if you want something non-clinical

- **Does deep RL still sit on the statistical precipice?** Re-analyze per-seed results from 2024–26 RL papers (repos, W&B) with IQM / stratified bootstrap / anytime-valid tests; report how many headline wins survive and a "seeds-to-decision" number. CPU only, builds on your S4-46 tooling. NeurIPS 2026 accepted "Auditing Evidence Framing at NeurIPS" and "The Heel of RLVR," so there is appetite. Odds ~20–30%, higher as a Datasets & Benchmarks paper. No moat: anyone can do it.
- **Why does γ→1 still hurt after reward centering?** 2501.06937 fixes the reward-offset problem and still sees degradation near γ=1 with no explanation. A controlled decomposition (variance vs. optimization conditioning vs. approximation) using your TD-Jacobian/sharpness tools on MinAtar/MuJoCo continuing tasks. 3080 is enough. Odds ~15–25%; RLC-shaped.

## Dropped
- NFL tracking OPE benchmark: realized outcomes are confounded by team quality, and JointDiff (ICLR 2026) / TacSim already own the trajectory-benchmark angle.
- Action-chunking, long-horizon TD bias, streaming RL mechanism, successor-feature variants, delayed-observation theory: all have 2025–26 papers from the groups that own them (Levine, Alberta, etc.).
