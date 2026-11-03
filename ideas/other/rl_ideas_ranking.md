# Ranking of the 250 RL ideas

Scores for every idea in `rl_ideas_50.md` (L1), `rl_ideas_novel_50.md` (N1), `rl_ideas_novel_50_set2.md` (S2), `rl_ideas_novel_50_set3.md` (S3), and `rl_ideas_novel_50_set4.md` (S4).

## Criteria and calibration

**Venue scores (1–100), one per conference.** The probability that a well-executed version of this idea, run by one person on one RTX 3080, produces a paper that gets accepted at that venue. This is not "fit"; it is acceptance odds, so it bakes in the base rate (~25% at all four), the venue's taste, and how much a single-GPU result is likely to convince reviewers. Calibration:

- 50+: a top pick. Clean mechanism or measurement, anchor papers that make the question obviously open, result informative either way, benchmarks reviewers accept at small scale.
- 40–49: solid. Publishable if the result is clean, but one of: moderate novelty risk, a narrow audience, or a benchmark reviewers may call small.
- 30–39: a coin flip at best. Usually a workshop paper or an AAAI paper unless the effect is surprisingly large.
- <30: incremental, likely scooped, or too narrow for the main track.

Venue taste, as applied: **NeurIPS** rewards surprise and mechanism and is the hardest on small-scale empirics. **ICML** rewards learning dynamics, optimization, estimators, and anything with a statistical or theoretical spine. **ICLR** rewards representation learning, LLM-RL, memory, world models, and provocative small-model findings. **AAAI** is the most forgiving of small benchmarks and rewards safety, multi-agent, evaluation methodology, and applied framing; it also penalizes niche theory.

**Feasibility (1–100).** Can one RTX 3080 (10 GB) plus freely available data finish the full experiment, with enough seeds to be believable, in roughly one to three weeks? Reference points: gridworld / gymnax / MinAtar diagnostics 85–95; state-based DMC or D4RL 80–90; Mujoco Playground / Brax / JaxMARL GPU-parallel 75–85; 0.5B LoRA-GRPO on procedural data 65–80 (1.5B drops to ~60); DreamerV3 / pixel world models 45–60; Atari-100k with several games × seeds 30–55; anything needing EfficientZero-V2 reproduction, human raters, Isaac Lab, or a real robot 25–45.

**EV** = best venue score × feasibility / 100. A rough "probability this turns into an accepted paper" that combines the two axes. Use it for triage only; the individual columns are the real answer to the question.

---

## L1: `rl_ideas_50.md` (follow-ups to recent papers)

| # | Idea | NeurIPS | ICML | ICLR | AAAI | Feas | EV | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Depth in TD vs contrastive RL | 45 | 48 | 50 | 40 | 60 | 30 | Strong anchor; depth-256 sweep on HumanoidBench strains 10 GB, keep it to DMC |
| 2 | Predictor for critic size | 35 | 40 | 42 | 40 | 70 | 29 | Many short runs; the regression may not hold across suites |
| 3 | Compute-optimal actor-critic | 40 | 45 | 45 | 38 | 55 | 25 | Scaling laws from one GPU look thin to reviewers |
| 4 | Why scaling the actor hurts | 42 | 45 | 48 | 40 | 80 | 38 | Documented puzzle, cheap, clean yes/no |
| 5 | Which normalization does the work | 38 | 42 | 45 | 40 | 70 | 32 | Useful ablation; risk of "just an ablation" review |
| 6 | Deep critics and seed variance | 32 | 35 | 35 | 35 | 85 | 30 | Methodology; AAAI/workshop shape |
| 7 | WM component losing plasticity first | 40 | 40 | 45 | 38 | 50 | 22 | DreamerV3 continual is slow on a 3080 |
| 8 | Taxonomy of nonstationarity × fix | 35 | 38 | 42 | 42 | 80 | 34 | Lookup-table paper; AAAI likes it |
| 9 | Predicting plasticity collapse | 35 | 38 | 40 | 40 | 80 | 32 | Forecaster may be trivially the return slope |
| 10 | Plasticity under self-play | 40 | 40 | 42 | 40 | 75 | 32 | Connects two literatures; JaxMARL cheap |
| 11 | Redundancy reduction as plasticity reg | 30 | 32 | 38 | 38 | 75 | 28 | Incremental method |
| 12 | Auto UTD from spectral signal | 33 | 36 | 40 | 38 | 70 | 28 | Adaptive-UTD papers exist; must beat them |
| 13 | When decoder-free WMs fail | 35 | 35 | 42 | 36 | 45 | 19 | Three pixel Dreamer variants × dials is weeks |
| 14 | Where optimism lives in a WM | 35 | 35 | 40 | 36 | 35 | 14 | Dreamer on 8–10 Atari games × 3 variants × seeds is not a 3080 project |
| 15 | Adaptive imagination horizon | 30 | 32 | 38 | 38 | 45 | 17 | Likely done in some form |
| 16 | WM hallucination audit | 35 | 35 | 40 | 38 | 55 | 22 | Good metric idea; pixel Dreamer cost |
| 17 | DR inside imagination | 30 | 32 | 38 | 38 | 55 | 21 | Modest novelty |
| 18 | Test-time search compute-matched | 35 | 35 | 38 | 33 | 30 | 11 | Requires EZ-V2 + Dreamer on Atari |
| 19 | Quasimetric heads in WMs | 32 | 35 | 40 | 35 | 55 | 22 | Combination paper |
| 20 | Does depth buy horizon generalization | 45 | 45 | 50 | 38 | 80 | 40 | CRL is cheap; mechanistic follow-up to a best paper |
| 21 | MI-driven relabeling horizon | 32 | 35 | 40 | 38 | 75 | 30 | Diagnostic→method; moderate |
| 22 | Horizon-generalization benchmark | 30 | 30 | 38 | 35 | 65 | 25 | Benchmark papers need breadth one GPU lacks |
| 23 | Depth vs relabeling horizon | 28 | 32 | 36 | 35 | 85 | 31 | Small interaction study; workshop |
| 24 | Jacobian distillation for RL | 25 | 28 | 33 | 35 | 80 | 28 | Direct extension, little surprise |
| 25 | Why intra-trajectory negatives help | 30 | 35 | 38 | 33 | 75 | 28 | Explains a trick; narrow audience |
| 26 | Rank flips at fine-tune onset | 40 | 40 | 45 | 40 | 80 | 36 | Live problem (SOAR withdrawn); D4RL cheap |
| 27 | How many warmup rollouts | 25 | 28 | 32 | 35 | 80 | 28 | Hyperparameter study |
| 28 | O2O with changed dynamics | 32 | 33 | 38 | 40 | 80 | 32 | Realistic setting; benchmark-ish |
| 29 | Pathwise vs score-function switching rule | 38 | 45 | 42 | 36 | 80 | 36 | Practitioner value; GPU-parallel fast |
| 30 | Discrete Value Gradient Flow | 30 | 35 | 40 | 35 | 50 | 20 | Atari-100k subset cost |
| 31 | Test-time flow steps in RL | 30 | 33 | 40 | 35 | 75 | 30 | Hot framing, small result |
| 32 | Pathwise via WM vs via Q | 38 | 42 | 42 | 35 | 65 | 27 | Clean comparison; bias measurement is the selling point |
| 33 | Actor sees critic quantiles | 25 | 28 | 33 | 35 | 85 | 30 | Probably already tried |
| 34 | Dense-to-sparse RLVR curriculum | 28 | 30 | 38 | 40 | 55 | 22 | Needs retrieval/search infra for MuSiQue |
| 35 | Difficulty-mixing law | 32 | 38 | 40 | 38 | 55 | 22 | Grid × domains × budgets is heavy at 1B |
| 36 | Adaptive GRPO group size | 28 | 32 | 38 | 38 | 65 | 25 | Small method |
| 37 | Hackable pretext games | 30 | 30 | 38 | 35 | 45 | 17 | Needs human-rated quality |
| 38 | Symbolic↔verbal RLVR transfer | 30 | 32 | 40 | 40 | 65 | 26 | Good question; answer may be mundane |
| 39 | Step-level advantage with oracle states | 33 | 36 | 42 | 38 | 70 | 29 | Controlled credit-assignment study |
| 40 | RL new behaviors in interactive envs | 42 | 40 | 48 | 38 | 60 | 29 | Direct test of the NeurIPS runner-up; k=256 rollouts are slow |
| 41 | Auto difficulty controller | 25 | 28 | 35 | 38 | 65 | 25 | Curricula at 50% success are standard |
| 42 | Latch units for SSM horizon generalization | 40 | 45 | 48 | 38 | 90 | 43 | Anchor paper names it; tiny compute |
| 43 | Memory in WM vs in policy | 30 | 33 | 38 | 38 | 70 | 27 | Clean but small |
| 44 | ZSC via shared world models | 32 | 33 | 38 | 38 | 55 | 21 | WM + MARL infra |
| 45 | Population size vs plasticity | 30 | 32 | 35 | 38 | 70 | 27 | Extension of #10 |
| 46 | How many seeds Atari-100k needs | 35 | 35 | 35 | 32 | 25 | 9 | 20 seeds × 8 games × 3 methods incl. EZ-V2 is far beyond a 3080 |
| 47 | Wall-clock-matched leaderboard | 33 | 33 | 33 | 30 | 25 | 8 | Same runs as #46 |
| 48 | Sample efficiency when steps are free | 35 | 40 | 38 | 35 | 80 | 32 | Timely; REPPO raised it |
| 49 | Spectral health as universal signal | 30 | 33 | 38 | 38 | 55 | 21 | Needs Dreamer + Atari piggyback |
| 50 | Depth × norm × plasticity sweep | 40 | 42 | 45 | 38 | 70 | 32 | One heatmap; many runs |

## N1: `rl_ideas_novel_50.md` (mechanisms and measurements)

| # | Idea | NeurIPS | ICML | ICLR | AAAI | Feas | EV | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Curl of the TD update field | 48 | 55 | 45 | 35 | 80 | 44 | Theory-backed, never measured; JVPs cheap |
| 2 | Leverage scores in the deadly triad | 45 | 50 | 45 | 35 | 75 | 38 | Classical diagnostic, new domain |
| 3 | Ordering-only critic | 40 | 42 | 45 | 38 | 80 | 36 | Ranking-loss critics exist; the decomposition is the novelty |
| 4 | Activations as pessimism knob | 42 | 45 | 45 | 40 | 85 | 38 | Offline MuJoCo cheap; clear mechanism |
| 5 | Lazy critics: action sensitivity onset | 45 | 45 | 48 | 38 | 85 | 41 | Diagnostic + fix; shared with many later ideas |
| 6 | Gradient starvation across action dims | 35 | 38 | 40 | 38 | 70 | 28 | Hand tasks are finicky |
| 7 | Residual gradients under modern archs | 38 | 45 | 42 | 38 | 70 | 32 | Retest of a classic |
| 8 | Stale momentum in on-policy | 30 | 38 | 35 | 35 | 90 | 34 | Days of compute; effect may be small |
| 9 | γ–learning-rate scaling law | 35 | 42 | 38 | 38 | 85 | 36 | Simple rule, easy to validate |
| 10 | Policy-invariant critic subspace | 35 | 40 | 42 | 35 | 70 | 29 | 50 policies × critics; CCA analysis |
| 11 | Within-task forgetting of stepping stones | 42 | 40 | 45 | 40 | 75 | 34 | Intuitive, unmeasured, actionable |
| 12 | Half-life of a transition | 35 | 38 | 40 | 38 | 80 | 32 | Buffer-size rule |
| 13 | Replay deduplication in GPU envs | 25 | 28 | 32 | 35 | 80 | 28 | Systems-flavored, narrow |
| 14 | Agents learn clocks | 45 | 42 | 50 | 40 | 90 | 45 | Cheap probe, clear story, horizon-generalization tie-in |
| 15 | Truncation as right-censoring | 40 | 48 | 42 | 40 | 90 | 43 | Statistics transplant; at risk of prior work |
| 16 | Control-frequency transfer | 32 | 38 | 38 | 40 | 80 | 32 | Robotics-relevant |
| 17 | How many parallel envs | 30 | 35 | 38 | 35 | 80 | 30 | Partially studied in PPO scaling work |
| 18 | Vine at scale | 42 | 48 | 45 | 38 | 75 | 36 | Cloning is a pytree copy in Brax/MJX |
| 19 | Ground-truth ∇ₐQ vs learned | 45 | 48 | 48 | 38 | 75 | 36 | Measures a folk explanation |
| 20 | Counterfactual info gain via cloning | 35 | 38 | 40 | 40 | 85 | 34 | Noisy-TV solution; small envs |
| 21 | Vine advantages for LLM text agents | 38 | 38 | 45 | 38 | 55 | 25 | Branching K actions × LLM rollouts is slow |
| 22 | γ-disagreement exploration | 30 | 35 | 38 | 40 | 85 | 34 | May be redundant with ensembles |
| 23 | Flow noise as intention | 32 | 35 | 42 | 38 | 70 | 29 | Needs online flow-policy codebase |
| 24 | Denoising steps as temperature | 28 | 32 | 38 | 36 | 75 | 28 | Small |
| 25 | OPE for exploration coverage | 32 | 40 | 36 | 35 | 70 | 28 | New OPE target; theory-light |
| 26 | Strategy modes across seeds | 38 | 35 | 40 | 40 | 80 | 32 | Reporting-practice paper |
| 27 | MDL of V* predicts sample complexity | 35 | 42 | 35 | 38 | 85 | 36 | Tabular; ICML-shaped |
| 28 | Reward dropout | 30 | 32 | 38 | 40 | 85 | 34 | One-liner; needs a big effect |
| 29 | Reward-term equalization | 28 | 30 | 35 | 38 | 85 | 32 | Hand-tuned defaults may win |
| 30 | Hacking detection from return distribution | 35 | 33 | 40 | 42 | 60 | 25 | Needs envs with known exploits |
| 31 | How often is Q multimodal | 35 | 38 | 42 | 36 | 75 | 32 | Settles a motivation claim for flow policies |
| 32 | Seed-variance ANOVA | 38 | 38 | 36 | 38 | 90 | 34 | Null result still informative |
| 33 | Cross-fitted Bellman error | 42 | 48 | 42 | 40 | 85 | 41 | Validation signal for RL; causal-inference transplant |
| 34 | Early kill from first 5% | 28 | 32 | 33 | 38 | 75 | 28 | AutoML flavor |
| 35 | Cross-sim proxy for sim2real | 30 | 28 | 32 | 32 | 30 | 10 | Needs Isaac Lab and a real robot to validate |
| 36 | Do fast learners converge worse | 35 | 38 | 40 | 36 | 55 | 22 | Running BRO/SimbaV2 to convergence is weeks |
| 37 | Cross-action interference | 35 | 38 | 40 | 38 | 55 | 22 | Atari subset |
| 38 | Churn hotspots = bottlenecks | 30 | 32 | 35 | 40 | 90 | 36 | Cheap; HRL audience |
| 39 | Value-preserving replay compression | 25 | 28 | 33 | 36 | 55 | 20 | Systems paper |
| 40 | Learned masks for sensor selection | 25 | 28 | 32 | 38 | 85 | 32 | Known technique |
| 41 | Stigmergy | 35 | 32 | 38 | 42 | 70 | 29 | Original; results may be messy |
| 42 | Hamiltonian component in self-play | 40 | 50 | 40 | 38 | 75 | 38 | Game-dynamics theory meets deep self-play |
| 43 | Exact counterfactual credit in MARL | 40 | 42 | 42 | 42 | 70 | 29 | JaxMARL cloning easy; COMA baselines fiddly |
| 44 | RL teaches when to stop | 42 | 40 | 50 | 38 | 75 | 38 | Cheap, decisive ablation; adjacent work exists |
| 45 | Verifier vs policy sample allocation | 35 | 40 | 42 | 38 | 75 | 32 | Clean crossover law |
| 46 | Spaced repetition for RLVR | 30 | 32 | 40 | 40 | 70 | 28 | Retention angle is new |
| 47 | RLVR → in-context exploration | 35 | 35 | 45 | 38 | 80 | 36 | Mostly evaluation |
| 48 | Answer-first RL | 42 | 40 | 50 | 38 | 80 | 40 | Both outcomes informative |
| 49 | Interleaved vs blocked domains | 30 | 33 | 40 | 38 | 70 | 28 | Gradient-conflict schedulers exist |
| 50 | Prompt-level learned baseline | 35 | 38 | 45 | 38 | 70 | 32 | At risk of prior work |

## S2: `rl_ideas_novel_50_set2.md`

| # | Idea | NeurIPS | ICML | ICLR | AAAI | Feas | EV | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Init as exploration hyperparameter | 35 | 38 | 40 | 40 | 90 | 36 | Could join a "hidden hyperparameters" paper |
| 2 | Numerical precision in TD | 38 | 42 | 42 | 38 | 90 | 38 | Days of compute; effect must exist in bf16 |
| 3 | Entropy collapse hysteresis | 40 | 45 | 42 | 38 | 85 | 38 | Measurable loop; clean mechanism |
| 4 | Bits of advantage the actor uses | 38 | 42 | 42 | 38 | 90 | 38 | One-line ablation; sign-PG variants exist |
| 5 | n-step by transition age | 30 | 35 | 38 | 40 | 65 | 26 | Atari subset |
| 6 | μP under bootstrapping | 40 | 50 | 45 | 35 | 70 | 35 | ICML-shaped; at risk |
| 7 | Normalizer drift | 30 | 33 | 36 | 38 | 90 | 34 | Infrastructure finding |
| 8 | Policy-evaluation lag | 35 | 42 | 40 | 38 | 80 | 34 | Freeze-and-refit protocol is expensive but small nets |
| 9 | Tanh saturation | 28 | 32 | 35 | 40 | 85 | 34 | Known folklore |
| 10 | Exploration order | 32 | 33 | 36 | 42 | 90 | 38 | Your own question; AAAI-friendly |
| 11 | Exploration waste | 32 | 35 | 36 | 40 | 85 | 34 | Needs computable V* |
| 12 | Luby restarts for RL seeds | 32 | 40 | 35 | 40 | 85 | 34 | Heavy tails must show up |
| 13 | State-dependent entropy target | 30 | 35 | 38 | 38 | 80 | 30 | Method paper |
| 14 | Bellman wavefront | 35 | 40 | 38 | 40 | 90 | 36 | Nice visual; predictive claim is the risk |
| 15 | Hidden curriculum in replay datasets | 35 | 38 | 42 | 42 | 90 | 38 | One week on D4RL; surprising if true |
| 16 | Dataset mode count picks the method | 32 | 35 | 40 | 42 | 80 | 34 | Controlled dataset construction |
| 17 | Data attribution for RL | 38 | 40 | 45 | 40 | 65 | 29 | At risk; influence in moving-target setting is hard |
| 18 | Hot-swapping reward mid-training | 30 | 30 | 35 | 40 | 85 | 34 | Practical, low prestige |
| 19 | Bursty vs steady updates | 32 | 38 | 38 | 38 | 85 | 32 | Clean factorial |
| 20 | Origin of ensemble diversity | 35 | 40 | 40 | 38 | 80 | 32 | Explains inconsistent results |
| 21 | Value discontinuities at contact | 35 | 38 | 40 | 38 | 65 | 26 | Manipulation tasks are slow to learn |
| 22 | Control margins for RL policies | 35 | 38 | 38 | 40 | 65 | 26 | Linearizing through the sim is fiddly |
| 23 | Weight quantization closed-loop | 25 | 28 | 32 | 38 | 80 | 30 | Embedded audience |
| 24 | How open-loop is a policy | 32 | 33 | 38 | 40 | 90 | 36 | Cheap and interpretable |
| 25 | Emergent equivariance | 35 | 40 | 42 | 38 | 80 | 34 | Representation story |
| 26 | WM error is policy-conditioned | 35 | 38 | 42 | 38 | 55 | 23 | Dreamer/TD-MPC2 cost |
| 27 | Exact double sampling via cloning | 38 | 48 | 40 | 36 | 75 | 36 | Settles an old debate properly |
| 28 | Bandwidth law for coordination | 35 | 40 | 38 | 40 | 70 | 28 | Information-theoretic MARL |
| 29 | Specialization vs team size | 30 | 32 | 35 | 42 | 75 | 32 | AAAI MARL track |
| 30 | Role adapters | 28 | 30 | 36 | 42 | 85 | 36 | At risk; simple |
| 31 | Legibility tax | 38 | 36 | 45 | 38 | 60 | 27 | Human-rated subset |
| 32 | Learning to ask for a hint | 32 | 33 | 42 | 40 | 70 | 29 | Calibration angle is good |
| 33 | Planted-error recovery metric | 35 | 35 | 45 | 40 | 75 | 34 | Transfer claim is the hook |
| 34 | Mixed-temperature groups | 25 | 30 | 38 | 38 | 75 | 28 | Small |
| 35 | Notes across episodes | 32 | 30 | 42 | 40 | 55 | 23 | Multi-episode LLM RL is slow |
| 36 | Inverse-task auxiliary | 30 | 32 | 40 | 38 | 65 | 26 | Problem generation is well trodden |
| 37 | Paraphrase groups | 25 | 28 | 38 | 38 | 75 | 28 | Small |
| 38 | Activation steering for policies | 35 | 33 | 45 | 38 | 75 | 34 | At risk; SAE tooling exists |
| 39 | Dormant units are fossils | 32 | 35 | 40 | 38 | 85 | 34 | Charming; may be false |
| 40 | Deterministic-eval gap | 28 | 30 | 35 | 38 | 90 | 34 | Piggyback diagnostic |
| 41 | Minimal continuous-control task subset | 30 | 32 | 32 | 35 | 50 | 18 | Needs results for 15+ methods |
| 42 | Sign accuracy of PG estimators | 35 | 45 | 40 | 36 | 80 | 36 | ICML-shaped estimator study |
| 43 | Do seeds learn the same representation | 32 | 35 | 40 | 36 | 85 | 34 | CKA across seeds |
| 44 | IRL recovers reward or shaping | 38 | 42 | 40 | 40 | 70 | 29 | Theory known; empirics missing |
| 45 | Learning-path imitation | 28 | 30 | 35 | 40 | 85 | 34 | Simple |
| 46 | Anti-goals | 32 | 30 | 40 | 42 | 70 | 29 | Safety interface for GCRL |
| 47 | How much memory a task needs | 35 | 38 | 42 | 40 | 75 | 32 | Useful statistic |
| 48 | Hack discovery time vs capacity | 35 | 35 | 42 | 40 | 60 | 25 | Exploit construction effort |
| 49 | LoRA rank for dynamics adaptation | 25 | 28 | 35 | 40 | 85 | 34 | At risk |
| 50 | PEGASUS at scale | 40 | 48 | 42 | 38 | 80 | 38 | Revival with a real reason |

## S3: `rl_ideas_novel_50_set3.md`

| # | Idea | NeurIPS | ICML | ICLR | AAAI | Feas | EV | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Weight decay is implicit pessimism | 40 | 48 | 45 | 38 | 85 | 41 | Sign-flip prediction is a crisp test |
| 2 | Adam ε as hidden schedule | 35 | 42 | 40 | 36 | 90 | 38 | Days of compute |
| 3 | Empirical contraction modulus | 42 | 52 | 42 | 35 | 80 | 42 | Classical quantity, never measured for deep critics |
| 4 | Critical batch size of TD | 38 | 48 | 42 | 36 | 85 | 41 | LLM-training concept transplanted |
| 5 | Fraction of actor updates that improve | 40 | 45 | 42 | 36 | 75 | 34 | CRN oracle is the trick |
| 6 | Trust region on state distribution | 38 | 45 | 42 | 38 | 75 | 34 | At risk |
| 7 | Plateaus as grokking | 42 | 42 | 48 | 38 | 85 | 41 | Timely framing; precursors must exist |
| 8 | Critic error tolerance curves | 35 | 40 | 40 | 38 | 85 | 34 | Map, not method |
| 9 | Effective horizon of a critic | 38 | 42 | 42 | 38 | 80 | 34 | Reward-impulse response |
| 10 | Actor's state distribution | 35 | 40 | 40 | 38 | 90 | 36 | One-line change |
| 11 | Two-speed critics | 32 | 38 | 40 | 38 | 85 | 34 | Replaces τ |
| 12 | Durbin–Watson for TD | 38 | 45 | 42 | 40 | 90 | 40 | POMDP detector; cheap |
| 13 | Emergent bisimulation | 35 | 40 | 45 | 36 | 70 | 32 | At risk |
| 14 | How linear is a trained policy | 32 | 35 | 38 | 40 | 85 | 34 | Deployment relevance |
| 15 | Actor–critic feature misalignment | 30 | 35 | 38 | 38 | 80 | 30 | Attribution noise |
| 16 | Recurrent policies: belief vs mode | 38 | 40 | 45 | 38 | 85 | 38 | Computable beliefs; clean |
| 17 | Emergent temporal abstraction | 30 | 33 | 38 | 38 | 80 | 30 | Measurement only |
| 18 | Replay-time DR via re-simulation | 38 | 40 | 45 | 40 | 75 | 34 | Robotics-relevant and new |
| 19 | Coarse rewards | 28 | 32 | 36 | 38 | 90 | 34 | Needs a big effect |
| 20 | Reward-term decorrelation alarm | 30 | 30 | 38 | 42 | 75 | 32 | Safety framing |
| 21 | Rarity vs distance sparsity | 35 | 36 | 40 | 42 | 85 | 36 | Benchmark grid with a thesis |
| 22 | Null-space exploration | 32 | 36 | 40 | 38 | 75 | 30 | High-DoF tasks slow |
| 23 | Random networks as macro-actions | 32 | 33 | 40 | 40 | 85 | 34 | Simple, could be strong |
| 24 | Buffer transplant decomposition | 40 | 40 | 45 | 40 | 80 | 36 | Reframes algorithm comparison |
| 25 | First-success lock-in | 38 | 36 | 42 | 40 | 85 | 36 | Causal intervention; nice |
| 26 | Algorithm fingerprinting | 35 | 35 | 40 | 40 | 75 | 30 | Deployment story |
| 27 | Sobolev critics from diff sims | 38 | 42 | 45 | 36 | 65 | 29 | MJX contact gradients are noisy |
| 28 | Shield reliance | 35 | 33 | 40 | 45 | 75 | 34 | Safety; AAAI strong |
| 29 | Safety is forgotten first | 33 | 33 | 40 | 45 | 70 | 32 | Safety; AAAI strong |
| 30 | How often a policy needs to look | 30 | 33 | 36 | 40 | 85 | 34 | Lyapunov tie-in |
| 31 | Perfect-model gap | 42 | 42 | 48 | 38 | 60 | 29 | Sim-as-model in TD-MPC2 is doable; in Dreamer awkward |
| 32 | Correlated noise as coordination channel | 35 | 35 | 38 | 42 | 85 | 36 | Audit finding |
| 33 | Inaction baselines | 30 | 32 | 35 | 42 | 75 | 32 | Interpretable credit |
| 34 | RL gain vs pretraining exposure (from scratch) | 48 | 48 | 55 | 40 | 65 | 36 | Controlled answer to sharpen-vs-create; 10 small pretrains |
| 35 | RLVR size floor | 38 | 38 | 48 | 38 | 65 | 31 | 1.5B GRPO tight on 10 GB |
| 36 | Rewarding lucky guesses | 38 | 36 | 48 | 40 | 75 | 36 | Checkable intermediates make it cheap |
| 37 | Decision depth before/after RL | 32 | 33 | 42 | 36 | 80 | 34 | At risk |
| 38 | Exposure bias in multi-turn RL | 35 | 35 | 45 | 40 | 60 | 27 | Text-env LLM rollouts slow |
| 39 | Restating as an action | 30 | 30 | 40 | 38 | 70 | 28 | Transfer test is the hook |
| 40 | Groups without replacement | 28 | 32 | 40 | 36 | 75 | 30 | At risk |
| 41 | Does RLVR compose | 40 | 40 | 50 | 40 | 70 | 35 | The generalization test reviewers want |
| 42 | Wrapper sensitivity audit | 35 | 35 | 38 | 38 | 85 | 32 | Implementation-matters lineage |
| 43 | Stratified evaluation | 28 | 33 | 33 | 38 | 90 | 34 | Small tool |
| 44 | Seeds vs evaluation episodes | 32 | 35 | 33 | 38 | 90 | 34 | Methodology |
| 45 | Start-state generalization offline | 33 | 35 | 40 | 42 | 85 | 36 | Cheap new axis for offline RL |
| 46 | Curse of context | 35 | 38 | 42 | 38 | 80 | 34 | At risk |
| 47 | Exploration bonuses create habits | 32 | 33 | 40 | 40 | 85 | 34 | Simple, likely true |
| 48 | Shaping ruggedness | 30 | 33 | 36 | 38 | 75 | 28 | Clustering endpoints is noisy |
| 49 | Reset subsidy | 32 | 35 | 38 | 38 | 85 | 32 | Reset-free RL audience |
| 50 | Hyperparameter drift by forking | 35 | 40 | 42 | 38 | 70 | 29 | Many forks |

## S4: `rl_ideas_novel_50_set4.md`

| # | Idea | NeurIPS | ICML | ICLR | AAAI | Feas | EV | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Edge of stability in TD | 48 | 55 | 48 | 35 | 80 | 44 | Hot supervised result; at risk of parallel work |
| 2 | Anticipatory (extragradient) targets | 35 | 42 | 40 | 36 | 85 | 36 | Method with a mechanism |
| 3 | Revealed time preference | 40 | 42 | 42 | 42 | 90 | 38 | Economist's method; tiny compute |
| 4 | Frequency response of the learner | 40 | 45 | 40 | 36 | 85 | 38 | Very original; "so what" risk |
| 5 | Where the regularizer enters | 30 | 35 | 36 | 38 | 90 | 34 | At risk |
| 6 | Frame-skip decomposition | 38 | 40 | 42 | 40 | 80 | 34 | Universal knob explained |
| 7 | Range compression in critics | 38 | 42 | 42 | 38 | 85 | 36 | Slope, not intercept; new |
| 8 | Fresh-critic gap as reset trigger | 32 | 36 | 40 | 38 | 80 | 32 | At risk |
| 9 | Dimensionless hyperparameters | 35 | 42 | 40 | 38 | 65 | 27 | Many rescaled families |
| 10 | Advantage-sign agreement uncertainty | 32 | 36 | 40 | 38 | 85 | 34 | Ensemble-free uncertainty |
| 11 | Budget-aware exploration | 32 | 36 | 40 | 40 | 80 | 32 | Bandit theory tie-in |
| 12 | Reward-epistemic exploration | 30 | 32 | 38 | 40 | 85 | 34 | Must beat RND |
| 13 | Recovery paradox | 35 | 33 | 40 | 42 | 80 | 34 | Robotics-relevant |
| 14 | Decide only when it matters | 30 | 33 | 38 | 40 | 80 | 32 | Adaptive repeat |
| 15 | Junk tolerance curve | 35 | 38 | 42 | 38 | 85 | 36 | Cliff shape is the result |
| 16 | State vs action coverage offline | 35 | 38 | 42 | 42 | 80 | 34 | Dataset construction |
| 17 | Offline RL with known π_β | 33 | 38 | 40 | 40 | 85 | 34 | Oracle comparison |
| 18 | PPO clip vs action dimension | 30 | 40 | 38 | 36 | 90 | 36 | At risk; fast |
| 19 | Action parameterization and noise color | 33 | 36 | 40 | 40 | 80 | 32 | Unifying predictor |
| 20 | Developmental DoF curriculum | 33 | 33 | 40 | 42 | 70 | 29 | Humanoid runs slow |
| 21 | Basin of attraction shrinks | 38 | 38 | 45 | 40 | 85 | 38 | Non-monotonicity claim is the hook |
| 22 | Is hierarchy just long credit | 40 | 40 | 45 | 40 | 65 | 29 | HRL baselines are finicky |
| 23 | Latency identification time | 30 | 33 | 36 | 40 | 80 | 32 | RMA-adjacent |
| 24 | Model-tolerant actors | 30 | 33 | 40 | 36 | 55 | 22 | Dreamer cost |
| 25 | Where to imagine from | 28 | 30 | 38 | 38 | 50 | 19 | At risk; Dreamer cost |
| 26 | Asymmetric capacity teams | 30 | 30 | 35 | 42 | 80 | 34 | AAAI MARL |
| 27 | Partner identification time | 32 | 33 | 40 | 40 | 70 | 28 | ZSC audience |
| 28 | Predictability bonus | 30 | 30 | 38 | 40 | 75 | 30 | At risk |
| 29 | Prefix-only GRPO | 40 | 40 | 50 | 38 | 80 | 40 | Cheap elicitation handle; at risk |
| 30 | Tool-induced skill atrophy | 38 | 36 | 48 | 42 | 75 | 36 | Timely; clean |
| 31 | How reversible is RLVR | 40 | 38 | 50 | 38 | 75 | 38 | Reversibility as a lens; new |
| 32 | Counterfactual token importance | 35 | 35 | 45 | 38 | 65 | 29 | Resampling cost |
| 33 | Information content of code feedback | 32 | 32 | 45 | 38 | 60 | 27 | Sandbox infra; 1.5B tight |
| 34 | Solver–verifier asymmetry | 35 | 35 | 45 | 38 | 75 | 34 | 2×2 matrix |
| 35 | Stubborn or sycophantic | 32 | 30 | 45 | 40 | 80 | 36 | Cheap eval |
| 36 | Answer-mode collapse | 35 | 35 | 45 | 40 | 80 | 36 | Enumerable solution sets |
| 37 | Scoring rules for numeric answers | 32 | 38 | 40 | 38 | 80 | 32 | Decision-theoretic |
| 38 | Temperature as an action | 32 | 33 | 42 | 38 | 75 | 32 | Small |
| 39 | Baldwin effect in meta-RL | 42 | 42 | 45 | 40 | 85 | 38 | Tiny compute; strong framing |
| 40 | Emergent quasimetric structure | 38 | 40 | 45 | 36 | 80 | 36 | Penalty vs architecture |
| 41 | Cognitive-bias battery | 38 | 33 | 40 | 45 | 85 | 38 | AAAI loves it; NeurIPS may call it cute |
| 42 | Out-of-span penalty for skills | 33 | 35 | 42 | 38 | 70 | 29 | Negative-transfer map |
| 43 | Teacher cloneability | 30 | 33 | 38 | 40 | 85 | 34 | Distillation audience |
| 44 | Transfer actor, reset critic | 30 | 33 | 38 | 40 | 85 | 34 | At risk |
| 45 | Structured label noise in BT | 38 | 42 | 42 | 40 | 80 | 34 | RLHF-adjacent; small data |
| 46 | Anytime-valid comparisons | 38 | 42 | 35 | 38 | 90 | 38 | Ship a tool |
| 47 | Mutation testing for RL | 32 | 32 | 35 | 40 | 85 | 34 | Engineering audience |
| 48 | Checkpoint selection inflation | 30 | 32 | 32 | 36 | 90 | 32 | Short paper |
| 49 | Determinism in defaults | 32 | 36 | 38 | 38 | 70 | 27 | Re-tuning cost |
| 50 | Entropy conversion efficiency | 38 | 42 | 42 | 38 | 80 | 34 | Task property predicts method |

---

## Leaderboards

### Top 30 by EV (best venue × feasibility)

| Rank | ID | Idea | Best venue | NeurIPS | ICML | ICLR | AAAI | Feas | EV |
|---|---|---|---|---|---|---|---|---|---|
| 1 | N1-14 | Agents learn clocks | ICLR | 45 | 42 | 50 | 40 | 90 | 45 |
| 2 | N1-1 | Curl of the TD update field | ICML | 48 | 55 | 45 | 35 | 80 | 44 |
| 2 | S4-1 | Edge of stability in TD | ICML | 48 | 55 | 48 | 35 | 80 | 44 |
| 4 | L1-42 | Latch units for SSM horizon generalization | ICLR | 40 | 45 | 48 | 38 | 90 | 43 |
| 4 | N1-15 | Truncation as right-censoring | ICML | 40 | 48 | 42 | 40 | 90 | 43 |
| 6 | S3-3 | Empirical contraction modulus | ICML | 42 | 52 | 42 | 35 | 80 | 42 |
| 7 | N1-5 | Lazy critics: action-sensitivity onset | ICLR | 45 | 45 | 48 | 38 | 85 | 41 |
| 7 | N1-33 | Cross-fitted Bellman error | ICML | 42 | 48 | 42 | 40 | 85 | 41 |
| 7 | S3-1 | Weight decay is implicit pessimism | ICML | 40 | 48 | 45 | 38 | 85 | 41 |
| 7 | S3-4 | Critical batch size of TD | ICML | 38 | 48 | 42 | 36 | 85 | 41 |
| 7 | S3-7 | Plateaus as grokking | ICLR | 42 | 42 | 48 | 38 | 85 | 41 |
| 7 | S3-12 | Durbin–Watson for TD | ICML | 38 | 45 | 42 | 40 | 90 | 41 |
| 13 | L1-20 | Does depth buy horizon generalization | ICLR | 45 | 45 | 50 | 38 | 80 | 40 |
| 13 | N1-48 | Answer-first RL | ICLR | 42 | 40 | 50 | 38 | 80 | 40 |
| 13 | S4-29 | Prefix-only GRPO | ICLR | 40 | 40 | 50 | 38 | 80 | 40 |
| 16 | L1-4 | Why scaling the actor hurts | ICLR | 42 | 45 | 48 | 40 | 80 | 38 |
| 16 | N1-2 | Leverage scores in the deadly triad | ICML | 45 | 50 | 45 | 35 | 75 | 38 |
| 16 | N1-4 | Activations as pessimism knob | ICML/ICLR | 42 | 45 | 45 | 40 | 85 | 38 |
| 16 | N1-42 | Hamiltonian component in self-play | ICML | 40 | 50 | 40 | 38 | 75 | 38 |
| 16 | N1-44 | RL teaches when to stop | ICLR | 42 | 40 | 50 | 38 | 75 | 38 |
| 16 | S2-2 | Numerical precision in TD | ICML/ICLR | 38 | 42 | 42 | 38 | 90 | 38 |
| 16 | S2-3 | Entropy collapse hysteresis | ICML | 40 | 45 | 42 | 38 | 85 | 38 |
| 16 | S2-4 | Bits of advantage | ICML/ICLR | 38 | 42 | 42 | 38 | 90 | 38 |
| 16 | S2-10 | Exploration order | AAAI | 32 | 33 | 36 | 42 | 90 | 38 |
| 16 | S2-15 | Hidden curriculum in replay datasets | ICLR/AAAI | 35 | 38 | 42 | 42 | 90 | 38 |
| 16 | S2-50 | PEGASUS at scale | ICML | 40 | 48 | 42 | 38 | 80 | 38 |
| 16 | S3-2 | Adam ε as hidden schedule | ICML | 35 | 42 | 40 | 36 | 90 | 38 |
| 16 | S3-16 | Recurrent policies: belief vs mode | ICLR | 38 | 40 | 45 | 38 | 85 | 38 |
| 16 | S4-3 | Revealed time preference | ICML/ICLR/AAAI | 40 | 42 | 42 | 42 | 90 | 38 |
| 16 | S4-4 | Frequency response of the learner | ICML | 40 | 45 | 40 | 36 | 85 | 38 |
| 16 | S4-21 | Basin of attraction shrinks | ICLR | 38 | 38 | 45 | 40 | 85 | 38 |
| 16 | S4-31 | How reversible is RLVR | ICLR | 40 | 38 | 50 | 38 | 75 | 38 |
| 16 | S4-39 | Baldwin effect in meta-RL | ICLR | 42 | 42 | 45 | 40 | 85 | 38 |
| 16 | S4-41 | Cognitive-bias battery | AAAI | 38 | 33 | 40 | 45 | 85 | 38 |
| 16 | S4-46 | Anytime-valid comparisons | ICML | 38 | 42 | 35 | 38 | 90 | 38 |

### Top 10 per venue (acceptance odds only, ignoring feasibility)

**NeurIPS**
1. N1-1 TD curl (48), S4-1 edge of stability (48), S3-34 RL gain vs pretraining exposure (48)
4. L1-1 depth in TD (45), L1-20 depth → horizon generalization (45), N1-2 leverage (45), N1-5 lazy critics (45), N1-14 clocks (45), N1-19 ground-truth ∇ₐQ (45)
10. L1-4, L1-40, N1-4, N1-11, N1-18, N1-33, N1-44, N1-48, S3-3, S3-7, S3-31, S4-39 (42)

**ICML**
1. N1-1 TD curl (55), S4-1 edge of stability (55)
3. S3-3 contraction modulus (52)
4. N1-2 leverage (50), N1-42 self-play Hamiltonian (50), S2-6 μP under bootstrapping (50)
7. L1-1 (48), N1-15 censoring (48), N1-18 vine (48), N1-19 (48), N1-33 cross-fit (48), S2-27 exact double sampling (48), S2-50 PEGASUS (48), S3-1 weight decay (48), S3-4 critical batch (48), S3-34 (48)

**ICLR**
1. S3-34 RL gain vs pretraining exposure (55)
2. L1-1 (50), L1-20 (50), N1-14 clocks (50), N1-44 when to stop (50), N1-48 answer-first (50), S3-41 does RLVR compose (50), S4-29 prefix-only GRPO (50), S4-31 reversible RLVR (50)
10. L1-4, L1-40, L1-42, N1-5, N1-19, S3-7, S3-31, S3-35, S3-36, S4-1, S4-30 (48)

**AAAI**
1. S3-28 shield reliance (45), S3-29 safety forgotten first (45), S4-41 cognitive-bias battery (45)
4. L1-8 taxonomy of nonstationarity (42), N1-30 hacking detector (42), N1-41 stigmergy (42), N1-43 exact MARL credit (42), S2-10 exploration order (42), S2-15 hidden curriculum (42), S2-16 dataset mode count (42), S2-29 specialization vs team size (42), S2-30 role adapters (42), S2-46 anti-goals (42), S3-20 reward-term alarm (42), S3-21 rarity vs distance (42), S3-32 correlated noise (42), S3-33 inaction baselines (42), S3-45 start-state generalization (42), S4-3 revealed time preference (42), S4-13 recovery paradox (42), S4-16 coverage grid (42), S4-20 DoF curriculum (42), S4-26 asymmetric capacity (42), S4-30 tool atrophy (42)

### Lowest feasibility (do not start these on a 3080)

| ID | Idea | Feas | Why |
|---|---|---|---|
| L1-46 | How many seeds Atari-100k needs | 25 | 20 seeds × 8 games × 3 methods including EZ-V2 |
| L1-47 | Wall-clock-matched leaderboard | 25 | Same runs |
| L1-18 | Test-time search compute-matched | 30 | EZ-V2 reproduction |
| N1-35 | Cross-sim proxy for sim2real | 30 | Isaac Lab + real robot for validation |
| L1-14 | Where optimism lives in a WM | 35 | Dreamer on Atari × variants × seeds |
| L1-13, L1-15 | Decoder-free WM boundary; adaptive horizon | 45 | Pixel Dreamer sweeps |
| L1-37 | Hackable pretext games | 45 | Human raters |
| L1-7 | WM component plasticity | 50 | DreamerV3 continual |
| L1-30, S2-41, S4-25 | Discrete VGF; task subset; where to imagine from | 50 | Atari subset / data collection / Dreamer |

---

## Patterns worth knowing

- **ICML is the natural home for ~40% of these lists.** The learning-dynamics ideas (TD curl, edge of stability, contraction modulus, leverage, critical batch size, censoring, cross-fitting, weight-decay bias, Adam ε, Durbin–Watson) are cheap, run on tiny nets, and fit ICML's taste exactly. If you want the highest EV per GPU-hour, pick from this cluster and bundle two or three into one "the critic as a measured system" paper (S4's own suggestion).
- **ICLR is the home for the LLM-RL ideas,** but feasibility caps them around 75–80 because GRPO rollouts on a 10 GB card are slow and 1.5B is marginal. Stay at 0.5B–0.8B, procedural data, short contexts. The best ones ask a question about RL itself (answer-first, when-to-stop, reversibility, prefix-only, composition, from-scratch exposure) rather than proposing a new GRPO variant; the GRPO-variant ideas (group size, temperature mixing, paraphrase groups, without-replacement) all score under 40 everywhere.
- **NeurIPS never exceeds ~48 here.** That is honest: a single-GPU empirical paper needs an unusually surprising result to clear NeurIPS. The ideas that could do it are the ones where a null result would also surprise (clocks, TD curl, edge of stability, from-scratch exposure, lazy critics, ground-truth action gradients).
- **AAAI is where safety, MARL, and methodology go.** Shield reliance, safety-forgotten-first, the bias battery, exploration order, and the evaluation-budget ideas score higher at AAAI than anywhere else and are all highly feasible.
- **Feasibility killers are consistent:** Atari-100k at multiple games and seeds, any pixel-based world model sweep, EfficientZero-V2, human raters, Isaac Lab, real robots, and 1.5B-parameter GRPO. Everything built on gymnax / MinAtar / state-DMC / D4RL / Mujoco Playground / JaxMARL / Brax is in the 80–90 band.
- **The cloning family (N1-18/19/20/43, S2-27/50, S3-5/18/31/33) shares one harness** and most of it scores 36–48 at ICML. Building the save/restore + common-random-numbers infrastructure once and writing two or three papers from it is the highest-leverage program in these five files.
