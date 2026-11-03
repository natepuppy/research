# 50 RL research ideas for one RTX 3080

Each idea builds on a 2025–2026 paper that had impact, asks a question that paper left open, and can be run on one 10 GB GPU in days to a few weeks. None overlaps a stub in the ADEPT catalog (dead-ends, delayed rewards, offline data amount, action quantization, RLVR rigidity, cross-family distillation, borrowing experience, etc.).

Two honest caveats. "Would be accepted" means: if the result is clean and the finding is real, the shape of the paper fits what NeurIPS / ICML / ICLR / AAAI accept. No idea is guaranteed. And I checked each against what I could find, but you must run your own literature search before committing; a close paper may have appeared in the last few months.

Format: **Builds on** (the anchor paper), **Question**, **Experiment**, **Why it lands**, **Compute**.

---





Other:
- Should a agent purposely try to go to every square it does not understand before traversing ones it doesn't? Will this help it learn faster??? Has that deen done before?




## A. Scaling and architecture

**1. Depth helps self-supervised RL. Does it help bootstrapped RL, and if not, why?**
- Builds on: *1000 Layer Networks for Self-Supervised RL* (NeurIPS 2025 best paper); SimbaV2 (hyperspherical normalization).
- Question: the 1000-layer result used contrastive goal-reaching, which has a stationary target. TD learning has a moving, bootstrapped target. Does depth still help, and where does it break?
- Experiment: same architecture (SimbaV2 blocks), same tasks, depth swept 2 → 256. Train with (a) contrastive RL, (b) TD-based SAC. Track eNTK effective rank, churn, and target drift. Test fixes: stop-gradient schedules, target-network depth decoupling.
- Why it lands: it either extends the best paper or explains its limits. Both are publishable.
- Compute: DMC and a few HumanoidBench tasks, GPU-parallel via Mujoco Playground. Days.

**2. A predictor for how big the critic should be.**
- Builds on: SimBa (ICLR 2025), BRO: critic scaling saturates around 5M parameters, but nobody knows why there.
- Question: can a cheap measurement of the task (effective dimension of the visited state distribution, reward smoothness) predict the critic size where returns stop improving?
- Experiment: 15–20 DMC/MyoSuite tasks, critic width swept, find the saturation point per task, regress it on task statistics measured from a random-policy rollout. Validate on held-out tasks.
- Why it lands: scaling papers with a predictive rule get cited; "it saturates at 5M" with no explanation is the current state.
- Compute: small nets, many short runs. One to two weeks.

**3. Compute-optimal actor-critic: parameters vs. update-to-data ratio vs. env steps at fixed GPU-hours.**
- Builds on: BRO showed model size and replay ratio are "interchangeable to some degree"; SimbaV2 scaled UTD without resets.
- Question: for a fixed wall-clock budget, how should it be split? Is there a Chinchilla-style law for RL?
- Experiment: grid over critic size × UTD × env steps on 6 tasks at matched GPU-seconds. Fit a parametric frontier, test it on 3 unseen tasks.
- Why it lands: every practitioner wants this number. The LLM community rewarded the analogous paper heavily.
- Compute: many small runs. Two weeks.

**4. Why does scaling the actor hurt?**
- Builds on: SimBa, SimbaV2, and BRO all report that scaling the actor gives little or negative benefit, and all leave it unexplained.
- Question: is it exploration collapse (entropy falls faster in bigger actors), or overfitting to critic errors?
- Experiment: scale actor only; log entropy, action-gradient norm, and critic-error exploitation (action value under true return vs. critic). Test two fixes: entropy target scaled with actor size, and actor update delay.
- Why it lands: resolves a documented open puzzle in three high-profile papers.
- Compute: DMC. Days.

**5. Which normalization is doing the work in plasticity methods?**
- Builds on: SimBa, SimbaV2, SPHERE (ICML 2026), AdamO, SingularClip all involve normalization or spectral control.
- Question: strip every method down to its normalization/spectral component. How much of the plasticity benefit survives?
- Experiment: continual RL on Procgen/MinAtar/DMC; isolate layer norm, hyperspherical norm, Parseval penalty, singular-value clipping. Report a single dependency chart.
- Why it lands: unifying ablations of a crowded area are high-value when clean (SimBa itself was partly this).
- Compute: small nets. One to two weeks.

**6. Deep critics and seed variance.**
- Builds on: SimBa's observation that bigger SimBa nets train more reliably.
- Question: does scaling the critic reduce run-to-run variance, and does that change how many seeds a benchmark needs?
- Experiment: 50–100 seeds per config on gymnax/Craftax (fast on GPU), critic size swept. Report variance vs. size and the implied seed count for a given statistical power.
- Why it lands: methodology papers on RL evaluation (rliable, "deep RL at the edge of the statistical precipice") are well cited. This one adds a scaling axis.
- Compute: tiny envs, many seeds. One week.

## B. Plasticity

**7. Which part of a world-model agent loses plasticity first?**
- Builds on: SPHERE, C-CHAIN (ICML 2025), SingularClip: all study model-free agents.
- Question: in DreamerV3 under task switching, does the encoder, the dynamics model, the reward head, or the actor-critic lose plasticity first? Does fixing only that component recover performance?
- Experiment: continual DMC/Craftax with DreamerV3. Measure eNTK rank and dormant units per component. Apply resets or spectral clipping to one component at a time.
- Why it lands: first component-level account of plasticity in model-based RL.
- Compute: DreamerV3 small config fits a 3080. Two weeks.

**8. A taxonomy of nonstationarity and which plasticity fix works for which.**
- Builds on: the plasticity literature tests on mixed benchmarks; nobody separates the kind of change.
- Question: reward shift, dynamics shift, observation shift, and action-space growth are different. Do existing fixes (resets, SingularClip, AdamO, C-CHAIN, Parseval) help differently for each?
- Experiment: one gridworld/Craftax family where you control the type of change exactly. Run every method on every type. Report a method × type table with clear winners.
- Why it lands: turns "plasticity loss" into a structured problem; gives practitioners a lookup.
- Compute: small. One to two weeks.

**9. Predicting plasticity collapse before it happens.**
- Builds on: SingularClip found singular-value anisotropy grows before plasticity loss.
- Question: can spectral statistics forecast a performance drop N thousand steps ahead well enough to trigger an intervention only when needed?
- Experiment: continual RL runs; log spectral stats; train a tiny forecaster; use it to gate resets or clipping. Compare against always-on and never-on.
- Why it lands: adaptive, cheap, and it answers "when should I intervene," which no current method does.
- Compute: small. One week.

**10. Plasticity under opponent nonstationarity.**
- Builds on: all plasticity work is single-agent.
- Question: in self-play, the opponent is a nonstationary environment. Do self-play agents lose plasticity faster, and does that explain self-play plateaus?
- Experiment: JAX multi-agent envs (Overcooked-JAX, MPE). Measure plasticity metrics over self-play; apply SingularClip/AdamO; see if the plateau moves.
- Why it lands: connects two literatures that have never met, with a practical payoff for self-play.
- Compute: small. One to two weeks.

**11. Redundancy reduction as a plasticity regularizer.**
- Builds on: R2-Dreamer (2026) used a Barlow Twins objective to prevent representation collapse without augmentation.
- Question: representation collapse and plasticity loss both look like rank collapse. Does the redundancy-reduction objective preserve plasticity in continual RL?
- Experiment: add the R2 objective to a model-free agent in continual Procgen/DMC; compare against standard plasticity fixes; measure eNTK rank.
- Why it lands: a cheap new plasticity method with a clear mechanism, borrowed from a different line of work.
- Compute: small. One week.

**12. Automatic update-to-data ratio from a spectral signal.**
- Builds on: SimbaV2 scales UTD without resets; BRO uses UTD 10 with resets; both pick UTD by hand.
- Question: can eNTK rank or churn be used online to raise UTD when the network has capacity to spare and lower it when overfitting begins?
- Experiment: controller on DMC/HumanoidBench; compare fixed UTD 1, 2, 8, 10 vs. adaptive at matched compute.
- Why it lands: removes a hyperparameter that everyone tunes, with a principled signal.
- Compute: DMC. One to two weeks.

## C. World models

**13. When do decoder-free world models fail?**
- Builds on: R2-Dreamer wins on DMC-Subtle (tiny task-relevant objects) but the paper doesn't map the boundary.
- Question: as task-relevant pixels shrink, distractor pixels grow, or the relevant signal becomes temporal rather than spatial, where does reconstruction beat decoder-free, and can a cheap switch pick the right one?
- Experiment: a controlled DMC/Procgen variant with dials on object size, distractor fraction, and temporal cue. Run DreamerV3, DreamerPro, R2-Dreamer. Propose a selector based on reconstruction error variance.
- Why it lands: characterization papers with a decision rule are valued; this one settles a live debate.
- Compute: DreamerV3 small. Two weeks.

**14. Where should optimism live in a world model?**
- Builds on: Optimistic DreamerV3 (2026) reports 152% vs 97% mean HNS on Atari-100k but puts optimism in one place.
- Question: optimism can be added to the reward head, the dynamics, or the value. Which placement gives the exploration benefit, and which just inflates returns?
- Experiment: three variants on sparse Atari-100k games (Private Eye, Freeway, Montezuma subset) and DMC sparse tasks. Report true return and imagined-vs-real return gap.
- Why it lands: a direct, informative follow-up to a strong recent result.
- Compute: 8–10 Atari-100k games, 3 seeds each. Two to three weeks.

**15. Adaptive imagination horizon.**
- Builds on: DreamerV3 uses a fixed 15-step imagination horizon.
- Question: can the horizon be chosen per state from model disagreement or latent uncertainty, so imagination goes far where the model is good and short where it is not?
- Experiment: ensemble-disagreement truncation vs. fixed horizons 5/15/30 on Atari-100k subset and DMC. Measure return and model exploitation.
- Why it lands: simple, principled, broadly applicable.
- Compute: Dreamer small. Two weeks.

**16. An audit of world-model hallucination.**
- Builds on: everyone knows policies exploit model errors; nobody measures it systematically.
- Question: how often do imagined rollouts visit states the real environment can't produce, and does a cheap consistency check (cycle consistency, physics-plausibility classifier) reduce exploitation?
- Experiment: DMC with ground-truth state; classify imagined latents as reachable/unreachable; correlate with policy over-optimism; add a consistency penalty.
- Why it lands: gives the field a metric it lacks and a fix that uses it.
- Compute: DMC. Two weeks.

**17. Domain randomization inside imagination.**
- Builds on: sim-to-real randomizes the simulator; nobody randomizes the learned model.
- Question: perturbing the RSSM's latent dynamics during imagination (parameter noise, dropout on transitions) should produce policies robust to real dynamics shift. Does it, and what does it cost in nominal performance?
- Experiment: train in imagination with randomized latent dynamics; test on DMC with perturbed mass/friction; compare against standard Dreamer and explicit DR in the real sim.
- Why it lands: robustness without touching the simulator is a useful capability with a clear story.
- Compute: DMC. Two weeks.

**18. Search at test time vs. search at train time, compute-matched.**
- Builds on: EfficientZero V2 (search), BBF (model-free scaling), DreamerV3 (imagination) all top Atari-100k by different routes.
- Question: at equal GPU-seconds including inference, does adding a small MCTS only at test time to a trained Dreamer close the gap to EZ-V2?
- Experiment: 8 Atari-100k games, strict compute accounting, Dreamer + test-time search with budgets 0/8/32/128 simulations.
- Why it lands: a fair-comparison paper that reframes the leaderboard around compute.
- Compute: 8 games, 3 seeds. Two to three weeks.

**19. Quasimetric value heads in world models.**
- Builds on: horizon generalization work shows quasimetric (triangle-inequality) value functions plan-invariantly; Dreamer's value head has no such structure.
- Question: does enforcing a quasimetric structure on Dreamer's goal-conditioned value head give horizon generalization in imagination?
- Experiment: goal-conditioned Dreamer on maze/manipulation tasks; train on near goals, test on far ones; quasimetric head vs. standard.
- Why it lands: combines two active 2025–26 lines in a way neither has tried.
- Compute: small Dreamer. Two weeks.

## D. Goal-conditioned RL and horizon generalization

**20. Does depth buy horizon generalization?**
- Builds on: 1000-layer CRL; *Horizon Generalization in RL* (planning invariance).
- Question: the deep CRL networks reached far goals from training on near ones. Is depth causing planning invariance, and can you measure it?
- Experiment: planning-invariance metric (action agreement between goal and waypoint) vs. depth 4 → 512 on point-maze and manipulation.
- Why it lands: a mechanistic explanation for the best paper's headline.
- Compute: CRL is cheap. One to two weeks.

**21. A learned curriculum over relabeling horizon.**
- Builds on: *An Informational Curse of Horizon* (2026): long relabeling horizons reduce action–goal mutual information.
- Question: can an online estimate of that mutual information drive the relabeling horizon so it grows only as fast as the policy can use it?
- Experiment: HER/CRL on manipulation with MI-driven horizon schedule vs. fixed and linear schedules.
- Why it lands: turns a diagnostic finding into a method.
- Compute: small. One to two weeks.

**22. Horizon generalization benchmark with controlled horizon ratio.**
- Builds on: current evidence is scattered across different envs.
- Question: which of CRL, quasimetric RL, HIQL, and GCBC generalize over horizon, measured on a procedural family where train/test horizon ratio is a dial?
- Experiment: procedural mazes and block-stacking with horizon ratio 1 → 16. Report curves per method.
- Why it lands: benchmark papers land when they produce a clear ranking and a surprise.
- Compute: small. Two weeks.

**23. Depth vs. relabeling horizon: do deep nets need fewer relabeled goals?**
- Builds on: 1000-layer CRL and the informational curse.
- Question: if depth buys horizon generalization, deeper nets should tolerate shorter relabeling horizons. Is there a trade-off curve?
- Experiment: depth × relabeling horizon grid on two GCRL tasks.
- Why it lands: a cheap interaction study between two headline results.
- Compute: small. One week.

**24. Jacobian distillation for RL (not just BC) policies.**
- Builds on: the informational curse paper distilled short-horizon input Jacobians into long-horizon BC policies.
- Question: does the same trick help RL-trained goal-conditioned policies, and does it interact with the critic?
- Experiment: CRL/HIQL with and without Jacobian distillation on manipulation.
- Why it lands: direct extension with a clear yes/no.
- Compute: small. One week.

**25. Intra-trajectory negatives: why do they stabilize contrastive RL?**
- Builds on: StableCRL found intra-trajectory negative sampling regularizes CRL.
- Question: is the benefit from harder negatives, from implicit horizon reduction, or from reducing false negatives? Separate them.
- Experiment: controlled negative-sampling variants on BuilderBench-GPU and point-maze.
- Why it lands: explains a working trick; explanations of tricks are publishable when crisp.
- Compute: BuilderBench runs on GPU. One to two weeks.

## E. Offline-to-online fine-tuning

**26. Detecting spurious Q-optimism at fine-tune onset.**
- Builds on: SOAR (withdrawn ICLR 2026) identified spurious Q-optimism as the cause of collapse, leaving the problem open.
- Question: can you measure rank flips (actions the offline policy preferred that Q now ranks below worse actions) at the start of fine-tuning and use the flip rate to gate policy updates?
- Experiment: D4RL/OGBench kitchen and AntMaze; flip-rate gate vs. WSRL, Cal-QL, SUF.
- Why it lands: a diagnostic turned into a mechanism, in a field where the leading fix was withdrawn.
- Compute: MuJoCo. Two weeks.

**27. How many warmup rollouts?**
- Builds on: WSRL (ICLR 2025) seeds online training with "a very small number" of rollouts.
- Question: how does the required warmup size scale with the offline–online distribution gap, measured cheaply?
- Experiment: vary dataset quality and env perturbation; sweep warmup size; fit a rule.
- Why it lands: converts a hyperparameter into a predictable quantity.
- Compute: MuJoCo. One to two weeks.

**28. Fine-tuning when the dynamics changed, not just the distribution.**
- Builds on: offline-to-online benchmarks assume the same environment.
- Question: do unlearning and forgetting behave differently when the online environment has perturbed physics? Which methods survive?
- Experiment: MuJoCo with mass/friction/actuator shifts between offline data and online env; run WSRL, Cal-QL, IQL+online.
- Why it lands: the realistic case nobody benchmarks.
- Compute: MuJoCo. Two weeks.

## F. Policy optimization

**29. When does pathwise beat score-function on-policy?**
- Builds on: REPPO (ICLR 2026): on-policy Q-gradient learning beats PPO on GPU-parallel benchmarks.
- Question: characterize by action dimension, reward smoothness, and contact-richness when pathwise wins. Derive a switching rule.
- Experiment: Mujoco Playground tasks binned by those properties; REPPO vs. PPO at matched compute.
- Why it lands: tells practitioners which estimator to use and why.
- Compute: GPU-parallel. Days to one week.

**30. Value Gradient Flow for discrete actions.**
- Builds on: VGF (ICLR 2026) avoids parameterizing a policy, but is continuous-only.
- Question: can a discrete version (Gumbel relaxation or straight-through) keep the test-time scaling benefits on Atari-100k and gridworlds?
- Experiment: discrete VGF vs. DQN variants and policy-gradient baselines.
- Why it lands: extends a new method to the other half of RL.
- Compute: Atari-100k subset, MinAtar. Two weeks.

**31. Test-time compute scaling for flow policies in RL.**
- Builds on: GoRL (ICML 2026) trains flow/diffusion policies online.
- Question: does using more flow steps at test than at train improve return, and is there a compute-optimal split?
- Experiment: flow steps at train × test grid on DMC/MyoSuite.
- Why it lands: the test-time-compute story is hot; RL control is an untested venue for it.
- Compute: DMC. One to two weeks.

**32. Pathwise gradients through a world model vs. through Q.**
- Builds on: REPPO (through Q) and Dreamer (through the model) are both pathwise.
- Question: same tasks, same compute: which bias hurts more, learned Q or learned dynamics? When does each win?
- Experiment: controlled comparison on DMC with ground-truth dynamics available for bias measurement.
- Why it lands: clean methodological comparison with a measurable quantity.
- Compute: DMC. Two weeks.

**33. Should the actor see the critic's uncertainty?**
- Builds on: distributional critics in SimbaV2 and BRO improve stability, but the actor only sees the mean.
- Question: does feeding the critic's return distribution (quantiles) into the actor's objective improve exploration or risk-aware behavior at no extra cost?
- Experiment: SAC/BRO with quantile-aware actor loss on DMC and a few risk-sensitive tasks.
- Why it lands: small change, clear hypothesis, cheap.
- Compute: DMC. One week.

## G. Small-model RL for language and reasoning

All of these use 0.5B–1.5B models with LoRA and GRPO, short contexts, procedural data. That fits a 3080.

**34. Dense-to-sparse reward curricula for sub-1B RLVR.**
- Builds on: *RLVR for Small Search Agents* (2026): dense F1 reward beats exact-match even on the exact-match metric.
- Question: does training dense-first then switching to sparse recover the exact-match ceiling while keeping the robustness?
- Experiment: Qwen-0.5B/0.8B on MuSiQue-style QA with three schedules × three seeds.
- Why it lands: the anchor paper explicitly calls for a small-model reward-design study.
- Compute: 0.8B LoRA GRPO fits. One to two weeks.

**35. A difficulty-mixing law for low-data RLVR.**
- Builds on: *Learning from Less* (MLSys 2026): mixed-difficulty data gives 5× sample efficiency; the authors call for scaling laws.
- Question: what fraction of hard problems maximizes generalization as a function of budget, across three procedural domains?
- Experiment: procedural generators for counting, graph, and spatial reasoning; mixing fraction × budget grid; fit a law; test on a fourth domain.
- Why it lands: directly answers the anchor paper's stated future work.
- Compute: 0.5–1.5B LoRA. Two weeks.

**36. Adaptive group size in GRPO from within-group reward variance.**
- Builds on: the small-search-agent paper proposes that reward spread within a GRPO group explains the sparse-reward deficit.
- Question: if you grow the group when variance is low and shrink it when high, do you get sparse-reward learning at dense-reward cost?
- Experiment: adaptive vs. fixed group sizes on procedural math and QA.
- Why it lands: cheap method with a mechanism, testable on small models.
- Compute: 0.5B. One week.

**37. Which self-verifiable pretext games are hackable?**
- Builds on: RLSVR / SpyRL (2026) turns open-ended tasks into verifiable games; the paper flags verifier–policy collusion as a risk.
- Question: propose two alternative pretext games (prompt reconstruction, paired-consistency) and measure how fast each one is gamed versus how well it tracks held-out quality.
- Experiment: small models, summarization and creative writing, with human-rated held-out quality on a small set.
- Why it lands: the open risk named in the anchor paper, measured.
- Compute: 0.5–1B. Two weeks.

**38. Does RLVR on procedural tasks transfer to natural-language phrasings of the same tasks?**
- Builds on: *Learning from Less* trains and tests on procedural data.
- Question: if a model learns graph reasoning on symbolic inputs, does it get better at the same problems stated in words, and vice versa?
- Experiment: paired symbolic/verbal versions of three procedural domains; cross-evaluate.
- Why it lands: tells us whether cheap procedural RLVR buys real reasoning or format fitting.
- Compute: 0.5–1.5B. One to two weeks.

**39. Token-level credit assignment with ground-truth intermediate states.**
- Builds on: GRPO assigns one advantage per sequence; procedural tasks give you correct intermediate states for free.
- Question: using known intermediate correctness to build a step-level advantage, how much does learning speed up, and does the gain persist when you remove the oracle at test?
- Experiment: counting/graph tasks where every intermediate step is checkable; compare sequence-level vs. step-level advantages.
- Why it lands: a controlled credit-assignment study that LLM-RL lacks, because real tasks don't have step labels.
- Compute: 0.5B. One week.

**40. Does RL create new behaviors in interactive environments, or only sharpen them?**
- Builds on: *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?* (NeurIPS 2025 runner-up) found RLVR narrows the pass@k boundary on static problems.
- Question: in an interactive text environment (BabyAI-text, TextWorld), where exploration matters, does RL produce action sequences the base model never samples at any k?
- Experiment: base vs. RL-trained 0.5–1B models; measure pass@k and trajectory coverage at k up to 256.
- Why it lands: the most direct test of whether the runner-up's finding holds outside static reasoning.
- Compute: text envs are cheap. Two weeks.

**41. Automatic difficulty control for procedural RLVR.**
- Builds on: *Learning from Less*; curriculum results in low-data regimes.
- Question: a controller that keeps the model's success rate near 50% by adjusting the generator's difficulty. Does it beat fixed mixtures at equal budget?
- Experiment: three procedural domains, controller vs. best fixed mixture from idea 35.
- Why it lands: a practical method with the anchor paper's generators already in hand.
- Compute: 0.5–1B. One week.

## H. Memory and partial observability

**42. A few latch units make state-space models generalize over horizon.**
- Builds on: *Multistability for Horizon Generalization* (2026): SSMs and gated linear RNNs are monostable and fail; multistability plus transient dynamics are needed.
- Question: can you add a small number of explicitly bistable units to an SSM and recover horizon generalization while keeping parallel training?
- Experiment: POMDP memory tasks (T-maze, passive visual match) with horizon 10 → 1000; SSM vs. SSM+latches vs. LSTM.
- Why it lands: the anchor paper names this as the key direction.
- Compute: tiny. Days.

**43. Memory in the world model vs. memory in the policy.**
- Builds on: Dreamer carries memory in the RSSM; model-free agents carry it in the policy's RNN.
- Question: which generalizes over temporal horizon, and does the answer depend on whether the memory is needed for prediction or for control?
- Experiment: POMDP tasks where the remembered cue affects dynamics vs. only reward.
- Why it lands: a clean structural question with a cheap testbed.
- Compute: tiny. One week.

## I. Multi-agent

**44. Zero-shot coordination via shared world models.**
- Builds on: world models are single-agent; zero-shot coordination work is model-free.
- Question: if two agents each learn a world model that includes the partner, do they coordinate with novel partners better than model-free baselines?
- Experiment: Overcooked-JAX cross-play with held-out partners.
- Why it lands: new tool for a benchmark with a known, hard metric.
- Compute: JAX envs. Two weeks.

**45. Opponent-induced plasticity loss and population size.**
- Builds on: idea 10; population-based training.
- Question: does a larger population (more diverse opponents) slow plasticity loss, and is there a population size that is compute-optimal for that reason?
- Experiment: population size × plasticity metrics on MPE/Overcooked.
- Why it lands: gives a mechanistic reason for a known empirical practice.
- Compute: small. One to two weeks.

## J. Evaluation and methodology

**46. How many seeds does Atari-100k need now?**
- Builds on: rliable; the leaderboard now has many methods within a few points of each other.
- Question: with 20+ seeds on a subset of games, what is the actual confidence interval on EZ-V2 vs. BBF vs. DreamerV3, and are the published gaps significant?
- Experiment: 6–8 games, 20 seeds each, three open implementations.
- Why it lands: methodology papers that puncture a leaderboard get attention, if the statistics are rigorous.
- Compute: significant but bounded; pick games carefully. Three weeks.

**47. Wall-clock-matched leaderboard for sample-efficient RL.**
- Builds on: Atari-100k matches env steps, not compute. EZ-V2's search is far more expensive than BBF.
- Question: at matched GPU-seconds, how does the ranking change?
- Experiment: same as 46 with strict compute accounting; publish both rankings.
- Why it lands: reframes a benchmark everyone uses.
- Compute: shares runs with 46.

**48. What does "sample efficiency" mean when env steps are free?**
- Builds on: Mujoco Playground and gymnax make millions of steps cost seconds.
- Question: on GPU-native envs, does the sample-efficient algorithm (high UTD, big critic) still win on wall-clock, or does PPO-style throughput win?
- Experiment: REPPO, PPO, SimbaV2-SAC, BRO at matched wall-clock on Playground.
- Why it lands: a question the GPU-env shift forces; REPPO raised it without settling it.
- Compute: GPU-parallel. One week.

## K. Cross-cutting

**49. Spectral health as a universal early-stopping signal for RL.**
- Builds on: SingularClip, AdamO, SPHERE all use spectral quantities as regularizers.
- Question: without regularizing, can eNTK rank or singular-value spread alone tell you when to stop increasing UTD, when to reset, and when a run has silently died? One signal, three uses.
- Experiment: log the signal across SAC/PPO/Dreamer on DMC and Atari-100k; test each use against hand-tuned baselines.
- Why it lands: a practical diagnostic with a theory story, across algorithms.
- Compute: piggybacks on other runs. One week extra.

**50. Depth, normalization, and plasticity in one controlled sweep.**
- Builds on: 1000-layer RL (depth), SimbaV2 (normalization), SPHERE/SingularClip (plasticity).
- Question: these three results come from three communities. In one continual-RL sweep over depth × normalization × task sequence, do deep normalized nets lose plasticity faster or slower than shallow ones?
- Experiment: continual DMC/Craftax, depth 2 → 128, with and without hyperspherical norm. One heatmap.
- Why it lands: the three biggest architecture results of the last year have never been tested together.
- Compute: small nets, many runs. Two weeks.

---

## How to pick

Rank by (a) how clean the yes/no is, (b) how directly a recent paper asked for it, and (c) how little infrastructure you need. By that ranking, the strongest first picks are **1, 4, 20, 26, 29, 34, 35, 40, 42, 50**. Each has an anchor paper that practically wrote the question for you, a controlled experiment, and a result either way.

Before committing to any one: search arXiv for the last six months on the anchor paper's title plus your variable. If someone has done it, pick the next one.
