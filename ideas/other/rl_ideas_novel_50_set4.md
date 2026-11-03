# 50 more novel RL research ideas (set 4)

A fifth list, disjoint from `rl_ideas_50.md` (follow-ups to recent papers), `rl_ideas_novel_50.md` (mechanisms and measurements), `rl_ideas_novel_50_set2.md` (hidden hyperparameters, data, control, LLM RL), `rl_ideas_novel_50_set3.md` (optimizer defaults, representation, safety, evaluation budgets), and the ADEPT stubs (dead-ends, irreversibility, empowerment, backtracking, pretraining transfer, action quantization, offline data amount, single-network uncertainty, untrusted graders, delayed rewards, environment design, borrowing experience, RLVR rigidity, distillation). Where an idea sits next to an earlier one, the Gap line names it and says how this differs.

Same caveats. Novelty is as of my knowledge; run a literature search on each before committing, and I flag the ones most at risk at the end. "NeurIPS candidate" means the paper has a mechanism or a measurement, a falsifiable claim, and a result that is informative either way.

Everything fits one RTX 3080 (10 GB). Control ideas use gymnax, MinAtar, Craftax, DMC, MyoSuite, Mujoco Playground / MJX, Brax, or JaxMARL. LLM ideas use 0.5–1.5B models with LoRA or soft prompts and GRPO-style training on procedural data with short contexts.

Format for each: **Hypothesis**, **Gap**, **Experiment**, **Kill test**, **Compute**.

---

## A. Value learning and optimization

**1. Does TD learning sit at an edge of stability?**
- Hypothesis: in supervised learning with gradient descent, the loss Hessian's top eigenvalue rises until it reaches 2/η and then hovers there. For a bootstrapped loss the relevant operator is the Jacobian of the full update map (prediction gradient plus the bootstrap term through the target network), and its sharpness also equilibrates, but at an edge lower than 2/η that moves with τ and γ. Divergence events are excursions past that edge, and hyperspherical normalization works partly by capping sharpness below it.
- Gap: edge-of-stability is characterized for supervised learning; sharpness appears in RL only as a plasticity statistic. Whether the equilibrium exists for bootstrapped losses and where it sits has not been tested.
- Experiment: DQN on MinAtar and SAC on DMC; power iteration on the TD-loss Hessian and on the update-map Jacobian (JVPs) every few hundred steps; sweep η, τ, γ; relate sharpness excursions to divergence; repeat with SimbaV2-style critics.
- Kill test: sharpness does not equilibrate, or equilibrates at 2/η regardless of τ and γ.
- Compute: small nets, cheap power iteration. One to two weeks.

**2. Anticipatory targets: compute TD targets with the actor you will have, not the one you have.**
- Hypothesis: by the time the critic has fit targets computed under the current actor, the actor has moved. Extrapolating the actor's parameters one step along its recent update direction (θ + κ·Δθ) and using the extrapolated actor to compute targets reduces policy-evaluation lag, with the largest gain at high actor learning rate and low UTD. This is the extragradient idea from game optimization applied inside the Bellman target.
- Gap: set 2 idea 8 measures the lag; this is a method to remove it. Extragradient and optimistic updates are used for GANs and self-play, never for the actor inside a TD target.
- Experiment: SAC and TD3 on DMC and Playground; κ ∈ {0, 0.5, 1, 2}; measure lag with the freeze-and-refit protocol, return, and stability.
- Kill test: κ > 0 never helps or destabilizes.
- Compute: GPU-parallel. One week.

**3. Revealed time preference: what discount does a trained agent actually use?**
- Hypothesis: the discount an agent expresses in its choices differs from the γ it was trained with. Build choice gadgets (left gives r₁ at delay d₁, right gives r₂ at delay d₂), fit the implied discount function from the agent's choices, and find that it is steeper than γ, that the gap grows with bootstrap error, and that it is hyperbolic in shape (near rewards discounted more steeply than exponential, far rewards less) because n-step returns treat near and far rewards differently.
- Gap: set 3 idea 9 measures how far a reward's influence reaches into Q. This measures behavior by the economist's method, revealed preference, which is what matters at deployment, and tests whether hyperbolic discounting emerges rather than imposing it (Fedus et al. 2019).
- Experiment: gridworlds and a DMC task with a delay-choice gadget; DQN and SAC at γ ∈ {0.9, 0.99, 0.999}, n ∈ {1, 3, 10}; fit exponential and hyperbolic forms to choices; relate the gap to critic error.
- Kill test: implied discount equals γ and is exponential.
- Compute: tiny. One week.

**4. Frequency response of the learner.**
- Hypothesis: modulate the reward scale sinusoidally over training time, r ← (1 + A sin ωt)·r with small A, and measure the amplitude and phase of the induced oscillation in mean Q and in return. The learning loop behaves as a filter whose cutoff and phase lag are set by lr, τ, UTD, and buffer size. Resonance peaks (gain > 1 at some ω) indicate under-damped dynamics and predict instability under natural nonstationarity (curricula, reward edits, moving opponents). The Bode plot is a compact signature of an algorithm's tracking ability.
- Gap: control engineers characterize systems this way; RL algorithms have never been system-identified. Set 2 idea 18 applies one step edit; set 3 idea 9 injects impulses along the trajectory. This is a frequency sweep over training time.
- Experiment: SAC, DQN, PPO on DMC and MinAtar; sweep ω; measure gain and phase; verify that the fitted filter predicts the step response to an actual reward edit; test whether the τ that flattens the resonance improves performance under nonstationarity.
- Kill test: no resonance, and gain and phase do not predict the step response.
- Compute: small. One to two weeks.

**5. Where the regularizer enters: target or actor.**
- Hypothesis: entropy (SAC) and pessimism (clipped double-Q) each appear in two places, the bootstrap target and the actor's objective. A full 2 × 2 × 2 factorial separates them: target-side entries govern critic stability and the value scale; actor-side entries govern exploration and policy conservatism. Entropy in the actor only, with pessimism in the target only, matches full SAC and removes the coupling between α-tuning and Q-scale.
- Gap: pieces exist (TD3 has pessimism only in the target; some SAC variants drop entropy from the target), but no paper runs the factorial and attributes effects. Flagged as at-risk below.
- Experiment: DMC and Playground; the eight configurations; return, Q bias against Monte Carlo, entropy trajectory, α dynamics.
- Kill test: all placements equivalent, or full SAC strictly dominates.
- Compute: GPU-parallel. Days to one week.

**6. A mechanistic decomposition of frame skip.**
- Hypothesis: action repeat k helps through four separable channels: temporally correlated exploration (the noise is held for k steps), horizon shortening (γ^k per decision), fewer decisions per return (variance), and fewer updates per environment step at fixed UTD-by-decision. Each can be isolated: hold the noise but update the mean every step; decide every step but use γ^k with k-step returns; and so on. On locomotion the correlated-noise channel dominates; on Atari-like tasks horizon shortening does; the optimal k is predictable from the task's Lyapunov time and reward density.
- Gap: frame skip is universal and tuned by sweep. FiGAR and learned repeat exist; nothing explains which channel the benefit comes from.
- Experiment: DMC with repeat 1 to 8, MinAtar, Playground; four controlled variants; attribute the gain.
- Kill test: the channels are not separable or contribute equally everywhere.
- Compute: GPU-parallel. One week.

**7. Range compression in critics.**
- Hypothesis: regressing on noisy bootstrapped targets shrinks predictions toward the mean: the calibration slope of Q against Monte Carlo return is below 1 and falls with target noise and regularization. Compression makes the actor undervalue rare high-return actions relative to common medium ones, and the amount of compression predicts the fraction of seeds that reach high-return modes. Post-hoc linear or isotonic recalibration of targets removes it.
- Gap: overestimation bias (the intercept) is studied exhaustively; the slope (regression dilution) has not been measured in deep RL or linked to exploitation of rare good outcomes.
- Experiment: DMC and Craftax; Monte Carlo ground truth by rollouts; calibration slope per checkpoint; relate to mode reach across seeds; test recalibration.
- Kill test: slope near 1, or uncorrelated with mode reach.
- Compute: small. One week.

**8. The fresh-critic gap as a reset trigger.**
- Hypothesis: periodically train a fresh critic from scratch on the current buffer at fixed compute and compare it to the running critic on held-out transitions and on the actor gradient direction. The gap directly measures accumulated bias (primacy, stale targets, lost plasticity). Resets triggered when the gap is large beat scheduled resets at matched compute, and the gap's trajectory differs across algorithms in a way that explains which ones benefit from resets.
- Gap: resets run on a schedule and plasticity metrics are proxies. The refit oracle is cheap on small nets and has not been used. List 1 idea 33 cross-fits for validation error; this measures bias relative to a refit, a different quantity. Flagged as at-risk below.
- Experiment: SAC and DQN at UTD ∈ {1, 8}; refit every N steps; gap trajectory; gated vs scheduled resets.
- Kill test: the gap is flat or uncorrelated with reset benefit.
- Compute: small. One to two weeks.

**9. Dimensionless hyperparameters.**
- Hypothesis: RL hyperparameters carry units. Learning rate scales with reward magnitude, γ with the time step, batch size with episode length, the entropy coefficient with reward scale, the PPO clip with action dimension. Forming dimensionless groups (Buckingham-π style) and expressing optima in them makes hyperparameters transfer across tasks; raw values do not. A table of dimensionless defaults replaces per-suite tuning.
- Gap: Pop-Art handles reward scale, list 1 idea 9 handles γ–lr, set 2 idea 6 handles width. The systematic dimensional analysis across all axes with a transfer test has not been done. Flagged as at-risk below.
- Experiment: task families made by rescaling (reward × c, dt × c, episode length × c, redundant actuators for action dimension); find optima per task; test invariance of dimensionless optima; validate across Playground and DMC.
- Kill test: dimensionless optima transfer no better than raw ones.
- Compute: many small runs. Two weeks.

**10. Advantage sign agreement across checkpoints as free uncertainty.**
- Hypothesis: the sign of A(s, a) = Q(s, a) − Q(s, π(s)) for buffer transitions flips across consecutive critic snapshots often; the flip rate is a per-sample uncertainty estimate with no ensemble. Weighting updates by cross-snapshot agreement matches ensemble-based stabilization at zero extra forward cost, and the flip map localizes where the critic is unreliable.
- Gap: uncertainty comes from ensembles or bootstrapped heads; temporal self-consistency of the critic has not been used. Set 3 idea 8 maps which errors matter; this is a cheap detector of them.
- Experiment: SAC and TD3 on DMC; keep K recent critic snapshots; flip rate vs true error against Monte Carlo; agreement-weighted updates vs REDQ-style ensembles at matched compute.
- Kill test: flip rate is uncorrelated with true error.
- Compute: small. One week.

## B. Exploration

**11. Budget-aware exploration.**
- Hypothesis: the right explore–exploit trade-off depends on how much training remains. Giving the agent the remaining budget fraction as an input, with the entropy or bonus coefficient a learned function of it, yields agents that anneal exploration themselves and beat the best hand-tuned schedule across budgets. The learned schedule's shape is task-dependent: front-loaded on sparse tasks, flat on dense ones.
- Gap: finite-horizon bandit theory makes exploration budget-dependent; deep RL anneals by hand or with budget-blind auto-tuning. Conditioning on remaining budget has not been tried.
- Experiment: MinAtar, Craftax, sparse DMC; budgets {0.5M, 2M, 8M}; budget-conditioned vs schedules tuned per budget vs auto-α.
- Kill test: budget conditioning never beats a per-budget tuned schedule.
- Compute: small. One to two weeks.

**12. Reward-epistemic exploration.**
- Hypothesis: state-novelty bonuses keep paying for regions the agent has already confirmed are reward-free. A bonus from the epistemic uncertainty of a learned reward model (ensemble disagreement over "is there reward here") stops paying in confirmed-empty regions and directs the agent toward unvisited states in proportion to how reward-plausible they look, beating RND on tasks with large reward-free regions and matching it elsewhere.
- Gap: reward uncertainty drives exploration in preference-learning settings; in standard sparse RL it has not been compared to state novelty. Set 2 idea 11 gates by value error, which conflates reward and dynamics uncertainty.
- Experiment: MinAtar, Craftax, maze families with controllable reward-free fraction; reward-ensemble bonus vs RND vs counts; time to first reward and bonus spend by region.
- Kill test: no advantage over RND at any reward-free fraction.
- Compute: small. One week.

**13. The recovery paradox.**
- Hypothesis: shaped penalties on near-failure states (tilt, joint limits, low height) reduce the agent's exposure to recovery situations during training. The resulting policies have less recovery competence and are less robust to pushes and dynamics shift at deployment than policies trained with terminal-failure penalties only, despite higher training return. Robustness tracks time-spent-near-failure during training.
- Gap: shaping is evaluated on nominal return; its effect on recovery skill has not been isolated. Set 3 idea 28 is about external shields; here the shaping itself removes the experience.
- Experiment: Playground quadruped and humanoid; penalty-strength sweep; log near-failure exposure; evaluate under pushes and friction shift; recovery success rate.
- Kill test: robustness is independent of near-failure exposure.
- Compute: GPU-parallel. One week.

**14. Decide only when it matters.**
- Hypothesis: use the critic's action sensitivity ‖∂Q/∂a‖ at the current state to decide whether to compute a new action or repeat the last one. In insensitive states repetition gives temporally extended exploration and fewer wasted decisions; in sensitive states the agent acts every step. This adaptive repeat beats fixed repeat and learned-repeat heads, and the decision states it identifies coincide with bottlenecks.
- Gap: list 1 idea 5 measures action sensitivity; learned action repeat trains a separate head on return. The critic's gradient as a free decision-timing signal is new.
- Experiment: SAC on DMC, Playground, MyoSuite; thresholded repeat; compare to k ∈ {1, 2, 4} and a learned-repeat head.
- Kill test: no gain over the best fixed k.
- Compute: GPU-parallel. One week.

## C. Data and offline RL

**15. The junk tolerance curve.**
- Hypothesis: diluting a replay buffer with transitions from other tasks, other embodiments, or random play degrades learning along a curve whose shape reveals the mechanism: a flat region where the critic identifies and ignores junk, then a cliff when junk saturates capacity. The cliff moves right with network width, and junk that shares observation statistics with the task is far more harmful than obviously different junk.
- Gap: the ADEPT borrowing stub asks which unit of experience transfers; this asks how much irrelevant data a learner can absorb, the denominator of any shared-buffer scheme, and no dilution curve exists.
- Experiment: DMC and Playground; dilution fraction ∈ {0, 10, 50, 90, 99}% with three junk types; width sweep; return and critic error on task data.
- Kill test: degradation is linear in dilution with no flat region.
- Compute: small. One week.

**16. State coverage versus action coverage in offline RL.**
- Hypothesis: "coverage" conflates the set of states visited and the diversity of actions at those states, and methods fail differently along the two axes: BC-regularized methods need action diversity at visited states; in-sample methods need state coverage along the optimal path; model-based methods need both. A 2D coverage grid reorders methods, and two dataset numbers choose the method.
- Gap: set 2 idea 16 is mode count; the ADEPT stubs vary dataset size. Coverage by axis has not been constructed or linked to method choice.
- Experiment: datasets on MuJoCo and mazes with independently controlled state coverage and per-state action entropy; five methods; rank per cell.
- Kill test: ranking is constant across the grid.
- Compute: MuJoCo offline. Two weeks.

**17. Offline RL with the behavior policy known.**
- Hypothesis: offline methods implicitly estimate π_β through BC terms, in-sample constraints, or density models. When you generated the data you know π_β exactly. Giving methods the true π_β bounds what is lost to estimation; the gap is large for narrow data and small for mixtures, which says where effort should go (behavior modeling vs value learning).
- Gap: benchmarks hide π_β; the oracle comparison has never been run.
- Experiment: D4RL-style datasets generated with known π_β; TD3+BC, IQL, AWAC, CQL with estimated vs oracle π_β; gap by dataset type.
- Kill test: oracle π_β gives no gain.
- Compute: MuJoCo offline. One to two weeks.

## D. Policy parameterization and control

**18. The PPO clip should scale with action dimension.**
- Hypothesis: PPO's ratio is a product over independent action dimensions, so for a fixed per-dimension change its deviation from 1 grows with √d. A fixed ε therefore enforces a trust region that tightens with dimension, which is one reason PPO underperforms on high-DoF control. Scaling ε with √d, or clipping per-dimension ratios, restores performance; the same holds for the KL coefficient.
- Gap: PPO tuning sweeps ε per suite; the dimension dependence is folklore without a controlled test. Flagged as at-risk below.
- Experiment: Playground tasks spanning 1 to 60 action dimensions plus synthetic redundant-actuator variants; ε sweep per d; test √d scaling and per-dimension clipping.
- Kill test: optimal ε is independent of d.
- Compute: GPU-parallel. Days.

**19. Action parameterization sets the color of exploration.**
- Hypothesis: whether the policy outputs absolute actions, deltas (integrated), or low-pass-filtered actions changes two things at once: the spectral color of exploration noise (white, brown, band-limited) and the smoothness of deployable behavior. The winner is predicted by the ratio of plant bandwidth to control frequency: when control is much faster than the plant, integrated or filtered actions win because noise above the plant's bandwidth is wasted exploration. Pink-noise exploration is a special case.
- Gap: delta actions are manipulation folklore, pink noise is studied as noise, filtering as a smoothness regularizer. Nobody has unified them as one design choice or given a predictor.
- Experiment: DMC, Playground, MyoSuite at varied control frequencies; three parameterizations; return, action spectrum, coverage; relate to measured plant bandwidth.
- Kill test: parameterization effects are unrelated to the bandwidth ratio.
- Compute: GPU-parallel. One week.

**20. A developmental curriculum over degrees of freedom.**
- Hypothesis: infants learn motor control by progressively freeing degrees of freedom. Training high-DoF policies with distal joints initially locked (held by PD) and released on a schedule reaches higher final performance at matched compute than training all DoFs at once, because early exploration in a low-dimensional space finds reward faster. The best unlock order follows the kinematic chain from proximal to distal.
- Gap: progressive action spaces exist for discrete actions; the continuous high-DoF case with a developmental ordering has not been tested. List 1 idea 6 balances per-dimension gradients, a different fix for the same symptom.
- Experiment: MyoSuite hand and Playground humanoid; schedules (proximal-to-distal, random, none) at matched compute.
- Kill test: no schedule beats all-at-once.
- Compute: GPU-parallel. One to two weeks.

**21. The basin of attraction shrinks late in training.**
- Hypothesis: define a policy's basin as the set of initial states from which it reaches the goal. Over training the basin first grows (generalization) then shrinks (specialization to ρ₀) even as return on ρ₀ keeps rising. The shrink is predicted by actor entropy collapse and critic overconfidence outside the visited set. Return on ρ₀ hides this; basin volume is cheap to estimate in GPU simulators and predicts robustness.
- Gap: set 3 idea 45 evaluates offline methods once from out-of-coverage starts; this tracks the basin as a training-time trajectory and claims non-monotonicity.
- Experiment: Playground and DMC; sample a wide initial-state distribution; estimate basin volume per checkpoint; relate to entropy and to perturbation robustness.
- Kill test: basin volume is monotone in training.
- Compute: GPU-parallel. One week.

**22. Is hierarchy just long credit?**
- Hypothesis: much of the long-horizon advantage of hierarchical RL comes from the high-level controller receiving k-step returns, not from the abstraction itself. A flat agent with n-step returns of length k and action repeat (or correlated noise) over k steps closes most of the gap to HIRO-, HAC-, and option-critic-style methods on standard long-horizon benchmarks; the residual gap is what hierarchy actually buys.
- Gap: HRL papers compare against flat agents with n = 1 or 3; matched-credit-length flat baselines have not been run systematically.
- Experiment: AntMaze-style tasks and Craftax; HRL vs flat with matched n and repeat; decompose the gap.
- Kill test: flat plus n-step closes nothing.
- Compute: small to medium. Two weeks.

**23. How fast does a policy identify its own latency?**
- Hypothesis: recurrent policies trained with randomized observation and action latency implicitly infer the current latency within the first few steps of an episode (a linear probe recovers it), and the identification time, not the latency magnitude, determines performance loss. Tasks whose Lyapunov time is shorter than the identification time are where latency randomization fails, and an explicit latency-estimation auxiliary shortens it.
- Gap: RMA-style methods estimate extrinsics explicitly; whether plain recurrent policies identify latency, how fast, and whether identification speed is the bottleneck has not been measured.
- Experiment: DMC and Playground with latency ∈ {0..8} steps randomized per episode; probe hidden states; identification time vs return loss vs Lyapunov time.
- Kill test: probes cannot recover latency, or identification time does not predict loss.
- Compute: GPU-parallel. One week.

## E. Model-based RL

**24. Model-tolerant actors.**
- Hypothesis: return error from imagination is roughly model error times the policy's sensitivity to state. Penalizing the actor's state-Lipschitz constant during imagined rollouts (not the critic's) makes the policy tolerant to model error: at a fixed model, smoother actors have a smaller imagined-vs-real return gap and higher real return. The trade-off against expressiveness is favorable on locomotion and unfavorable on contact-rich manipulation.
- Gap: smoothness regularizers target action smoothness for sim-to-real; their role as model-error tolerance inside model-based RL has not been proposed or measured.
- Experiment: DreamerV3 small and TD-MPC2 on DMC; penalty sweep; imagined-vs-real gap and real return.
- Kill test: smoothness does not reduce the gap, or costs more than it saves.
- Compute: Dreamer small. Two weeks.

**25. Where to imagine from.**
- Hypothesis: Dreamer-style agents start imagined rollouts from uniformly replayed states, most of which are states where the critic is already accurate. Prioritizing start states by critic disagreement with real returns (or model-uncertainty-adjusted TD error) concentrates imagination where it changes the policy; the optimal prioritization exponent falls as the model improves.
- Gap: prioritized replay exists for real transitions; prioritizing where to spend imagination has barely been explored.
- Experiment: DreamerV3 small on DMC and an Atari-100k subset; uniform vs prioritized start states; return and where imagination is spent.
- Kill test: no gain.
- Compute: Dreamer small. Two weeks.

## F. Multi-agent

**26. Asymmetric capacity in cooperative teams.**
- Hypothesis: at fixed total parameters, teams with one large agent and several small ones outperform equal splits on tasks with a planning bottleneck and underperform on symmetric tasks. The benefit appears without a communication channel because small agents learn to follow the large agent's observable behavior. Capacity allocation is a hidden design variable in MARL.
- Gap: MARL studies parameter sharing and team size; heterogeneous capacity at matched total compute has not been studied.
- Experiment: JaxMARL Overcooked, SMAX, MPE; capacity splits at fixed total; return and leader–follower structure (Granger causality between agents' actions).
- Kill test: equal split is always best.
- Compute: small. One to two weeks.

**27. Partner identification time predicts zero-shot coordination.**
- Hypothesis: in zero-shot coordination the agent must infer the partner's convention from behavior. The number of steps a trained agent needs to identify partner type (probe on hidden state) predicts cross-play return better than population diversity does, and a partner-identification auxiliary loss shortens it and improves ZSC.
- Gap: ZSC methods are evaluated on return; the inference dynamics inside the agent have not been measured.
- Experiment: JaxMARL Overcooked and small Hanabi; partner populations; probe identification over steps; correlate with cross-play; test the auxiliary.
- Kill test: identification time does not predict cross-play.
- Compute: small. Two weeks.

**28. A predictability bonus for coordination.**
- Hypothesis: rewarding each agent for actions its teammate's model of it predicts (the opposite of an exploration bonus) drives agents toward the most obvious strategy, a learned analog of other-play's symmetry breaking that needs no hand-specified symmetries, and the resulting conventions transfer to novel partners trained the same way.
- Gap: legibility is studied in human–robot motion; as a trained bonus for ZSC with cross-play evaluation it has not been tested. Flagged as at-risk below.
- Experiment: JaxMARL; predictability bonus vs other-play vs population training; cross-play with held-out seeds and methods.
- Kill test: no cross-play gain.
- Compute: small. One to two weeks.

## G. RL for language models

**29. Prefix-only GRPO: how much of RLVR fits in a soft prompt?**
- Hypothesis: training only a soft prefix (20 to 200 virtual tokens) with GRPO, all weights frozen, recovers most of LoRA-GRPO's gain on easy tasks and little on hard ones. The recoverable fraction is a direct measurement of elicitation versus learning per task, and learned prefixes transfer across model sizes within a family when the gain was elicitation.
- Gap: the sharpening-versus-creating debate uses pass@k; prefix capacity is a different and cheaper handle that has not been used.
- Experiment: 0.5–1.5B; prefix lengths × tasks; compare to LoRA ranks and full fine-tuning; transfer prefixes across sizes.
- Kill test: prefix recovers little on every task.
- Compute: 0.5–1.5B. One to two weeks.

**30. Tool-induced skill atrophy.**
- Hypothesis: when RL gives the model a calculator or code tool, it offloads; the model's no-tool accuracy on the offloaded operations declines over training, concentrated exactly on those operations. A small fraction of tool-disabled rollouts during training prevents the decline at no cost to tool-enabled accuracy.
- Gap: tool-use RL reports tool-enabled accuracy; the effect on internal skills has not been measured.
- Experiment: 0.5–1B on arithmetic and graph tasks with a tool action; evaluate with the tool disabled across training; tool-dropout.
- Kill test: no decline without the tool.
- Compute: 0.5–1B. One week.

**31. How reversible is RLVR?**
- Hypothesis: a small SFT on the base model's own samples undoes most of RLVR's accuracy gain quickly, but a residual persists, concentrated in specific layers or heads, and the residual is the part that transfers across domains. Reversibility separates mode or format learning (reversible) from capability (irreversible).
- Gap: forgetting of RL under SFT is anecdotal; a controlled reversion protocol with residual analysis is new. The ADEPT rigidity stub asks about plasticity after RLVR, not whether RL can be undone.
- Experiment: 0.5–1.5B; RLVR on three procedural domains; revert with k base samples; accuracy curves; layerwise diff of the residual; transfer test.
- Kill test: reversion is total, or does not happen.
- Compute: 0.5–1.5B. One to two weeks.

**32. Counterfactual token importance as the credit signal.**
- Hypothesis: deleting or masking a reasoning span and re-sampling the answer reveals which tokens are load-bearing. Weighting token-level advantages by this counterfactual importance speeds learning and reduces length inflation relative to uniform credit, and the load-bearing fraction of reasoning tokens falls over training as the model learns to pad. At 0.5B the counterfactuals are cheap because answers are short.
- Gap: token-level credit uses process reward models or entropy heuristics; direct intervention as the credit signal has not been tried.
- Experiment: procedural tasks; measure load-bearing fraction; importance-weighted GRPO vs standard vs entropy-weighted; accuracy and length.
- Kill test: nearly all tokens are load-bearing, or weighting does not help.
- Compute: 0.5–1B. Two weeks.

**33. The information content of feedback in code RL.**
- Hypothesis: in multi-turn code RL, ablating the feedback to {binary pass/fail, error type only, full traceback, traceback with line numbers} at matched budget shows that traceback content only helps models above a size threshold; below it the model cannot use the text and binary feedback suffices. The threshold is a measurable property of the feedback–model pair.
- Gap: agentic code RL pipelines assume rich feedback helps; the information–benefit curve has not been measured.
- Experiment: 0.5–1.5B on small procedural coding tasks; feedback levels; accuracy and error-recovery rate; size sweep.
- Kill test: feedback content never matters.
- Compute: 0.5–1.5B. Two weeks.

**34. Solver–verifier asymmetry under RL.**
- Hypothesis: RLVR on solving improves the model's ability to judge solutions less than it improves solving, while RL on verifying (reward for correct judgments) transfers partially back to solving. The 2 × 2 transfer matrix quantifies the generation–verification gap at small scale and tells whether joint training is worth it.
- Gap: generation–verification gaps are measured on frozen models; how RL on one changes the other has not been.
- Experiment: 0.5–1B; train solver, verifier, both; evaluate each on each; three domains.
- Kill test: symmetric transfer.
- Compute: 0.5–1B. One to two weeks.

**35. Does RLVR make models stubborn or sycophantic?**
- Hypothesis: RLVR makes models more resistant to incorrect assertions in the prompt ("I think the answer is 42") in-domain but not out-of-domain, and including adversarial assertions in a fraction of training prompts generalizes the resistance. The flip rate under assertion is a cheap robustness metric.
- Gap: sycophancy is studied under RLHF; its change under RLVR and the adversarial-assertion fix are untested.
- Experiment: 0.5–1.5B; assertion-perturbed evaluation before and after RL; training with assertions; in- and out-of-domain.
- Kill test: no change in flip rate.
- Compute: 0.5–1.5B. One week.

**36. Answer-mode collapse on multi-solution problems.**
- Hypothesis: on problems with several correct answers, GRPO collapses to one answer per problem type (not just one reasoning style), which is invisible on single-answer benchmarks and harmful for program synthesis and planning. A coverage-aware group advantage that rewards distinct correct answers maintains answer diversity at no accuracy cost.
- Gap: diversity collapse is measured on reasoning paths; collapse of the answer set on multi-solution problems is a distinct phenomenon.
- Experiment: procedural tasks with enumerable solution sets (paths, factorizations, programs); mode coverage over training; coverage-aware advantage.
- Kill test: no collapse.
- Compute: 0.5–1B. One week.

**37. Scoring rules for numeric answers.**
- Hypothesis: for numeric-answer tasks, the reward shape (exact match, −|error|, −error², threshold) determines which statistic of the model's belief the policy outputs. Under uncertainty, L1-trained models hedge toward the median and L2-trained toward the mean, losing exact-match accuracy; a proper scoring rule on a verbalized distribution yields calibrated outputs and the best exact-match pass@k.
- Gap: reward shaping in RLVR is studied as dense versus sparse; its decision-theoretic effect on numeric outputs has not been.
- Experiment: estimation tasks with known answer distributions; reward shapes; analyze the output statistic; pass@k.
- Kill test: all shapes give the same output distribution.
- Compute: 0.5–1B. One week.

**38. Temperature as an action.**
- Hypothesis: letting the model emit a temperature token per problem before solving, trained with GRPO, yields a policy that picks high temperature on hard or unfamiliar problems and low on easy ones, improving pass@1 at fixed compute and giving a calibrated difficulty signal for free. The chosen temperature also improves pass@k at test.
- Gap: temperature is fixed or scheduled globally; set 2 idea 34 mixes temperatures within the group. Letting the policy choose it as an action is new.
- Experiment: 0.5–1B; temperature-action vs best fixed vs oracle per-problem temperature; accuracy; correlation of the chosen temperature with true difficulty.
- Kill test: the model picks a constant temperature.
- Compute: 0.5–1B. One week.

## H. Representation, cognition, and skills

**39. The Baldwin effect in meta-RL.**
- Hypothesis: on a task distribution with a stable component and a varying component, an in-context RL agent's reliance on adaptation for the stable component shrinks over meta-training as the behavior becomes innate, measurable as the performance drop when context is ablated. Distributions with no stable component show no assimilation, and the assimilated fraction predicts loss of out-of-distribution adaptability.
- Gap: in-context RL work measures adaptation ability at the end of training; the dynamic trade between innate and adapted behavior over meta-training has not been measured.
- Experiment: RL² and transformer in-context RL on procedural bandits and gridworlds with a controlled stable/varying split; ablate context at checkpoints; OOD tests.
- Kill test: context reliance is constant over meta-training.
- Compute: tiny. One to two weeks.

**40. Emergent quasimetric structure in goal-conditioned critics.**
- Hypothesis: goal-conditioned critics trained with standard TD approximately satisfy the triangle inequality on trained-on goal pairs and violate it on untrained pairs. The violation rate on held-out triples predicts stitching failure better than training-goal return does, and a soft triangle-inequality penalty on random triples (no architecture change) recovers most of what quasimetric architectures give.
- Gap: quasimetric methods impose the structure; whether it emerges, where it fails, and whether a penalty suffices has not been measured.
- Experiment: OGBench and point/ant mazes; HER-TD critics; violation maps; penalty vs quasimetric architecture.
- Kill test: violation does not predict stitching.
- Compute: small. One to two weeks.

**41. A cognitive-bias battery for control agents.**
- Hypothesis: deep RL agents show systematic decision biases that mirror human ones, each traceable to a mechanism: sunk-cost persistence (recurrent state and churn hysteresis), probability matching (entropy regularization), loss aversion (clipped double-Q pessimism), framing sensitivity to reward sign (initialization). A standardized battery of small choice tasks quantifies each bias per algorithm, and the scores predict deployment failures such as not abandoning a failing plan.
- Gap: bias batteries exist for LLMs; for control agents, with mechanism attribution by ablation, they do not.
- Experiment: gridworld and continuous gadgets per bias; SAC, TD3, PPO, DQN variants; bias scores; ablations to confirm attribution.
- Kill test: no stable biases.
- Compute: tiny. Two weeks.

**42. The out-of-span penalty for pretrained skills.**
- Hypothesis: skill-discovery pretraining helps downstream only when the downstream optimal behavior lies near the span of discovered skills; outside it, pretrained initialization is worse than scratch because the policy must unlearn a skill prior. A span-coverage statistic computed before fine-tuning predicts the sign of transfer.
- Gap: skill pretraining is reported on showcase tasks; the negative-transfer region and a predictor for it have not been mapped.
- Experiment: DMC and mazes; pretrain skills (DIAYN, METRA); downstream tasks designed in and out of span; transfer sign vs span statistic.
- Kill test: pretrained is always at least as good as scratch.
- Compute: small. Two weeks.

**43. Teacher cloneability.**
- Hypothesis: at matched return, RL-trained teachers differ widely in how well they can be behavior-cloned, and cloneability is predicted by the teacher's action smoothness, entropy, and state coverage. Choosing the teacher by cloneability rather than return gives better students at the same demonstration budget, and smoothness-regularized teachers distill better than higher-return jittery ones.
- Gap: distillation takes the best teacher by return; teacher properties governing BC sample efficiency have not been studied in RL. Set 2 idea 45 chooses which part of the buffer to clone; set 3 idea 26 characterizes behavioral differences, not their effect on students.
- Experiment: Playground; teachers trained with varied regularization; BC students at several demo budgets; cloneability vs teacher statistics.
- Kill test: student return tracks teacher return only.
- Compute: GPU-parallel. One week.

**44. Transfer the actor, reset the critic.**
- Hypothesis: when transferring to a new reward on the same dynamics, the actor and critic have opposite transfer value: the critic's values are wrong and bias early learning, while the actor's behaviors are a good exploration prior. Transferring the actor and resetting the critic (with a short Monte Carlo warmup) beats transferring both, resetting both, or the reverse. The ordering flips for a pure dynamics shift with the same reward.
- Gap: fine-tuning transfers everything; which network to transfer is folklore. The ADEPT pretraining stub asks whether transfer happens, not which component carries it. Flagged as at-risk below.
- Experiment: DMC and Playground task pairs of both kinds; 2 × 2 transfer matrix; time to threshold.
- Kill test: all configurations equal.
- Compute: GPU-parallel. One week.

## I. Reward learning

**45. Structured label noise in preference learning.**
- Hypothesis: human preference labels are noisier on pairs with similar true reward. Bradley–Terry fitting under difficulty-dependent noise systematically compresses the reward's range in dense regions and inflates it in sparse ones, and the downstream policy over-weights distinctions the labels never resolved. A noise-aware BT (per-pair noise as a function of predicted gap) recovers the true scale and improves policy return.
- Gap: label noise in RLHF is modeled as uniform; structured noise and its downstream effect have not been studied.
- Experiment: control tasks with known reward and simulated labelers with gap-dependent noise; BT vs noise-aware BT; reward recovery and policy return; validate on a public preference dataset with a difficulty proxy.
- Kill test: uniform-noise BT is unbiased under structured noise.
- Compute: small. One to two weeks.

## J. Evaluation and methodology

**46. Anytime-valid comparisons of RL algorithms.**
- Hypothesis: algorithm comparisons use a fixed seed count chosen in advance or, worse, grown until significant. Sequential testing with anytime-valid confidence sequences lets you add seeds until a conclusion is reached and stop early with valid error control, cutting compute for the same false-positive rate by 2 to 5× on typical gaps.
- Gap: rliable gives fixed-sample intervals; anytime-valid inference is standard in A/B testing and unused in RL evaluation.
- Experiment: large seed pools on gymnax and Craftax; simulate sequential vs fixed protocols; compute saved and error rates; ship a drop-in tool.
- Kill test: no compute saving at matched error.
- Compute: many tiny runs. One week.

**47. Mutation testing for RL: which bugs can diagnostics catch?**
- Hypothesis: most RL implementation bugs (off-by-one in GAE, wrong done handling, stale normalizer, clipping before log-prob) are silent; return drops modestly. A benchmark of injected mutations tests which diagnostics (TD-error whiteness, advantage statistics, entropy trajectory, value calibration) detect each bug, and a small set of cheap monitors catches most with few false alarms.
- Gap: implementation-matters papers show bugs matter; a systematic detectability study does not exist.
- Experiment: CleanRL-style PPO and SAC with 30 injected mutations; run diagnostics; detection matrix and a recommended monitor set.
- Kill test: diagnostics cannot separate bugged from clean runs.
- Compute: small. Two weeks.

**48. Checkpoint selection inflation.**
- Hypothesis: reporting the best checkpoint's return, selected on the same evaluation episodes, inflates results relative to last-checkpoint or held-out-selected reporting by an amount that grows with evaluation noise and checkpoint count, and across common protocols the inflation is comparable to typical claimed improvements. Averaging the last k checkpoints is a cheap unbiased alternative.
- Gap: the practice varies and is rarely stated; its magnitude has not been quantified for RL.
- Experiment: many runs on DMC and Playground; simulate reporting protocols on the same runs; inflation distribution.
- Kill test: inflation is negligible.
- Compute: piggybacks on other runs. Days.

**49. The determinism assumption in default hyperparameters.**
- Hypothesis: defaults for γ, n-step, UTD, and target-smoothing noise were tuned on deterministic simulators. As environment stochasticity is dialed up, the optima shift predictably (shorter n, lower UTD, lower γ), and the gap between default and re-tuned performance grows with noise, a hidden fragility in benchmark-tuned defaults. A noise-aware default rule closes most of the gap.
- Gap: stochasticity is a neglected benchmark axis; set 2 idea 50 dials it for variance reduction. The hyperparameter shift law is new.
- Experiment: DMC and Playground with injected dynamics and observation noise at several levels; re-tune per level; fit shift laws; validate on held-out tasks.
- Kill test: optima are independent of noise.
- Compute: GPU-parallel. Two weeks.

## K. Cross-cutting

**50. Entropy conversion efficiency.**
- Hypothesis: max-ent RL maximizes action entropy, but exploration needs state-visitation entropy. The conversion ratio (state entropy gained per unit of action entropy) is a task property set by how fast the dynamics damp action perturbations. Tasks with low conversion are exactly where SAC-style exploration fails and where state-entropy methods win. The ratio can be measured from random-policy rollouts and used to choose the exploration method before training.
- Gap: action-entropy and state-entropy methods are compared by return; the conversion between the two quantities as a task property is new, and it is distinct from set 2 idea 13, which redistributes a fixed entropy budget across states.
- Experiment: DMC, Playground, MinAtar; measure the ratio; correlate with the SAC-vs-state-entropy performance gap; validate the predictor on held-out tasks.
- Kill test: the ratio does not predict the gap.
- Compute: GPU-parallel. One week.

---

## How to pick

Rank by how clean the yes/no is, how little infrastructure it needs, and how surprised the field would be. By that ranking the strongest first picks are **3, 4, 6, 7, 13, 15, 17, 22, 29, 30, 36, 38, 46**.

Groups that share infrastructure:
- Ideas 1, 4, 7, 8, and 10 are cheap diagnostics on SAC and DQN with Monte Carlo ground truth, and could be one "the critic as a measured system" paper.
- Ideas 6, 14, 19, and 22 all vary decision timing and action structure and share a frame-skip/repeat harness.
- Ideas 13, 21, 23, and 49 share a perturbation and noise-injection harness on Playground.
- Ideas 16 and 17 share dataset generators with known behavior policies.
- Ideas 29, 30, 31, 34, and 35 share base models and three procedural domains.
- Ideas 46, 47, and 48 are methodology and share a large pool of seeds and checkpoints.

Ideas most at risk of having been done recently; search before committing: 1 (sharpness or edge-of-stability in TD learning), 5 (factorial placement of entropy and pessimism), 8 (refit-based reset triggers), 9 (dimensional analysis of RL hyperparameters), 18 (PPO clip versus action dimension), 22 (matched n-step baselines for HRL), 28 (legibility or predictability bonuses in MARL), 29 (prompt tuning with RL), 44 (component-wise transfer in fine-tuning), 46 (sequential testing for RL evaluation).
