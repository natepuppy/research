# 50 novel RL research ideas

This list is different from `rl_ideas_50.md`. That file took a recent paper and asked the question it left open. These start from a mechanism, a measurement, or a reframing that, to my knowledge, nobody has made. None overlaps the ADEPT stubs (dead-ends, delayed rewards, offline data amount, action quantization, RLVR rigidity, cross-family distillation, borrowing experience, empowerment, environment design, trusted graders) or the 50 in the other file.

Two caveats. Novelty is as of my knowledge; papers from the last year may have closed some of these, so run a literature search on each before committing. And "NeurIPS candidate" means the shape of the paper fits: a mechanism, a clean measurement, a falsifiable claim, and a result that is informative either way. Nothing is guaranteed.

Format for each: **Hypothesis**, **Gap** (what exists and what is missing), **Experiment**, **Kill test** (the result that would end it), **Compute** (one 10 GB GPU).

---

## A. Learning dynamics of temporal-difference learning

**1. The TD update is not a gradient. Measure its curl.**
- Hypothesis: semi-gradient TD defines a vector field on parameters with a rotational (non-conservative) component. Its local magnitude predicts divergence and oscillation better than TD error or Q magnitude.
- Gap: known in theory (Baird's counterexample; the game-dynamics literature on symplectic gradient adjustment) but never measured in deep RL agents or used as a control signal.
- Experiment: estimate the antisymmetric part of the Jacobian of the TD update field with a few Hessian-vector products per step, for DQN and SAC on MinAtar and DMC. Correlate with divergence events. Then use the measured rotation to set target-network τ and to apply a symplectic-style correction to the update.
- Kill test: curl magnitude is flat across stable and diverging runs.
- Compute: small nets, JVPs are cheap. One to two weeks.

**2. The deadly triad has high-leverage transitions.**
- Hypothesis: divergence in deep TD is driven by a small fraction of transitions with large eNTK leverage scores (updating on them moves the function far at other inputs). Capping leverage kills divergence without hurting learning.
- Gap: leverage is a standard regression diagnostic; it has not been applied to TD, where the moving target makes high-leverage points especially dangerous. PER prioritizes by TD error, a different quantity.
- Experiment: approximate leverage with random-projection NTK features for buffer samples; track its distribution on Baird-like tasks, an Atari-100k subset, and DMC. Intervene by downweighting the top 1%.
- Kill test: leverage is uncorrelated with divergence onset.
- Compute: one to two weeks.

**3. Does the actor need a correct critic, or just a correctly ordered one?**
- Hypothesis: the policy gradient uses the critic's ordering over actions at each state, not its absolute values. Late in training most critic capacity goes to calibration the actor never uses.
- Gap: ranking losses have appeared in RL, but nobody has decomposed critic error into rank and scale components over training, or trained an ordering-only critic alongside a tiny scalar head for bootstrapping.
- Experiment: at checkpoints, measure Kendall-τ between Q and Monte Carlo orderings over sampled actions, separately from MSE. Then train a critic with a pairwise ranking loss over actions plus a scalar V head for targets. DMC and MyoSuite.
- Kill test: the ordering-only critic underperforms at matched compute on every task.
- Compute: DMC. Two weeks.

**4. Activation functions are a hidden pessimism knob in offline RL.**
- Hypothesis: how a critic extrapolates to out-of-distribution actions is set by the activation's tail (ReLU grows linearly, tanh saturates, periodic activations are bounded). Part of what CQL and IQL penalties correct is the extrapolation geometry of ReLU networks.
- Gap: offline RL compares regularizers and never the extrapolation geometry of the function class.
- Experiment: fix the algorithm (TD3+BC, IQL), sweep activations, and measure OOD Q-overestimation against ground-truth Monte Carlo values on D4RL and OGBench. Then check whether the optimal penalty weight tracks the activation's extrapolation slope.
- Kill test: no relationship between extrapolation slope and overestimation.
- Compute: MuJoCo offline. One to two weeks.

**5. Lazy critics: how much of Q depends on the action?**
- Hypothesis: early in training ‖∂Q/∂a‖ ≪ ‖∂Q/∂s‖ on many tasks, so the actor follows noise. The time to "action-sensitivity onset" explains much of the sample-efficiency gap across tasks and algorithms.
- Gap: no paper measures the critic's action sensitivity over training. It would explain why critic warm-up and high UTD help.
- Experiment: log the ratio ‖∂Q/∂a‖ / ‖∂Q/∂s‖ and the signal-to-noise of the actor gradient on DMC, MyoSuite, and Mujoco Playground. Intervene with an auxiliary contrastive-action loss that makes the critic distinguish actions before TD targets become informative.
- Kill test: sensitivity onset is immediate everywhere, or uncorrelated with learning speed.
- Compute: DMC. One week for the diagnostic, one more for the fix.

**6. Gradient starvation across action dimensions.**
- Hypothesis: in high-DoF control, a few action dimensions with large ∂Q/∂a dominate the actor gradient and the rest (distal joints, fingers) stop learning. Per-dimension normalization of ∂Q/∂a recovers them and improves fine manipulation.
- Gap: gradient starvation is studied in supervised feature learning, not in the action space of RL policies.
- Experiment: MyoSuite hand tasks and Shadow Hand in Playground; log per-dimension gradient share and per-dimension policy change. Intervene with per-dimension Adam-style normalization.
- Kill test: gradient share is roughly uniform, or balancing it does not change return.
- Compute: GPU-parallel. One to two weeks.

**7. Residual gradients, revisited under modern architecture.**
- Hypothesis: the classic finding that full-gradient (residual) TD loses to semi-gradient was measured on shallow unnormalized nets. With hyperspherical normalization and modern optimizers, residual gradients may win, especially at high UTD where semi-gradient diverges.
- Gap: the comparison has not been redone since 2017-era architectures.
- Experiment: SimbaV2-style critic; semi-gradient vs residual vs double-sampled residual; UTD 1 to 16; DMC and an Atari-100k subset.
- Kill test: residual still loses everywhere.
- Compute: DMC. One to two weeks.

**8. Optimizer momentum carries gradients from a dead policy.**
- Hypothesis: in on-policy methods, Adam's first moment at iteration k contains directions computed on data from policies k−1, k−2, ... toward targets that no longer exist. Resetting or decaying momentum at each rollout changes PPO's effective trust region.
- Gap: optimizer state in RL is studied for plasticity and second-moment staleness, not for stale-momentum bias in on-policy learning.
- Experiment: PPO and REPPO on Playground; β₁ ∈ {0, 0.5, 0.9} and momentum-reset-per-rollout variants; measure KL per update and return at matched compute.
- Kill test: no difference.
- Compute: GPU-parallel. Days.

**9. A scaling law between discount and critic learning rate.**
- Hypothesis: the TD contraction factor implies the critic's effective step size should scale with (1−γ); the best learning rate at γ = 0.999 is about 10× smaller than at γ = 0.99. The two are tuned independently, and the interaction is why "high γ is unstable."
- Gap: the theory suggests the form; nobody has checked it on deep nets or offered a rule.
- Experiment: learning rate × γ grid on 6 tasks; fit optimal learning rate as a function of γ; validate on 3 held-out tasks.
- Kill test: optimal learning rate is flat in γ.
- Compute: small. One week.

**10. Which part of a critic is about the policy, and which about the world?**
- Hypothesis: a trained Q^π has a subspace invariant across policies (dynamics, reward structure) and a policy-specific subspace. Pretraining the invariant part and relearning only the policy-specific part accelerates policy evaluation after a policy change.
- Gap: successor features separate dynamics from reward, not policy-invariant from policy-specific representation. Probing critics trained under many policies for shared structure has not been done.
- Experiment: train critics for 50 diverse policies on one environment; CCA across penultimate layers; measure the invariant dimension. Then train a two-part critic and compare re-evaluation speed after a policy switch.
- Kill test: the invariant subspace is empty or is just the state encoding.
- Compute: small. Two weeks.

## B. Data, replay, and time

**11. Within-task forgetting: the agent forgets its stepping stones.**
- Hypothesis: as the buffer fills with late high-reward data, critic error on early bottleneck transitions (first door, first contact) grows; they are rare in the buffer but critical for value propagation. Protecting a small set of anchor transitions prevents mid-training collapses.
- Gap: forgetting is studied across tasks; the loss of accuracy on early, now-rare transitions within one task has not been measured.
- Experiment: tag transitions by first-visit time; track per-transition TD error over training on Montezuma-style, Craftax, and maze tasks. Identify forgotten transitions; add an anchor buffer sampled at a fixed rate.
- Kill test: error on early transitions does not rise, or anchors do not help.
- Compute: one to two weeks.

**12. The half-life of a transition.**
- Hypothesis: each transition's usefulness decays with a measurable half-life that depends on the environment and on how fast the policy changes. Buffer size should be set by that half-life, not fixed at 1M.
- Gap: buffer size has been studied as a knob; per-transition usefulness decay has not been defined or shown to predict the optimal size.
- Experiment: define usefulness as alignment between a transition's TD gradient and the mean batch gradient; fit decay curves across environments; predict and test the optimal buffer size.
- Kill test: half-life does not predict optimal buffer size across environments.
- Compute: small. One to two weeks.

**13. Deduplicating replay in GPU-native environments.**
- Hypothesis: with thousands of parallel environments, a large fraction of stored transitions are near-duplicates and effective sample size is far below buffer size. Learned-hash deduplication gives the same return at a fraction of memory and update cost.
- Gap: nobody measures duplication rate or effective sample size in GPU-environment buffers.
- Experiment: Playground tasks; measure near-duplicate fraction by latent hashing; compare deduplicated buffers at matched compute.
- Kill test: duplication is low, or deduplication hurts.
- Compute: GPU-parallel. One week.

**14. Agents learn clocks, and clocks break horizon generalization.**
- Hypothesis: policies trained on fixed-length episodes encode an implicit step counter in their hidden state even when time is not observed, and that is why they fail at longer horizons. Removing the clock (adversarial time-prediction penalty, or randomized episode lengths) restores generalization.
- Gap: time-awareness work studied giving the agent the time; nobody has probed for implicit time or linked it to horizon generalization.
- Experiment: train recurrent and frame-stack policies on episodes of length T; linear-probe hidden states for t; test at 2T to 10T; apply the clock-removal fixes.
- Kill test: probes cannot recover t, or removing it does not help.
- Compute: tiny. One week.

**15. Time-limit truncation is right-censoring. Use survival analysis.**
- Hypothesis: when an episode is cut by a time limit, the return is censored, not observed. Monte Carlo and GAE targets then carry censoring bias. Censored regression (Tobit or Kaplan–Meier style) for value targets removes the bias without bootstrapping.
- Gap: partial-episode bootstrapping handles truncation by bootstrapping; treating it as censored data is standard statistics and untried in RL.
- Experiment: PPO with censored-regression value targets vs standard vs bootstrap-at-truncation, on tasks whose time limit is short relative to the natural horizon.
- Kill test: no gain over partial-episode bootstrapping.
- Compute: small. One week.

**16. Control-frequency transfer: train at 50 Hz, deploy at 20 Hz.**
- Hypothesis: a critic transfers across control frequencies by rescaling γ → γ^(Δt'/Δt) plus a small correction, while the policy transfers much worse. Characterize what breaks.
- Gap: continuous-time RL theory predicts the rescaling; nobody has tested cross-frequency transfer on standard benchmarks, which matters for robots whose control rate varies.
- Experiment: DMC and Playground with varied dt; train at one, test at others with and without rescaling and a short fine-tune.
- Kill test: nothing transfers regardless of rescaling.
- Compute: GPU-parallel. One week.

**17. How many parallel environments is too many?**
- Hypothesis: beyond a point, more parallel environments per update hurts learning (not throughput) because the batch becomes highly correlated same-policy data and the number of policy updates per environment step falls. There is a compute-optimal curve that depends on task stochasticity.
- Gap: practitioners pick 2048 to 8192 by throughput; no systematic law exists.
- Experiment: PPO and REPPO on Playground; num_envs × rollout length grid at matched steps and matched wall-clock; fit a law; test on held-out tasks.
- Kill test: return is monotone in num_envs.
- Compute: GPU-parallel. One week.

## C. Cloneable simulators as a new primitive

GPU-native simulators make it cheap to save a state, branch, and roll out alternatives. Several estimators that were abandoned for cost become free. These four share one piece of infrastructure.

**18. Vine at scale: zero-variance counterfactual advantages.**
- Hypothesis: TRPO's abandoned "vine" estimator (branch N actions from the same state and compare returns) is nearly free in a cloneable simulator and gives a per-state exact baseline, removing most of the variance GAE trades for bias.
- Gap: vine was dropped in 2015 for cost and never revisited.
- Experiment: state cloning in Playground or Brax; vine-PPO vs GAE-PPO vs REPPO at matched compute; measure advantage variance and return.
- Kill test: no compute-matched gain.
- Compute: GPU-parallel. One to two weeks.

**19. Ground-truth action gradients expose the actor exploiting the critic.**
- Hypothesis: with cloning you can estimate ∇ₐQ_true by finite differences and compare it to the learned ∇ₐQ. Their cosine is a direct measurement of "the actor exploits critic error," and its trajectory explains instability episodes.
- Gap: actor-exploits-critic is a folk explanation that has never been measured against a ground-truth gradient.
- Experiment: SAC and TD3 on cloneable DMC; log the cosine; correlate with return drops; test fixes (ensemble min, clipping ∂Q/∂a, critic training on perturbed actions).
- Kill test: the cosine stays high throughout.
- Compute: GPU-parallel. One to two weeks.

**20. Counterfactual information gain per action.**
- Hypothesis: the exploration value of an action can be measured directly as the spread of outcomes across cloned branches, instead of approximated by prediction error. The resulting bonus is zero in stochastic-but-uninformative states, which curiosity methods cannot achieve.
- Gap: the noisy-TV problem is a direct consequence of estimating informativeness without counterfactuals.
- Experiment: gridworld and MinAtar with a noisy-TV region, plus Craftax; cloning-based bonus vs RND and ICM.
- Kill test: the cloning bonus is no better on the noisy-TV test.
- Compute: small. One week.

**21. Vine advantages for LLM agents in text environments.**
- Hypothesis: TextWorld and BabyAI-text allow state cloning. Branching K actions from the same environment state gives per-step advantages for a small LLM agent without a learned critic, and beats GRPO's sequence-level advantage.
- Gap: LLM RL uses prompt-level group baselines; per-step counterfactual branching in a cloneable environment has not been tried.
- Experiment: 0.5–1B LoRA agent; vine vs GRPO at matched rollout budget; analyze which steps benefit most.
- Kill test: no gain over GRPO at matched budget.
- Compute: 0.5B. Two weeks.

## D. Exploration

**22. Disagreement across discount factors as an exploration signal.**
- Hypothesis: train Q heads at several γ values; where their action orderings disagree, the long-horizon consequences of actions are unresolved. Exploring there targets uncertainty that matters for the task, unlike state-novelty bonuses.
- Gap: multi-γ heads exist for stability; their disagreement has not been used for exploration.
- Experiment: MinAtar, Craftax, sparse DMC; bonus = rank disagreement across γ heads; compare to RND and ensemble disagreement.
- Kill test: γ-disagreement is redundant with ensemble disagreement.
- Compute: small. One week.

**23. Flow-policy noise as a temporally correlated exploration variable.**
- Hypothesis: in flow and diffusion policies the initial noise z is a latent intention. Holding z fixed within an episode, or resampling it slowly, gives structured exploration like gSDE with the multimodal expressiveness of flows.
- Gap: gSDE and parameter noise do this for Gaussian policies; nothing equivalent exists for generative policies in online RL.
- Experiment: online flow policy; per-step vs per-episode vs OU-correlated z; sparse DMC and maze tasks.
- Kill test: no difference in coverage or return.
- Compute: DMC. One week.

**24. Denoising steps as temperature.**
- Hypothesis: the number of denoising steps controls how sharp a flow policy's action distribution is; annealing the step count over training can replace the entropy bonus in max-ent RL for flow policies.
- Gap: entropy for flow policies is hard to compute; a step-count schedule is a cheap substitute nobody has tested.
- Experiment: flow policy with steps annealed 1 → 16 vs fixed vs entropy-regularized; measure action spread and return.
- Kill test: step count does not act like temperature.
- Compute: DMC. One week.

**25. Off-policy evaluation of exploration policies.**
- Hypothesis: the coverage an exploration policy would achieve can be estimated from logged data of another policy, via density-ratio estimation on a learned state abstraction, well enough to pick exploration bonuses without running them.
- Gap: OPE is about return; coverage and information objectives have no OPE theory or practice.
- Experiment: log data under one policy on Craftax and MinAtar; estimate coverage of 5 candidate exploration policies; compare to the coverage they actually achieve.
- Kill test: poor rank correlation.
- Compute: small. Two weeks.

**26. What do different seeds learn? Strategy modes.**
- Hypothesis: seeds of the same algorithm converge to a small number of qualitatively distinct strategies, and the number of modes, not performance noise, explains most inter-seed return variance. Mode count is a task property worth reporting alongside return.
- Gap: seed variance is treated as noise; nobody has clustered what seeds learn.
- Experiment: 30+ seeds on Craftax, Procgen, and DMC; embed trajectories; cluster; report modes and their return distribution; test whether initializing from a mode is heritable.
- Kill test: behaviors form a continuum with no clusters.
- Compute: many small runs. One to two weeks.

## E. Information, compression, and reward

**27. Description length of the value function predicts sample complexity.**
- Hypothesis: the compressed size of V* over the reachable state space (MDL under a fixed coder) predicts how many samples deep RL needs on a task better than state count or horizon do. MDL-regularized critics learn faster where V* is compressible.
- Gap: sample-complexity bounds use dimension and horizon; MDL of the value function has not been proposed as the relevant quantity.
- Experiment: tabular and gridworld families where V* is computable; compress V* with a fixed coder; correlate with DQN and PPO sample complexity across 50+ generated tasks.
- Kill test: MDL adds nothing over state count and horizon.
- Compute: tiny. One to two weeks.

**28. Reward dropout.**
- Hypothesis: shaped robotics rewards have 5 to 20 terms; dropping terms at random during training prevents overfitting to the shaping and improves the sparse task metric.
- Gap: reward-term regularization does not exist; weights are tuned by hand.
- Experiment: Playground humanoid and quadruped tasks; dropout rate sweep; report task success and the sparse metric, not shaped return.
- Kill test: no change in sparse success.
- Compute: GPU-parallel. One week.

**29. Reward-term equalization.**
- Hypothesis: the right weights for a multi-term reward equalize each term's contribution to actor-gradient variance. A per-term online normalizer achieves this and removes most hand-tuning.
- Gap: Pop-Art normalizes the total reward; nothing normalizes per term by gradient contribution.
- Experiment: same tasks as 28; equalized vs hand-tuned vs uniform weights.
- Kill test: hand-tuned defaults dominate.
- Compute: GPU-parallel. One week.

**30. Detecting reward hacking from the shape of the return distribution.**
- Hypothesis: when a policy finds an exploit, the per-episode return distribution changes shape (a new mode, a variance collapse) in a way distinguishable from legitimate improvement, which is gradual and unimodal. A distributional critic can flag it online.
- Gap: reward hacking is detected by humans watching videos; there is no learned online detector.
- Experiment: environments with known exploits (classic MuJoCo bugs, Procgen glitch levels, injected exploits); train the detector on some, test on held-out exploits.
- Kill test: the detector cannot separate hacking from improvement.
- Compute: small. Two weeks.

**31. How often is the optimal action set multimodal?**
- Hypothesis: the move to diffusion and flow policies assumes multimodality matters, but nobody has measured how often the Q landscape over actions has multiple comparable maxima on standard benchmarks. The fraction is small on DMC and large on manipulation with symmetries.
- Gap: a measurement that says when the Gaussian policy is the bottleneck.
- Experiment: with a converged critic or cloning-based Monte Carlo, scan action space at sampled states; count modes above a threshold; correlate with the Gaussian-vs-flow gap per task.
- Kill test: mode count does not predict the gap.
- Compute: DMC and MyoSuite. One to two weeks.

## F. Variance, validation, and methodology

**32. Where does seed variance come from? A factorial decomposition.**
- Hypothesis: variance across seeds decomposes into initialization, environment stochasticity, replay sampling order, and hardware nondeterminism, with very different shares across algorithms. The shares tell you what to control to get reliable comparisons cheaply.
- Gap: rliable gives confidence intervals; nobody has done the ANOVA.
- Experiment: fix three sources and vary one, for PPO, SAC, and DQN on gymnax and Craftax. Report variance components.
- Kill test: none; a null decomposition is still informative.
- Compute: many tiny runs. One week.

**33. A validation set for RL: cross-fitted Bellman error.**
- Hypothesis: RL has no validation signal. Train two critics on disjoint halves of the buffer; each one's TD error evaluated with the other's targets on the other half tracks true value error, unlike in-sample Bellman error. This enables early stopping and model selection without environment evaluation.
- Gap: Bellman error is known to be a poor proxy; cross-fitting is standard in causal inference and untried here.
- Experiment: offline and online settings; correlate cross-fitted error with true value error and with return across checkpoints.
- Kill test: cross-fitted error is no better than in-sample.
- Compute: small. One to two weeks.

**34. Kill the run early: predicting final return from the first 5%.**
- Hypothesis: a few early signals (action sensitivity from idea 5, spectral statistics, gradient variance, return slope) predict final return well enough to prune 70% of hyperparameter configurations after 5% of training.
- Gap: learning-curve extrapolation exists in AutoML; RL sweeps do not use it, and RL curves have distinct early signatures.
- Experiment: 500+ short runs across tasks; train a predictor; measure the regret of pruned sweeps vs full.
- Kill test: early features have no power beyond return slope.
- Compute: many tiny runs. Two weeks.

**35. Cross-simulator generalization as a cheap proxy for sim-to-real.**
- Hypothesis: robustness when transferring between simulators of the same task (MuJoCo → Brax → Isaac Lab) predicts real-world transfer. If so, sim-to-sim becomes a free benchmark for sim-to-real methods.
- Gap: everyone evaluates sim-to-real on their own robot; no shared proxy exists.
- Experiment: 4 tasks in three simulators; train domain randomization, asymmetric actor-critic, and vanilla; measure sim-to-sim; compare the ranking to published sim-to-real results for the same methods, and one real-robot test if accessible.
- Kill test: sim-to-sim ranking disagrees with sim-to-real.
- Compute: GPU-parallel. Three weeks.

**36. Do sample-efficient learners converge to worse optima?**
- Hypothesis: high UTD, large critics, and resets reach a given return faster but converge to different, sometimes worse, final policies than slow on-policy learning. The field reports curves truncated before convergence.
- Gap: nobody runs sample-efficient methods to convergence and compares endpoints.
- Experiment: 6 tasks; run SAC, BRO, SimbaV2 and PPO, REPPO to plateau; compare final return and behavioral quality (smoothness, energy).
- Kill test: endpoints are identical.
- Compute: long runs on small tasks. Two to three weeks.

## G. Generalization and interference

**37. Cross-action interference in discrete-action critics.**
- Hypothesis: with a shared trunk, updating Q(s,a) moves Q(s,a') for other actions; the interference is large and anti-correlated with learning on tasks with many actions. Decoupled heads or action-conditioned trunks reduce it.
- Gap: interference across states is studied; across actions at the same state it is not.
- Experiment: measure ΔQ(s,a') when updating only (s,a) on an Atari-100k subset and Procgen; relate to action count; test architectures.
- Kill test: interference is small or helpful.
- Compute: Atari subset. One to two weeks.

**38. Churn hotspots are bottleneck states.**
- Hypothesis: the states where the policy's action flips most across consecutive checkpoints are the task's bottleneck or decision states. This is a free bottleneck detector for hierarchy and exploration.
- Gap: policy churn is studied as a nuisance; its spatial structure has not been used.
- Experiment: Four Rooms, MiniGrid, Craftax; compare churn hotspots to known bottlenecks; use them as subgoals for a hierarchical agent vs graph-based bottleneck discovery.
- Kill test: hotspots are uniformly distributed.
- Compute: tiny. One week.

**39. Value-preserving observation compression for replay.**
- Hypothesis: a codec trained to preserve the critic's output, not pixel reconstruction, compresses observations 10 to 50× with no loss in return, enabling 10× larger buffers in the same memory.
- Gap: replay compression uses generic codecs; value-aware compression has not been proposed.
- Experiment: Atari-100k and pixel DMC; value-preserving codec vs JPEG vs none; return and memory.
- Kill test: the value-aware codec is no better than generic.
- Compute: Atari subset. Two weeks.

**40. Learned observation masks as sensor selection.**
- Hypothesis: training with a learned, sparsity-penalized input mask discovers the minimal sensor set for a task, and the policy trained on that set is more robust to sensor noise and failure than one trained on all sensors.
- Gap: input dropout is used for robustness; a learned mask for sensor selection, with a generalization test, has not been done in RL.
- Experiment: proprioceptive tasks with 50 to 100 sensors; learned mask vs full vs random subsets; test under sensor noise and dropout.
- Kill test: the masked policy is not more robust.
- Compute: GPU-parallel. One week.

## H. Multi-agent

**41. Stigmergy: communication only through the environment.**
- Hypothesis: when agents can only communicate by modifying shared state (marks, object placement), they develop conventions that transfer to novel partners better than learned channel messages, because the channel is grounded.
- Gap: emergent communication uses explicit channels; stigmergy is studied in swarm robotics with hand-designed rules, not learned.
- Experiment: JAX gridworld with markable cells; cross-play with held-out partners; compare to CommNet-style channels.
- Kill test: no conventions emerge, or they do not transfer.
- Compute: small. Two weeks.

**42. Does self-play rotate or converge? Measuring the Hamiltonian component.**
- Hypothesis: self-play learning dynamics have a measurable rotational component whose magnitude predicts exploitability and plateaus better than Elo does. Opponent-pool sampling can be set to cancel the measured rotation.
- Gap: the mechanics-of-games theory was validated on small analytic games; nobody has measured rotation in deep self-play agents or used it as a controller.
- Experiment: small JAX games (Kuhn poker, simple Overcooked); estimate the antisymmetric part of the game Jacobian from gradients; correlate with exploitability; test the controller.
- Kill test: rotation is uncorrelated with exploitability.
- Compute: small. Two weeks.

**43. Exact counterfactual credit via cloning in cooperative MARL.**
- Hypothesis: COMA approximates counterfactual baselines with a critic; in a cloneable GPU MARL environment the counterfactual (replace one agent's action, rerun) is exact. Exact Shapley-style credit beats learned baselines, and the gap measures how much credit assignment is lost to approximation.
- Gap: counterfactual credit in MARL has always been approximated.
- Experiment: JaxMARL (SMAX, Overcooked); exact counterfactual vs COMA vs MAPPO at matched compute.
- Kill test: the approximation gap is negligible.
- Compute: GPU-parallel. Two weeks.

## I. RL for language models

All use 0.5–1.5B models with LoRA, GRPO-style training, short contexts, and procedural or small public datasets.

**44. RL mostly teaches when to stop.**
- Hypothesis: a large share of RLVR's accuracy gain comes from changing termination decisions (when to commit to an answer, when to keep checking), not from changing reasoning content. Freezing all logits except the answer-start and EOS decisions during RL recovers much of the gain.
- Gap: the "high-entropy minority tokens" finding is close but does not isolate termination. A frozen-content ablation is cheap and decisive.
- Experiment: GRPO on GSM8K-style and procedural tasks; full RL vs termination-only RL vs content-only RL.
- Kill test: termination-only recovers little.
- Compute: 0.5B LoRA. One to two weeks.

**45. Compute-optimal allocation between verifier samples and policy samples.**
- Hypothesis: when the verifier is noisy (LLM judge, test-case subsets), spending part of the rollout budget on more verifier samples per rollout beats spending it on more rollouts, with a crossover that depends on verifier noise.
- Gap: the policy-sample vs verifier-sample trade-off has not been formalized or measured.
- Experiment: small model, judge with controllable noise; grid over allocation; fit the crossover law.
- Kill test: more rollouts always wins.
- Compute: 0.5–1B. One to two weeks.

**46. Spaced repetition for RLVR.**
- Hypothesis: solved problems should be resampled on a forgetting-curve schedule (SM-2 style), not uniformly or dropped. This preserves early skills while focusing compute on the frontier and beats pass-rate filtering.
- Gap: RLVR curricula are about difficulty; retention scheduling has not been applied.
- Experiment: three procedural domains; spaced vs uniform vs frontier-only sampling at matched budget; measure held-out and retained accuracy.
- Kill test: no retention benefit.
- Compute: 0.5–1B. One to two weeks.

**47. Does RLVR make a model a better in-context explorer?**
- Hypothesis: RLVR on verifiable tasks changes how the model handles uncertainty, which should show up as lower regret on in-context bandits and small in-context RL tasks presented as text, even though those were never trained.
- Gap: in-context RL ability is measured on base and instruct models; its change under RLVR is unknown and is a clean transfer test.
- Experiment: base vs RLVR-tuned small models on text bandits and in-context gridworlds; measure regret and exploration statistics.
- Kill test: no change.
- Compute: 0.5–1.5B. One week.

**48. Answer-first RL: are reasoning tokens causal?**
- Hypothesis: force the model to emit its answer before its reasoning and train with RL. If accuracy improves comparably, the reasoning tokens were not load-bearing and RL's gains come from elsewhere. If it collapses, you have a clean measurement of how much the chain of thought matters under RL.
- Gap: chain-of-thought faithfulness studies use prompting or SFT models; an RL-trained answer-first control has not been run.
- Experiment: answer-first vs reasoning-first RL on procedural math and QA; measure accuracy and the mutual information between reasoning and answer.
- Kill test: none; both outcomes are informative.
- Compute: 0.5B. One week.

**49. Interleaved vs blocked domains in RLVR.**
- Hypothesis: mixed-domain batches and blocked-domain batches produce different interference, and the per-domain gradient cosine predicts which domains should be interleaved. A schedule set by online gradient conflict beats both.
- Gap: multi-domain RLVR mixes uniformly; no interference measurement exists.
- Experiment: three procedural domains; blocked, interleaved, and conflict-aware schedules; held-out accuracy per domain.
- Kill test: schedules do not matter.
- Compute: 0.5–1B. One to two weeks.

**50. A learned prompt-level baseline replaces the group.**
- Hypothesis: GRPO's group mean is an in-context baseline that costs G rollouts per prompt. A learned V(prompt), one scalar per prompt with no token-level critic, with G = 1 or 2 reaches the same accuracy at a fraction of the rollouts, with the crossover set by how predictable difficulty is from the prompt.
- Gap: token-level critics were abandoned for LLM RL; a prompt-level critic is a much easier regression nobody has benchmarked against group size.
- Experiment: group size × baseline type grid at matched rollout budget; measure accuracy and the variance of the advantage estimate.
- Kill test: the group baseline always wins at matched budget.
- Compute: 0.5–1B. One to two weeks.

---

## How to pick

Rank by how clean the yes/no is, how little infrastructure it needs, and how surprised the field would be. By that ranking the strongest first picks are **1, 2, 5, 14, 15, 18, 22, 26, 33, 44, 48**.

The cloning family (18, 19, 20, 21, 43) shares one piece of infrastructure, state save/restore in a GPU simulator, and could become a single program: counterfactual RL in cloneable simulators. Ideas 5, 19, and 34 share the action-sensitivity diagnostic. Ideas 28 and 29 share the same environments and reward decomposition.

Before committing, search for the closest prior work named in each Gap line plus your variable. The ones most at risk of having been done recently: 15 (censoring), 18 (vine), 42 (rotation in self-play), 44 (termination tokens), 50 (prompt-level baselines).
