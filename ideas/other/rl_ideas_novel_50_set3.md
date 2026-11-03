# 50 more novel RL research ideas (set 3)

A fourth list, disjoint from `rl_ideas_50.md` (follow-ups to recent papers), `rl_ideas_novel_50.md` (mechanisms and measurements), `rl_ideas_novel_50_set2.md` (hidden hyperparameters, data, control, LLM RL), and the ADEPT stubs (dead-ends, irreversibility, empowerment, backtracking, pretraining transfer, action quantization, offline data amount, single-network uncertainty, untrusted graders, delayed rewards, environment design, borrowing experience, RLVR rigidity, distillation). Where an idea is adjacent to an earlier one, the Gap line says which and how it differs.

Same caveats. Novelty is as of my knowledge; run a literature search on each before committing, and I flag the ones most at risk at the end. "NeurIPS candidate" means the paper has a mechanism or a measurement, a falsifiable claim, and a result that is informative either way.

Everything fits one RTX 3080 (10 GB). Control ideas use gymnax, MinAtar, Craftax, DMC, MyoSuite, Mujoco Playground / MJX, Brax, or JaxMARL. LLM ideas use 0.5–1.5B models with LoRA and GRPO-style training on procedural data, or 20–100M models pretrained from scratch on synthetic corpora.

Format for each: **Hypothesis**, **Gap**, **Experiment**, **Kill test**, **Compute**.

---

## A. Learning dynamics and optimization

**1. Weight decay is implicit pessimism.**
- Hypothesis: decoupled weight decay (AdamW) on a critic shrinks the target network's weights every step, biasing bootstrapped targets toward zero by an amount proportional to lr·wd·‖θ‖. For positive-reward tasks this is pessimism; for negative-reward tasks it is optimism. Part of why weight decay "helps offline RL" and "hurts online RL" is this sign-dependent bias, not regularization, and a bias-corrected decay (decay only the trunk, or rescale the output layer) separates the two effects.
- Gap: weight decay is studied in RL for plasticity; its interaction with bootstrapping and reward sign has not been isolated or measured against ground-truth Q.
- Experiment: SAC and IQL on DMC and D4RL with reward sign flipped by constant shift; sweep wd; measure Q − Q^π_MC per state; compare full decay, trunk-only decay, and no decay.
- Kill test: the Q bias does not track lr·wd·‖θ‖ or does not flip with reward sign.
- Compute: small. One week.

**2. Adam's ε is a hidden learning-rate schedule in TD learning.**
- Hypothesis: late in training TD gradients are tiny, so for many parameters √v̂ falls below ε and Adam's effective step collapses toward lr·g/ε, which is an implicit, unplanned decay that is sharper for critics than for supervised nets. The fraction of parameters in the ε-dominated regime rises over training, differs across tasks by reward scale, and explains part of the sensitivity of RL to reward scaling that Pop-Art and reward normalization fix.
- Gap: set 2 idea 2 is about floating-point precision of the TD residual. This is about the optimizer's denominator, which affects fp32 runs too. ε is almost never reported or swept in RL.
- Experiment: SAC, DQN, PPO on DMC and MinAtar; log the fraction of parameters with √v̂ < ε over training; sweep ε ∈ {1e-8, 1e-6, 1e-4} and reward scale; compare to Pop-Art.
- Kill test: the ε-dominated fraction stays near zero, or ε has no effect at any reward scale.
- Compute: tiny. Days.

**3. The empirical contraction modulus of deep fitted Q-iteration.**
- Hypothesis: theory says the Bellman operator contracts by γ; the projection onto a neural function class can expand. Measuring the ratio ‖Π T Q₁ − Π T Q₂‖ / ‖Q₁ − Q₂‖ for pairs of critics over training gives an empirical modulus. It exceeds 1 before divergence events, is lowered by normalization (hyperspherical, LayerNorm) and raised by UTD, and predicts which architecture–UTD combinations are safe.
- Gap: list 1 idea 1 measures the curl of the TD vector field; this measures the contraction factor of one full fitting step, a different and more classical quantity that has never been measured for deep critics.
- Experiment: at checkpoints, perturb the critic in random directions, apply one round of fitted-Q regression to both, measure the ratio across many directions; relate to divergence and to architecture.
- Kill test: the modulus is always well below 1 or does not predict divergence.
- Compute: small nets. One to two weeks.

**4. The critical batch size of TD learning changes over training.**
- Hypothesis: the gradient noise scale (McCandlish et al.) for the critic's loss grows over training as TD errors shrink and the target stabilizes, so the compute-optimal batch size rises; a schedule that grows the batch beats any fixed batch at matched gradient samples. Separately, the benefit of a larger batch comes from more unique states per update, not from variance reduction alone: a batch of 256 unique transitions beats 64 unique × 4 copies.
- Gap: critical batch size is standard for LLM training and unmeasured for bootstrapped losses, where the "label" also changes; list 1 idea 17 is about parallel environments, not the gradient batch.
- Experiment: SAC and DQN on DMC and MinAtar; estimate the noise scale per checkpoint; batch-growth schedules vs fixed; unique-vs-duplicated batch ablation.
- Kill test: noise scale is flat, or the duplicated batch matches the unique one.
- Compute: small. One week.

**5. What fraction of actor updates are actually improvements?**
- Hypothesis: with common random numbers in a GPU simulator you can evaluate J(π) before and after every actor step to within tight error. A large share of actor updates are neutral or negative; net progress comes from a minority. The improving fraction differs sharply between algorithms (PPO vs SAC vs flow policies) and predicts sample efficiency better than gradient variance does.
- Gap: set 2 idea 50 revives CRN for policy gradient estimation; this uses it only as an oracle to measure the sign of each update's true effect, which has never been reported.
- Experiment: Playground and Brax with RNG control; paired J before and after each actor update on a fixed set of start states and noise seeds; fraction improving, magnitude, and autocorrelation over training.
- Kill test: nearly all updates improve, in every algorithm.
- Compute: GPU-parallel. One to two weeks.

**6. Trust regions on the state distribution, not the action distribution.**
- Hypothesis: PPO constrains KL between consecutive policies' action distributions, but what the critic actually suffers from is the shift in the state visitation d^π. Action-KL is a loose proxy: small action changes near bottlenecks cause large d^π shifts, and large action changes in irrelevant states cause none. A trust region on d^π shift, estimated by a classifier between consecutive rollouts' states, gives a more stable and faster learner at matched compute.
- Gap: state-distribution divergence appears in theory bounds (e.g., performance-difference lemma analyses) but no on-policy method measures or constrains it directly; GPU simulators make the estimate cheap.
- Experiment: PPO on Playground and Procgen; log the classifier-estimated d^π shift per update and its relation to action-KL and to return drops; replace or augment the clip with a d^π constraint.
- Kill test: d^π shift tracks action-KL closely, or constraining it does not help.
- Compute: GPU-parallel. One to two weeks.

**7. Are plateaus-then-jumps in RL grokking?**
- Hypothesis: many RL learning curves show long plateaus followed by sudden jumps. In some cases the jump is preceded by a measurable representational change (drop in weight norm, rise in effective rank, sharpening of a probe for a task-relevant variable) while return is flat, matching the grokking signature in supervised learning, and weight decay modulates the plateau length the same way.
- Gap: grokking is studied on algorithmic tasks; RL plateaus are attributed to exploration. Nobody has checked whether some are representational phase transitions with the same signatures.
- Experiment: tasks with known plateaus (Craftax milestones, MinAtar Breakout, sparse DMC); track weight norm, effective rank, and probe accuracy during plateaus; sweep weight decay; distinguish exploration-limited plateaus (first-success time) from representation-limited ones (success found but not exploited).
- Kill test: no representational precursor exists; jumps coincide exactly with first success.
- Compute: tiny. One to two weeks.

**8. Critic error tolerance curves.**
- Hypothesis: the actor is tolerant of some critic errors and brittle to others. Inject controlled distortions into a converged critic (additive noise, multiplicative scale, state-dependent bias, rank-preserving monotone warps, rank-breaking local perturbations), retrain the actor only, and map return loss against error type and size. Rank-breaking errors at the actor's current action dominate; large global biases are nearly free.
- Gap: list 1 idea 3 asks whether an ordering-only critic suffices. This is the inverse experiment: how much and what kind of error the actor can absorb, which no paper has mapped.
- Experiment: SAC and TD3 on DMC; distortion type × magnitude grid; actor-only retraining; return loss surfaces.
- Kill test: return loss depends only on error magnitude, not type.
- Compute: DMC. One week.

**9. The effective horizon of a trained critic.**
- Hypothesis: perturbing a reward k steps into the future and measuring the change in Q(s₀, a₀) gives an influence curve over k. Its decay is much faster than γ^k would predict and depends on n-step, buffer age, and architecture. The point where influence drops below noise is the critic's effective horizon, and tasks where the required credit distance exceeds it are exactly the tasks the agent fails.
- Gap: set 2 idea 14 tracks how the accurate region grows outward from rewards; this measures how far a reward's influence reaches backward, a different quantity, and compares it to the task's required credit distance.
- Experiment: cloneable sims with editable rewards; inject a reward impulse at distance k from a probe state; measure ΔQ; relate the measured horizon to γ, n, UTD, and to failure on tasks with known credit distance.
- Kill test: influence decays as γ^k.
- Compute: small. One week.

**10. The actor's state distribution is a free hyperparameter.**
- Hypothesis: in off-policy actor-critic the actor's gradient is computed on states sampled uniformly from the whole buffer, but the policy-gradient theorem wants d^π. Sampling actor states from recent transitions only, while the critic uses the whole buffer, is a one-line change that speeds learning and reduces churn on states the policy will never revisit; the optimal recency window tracks how fast the policy moves.
- Gap: UTD, batch size, and critic sampling are studied; the actor's state sampling distribution is never varied separately from the critic's.
- Experiment: SAC and TD3 on DMC and Playground; actor-state window ∈ {all, last 50%, 10%, 1%, on-policy rollout}; return, churn, and the critic's error on actor-sampled states.
- Kill test: no effect of the actor's state distribution.
- Compute: GPU-parallel. Days.

**11. Two-speed critics: fast heads, slow trunks.**
- Hypothesis: the critic's output layer tracks the moving target and needs a fast learning rate; the trunk learns slowly varying features and is destabilized by target drift. Per-layer learning rates (or a target network applied only to the trunk, with a live head) give the stability of a target network with the speed of none, and remove the τ hyperparameter.
- Gap: target networks and learning rates are applied uniformly across layers; the layerwise structure of what moves in a critic has not been measured or exploited.
- Experiment: measure per-layer update magnitude and per-layer drift of the target over training on DMC and MinAtar; then compare uniform lr + target net, layerwise lr without target net, and trunk-only target.
- Kill test: all layers move at the same rate, or the layerwise variants do not match the target network.
- Compute: small. One week.

**12. Durbin–Watson for TD: residual autocorrelation along trajectories.**
- Hypothesis: if the critic is correct and the state is Markov, TD errors along a trajectory are white. Autocorrelated TD errors indicate either underfitting or missing state (partial observability). The autocorrelation statistic separates the two (it persists under more training only in the partially observed case), making it a cheap diagnostic for whether a task needs memory, and a loss penalty on autocorrelation improves the critic.
- Gap: residual autocorrelation is a standard regression diagnostic; it has never been applied to TD errors or used to detect partial observability online.
- Experiment: DMC, POPGym, and "accidentally partial" tasks (velocity-masked DMC); compute the lag-k autocorrelation of TD errors per checkpoint; verify it stays high only on POMDPs; test an autocorrelation penalty.
- Kill test: TD errors are white everywhere, or autocorrelation does not separate POMDPs from underfit MDPs.
- Compute: tiny. One week.

## B. Representation and interpretability

**13. Emergent bisimulation.**
- Hypothesis: without any bisimulation objective, the distance between a trained critic's penultimate features ‖φ(s) − φ(s′)‖ converges toward a bisimulation-like metric (states with the same reward and transition structure collapse together) on tasks where the agent generalizes, and does not on tasks where it memorizes. The degree of emergent bisimulation predicts Procgen-style train/test gaps.
- Gap: bisimulation metrics are imposed as auxiliary losses (DBC and successors); whether standard critics approximate them on their own has not been measured.
- Experiment: gridworlds with computable bisimulation metrics, Procgen, and DMC with distractors; correlate feature distance with the true metric over training; relate to generalization gap.
- Kill test: feature distances are unrelated to the bisimulation metric.
- Compute: small. One to two weeks.

**14. How linear is a trained policy?**
- Hypothesis: distilling a deep policy into a linear one at checkpoints gives a linear-fit R² over training. On many DMC and locomotion tasks R² is high throughout and the nonlinearity is used only in a small set of states (contact, recovery). Mapping where the nonlinearity is load-bearing predicts which tasks small or quantized policies can solve and where to spend capacity.
- Gap: linear policies were shown competitive on MuJoCo in 2017; nobody has measured how much of a deep policy is linear as a function of training time and state.
- Experiment: SAC and PPO on DMC, Playground, MyoSuite; linear distillation per checkpoint; per-state residual maps; correlate with return of linear-only and hybrid policies.
- Kill test: R² is low everywhere, or the residual is spread uniformly over states.
- Compute: GPU-parallel. One week.

**15. Actor–critic feature misalignment predicts instability.**
- Hypothesis: the actor and critic rely on different observation dimensions (measured by input-gradient attribution). Where the critic's value depends on features the actor ignores, the actor cannot follow the critic's gradient and learning stalls or oscillates; misalignment spikes precede return drops. Sharing or distilling the critic's attribution into the actor's input layer removes the stalls.
- Gap: shared vs separate encoders is studied for performance; the feature-level agreement between the two networks has not been measured or linked to instability.
- Experiment: SAC on DMC and MyoSuite; per-dimension attribution for Q and π over training; alignment score vs return drops; test attribution-matching regularizer.
- Kill test: alignment is uninformative about instability.
- Compute: DMC. One week.

**16. Do recurrent policies represent uncertainty or just the mode?**
- Hypothesis: on POMDPs the Bayes-optimal agent tracks a belief; trained GRU and Transformer policies instead track a point estimate of the hidden variable and discard the belief's entropy. A linear probe recovers the mode but not the belief entropy, and the agent's failures concentrate exactly in states where the true belief is multimodal. An auxiliary loss on belief entropy fixes those failures.
- Gap: probing recurrent RL agents for hidden variables is common; probing for the uncertainty about those variables, and linking it to failure, is not.
- Experiment: POMDPs with computable beliefs (T-maze variants, noisy MinAtar, partially observed gridworlds); probe hidden states for the belief mean and entropy; locate failures; test the auxiliary loss.
- Kill test: belief entropy is recoverable from hidden states, or failures are not concentrated where belief is multimodal.
- Compute: tiny. One week.

**17. Emergent temporal abstraction in flat policies.**
- Hypothesis: flat policies develop temporally extended behavior (long runs of similar actions, measured by action autocorrelation time and by the compressibility of action sequences) over training, and the degree of abstraction predicts transfer to tasks with longer horizons and robustness to action delay. Hierarchical methods buy little when the flat policy has already abstracted.
- Gap: set 2 idea 24 measures how much a policy depends on observations; this measures the temporal structure of its actions regardless of observation use, and relates it to when hierarchy helps.
- Experiment: DMC, Craftax, Playground; action autocorrelation time and sequence compressibility per checkpoint; relate to horizon transfer, delay robustness, and the flat-vs-hierarchical gap.
- Kill test: action sequences are white, or abstraction does not predict the hierarchy gap.
- Compute: GPU-parallel. One week.

## C. Data, replay, and reward

**18. Domain randomization at replay time via re-simulation.**
- Hypothesis: in a cloneable simulator, stored (s, a) pairs can be re-stepped under perturbed physics to produce augmented s′ at replay time, decoupling randomization from data collection. Replay-time randomization gives the same robustness as collection-time randomization with far less exploration damage (the agent explores in nominal physics), and the randomization range can be adapted per transition.
- Gap: the cloning family in lists 1 and 2 uses branching for advantages, bonuses, and credit. Using re-simulation as a data augmentation for off-policy critics has not been tried; all DR today happens at collection.
- Experiment: Playground/MJX tasks with save/restore; replay-time DR vs collection-time DR vs none at matched steps; nominal return and return under held-out physics shifts.
- Kill test: replay-time DR gives no robustness, or costs as much nominal return as collection-time DR.
- Compute: GPU-parallel. One to two weeks.

**19. Coarse rewards: quantizing shaped rewards helps.**
- Hypothesis: quantizing a continuous shaped reward to a few levels (3 to 8) per step speeds learning on many tasks because it removes low-amplitude noise the critic would otherwise fit, while preserving the ordering that drives improvement. The optimal number of levels is predictable from the reward's signal-to-noise along trajectories.
- Gap: set 2 idea 4 quantizes the advantage inside the actor update. This quantizes the reward signal itself before any learning, which also changes the critic's target, and nobody has tested it.
- Experiment: Playground and DMC with shaped rewards; levels ∈ {2, 3, 5, 8, continuous} with quantile-based bins; sparse task success and return; relate the best level count to reward SNR.
- Kill test: continuous reward is best everywhere.
- Compute: GPU-parallel. Days.

**20. Reward-term decorrelation as a hacking alarm.**
- Hypothesis: in multi-term shaped rewards the shaping terms are correlated with the sparse task term under intended behavior. When the agent finds an exploit, the correlation between shaping terms and the task term collapses, often before return changes, and the term whose correlation collapses first identifies the exploit. A monitor on these correlations flags hacking online with no labeled examples.
- Gap: list 1 idea 30 detects hacking from the shape of the return distribution; this uses the reward decomposition, which is always available in robotics and gives attribution to the exploited term.
- Experiment: Playground humanoid and quadruped tasks with injected exploits (e.g., a shaping term that can be satisfied without the task); log per-term correlations; detection latency and attribution accuracy.
- Kill test: correlations do not change before return does.
- Compute: GPU-parallel. One week.

**21. Two kinds of sparsity: rarity versus distance.**
- Hypothesis: "sparse reward" conflates reward that is rare under the current policy (low probability per episode) with reward that is far in time from the decisions that cause it (long credit distance). Exploration methods address rarity; credit-assignment methods address distance. A 2D benchmark grid with both dials independently controlled shows that each method family helps on only one axis, and that the hardest published tasks sit on different diagonals than assumed.
- Gap: benchmark papers rate tasks by one sparsity scalar; the two axes have not been separated, and the ADEPT delay stubs address only the distance axis.
- Experiment: procedural gridworld and MinAtar families where rarity and distance are set independently; run RND, counts, n-step, trace methods, and a transformer critic; a method × axis table.
- Kill test: all methods degrade identically along both axes.
- Compute: tiny. One to two weeks.

## D. Exploration

**22. Null-space exploration.**
- Hypothesis: exploration noise should be concentrated in action directions the critic says do not matter (small ∂Q/∂a), because those are the directions where the critic has no information; noise along the gradient direction only slows exploitation. Projecting Gaussian noise onto the null space of the critic's action gradient gives better coverage of unknown action effects and faster learning than isotropic noise, especially in high-DoF control.
- Gap: exploration noise is isotropic, parameter-space, or temporally correlated; its direction relative to the critic's gradient has not been used.
- Experiment: SAC and TD3 on MyoSuite and Shadow Hand; isotropic vs null-space-projected vs gradient-aligned noise at matched entropy; return and coverage of action space.
- Kill test: direction of the noise does not matter.
- Compute: GPU-parallel. One week.

**23. Random networks as macro-actions.**
- Hypothesis: sampling a small randomly initialized network and following it as the policy for k steps gives state-dependent, temporally extended, and structured exploration that outperforms OU, pink noise, and parameter noise on sparse tasks, because random networks produce coherent behaviors (sweeps, oscillations) rather than drift. The distribution of random networks (init scale, depth) is a cheap exploration prior.
- Gap: temporally correlated noise is the current best practice; using random function approximators as temporary behavior policies has not been studied.
- Experiment: sparse DMC, Craftax, MinAtar; random-network macro-actions vs pink noise vs parameter noise vs ε-greedy; time-to-first-reward and final return; sweep k and init scale.
- Kill test: random-network exploration is no better than pink noise.
- Compute: small. One week.

**24. Buffer transplant: decomposing algorithm differences into exploration and learning.**
- Hypothesis: when algorithm A beats B, the gap can be split into "A went to better places" and "A learned more from where it went." Transplanting A's early buffer into B (and vice versa) and continuing training measures each share. For most published gaps on sparse tasks the exploration share dominates; on dense tasks the learning share does.
- Gap: algorithm comparisons report return; nobody has run the swap experiment that separates data from learner.
- Experiment: SAC, TD3, PPO-with-buffer, and a world-model agent on DMC and Craftax; swap buffers at several points; 2 × 2 decomposition per task.
- Kill test: transplanting buffers has no effect in either direction.
- Compute: small. One to two weeks.

**25. First-success lock-in.**
- Hypothesis: the first rewarded trajectory an agent experiences disproportionately determines its final strategy. Intervening on that single trajectory (replacing it with a different successful trajectory of equal return, or deleting it) changes the final behavior mode in a large fraction of seeds. Lock-in strength is a task property that predicts inter-seed strategy variance and whether a seed can escape a suboptimal mode.
- Gap: list 1 idea 26 clusters what seeds learn; this is the causal intervention on the single transition sequence that may decide the cluster.
- Experiment: Craftax, Procgen, and multi-solution gridworlds; detect first success; branch runs with the trajectory kept, deleted, or swapped; measure final-mode divergence.
- Kill test: the intervention does not change final behavior.
- Compute: small. One week.

## E. Control and robotics

**26. Algorithm-induced behavioral bias.**
- Hypothesis: at matched return, policies trained by different algorithms have systematically different behavior (energy, smoothness, contact frequency, risk, gait), and a classifier can identify the training algorithm from a trajectory. The biases are explained by mechanism (entropy regularization → jitter, clipped double-Q → caution, flow policies → multimodal gaits) and matter for deployment more than the return difference does.
- Gap: algorithms are compared by return; the behavior they produce at equal return has not been characterized, and "algorithm fingerprinting" has not been attempted.
- Experiment: SAC, TD3, PPO, REPPO, and a flow policy on Playground locomotion and manipulation; match return; compute behavioral statistics; train a classifier; attribute biases to mechanisms by ablation.
- Kill test: behaviors are indistinguishable at matched return.
- Compute: GPU-parallel. One to two weeks.

**27. Sobolev critics from differentiable simulators.**
- Hypothesis: MJX and Brax provide analytic ∂r/∂a and ∂s′/∂a. Supervising the critic's action gradient ∇ₐQ with a simulator-derived estimate (Sobolev training) gives the actor a correct ascent direction long before TD targets converge, and avoids the gradient-variance problems that limit first-order policy methods like SHAC, because the sim gradient is used as a regression target rather than propagated through the trajectory.
- Gap: differentiable sims are used for pathwise policy gradients; using their gradients to shape the critic rather than the policy is largely untried.
- Experiment: SAC with and without Sobolev loss on MJX tasks of increasing contact-richness; return, actor-gradient cosine with ground truth, and robustness to sim-gradient noise.
- Kill test: the Sobolev critic is no better, or breaks under contact.
- Compute: GPU-parallel. Two weeks.

**28. Shield reliance.**
- Hypothesis: agents trained under a safety filter learn to depend on it (they drive toward the constraint and let the filter catch them), so unshielded performance decays over training while shielded performance improves. Shield dropout (removing the filter on a random fraction of steps with a penalty) produces policies that are safe with and without the shield at a small return cost.
- Gap: safe RL with shields reports shielded performance; the learned dependence on the shield and its growth over training have not been measured.
- Experiment: Safety Gymnasium-style tasks and Playground with velocity/joint constraints; shielded vs unshielded evaluation per checkpoint; shield dropout sweep.
- Kill test: unshielded safety does not degrade.
- Compute: GPU-parallel. One week.

**29. Safety is forgotten first.**
- Hypothesis: when a constrained policy is fine-tuned on a new reward or dynamics, constraint satisfaction degrades before return changes, because the constraint-relevant features are the first to be overwritten. The forgetting order is predictable from how concentrated the constraint-relevant features are in the network, and protecting a small constraint subspace (low-rank freeze) preserves safety through fine-tuning.
- Gap: forgetting during fine-tuning is measured on return; the ordering between return and constraint forgetting has not been studied.
- Experiment: train constrained policies; fine-tune under reward or dynamics shift; log constraint violation and return per step; identify constraint-relevant subspace by attribution; test the low-rank freeze.
- Kill test: constraints and return degrade together, or the freeze does not help.
- Compute: GPU-parallel. One to two weeks.

**30. How often does a policy need to look?**
- Hypothesis: training with randomized observation staleness (the agent sees a fresh observation only every k steps and acts on stale ones in between) reveals a per-task minimum observation rate below which return collapses, which is a property of the task's dynamics (Lyapunov time) rather than of the policy. The required rate predicts sensor and compute budgets for deployment and is far below the simulator rate on most tasks.
- Gap: set 2 idea 47 estimates how long a memory window a task needs; this measures how often a task needs fresh information, a distinct quantity relevant to sensor and compute scheduling.
- Experiment: DMC and Playground; staleness k ∈ {1, 2, 4, 8, 16, 32}; train and test matrices; relate the collapse point to the dynamics' Lyapunov exponent.
- Kill test: the collapse point is unrelated to dynamics and differs across policies trained on the same task.
- Compute: GPU-parallel. One week.

## F. Model-based RL

**31. The perfect-model gap.**
- Hypothesis: replacing the learned world model in TD-MPC2 or Dreamer with the real simulator (cloneable, so imagination and planning run in the true dynamics) gives an upper bound on what the algorithm can achieve with a perfect model. The gap between the perfect-model agent and the learned-model agent measures how much is lost to model error versus to the planning and learning machinery, and it is small on many benchmarks, implying model quality is no longer the bottleneck.
- Gap: model-based RL papers improve the model or the planner without an oracle to say which one is limiting. The oracle is free in GPU simulators and has not been used this way.
- Experiment: TD-MPC2 and small Dreamer on DMC and Playground; learned model vs sim-as-model vs sim-as-model with injected noise of controlled size; return and planning behavior.
- Kill test: the perfect model gives a large gain everywhere.
- Compute: GPU-parallel. Two weeks.

## G. Multi-agent

**32. Correlated exploration noise is a hidden coordination channel.**
- Hypothesis: in parameter-shared MARL, agents that share a random seed or sample exploration noise from correlated streams coordinate through the noise itself; when deployed with independent sampling or with a different partner, coordination falls. Part of the reported gap between shared and independent parameters, and part of the zero-shot coordination problem, is this noise correlation.
- Gap: exploration noise in MARL is treated as independent per agent and never audited; the coordination effect has not been identified.
- Experiment: JaxMARL cooperative tasks; shared vs independent noise streams during training; evaluate under independent noise, deterministic actions, and cross-play; measure the correlation of actions across agents at matched states.
- Kill test: noise correlation has no effect on coordination.
- Compute: small. One week.

**33. Inaction baselines for multi-agent credit.**
- Hypothesis: the counterfactual "what if this agent had done nothing" (a no-op or a default policy) is a cheaper and more interpretable baseline than COMA's marginalization and, in cloneable environments, is exact. Inaction credit identifies free-riders, correlates with human judgments of contribution, and trains policies that match or beat learned counterfactual baselines.
- Gap: list 1 idea 43 computes exact marginal counterfactuals over all actions; the inaction counterfactual is a single, interpretable branch that has not been studied as a baseline.
- Experiment: JaxMARL Overcooked and SMAX; inaction baseline vs COMA vs MAPPO; agreement with hand-labeled contribution on a small set.
- Kill test: inaction credit is no better than a shared baseline.
- Compute: GPU-parallel. One to two weeks.

## H. RL for language models

**34. RL gain as a function of pretraining exposure.**
- Hypothesis: pretrain 20–100M parameter LMs from scratch on synthetic corpora where the frequency of a target task (e.g., modular arithmetic, graph reachability) is controlled exactly, then run RLVR. The RL gain is a non-monotone function of exposure: zero at no exposure (nothing to sharpen), maximal at low-to-moderate exposure, and small at high exposure (already solved). This gives a controlled answer to whether RL creates capability or only amplifies it.
- Gap: the sharpening-vs-creating debate is run on pretrained models with unknown data; a from-scratch controlled study is feasible only at small scale and has not been done.
- Experiment: synthetic corpora with task frequency ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}; pretrain; RLVR with GRPO; pass@1 and pass@k before and after; repeat across two tasks.
- Kill test: RL gain is monotone in exposure or constant.
- Compute: 20–100M models. Two to three weeks.

**35. The RLVR size floor.**
- Hypothesis: there is a minimum model size below which RLVR on a given task fails regardless of compute, and the floor is set by base pass@k at large k: if the base model cannot find a correct answer in k samples, no group-based method can learn. The floor shifts with task difficulty and can be lowered by curriculum but not by more rollouts.
- Gap: RLVR studies use 0.5B and up; the small end, where the method breaks, has not been mapped, and it is where the mechanism is visible.
- Experiment: 30M–1.5B models (Pythia and SmolLM families) on three procedural tasks; measure base pass@k up to k = 1024; GRPO at matched compute; locate the floor per task.
- Kill test: all sizes learn, or the floor is unrelated to pass@k.
- Compute: 30M–1.5B LoRA. Two weeks.

**36. Rewarding lucky guesses.**
- Hypothesis: a measurable fraction of correct answers in RLVR come with wrong or incomplete reasoning (lucky guesses), detectable on procedural tasks where intermediate states are checkable. Rewarding them teaches the model to guess; filtering them (or down-weighting) at the same rollout budget improves held-out accuracy and reduces the pass@k collapse.
- Gap: step-level rewards (list 1 idea 39) use intermediate correctness for credit. This asks a different question: whether outcome-correct, process-wrong samples actively hurt, and how prevalent they are.
- Experiment: arithmetic, graph, and logic tasks with step checkers; measure lucky-guess rate over training; GRPO with and without lucky-guess filtering; held-out accuracy and pass@k.
- Kill test: lucky guesses are rare or harmless.
- Compute: 0.5–1B. One to two weeks.

**37. Where in depth is the answer decided, before and after RL?**
- Hypothesis: RLVR moves the layer at which the final answer becomes linearly decodable (logit lens) earlier in depth and shifts attention from the problem statement toward the model's own reasoning tokens. The size of these shifts tracks the RL gain per task and distinguishes tasks where RL changed computation from tasks where it changed only formatting.
- Gap: mechanistic analyses of RLVR focus on which parameters change; the change in internal computation (depth of decision, attention allocation) has not been measured.
- Experiment: base vs RLVR 0.5–1.5B on three procedural tasks; logit-lens decodability per layer, attention mass to prompt vs reasoning per layer; relate to accuracy gain.
- Kill test: no shift in decision depth or attention.
- Compute: 0.5–1.5B. One week.

**38. Exposure bias in multi-turn RL.**
- Hypothesis: multi-turn agent training (text games, tool use) conditions turn t on the model's own earlier turns during rollouts but many pipelines compute gradients only on the last turn or use teacher-forced histories. Agents trained with teacher-forced histories fail to recover from their own early mistakes; agents trained fully on-policy recover but learn slower. A scheduled mixture (as in scheduled sampling) beats both.
- Gap: exposure bias is a classic sequence-modeling result; its multi-turn RL analog has not been measured.
- Experiment: TextWorld and BabyAI-text; teacher-forced vs on-policy vs scheduled histories; success rate, recovery rate after planted early errors.
- Kill test: teacher-forced and on-policy training produce the same recovery behavior.
- Compute: 0.5–1B. Two weeks.

**39. Restating the problem as an action.**
- Hypothesis: giving the model a reward-free restatement step before solving (a canonicalization action, rewarded only through the final answer) causes RL to discover canonical forms that make the solving step easier, and the learned restatements transfer: a different model solving the restated problems is more accurate than on the originals.
- Gap: prompting work uses hand-written "restate the problem" instructions; letting RL shape the restatement and testing its transfer to another model is new.
- Experiment: procedural word problems and graph descriptions with paraphrase variation; restate-then-solve vs solve-only at matched rollouts; transfer restatements to a frozen second model.
- Kill test: restatement does not help or does not transfer.
- Compute: 0.5–1B. One to two weeks.

**40. Groups sampled without replacement.**
- Hypothesis: GRPO samples the group i.i.d., so groups are often redundant (G near-identical completions). Diversity-enforcing sampling (arithmetic or stratified sampling over the sequence distribution, or sampling-without-replacement approximations) yields groups with higher reward variance at equal cost, better advantage estimates, and faster learning, with the largest gain at small G.
- Gap: set 2 ideas 34 and 37 change the group's temperature and prompts; the sampling procedure for the group itself has not been varied.
- Experiment: three procedural tasks; i.i.d. vs stratified vs without-replacement sampling at G ∈ {2, 4, 8, 16}; within-group reward variance and accuracy.
- Kill test: no gain at any G.
- Compute: 0.5–1B. One week.

**41. Does RLVR compose?**
- Hypothesis: training on two procedural skills separately (A and B) produces a model that solves problems requiring A∘B at a rate predictable from the two single-skill accuracies only when the skills were interleaved in training; blocked training gives below-product composition. Composition is the clean generalization test the RLVR literature lacks.
- Gap: list 1 idea 49 studies interference between domains; this studies composition of skills into unseen combined problems, which is a different test and the one that matters for "reasoning."
- Experiment: skills with composable checkers (e.g., arithmetic ∘ graph reachability); train A, B, A+B interleaved, A then B; test on A∘B; measure composition gap.
- Kill test: composition tracks the product of single-skill accuracies regardless of training.
- Compute: 0.5–1B. Two weeks.

## I. Evaluation and methodology

**42. Wrapper sensitivity: how much do environment conventions move the leaderboard?**
- Hypothesis: a handful of implementation conventions (terminal vs truncation handling, observation and action clipping, reward clipping, frame-skip phase, reset-noise scale) shift reported returns by more than many published method differences. A sensitivity audit across these knobs on standard suites quantifies how much of the leaderboard is convention.
- Gap: implementation-matters papers study algorithm code; the environment-side conventions have not been audited systematically.
- Experiment: SAC and PPO on DMC and Gymnasium MuJoCo with a factorial over 6 wrapper conventions; report the effect size of each relative to the inter-method spread of recent papers.
- Kill test: wrapper effects are small relative to method differences.
- Compute: small. One to two weeks.

**43. Stratified evaluation: cut evaluation cost by sampling where policies disagree.**
- Hypothesis: most evaluation episodes are wasted on start states where all candidate policies behave identically. Stratifying start states by the disagreement between policies (or by return variance under one policy) and evaluating preferentially there gives the same ranking with 5–10× fewer episodes, and the saved budget can go to more seeds.
- Gap: active evaluation exists for supervised models; RL evaluation samples start states uniformly from ρ₀.
- Experiment: candidate policies on DMC and Playground; uniform vs stratified vs disagreement-weighted evaluation; ranking fidelity vs episode budget.
- Kill test: stratification gives no reduction in episodes for the same ranking fidelity.
- Compute: GPU-parallel. Days.

**44. Seeds versus evaluation episodes.**
- Hypothesis: for a fixed evaluation budget, the optimal split between more seeds and more evaluation episodes per seed is determined by the ratio of between-seed to within-seed return variance, which is a task property that can be estimated from a pilot. Most papers use far too many episodes per seed and too few seeds, and the misallocation inflates false-positive rates.
- Gap: list 1 idea 32 decomposes seed variance by source; this is the allocation decision under a fixed budget, with a pilot-based rule and a false-positive measurement.
- Experiment: 50+ seeds × 100 episodes on gymnax, Craftax, DMC; estimate variance components; simulate paper-style comparisons at various splits; false-positive and power curves.
- Kill test: within-seed variance dominates so that episodes are the right spend.
- Compute: many small runs. One week.

## J. Offline RL

**45. Start-state generalization in offline RL.**
- Hypothesis: offline RL methods are evaluated from the dataset's own start distribution; evaluating from start states outside the dataset's coverage (but reachable) separates methods that learned a value function over the state space from methods that learned to follow the data. In-sample methods (IQL) extrapolate better in state space than BC-regularized ones (TD3+BC), and the ordering flips relative to standard scores.
- Gap: the ADEPT data-amount stubs vary dataset size; this varies the evaluation start distribution, which no offline benchmark does, and it exposes a different failure.
- Experiment: D4RL and OGBench with controlled out-of-coverage start states (perturbed initial configurations, mid-trajectory restarts); run 5 offline methods; compare rankings.
- Kill test: rankings are unchanged from out-of-coverage starts.
- Compute: MuJoCo offline. One to two weeks.

## K. Cross-cutting

**46. The curse of context: extra history hurts on Markov tasks.**
- Hypothesis: giving a policy more observation history than the task needs degrades performance, not just efficiency, because the extra context admits spurious temporal correlations that the policy overfits; the degradation grows with context length and shrinks with data, which is why transformer policies underperform MLPs on standard DMC. A context-length penalty or a learned context gate removes it.
- Gap: set 2 idea 47 estimates how much memory a task needs; this measures the cost of exceeding it, which is relevant to the trend toward transformer policies on fully observed tasks.
- Experiment: DMC and Playground (Markov); frame-stack and transformer policies with context ∈ {1, 2, 4, 16, 64}; return vs context and vs data; test the gate.
- Kill test: longer context never hurts.
- Compute: GPU-parallel. One week.

**47. Do exploration bonuses create habits?**
- Hypothesis: after an intrinsic bonus is removed, the policy continues to exhibit bonus-seeking behavior (visiting novel-looking states) for a long time because the behavior has been baked into the actor and critic, and this residual exploration costs return on the task. The habit's half-life depends on the bonus scale and on whether the bonus entered the critic or the actor; delivering the bonus only to the actor's objective avoids the habit.
- Gap: set 2 idea 11 measures wasted exploration while the bonus is active; this measures what persists after it is removed, which matters for the standard practice of annealing bonuses.
- Experiment: RND and counts on MinAtar and Craftax; remove the bonus at checkpoints; measure residual novelty-seeking and return over the following steps; compare critic-bonus vs actor-only-bonus.
- Kill test: behavior returns to the extrinsic optimum immediately.
- Compute: small. One week.

**48. Shaping ruggedness: how many local optima does a dense reward create?**
- Hypothesis: dense shaped rewards create a landscape over policies whose number of distinct local optima (measured by clustering the endpoints of many short independent runs) is a property of the reward design, and it predicts the fraction of seeds that reach the sparse goal better than the shaping's correlation with the goal does. A ruggedness number lets you compare reward designs before full training.
- Gap: reward design is evaluated by final return of a few seeds; a landscape-level statistic from cheap short runs has not been proposed.
- Experiment: Playground tasks with several shaping variants; 100+ short runs per variant; cluster endpoints; relate the mode count to full-training success rate; validate on held-out shaping designs.
- Kill test: mode count is unrelated to success rate.
- Compute: many short GPU-parallel runs. One week.

**49. The reset subsidy.**
- Hypothesis: episodic resets contribute to learning even on tasks that do not need them (continuing locomotion) by repeatedly placing the agent in informative states; the size of the subsidy (return gap between reset-every-T and reset-never at matched steps) is large on some tasks and zero on others, and is predicted by how fast the state distribution under the current policy drifts from ρ₀. Reset-free RL methods should be evaluated against this subsidy, not against episodic baselines.
- Gap: reset-free RL is studied as a separate problem; the contribution of resets to standard episodic learning has not been quantified as a baseline quantity.
- Experiment: DMC and Playground locomotion; T ∈ {100, 1000, 10000, ∞}; return at matched steps; relate the subsidy to the drift of the state distribution.
- Kill test: the subsidy is zero everywhere.
- Compute: GPU-parallel. One week.

**50. Hyperparameter drift measured by forking.**
- Hypothesis: the optimal learning rate, UTD, and entropy coefficient change over the course of a single run. Forking a run at each checkpoint into short branches with different hyperparameters and measuring which branch improves most gives the optimal-hyperparameter trajectory directly. The trajectory is consistent across seeds and tasks in shape (e.g., UTD should rise then fall), and following it beats the best fixed setting.
- Gap: hyperparameter schedules in RL are hand-designed or absent; the forking measurement, which is cheap on GPU simulators, has not been used to reveal what the schedule should be.
- Experiment: SAC and PPO on Playground; fork every 5% of training into 5 branches per hyperparameter for a short window; record the winner; build the trajectory; validate a schedule derived from it on held-out tasks.
- Kill test: the optimal hyperparameters are constant over training.
- Compute: GPU-parallel. Two weeks.

---

## How to pick

Rank by how clean the yes/no is, how little infrastructure it needs, and how surprised the field would be. By that ranking the strongest first picks are **1, 2, 5, 10, 12, 19, 24, 32, 34, 36, 42, 43**.

Groups that share infrastructure:
- Ideas 5, 18, 31, and 33 all need save/restore and RNG control in a GPU simulator, the same harness as the cloning family in lists 1 and 2.
- Ideas 1, 2, 4, 10, and 11 are cheap diagnostics on SAC and DQN and could be one "optimizer and sampling defaults in TD learning" paper if several effects are real.
- Ideas 12, 16, 30, and 46 all concern partial observability and context and share POMDP testbeds.
- Ideas 28 and 29 share constrained tasks and a safety-evaluation harness.
- Ideas 34, 35, 36, 40, and 41 share the same procedural task generators and checkers.
- Ideas 43, 44, and 48 are all evaluation-budget methodology and share a large pilot of seeds × episodes.

Ideas most at risk of having been done recently; search before committing: 6 (state-distribution trust regions), 13 (emergent bisimulation), 27 (Sobolev or gradient-supervised critics with differentiable sims), 35 (RLVR scaling to very small models), 37 (mechanistic analyses of RLVR), 40 (diverse sampling for GRPO), 46 (context length in Markov tasks).
