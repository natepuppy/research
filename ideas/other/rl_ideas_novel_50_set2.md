# 50 more novel RL research ideas (set 2)

A third list, disjoint from `rl_ideas_50.md` (follow-ups to recent papers), `rl_ideas_novel_50.md` (mechanisms and measurements), and the ADEPT stubs (dead-ends, irreversibility, empowerment, backtracking, pretraining transfer, action quantization, offline data amount, single-network uncertainty, untrusted graders, delayed rewards, environment design, borrowing experience, RLVR rigidity, distillation).

Same caveats as before. Novelty is as of my knowledge; run a literature search on each before committing, and I flag the ones most at risk at the end. "NeurIPS candidate" means the paper has a mechanism or a measurement, a falsifiable claim, and a result that is informative either way.

Everything here fits one RTX 3080 (10 GB). Control ideas use gymnax, MinAtar, Craftax, DMC, MyoSuite, Mujoco Playground or JaxMARL. LLM ideas use 0.5–1.5B models with LoRA, GRPO-style training, short contexts, and procedural data.

Format for each: **Hypothesis**, **Gap**, **Experiment**, **Kill test**, **Compute**.

---

## A. Initialization, numerics, and optimizer behavior

**1. Initialization is an exploration hyperparameter.**
- Hypothesis: the critic's output bias at init, the actor's last-layer scale, and the sign convention of the reward (per-step penalty vs terminal bonus) jointly set the agent's implicit optimism. Much of the inter-task variance in "exploration difficulty" is really an interaction between reward sign and default init, and a zero-cost rule (set the output bias to the pessimistic bound for positive-reward tasks and the optimistic bound for negative-reward tasks) removes it.
- Gap: optimistic initialization is tabular folklore; nobody has measured how deep-net defaults interact with reward sign on standard benchmarks or offered a rule.
- Experiment: sparse MinAtar, Craftax, DMC sparse variants; sweep output-bias init × reward sign × actor last-layer scale; log coverage in the first 50k steps and final return.
- Kill test: init has no effect on time-to-first-reward beyond seed noise.
- Compute: tiny. One week.

**2. Numerical precision is an RL hyperparameter.**
- Hypothesis: late in training TD errors are small relative to Q magnitudes, so bf16 and fp16 critics lose the update signal entirely (the target and prediction round to the same value). Mixed precision that is safe in supervised learning silently stalls RL, and the stall point depends on γ and reward scale. Storing the TD residual in fp32 (or Kahan-style compensated targets) fixes it for free.
- Gap: precision studies exist for LLM training; none for bootstrapped targets, where the quantity of interest is a small difference of two large numbers.
- Experiment: SAC and DQN on DMC and MinAtar in fp32, bf16, fp16, and compensated-bf16; log the fraction of zero TD errors over training and the return.
- Kill test: bf16 matches fp32 at every γ.
- Compute: tiny. Days.

**3. Entropy collapse is hysteretic.**
- Hypothesis: once a policy's entropy collapses, raising the entropy coefficient does not restore exploration, because the actor's pre-activation weights have grown large and the critic has become flat around the collapsed actions. There is a hysteresis loop in (coefficient, entropy) space, and its width is a measurable property of the task and the architecture.
- Gap: entropy collapse is treated as a tuning failure; nobody has shown it is irreversible or asked what makes it so.
- Experiment: SAC and PPO; drive the coefficient down then up on a slow schedule; plot entropy against coefficient; measure the loop area; test fixes (weight-norm on the actor's last layer, critic action-perturbation).
- Kill test: entropy tracks the coefficient with no loop.
- Compute: DMC. One week.

**4. How many bits of the advantage does the actor use?**
- Hypothesis: on-policy policy gradient uses little more than the sign of the advantage. PPO trained on sign(A), or on A quantized to 2–3 levels, matches full PPO. If so, most of GAE's variance-reduction machinery is spent on information the update discards.
- Gap: the ranking-critic idea in the other list is about the critic's loss. This is about the actor's update, in on-policy methods, and it is a one-line ablation nobody has reported.
- Experiment: PPO and REPPO on Playground and Procgen; A ∈ {full, sign, 3-level, rank-within-batch}; matched compute.
- Kill test: sign-only is clearly worse everywhere.
- Compute: GPU-parallel. Days.

**5. The n in n-step returns should depend on the transition's age.**
- Hypothesis: older transitions in the buffer are more off-policy, so long n-step returns computed from them carry more bias; fresh transitions can use long n safely. Setting n per transition as a decreasing function of age beats any fixed n and removes a hyperparameter.
- Gap: n-step is fixed or tuned; off-policyness-aware correction (Retrace, V-trace) uses importance ratios, not age, and is rarely used in SAC/TD3-style agents.
- Experiment: SAC and DQN on DMC and Atari-100k subset; n = f(age) with a few functional forms vs fixed n ∈ {1,3,5,10}; also measure the bias of n-step targets by age against Monte Carlo ground truth.
- Kill test: bias does not grow with age, or adaptive n is no better than the best fixed n.
- Compute: small. One to two weeks.

**6. Does μP hyperparameter transfer survive bootstrapping?**
- Hypothesis: maximal-update parameterization gives width-independent optimal learning rates in supervised learning. With a moving target and a nonstationary data distribution, the optimal learning rate under μP still drifts with width, and the drift is explained by the target network's lag. A corrected parameterization that also scales τ restores transfer.
- Gap: μP has been applied to LLMs and some vision models; a careful test on TD-trained critics, with the target-network interaction, is missing.
- Experiment: critics of width 64 → 2048 under standard and μP parameterizations; learning-rate sweeps at each width; measure whether the optimum moves; test τ scaling.
- Kill test: μP transfers perfectly as is.
- Compute: small nets. One to two weeks.

**7. The observation normalizer is a hidden nonstationarity.**
- Hypothesis: running mean/std normalizers keep changing the input distribution the network sees, long after the environment has stabilized, and this drift accounts for a measurable share of what looks like plasticity loss or late-training instability. Freezing the normalizer after a warm-up, or normalizing with a fixed quantile transform, removes it.
- Gap: normalizers are infrastructure nobody reports on. No paper measures how much the inputs drift because of the normalizer alone.
- Experiment: PPO and SAC; log the KL between normalized-observation distributions across checkpoints with the env fixed; compare running vs frozen vs quantile normalizers on Playground and DMC.
- Kill test: normalizer drift is negligible after warm-up, or freezing does not change anything.
- Compute: GPU-parallel. Days.

**8. Measure the policy-evaluation lag.**
- Hypothesis: after the actor changes, the critic needs a measurable number of updates to track Q^π for the new policy; during that window the actor is climbing a stale landscape. The lag depends on actor learning rate, UTD, and τ, and setting the actor's step size so the lag stays under a threshold beats fixed delayed updates.
- Gap: TD3's "delayed policy updates" is a fixed heuristic. The lag has never been measured as a function of training knobs.
- Experiment: freeze the actor at checkpoints, train the critic to convergence to get Q^π ground truth, then measure how many updates the normal training loop needed to get within ε. Build a lag-aware actor schedule.
- Kill test: lag is constant or does not predict instability.
- Compute: DMC. One to two weeks.

**9. Tanh saturation as a failure mode.**
- Hypothesis: in squashed Gaussian policies a large fraction of actions end up at ±1 (saturated), where the policy gradient through tanh is near zero. Saturation fraction over training predicts stuck policies, and a mild penalty or a different squashing (e.g., clipped linear with a reflected gradient) removes the failure.
- Gap: practitioners know policies saturate; there is no systematic measurement of saturation fraction across tasks or a demonstration that it causes stalls.
- Experiment: SAC and PPO on DMC, MyoSuite, Playground; log per-dimension saturation fraction; correlate with learning stalls; test alternatives.
- Kill test: saturation is rare or uncorrelated with stalls.
- Compute: GPU-parallel. One week.

## B. Exploration

**10. Exploration order, not exploration magnitude.**
- Hypothesis: two agents with identical novelty bonuses but different visiting orders (exhaust what is nearby and poorly understood before moving on, vs sweep breadth-first) learn at different rates, and which order wins depends on how reward correlates with distance from the start. The field tunes bonus magnitude and never studies order.
- Gap: this is the question in your own note ("should the agent visit every square it does not understand before traversing ones it does?"). Count-based and prediction-error bonuses induce an implicit order nobody has characterized.
- Experiment: gridworlds, MinAtar, Craftax; implement depth-first (local-uncertainty-first) and breadth-first (frontier-first) exploration policies on top of the same bonus; measure time-to-first-reward and final return; vary where reward lives.
- Kill test: order has no effect at matched bonus.
- Compute: tiny. One week.

**11. Exploration waste.**
- Hypothesis: a large fraction of bonus-driven visits go to states whose value is already accurate (re-exploration after a policy change resets the novelty of states the agent understood), and this fraction is the real cost of intrinsic motivation. A bonus gated by value-error, not visitation, cuts the waste.
- Gap: exploration is scored by coverage and return; nobody measures how many exploratory steps were unnecessary against a ground-truth value.
- Experiment: small envs with computable V*; tag every visit as useful (value error > ε) or wasted; report the waste fraction for RND, ICM, counts; test a value-error-gated bonus.
- Kill test: waste is low for existing methods.
- Compute: tiny. One week.

**12. Restart schedules for RL.**
- Hypothesis: on sparse-reward tasks the time to first success across seeds is heavy-tailed, so a Luby-style restart schedule (as in SAT solvers) finishes more runs per GPU-hour than running each seed to the budget. Restart-with-buffer-carryover is better still.
- Gap: restart theory is standard in combinatorial search; it has not been applied to RL seed scheduling, and first-success-time distributions are not reported.
- Experiment: measure first-success-time distributions on 10 sparse tasks with 50+ seeds; fit tails; compare fixed-budget, Luby, and geometric restart schedules at matched compute.
- Kill test: first-success times are light-tailed.
- Compute: many tiny runs. One week.

**13. State-dependent entropy budgeting.**
- Hypothesis: a trained max-ent policy concentrates its randomness in states where it costs little, and SAC's single global entropy target fights this. A per-state entropy target that follows the critic's action-sensitivity (idea 5 in the other list measures it; here it is used) gives more exploration where it matters and sharper actions where it does not.
- Gap: all entropy-regularized methods use one target. Where entropy lives at convergence has not been mapped.
- Experiment: map per-state entropy of converged SAC policies vs ‖∂Q/∂a‖; then train with a state-dependent target and compare return and exploration efficiency.
- Kill test: entropy is uniform across states, or the state-dependent target does not help.
- Compute: DMC. One to two weeks.

**14. The Bellman wavefront.**
- Hypothesis: value information propagates from terminal and rewarding states outward at a measurable speed in state space (a wavefront). Its speed, measured against ground truth, predicts time-to-learn per task better than horizon or state count, and tells you the n-step or start-state distribution that would accelerate it.
- Gap: "reward propagation" is a verbal explanation. Nobody measures the front's position over training or uses it as a diagnostic.
- Experiment: gridworlds and low-dimensional DMC with ground-truth V*; at each checkpoint compute the set of states with value error < ε; track its boundary's distance from the goal; relate speed to n-step, γ, and buffer size.
- Kill test: the accurate region does not grow outward from rewards.
- Compute: tiny. One to two weeks.

## C. Data, replay, and offline datasets

**15. The hidden curriculum in replay-derived datasets.**
- Hypothesis: datasets generated by a learning agent (D4RL "replay" and "medium-replay", your own buffers) carry a chronological order that is a curriculum. Training offline RL in chronological order, or with time-weighted sampling, beats shuffled training; and the same holds when a fresh online agent is trained on the ordered log of a previous run.
- Gap: offline RL treats the dataset as a bag of transitions. The temporal structure has never been used or measured.
- Experiment: D4RL and OGBench replay datasets, plus self-generated logs; chronological, reverse-chronological, shuffled, and time-weighted sampling; IQL and TD3+BC.
- Kill test: ordering does not matter.
- Compute: MuJoCo offline. One week.

**16. Dataset mode count predicts which offline RL method wins.**
- Hypothesis: the number of distinct behavior policies that generated a dataset (estimable by clustering state-conditioned action distributions) predicts the ranking of offline methods: single-mode datasets favor BC-regularized methods, multi-mode favor in-sample maximization (IQL) or generative policies. A one-number dataset statistic chooses the algorithm.
- Gap: offline RL benchmarks report dataset quality as return; the mixture structure is not measured or linked to method choice.
- Experiment: construct datasets with controlled mode count from 1 to 8 behavior policies; run 5 offline methods; regress ranking on mode count; validate on D4RL.
- Kill test: ranking is unaffected by mode count.
- Compute: MuJoCo offline. Two weeks.

**17. Data attribution for RL.**
- Hypothesis: gradient-based influence (TracIn-style) can attribute a behavior in the final policy to the transitions that produced it. In reward-hacking cases a small set of transitions carries most of the influence, and removing them during training removes the hack while keeping the return.
- Gap: data attribution is active in supervised and LLM settings; in RL, where the data is self-generated and the target moves, it has barely been tried, and it has not been used to debug behavior.
- Experiment: MinAtar and Procgen with injected exploits; compute influence of buffer transitions on the critic's value of exploit states; remove top-k; measure hack rate and return.
- Kill test: influence is diffuse.
- Compute: small nets. Two weeks.

**18. Hot-swapping the reward function mid-training.**
- Hypothesis: when a designer edits reward weights during training (the normal workflow in robotics), relabeling the buffer with the new reward and continuing beats restarting, and the critic's recovery time is predictable from how much the relabeled targets changed. A "reward diff" tool that reports the expected recovery cost would change how people iterate.
- Gap: iterating on reward design is universal practice and has no study; everyone restarts.
- Experiment: Playground tasks with multi-term rewards; apply edits of increasing size at various points; compare restart, continue-without-relabel, continue-with-relabel; model recovery time.
- Kill test: restarting is always as good.
- Compute: GPU-parallel. One week.

**19. Bursty vs steady updates at fixed UTD.**
- Hypothesis: collecting 10k steps then doing 10k updates is not the same as interleaving 1:1, even at the same UTD. Bursty schedules act like fitted Q-iteration and are more stable at high UTD but worse at exploration; the optimal burst length depends on how fast the policy changes per update.
- Gap: UTD is studied as a ratio; the temporal arrangement of the same ratio has not been.
- Experiment: SAC on DMC and Playground; burst length ∈ {1, 10, 100, 1k, 10k} at UTD ∈ {1, 4, 16}; return and stability.
- Kill test: return depends only on UTD.
- Compute: GPU-parallel. One week.

**20. Where does critic-ensemble diversity come from?**
- Hypothesis: disagreement among ensemble critics is driven mostly by initialization early and by data order late, and the two sources have different value: init diversity helps exploration, data-order diversity helps pessimism. Cheaper ensembles (shared trunk, diverse heads, or dropout) reproduce one source but not the other, which is why they work in some papers and not others.
- Gap: ensembles are used everywhere; the origin of their disagreement has not been decomposed.
- Experiment: train ensembles with shared vs independent init and shared vs independent batches; measure disagreement over training; test each configuration for exploration (sparse tasks) and for offline pessimism.
- Kill test: all configurations behave the same.
- Compute: small. One to two weeks.

## D. Control and robotics

**21. Value discontinuities at contact.**
- Hypothesis: in contact-rich tasks Q is nearly discontinuous across contact boundaries, and smooth critics (spectral-norm, Lipschitz-regularized, hyperspherical) systematically blur them, which shows up as poor fine manipulation. A critic with a contact-gated mixture head recovers the discontinuity.
- Gap: smoothness regularizers are applied globally; where Q is actually discontinuous has not been mapped.
- Experiment: MyoSuite and Playground manipulation with contact flags available; measure ‖∇_s Q‖ on either side of contact events; compare smooth critics vs gated critics on fine manipulation success.
- Kill test: no discontinuity, or the gated critic does not help.
- Compute: GPU-parallel. Two weeks.

**22. Classical control margins for RL policies.**
- Hypothesis: linearizing the closed loop around trajectories gives gain and phase margins and Lyapunov exponents for a learned controller; these cheap numbers predict robustness to dynamics shift and sim-to-sim transfer better than training-time domain randomization range does.
- Gap: control theory has robustness metrics; deep RL evaluates robustness by running perturbed environments. Nobody has checked whether the classical numbers predict the empirical ones.
- Experiment: train policies on DMC and Playground with various regularizers; compute margins and exponents numerically via finite differences through the simulator; correlate with return under mass/friction/delay shifts.
- Kill test: margins are uncorrelated with empirical robustness.
- Compute: GPU-parallel. Two weeks.

**23. Weight quantization in closed loop.**
- Hypothesis: quantizing a policy's weights (int8, int4) introduces small per-step action errors that compound through the dynamics; the compounding rate is set by the closed loop's Lyapunov exponent (idea 22), so some policies tolerate int4 and others break at int8. Quantization-aware RL training (noise injection on weights) produces policies that are robust at deployment.
- Gap: policy quantization for embedded deployment is done post hoc; the compounding mechanism and its predictor have not been studied. This is weight quantization, distinct from the ADEPT action-quantization stub.
- Experiment: quantize trained policies at several bit-widths; measure return vs open-loop action error; test quantization-aware training.
- Kill test: closed-loop return tracks open-loop error with no compounding.
- Compute: GPU-parallel. One week.

**24. How open-loop is a trained policy?**
- Hypothesis: a substantial share of a trained policy's behavior is open-loop (it would produce near-identical action sequences with observations frozen or noised), and the open-loop fraction rises over training on deterministic tasks. It predicts fragility under observation noise and transfer.
- Gap: blind locomotion policies exist by design; how much ordinary policies actually use their observations has not been measured as a training-time quantity.
- Experiment: at checkpoints, replay episodes with observations frozen, delayed, or replaced by their running mean; measure action divergence and return retention; relate to robustness tests.
- Kill test: policies are fully closed-loop throughout.
- Compute: GPU-parallel. One week.

**25. Emergent equivariance.**
- Hypothesis: policies trained on tasks with symmetries (left-right mirror in locomotion, rotations in navigation) become approximately equivariant without being told, and the degree of emergent equivariance predicts generalization to held-out symmetric configurations. Equivariance emerges late and is fragile under fine-tuning.
- Gap: equivariant architectures impose symmetry; whether and when it emerges in standard networks has not been measured.
- Experiment: measure equivariance error (‖π(g·s) − g·π(s)‖) over training on mirror-symmetric Playground tasks and rotational gridworlds; relate to held-out performance; compare to equivariant baselines.
- Kill test: no equivariance emerges, or it does not predict generalization.
- Compute: GPU-parallel. One week.

## E. Model-based RL

**26. The world model is only good where the policy went.**
- Hypothesis: a learned world model's error is low on the training policy's distribution and rises sharply under policy shift; this policy-conditioned bias, not generic model error, is what limits imagination-based improvement. Spending a small fraction of real steps on deliberately diversified behavior (not exploration bonuses, just action perturbations) reduces the bias more than more data from the current policy.
- Gap: model error is reported as a scalar; its dependence on the policy that generated the data has not been measured or treated as the target of data collection.
- Experiment: DreamerV3 small and TD-MPC2 on DMC; measure model error under held-out policies; compare diversified collection vs standard at matched real steps.
- Kill test: model error is flat across policies.
- Compute: Dreamer small. Two weeks.

**27. Exact double sampling via cloning.**
- Hypothesis: the Bellman residual's double-sampling problem (two independent next states for an unbiased gradient) is solved for free in a cloneable simulator. With exact residual gradients, the classic verdict against residual methods can be retested fairly, and residual training may win at high UTD where semi-gradient diverges.
- Gap: the other list's residual-gradient idea retests under modern architecture. This removes the estimator's bias entirely with a cloneable sim, which has never been possible on nontrivial tasks.
- Experiment: Playground with state save/restore; semi-gradient vs single-sample residual vs double-sampled residual; UTD 1 to 16.
- Kill test: exact residual gradients still lose.
- Compute: GPU-parallel. One to two weeks.

## F. Multi-agent

**28. A bandwidth law for coordination.**
- Hypothesis: coordination performance as a function of communication bits per step follows a saturating curve whose knee is a task property; it is set by the entropy of the partner's hidden information that matters for the joint action, and can be predicted from an oracle-information measurement.
- Gap: emergent-communication work gives agents a fixed channel; the information–performance curve has not been measured or predicted.
- Experiment: JaxMARL (Overcooked, SMAX, Hanabi-small); channel capacity from 0 to 32 bits via discrete bottlenecks; fit curves; predict knees from mutual information between partner state and optimal joint action.
- Kill test: curves do not saturate or knees are unpredictable.
- Compute: small. Two weeks.

**29. How specialization scales with team size.**
- Hypothesis: with fixed total compute, role specialization (measured as divergence between agents' state-conditioned policies) grows with team size up to a point then collapses, because parameter sharing dominates; the point depends on whether agents are given identity inputs. This explains inconsistent findings on shared vs independent parameters.
- Gap: team-size scaling in MARL is studied for performance, not for what the policies become.
- Experiment: JaxMARL with N = 2 to 32 at matched compute; specialization metric vs N under shared, independent, and shared-plus-ID parameterizations.
- Kill test: specialization is independent of N.
- Compute: small. One to two weeks.

**30. Role adapters: low-rank per-agent deltas on a shared policy.**
- Hypothesis: a shared policy with per-agent low-rank adapters recovers the performance of independent networks at the cost of shared ones, and the required rank is a direct measure of how heterogeneous the roles are.
- Gap: MARL uses full sharing with an ID input or no sharing. Low-rank adapters are standard in LLMs and nearly untested here; rank as a heterogeneity measurement is new.
- Experiment: JaxMARL heterogeneous tasks; rank ∈ {0, 1, 4, 16, full}; return and the rank at which independent-network performance is matched.
- Kill test: ID input already matches independent networks everywhere.
- Compute: small. One week.

## G. RL for language models

**31. What does legibility cost?**
- Hypothesis: RLVR drifts reasoning toward compressed, language-mixed, or private notation unless constrained; forcing it to stay human-readable (a legibility classifier as a penalty, or a readability reward) costs a measurable accuracy tax that grows with task difficulty. The tax is the price of monitorability and nobody has measured it in a controlled setting.
- Gap: language-mixing drift was noted anecdotally in large reasoning models and fixed with a penalty; the accuracy cost of the fix has not been quantified, and small models make the experiment cheap.
- Experiment: 0.5–1.5B GRPO on procedural math and logic; unconstrained vs legibility-penalized at several strengths; accuracy, legibility (human-rated subset and classifier), and reasoning length.
- Kill test: the tax is zero at all difficulties.
- Compute: 0.5–1.5B LoRA. Two weeks.

**32. Learning when to ask for a hint.**
- Hypothesis: given a hint action that reveals a verified intermediate step at a reward cost, an RL-trained model learns to request hints on exactly the problems where it would otherwise fail, and the learned hint rate is a calibrated difficulty estimate. Hint-trained models also solve more problems without hints than models trained without the option.
- Gap: "asking for help" is studied in embodied agents and in RLHF-era clarification work; a costed, verifiable hint in RLVR with a calibration analysis has not been done.
- Experiment: procedural tasks with checkable intermediate steps; hint cost sweep; measure hint rate vs true difficulty, accuracy with and without hints at test.
- Kill test: the model requests hints uniformly or never.
- Compute: 0.5–1B. One to two weeks.

**33. Planted-error recovery as a transfer metric.**
- Hypothesis: RLVR's gains come partly from learning to detect and recover from its own mistakes. Inserting a wrong intermediate step into the model's reasoning at test time and measuring recovery rate separates this from improved first-pass accuracy, and recovery rate transfers across domains while first-pass accuracy does not.
- Gap: self-correction is measured by prompting; a controlled planted-error protocol on RL-trained vs base models, with cross-domain transfer, is new.
- Experiment: base vs RLVR 0.5–1.5B on three procedural domains; plant errors of controlled type and position; measure recovery; test transfer to a fourth domain.
- Kill test: recovery rate does not change under RL, or tracks first-pass accuracy exactly.
- Compute: 0.5–1.5B. One to two weeks.

**34. Heterogeneous-temperature groups.**
- Hypothesis: sampling GRPO's group at mixed temperatures (some low, some high) gives a baseline that spans the policy's exploratory range, producing better advantage estimates than a single temperature, and the best mixture shifts toward low temperature as training proceeds.
- Gap: the group is sampled at one temperature. Temperature is tuned, not mixed.
- Experiment: three procedural domains; fixed temperature vs mixed groups vs annealed; matched rollouts; accuracy and advantage variance.
- Kill test: mixing never beats the best single temperature.
- Compute: 0.5–1B. One week.

**35. Notes across episodes.**
- Hypothesis: a small model playing a text game across many episodes, allowed to write a short note that is prepended to the next episode's prompt, learns via RL to write notes that function as a transferable strategy, and the notes transfer to a different model playing the same game.
- Gap: external memory for LLM agents is prompted or hand-built; learning what to write via RL on episodic return, and testing note transfer across models, is new.
- Experiment: TextWorld or BabyAI-text families; RL over (actions, note) with return as reward; compare to no-note and to fixed human notes; test note transfer.
- Kill test: learned notes do not help the next episode.
- Compute: 0.5–1B. Two weeks.

**36. Inverse-task RL as an auxiliary.**
- Hypothesis: training a model with RL on the inverse of a verifiable task (given a solution, generate a problem it solves; checked by solving it) improves the forward task more than equal compute spent on more forward RL, because it forces the model to represent the task's structure.
- Gap: problem generation is used to make training data; using it as an auxiliary RL objective and measuring forward improvement at matched compute has not been tested.
- Experiment: arithmetic, graph, and logic domains; forward-only vs forward+inverse at matched rollouts; held-out forward accuracy.
- Kill test: inverse training does not help forward accuracy.
- Compute: 0.5–1B. One to two weeks.

**37. Paraphrase groups.**
- Hypothesis: forming the GRPO group from paraphrases of the same prompt (rather than repeated samples of one phrasing) yields policies that are robust to prompt wording at no extra rollout cost, and the advantage estimate's variance is lower because paraphrases decorrelate failure modes.
- Gap: prompt-wording robustness of RL-trained reasoners is measured but not trained for via the group structure.
- Experiment: procedural tasks with template paraphrasers; standard vs paraphrase groups; test on held-out phrasings.
- Kill test: no robustness gain or a cost in in-distribution accuracy.
- Compute: 0.5–1B. One week.

## H. Interpretability of RL agents

**38. Activation steering for control policies.**
- Hypothesis: sparse-autoencoder features on a trained policy's activations include interpretable behavioral directions (caution, speed, target preference), and adding a feature direction at inference steers behavior predictably without retraining. The steerability of a policy is a measurable, task-dependent property.
- Gap: steering vectors are a mature LLM tool; a systematic study on RL policies, with behavioral verification in the environment, has barely begun.
- Experiment: SAC and PPO on Playground and Procgen; train SAEs on hidden activations; label features by behavioral correlates; steer and measure the behavior change and return cost.
- Kill test: no monosemantic behavioral features, or steering produces noise rather than behavior.
- Compute: small. Two weeks.

**39. Dormant units are fossils.**
- Hypothesis: units that become dormant during training were the ones that encoded early-training strategies the agent abandoned. Their last active features are interpretable as the earlier behavior, and reviving them (resetting input weights but keeping output weights) recovers the earlier strategy when the environment changes back.
- Gap: dormant units are counted as a plasticity metric; what they used to do has never been examined.
- Experiment: Craftax and Procgen; track per-unit activations over training; for units that go dormant, probe their earlier receptive fields; test revival under task reversal.
- Kill test: dormant units were never informative.
- Compute: small. One week.

## I. Evaluation and methodology

**40. The deterministic-evaluation gap as a diagnostic.**
- Hypothesis: the difference between evaluating with the mean action and with sampled actions is not noise: a large positive gap signals an over-regularized policy, a negative gap signals a critic-exploiting mean, and the gap predicts transfer and fine-tunability better than return does.
- Gap: papers choose one evaluation mode by convention. The gap itself has never been studied as a signal.
- Experiment: SAC, PPO, and flow policies on DMC and Playground; log the gap over training; correlate with robustness tests and offline-to-online fine-tuning speed.
- Kill test: the gap is pure noise.
- Compute: piggybacks on other runs. Days.

**41. A minimal task subset for continuous control.**
- Hypothesis: method rankings on DMC, MyoSuite, and Playground are predicted by a handful of tasks; the rest are redundant. A principled 5-task subset preserves rankings of published methods and cuts benchmark cost by 5–10×.
- Gap: this was done for Atari (Atari-5); nothing comparable exists for the continuous-control suites that dominate recent RL papers, and the GPU-sim shift makes it tractable.
- Experiment: collect per-task results for 15+ methods from papers plus your own runs; fit a low-rank model of task × method; select the subset; validate on held-out methods.
- Kill test: ranking needs most tasks.
- Compute: mostly data collection; one week of runs.

**42. Sign accuracy of policy-gradient estimators.**
- Hypothesis: per update, the fraction of parameter directions on which an estimator agrees in sign with a high-sample ground-truth gradient varies more across estimators (GAE at various λ, vine, pathwise) than variance does, and sign accuracy predicts learning speed better than variance.
- Gap: estimators are compared by variance and return. Direction accuracy against a ground truth has not been measured in deep RL.
- Experiment: GPU sims where a 100k-sample ground-truth gradient is affordable; compute sign accuracy per estimator per checkpoint; correlate with learning speed.
- Kill test: sign accuracy tracks variance exactly.
- Compute: GPU-parallel. One week.

**43. Do seeds learn the same representation?**
- Hypothesis: representational similarity across seeds (CKA on hidden layers) is low early and converges late on tasks where seed variance in return is low, and stays low on high-variance tasks. Representation convergence, not return, is the earlier and cleaner indicator of a solved task, and it separates "same strategy, different features" from "different strategies."
- Gap: representation convergence across seeds is studied in supervised learning; in RL, where the data differs by seed too, it has not been measured, and the other list's strategy-modes idea is about behavior, not representation.
- Experiment: 20+ seeds on gymnax, Craftax, DMC; CKA across seeds per checkpoint; relate to return variance and to behavioral clustering.
- Kill test: CKA is uninformative about return variance.
- Compute: many small runs. One week.

## J. Imitation and reward learning

**44. Does IRL recover the reward or the shaping?**
- Hypothesis: potential-based shaping is unidentifiable from demonstrations, so deep IRL methods return an arbitrary mixture of reward and shaping. Which mixture they return is systematic (they prefer dense, shaped forms), and this is why IRL rewards transfer poorly to new dynamics, where shaping no longer helps.
- Gap: the theory is known; the empirical question of what deep IRL actually returns and how that affects transfer has not been answered.
- Experiment: tasks with known sparse reward plus known shaping; run AIRL, IQ-Learn, and a diffusion-based IRL; decompose the recovered reward; test transfer under dynamics shift.
- Kill test: recovered rewards transfer as well as the true reward.
- Compute: MuJoCo. Two weeks.

**45. Learning-path imitation.**
- Hypothesis: behavior cloning from the first half of an RL agent's buffer (diverse, suboptimal) gives a better initialization for a new RL run than cloning from the final policy (narrow, optimal), because the early data covers the states the learner will visit. The best source window is predictable from the learner's expected early state distribution.
- Gap: BC-then-RL always clones the expert. Cloning the learning path has not been tested.
- Experiment: DMC and Craftax; BC from windows of a previous run's buffer; fine-tune with RL; compare time-to-threshold.
- Kill test: cloning the final policy is always best.
- Compute: small. One week.

## K. Goals, memory, and specification

**46. Anti-goals.**
- Hypothesis: conditioning a goal-conditioned policy on what to avoid (an anti-goal embedding in the same space as goals) generalizes to novel hazards far better than training with penalties, because the hazard is represented, not just experienced. Anti-goal conditioning is a cheap safety interface for GCRL.
- Gap: GCRL conditions on targets; negative conditioning exists in generative modeling (negative prompts) and not in control.
- Experiment: maze and manipulation tasks with movable hazards; train with anti-goal conditioning on some hazards; test on held-out hazard positions and types; compare to penalty-trained policies.
- Kill test: anti-goal policies do not generalize to new hazards.
- Compute: small. Two weeks.

**47. How much memory does a task need?**
- Hypothesis: the memory horizon of a POMDP can be estimated from data by training predictors of the optimal action (or return) with increasing observation windows and finding the plateau. This statistic predicts the frame-stack vs recurrence gap across tasks and tells you when a Transformer or SSM is needed.
- Gap: memory architectures are compared by return; a cheap task statistic that predicts which architecture is needed does not exist.
- Experiment: POMDP suites (POPGym, MemoryGym) plus "accidentally partial" DMC and Atari tasks; estimate memory horizon; correlate with the return gap between frame-stack, GRU, and Transformer policies.
- Kill test: the estimated horizon does not predict the gap.
- Compute: small. One to two weeks.

**48. Hack discovery time vs compute.**
- Hypothesis: in environments with injected exploits, the time for an agent to find the exploit falls as a power law in agent capacity (critic size, UTD, exploration bonus strength), with an exponent that depends on how far the exploit lies from the intended behavior. This gives a quantitative version of "more capable agents hack more."
- Gap: reward hacking is reported as case studies; a scaling measurement with controlled exploits has not been done.
- Experiment: MinAtar, Procgen, and Playground with exploits of controlled distance from intended behavior; sweep capacity; fit discovery-time laws.
- Kill test: discovery time is flat in capacity.
- Compute: many small runs. Two weeks.

**49. How many parameters does adaptation need?**
- Hypothesis: adapting a control policy to shifted dynamics (mass, friction, actuator gain) needs a low-rank update, and the required LoRA rank is a measurement of how different the new dynamics are from the old. Rank-1 adaptation suffices for scalar parameter shifts; morphology changes need more.
- Gap: LoRA is standard for language models; its use for control-policy adaptation, with rank as a measurement, is new.
- Experiment: Playground and DMC; pretrain, shift dynamics, adapt with LoRA at ranks 1 to full; map the rank needed vs the kind of shift.
- Kill test: full fine-tuning is needed for every shift.
- Compute: GPU-parallel. One week.

**50. PEGASUS at scale: common random numbers in GPU simulators.**
- Hypothesis: GPU-native simulators make it free to evaluate two policies on identical random seeds, turning noisy policy comparisons into paired comparisons with far lower variance. Policy gradient with common random numbers across the batch and finite-difference policy search both become competitive with GAE-based methods on stochastic tasks.
- Gap: PEGASUS (Ng and Jordan, 2000) proposed this for simulators; it was impractical until thousands of parallel environments per GPU and has not been revisited. It is distinct from the cloning family in the other list, which branches from states; this fixes the randomness.
- Experiment: Playground and gymnax with explicit RNG control; paired evaluation variance vs unpaired; CRN policy gradient and CRN finite differences vs PPO at matched compute, with stochasticity dialed up.
- Kill test: variance reduction does not change the method ranking.
- Compute: GPU-parallel. One to two weeks.

---

## How to pick

Rank by how clean the yes/no is, how little infrastructure it needs, and how surprised the field would be. By that ranking the strongest first picks are **2, 4, 10, 12, 15, 18, 24, 33, 40, 44, 50**.

Groups that share infrastructure:
- Ideas 22, 23, and 24 share a closed-loop perturbation harness.
- Ideas 1, 3, 7, and 9 are all cheap diagnostics on SAC and PPO and could be one "hidden hyperparameters" paper if each effect is real.
- Ideas 27 and 50 share RNG and state control in a GPU simulator, and connect to the cloning family in the other list.
- Ideas 31, 33, and 36 share the same procedural domains and base models.

Ideas most at risk of having been done recently; search before committing: 6 (μP in RL), 17 (data attribution), 30 (adapters in MARL), 38 (SAEs on RL agents), 41 (benchmark subsets), 49 (LoRA for control adaptation), 50 (PEGASUS revivals).
