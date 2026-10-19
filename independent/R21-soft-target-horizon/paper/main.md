# Soft Target Updates Are an Implicit Discount Schedule

**Abstract.** DDPG, TD3 and SAC update their target critics by Polyak averaging with τ = 0.005, a default that is tuned but never derived. We show that τ is a horizon parameter. Between a decision and a consequence k steps away, the value signal passes through k/n segments (n the bootstrap length), and at each segment it is filtered twice: by the target average (rate τ) and by the critic's fit (rate α). The arrival time is the sum of 2k/n geometric delays, so the critic's effective horizon grows linearly with the update count, h(t) ≈ n·t/(1/τ + 1/α), and until h(t) reaches the task horizon the agent is greedy for a shorter-horizon problem. In an exact tabular pilot the horizon slope matches the closed form in 18/18 (τ, α, n) cells (R² = 1.000), the iteration at which a policy switches from an impatient to a patient option matches a one-line prediction across 54 cells (log–log slope 1.011, R² = 0.9997), τ and α are interchangeable, and a hard sync every K = 1/τ updates is the same schedule with a staircase instead of a line. At τ = 0.01 an agent is effectively a γ = 0.9 agent for its first thousand updates. [3080] We measure h(t) directly in TD3 and SAC on MuJoCo by regressing the learned critic on truncated Monte-Carlo returns, extract the fit rate α_eff that the theory leaves free, and test the τ schedule the law recommends. Together with the hard-sync law (R12) and the GAE law (R19) this gives one rule for all three target mechanisms of deep RL: credit assignment is a count of training iterations, and the count is set by hyperparameters nobody treats as horizons.

## 1. Introduction

Three mechanisms slow the movement of value information from a reward to the decisions that caused it in deep RL: hard target networks refreshed every K updates (DQN family), Polyak-averaged target networks with rate τ (DDPG/TD3/SAC family), and bootstrapped λ-returns refit each iteration (PPO family). For the first and third we have closed-form latency laws (R12, R19). This paper supplies the second, and in doing so shows that the three are the same statement: each mechanism moves the value front a fixed number of states per training iteration, so the critic's horizon grows linearly in iterations.

For Polyak averaging the derivation is a cascade. Let Q be the online table and Q̄ the target. With one-step targets, Q(s_k) relaxes toward γQ̄(s_{k+1}) at rate α and Q̄(s_k) relaxes toward Q(s_k) at rate τ. A unit reward injected k steps downstream therefore reaches s₀ through k filters of rate α and k − 1 filters of rate τ (the last segment is a direct reward, not a bootstrap). The arrival-time distribution is the convolution of 2k − 1 geometrics, with mean (k − 1)/τ + k/α and relative spread shrinking as k^{−1/2}. With n-step targets replace k by ⌈k/n⌉. The horizon of the critic — the truncation depth whose value equals the current estimate — therefore advances at n/(1/τ + 1/α) states per update.

Two consequences. First, τ and α are interchangeable in the latency: halving the fit rate costs the same as halving τ, which means that the default τ = 0.005 is only the binding delay if the critic fits faster than 200 updates per segment. Second, until h(t) exceeds 1/(1 − γ) the agent behaves as a shorter-horizon agent — soft target updates impose a discount schedule from γ_eff ≈ 0 to γ at a rate the practitioner did not choose.

Contributions. (1) The Polyak latency law with its α-symmetry, in closed form. (2) An exact pilot: horizon slopes, switch times, symmetry and the hard/soft equivalence all within a few percent of prediction. (3) The implicit-discount reading and its consequence for the τ default. (4) [3080] Direct measurement of h(t) in TD3/SAC, the extracted α_eff, and a τ schedule test.

## 2. Related work

Lillicrap et al. (2016) introduce τ without analysis; Fujimoto et al. (2018) and Haarnoja et al. (2018) inherit it. Target networks as stabilisers: Kim et al. (2019), Zhang et al. (2021, deadly triad), Lee & He (2019). Piché et al. (2022, functional regularisation) state qualitatively that hard sync needs N·H steps on an N-chain and that Polyak averaging also slows propagation; they do not quantify τ or the fit rate. Fedus et al. (2020) find n-step is what makes large replay buffers work, which the law predicts (n divides the latency). Discount scheduling: OpenAI Five annealed γ; François-Lavet et al. (2015) study increasing γ; our result says a schedule is already present, set by τ. R12 (hard sync) and R19 (GAE) are the sibling laws. Lab stubs on delayed reward concern reward delay within an episode; adjacent, not overlapping.

## 3. Setting

Deterministic synchronous tabular learning: all states updated every step, Q ← Q + α(y − Q), y the n-step target from Q̄, Q̄ ← Q̄ + τ(Q − Q̄), γ = 0.99. Chain: 200 states, reward 1 per step, horizon h(t) read off V(s₀) by inverting (1 − γ^h)/(1 − γ). Patience fork: `now` pays 1; `wait` enters a d-step corridor paying R = 2/γ^d at the end; t* is the first update with Q(wait) > Q(now), i.e. the median arrival time. Grid τ ∈ {0.1, 0.01, 0.001} × α ∈ {1, 0.1, 0.01} × n ∈ {1, 5} × d ∈ {10, 30, 100}; hard-sync controls at K = 1/τ.

Pre-registered: K1 horizon slope within [0.8, 1.2] of n/(1/τ + 1/α), R² ≥ 0.98, in ≥ 16/18 cells; K2 t* against t*_pred = (m − 1)/τ + m/α − max(1/τ, 1/α)/3, log–log slope in [0.9, 1.1], R² ≥ 0.98, ≥ 48/54 within [0.7, 1.4]; K3 τ = α cells match the predicted (2m − 1)/(m − 1) multiple; K4 hard K = 1/τ within [0.7, 1.3] of soft. SUPPORTED if K1–K3. (A smoke run caught an off-by-one in the stage count — the last segment needs no bootstrap — fixed by re-derivation before the pilot; README.)

## 4. Results

**Horizon (K1).** Measured/predicted slope 0.98–1.10 in all 18 cells, R² = 1.000. At τ = 0.01, α = 1, n = 1: h = 2 at t = 100, 11 at 1000, 31 at 3000, 100 at 10 000, i.e. γ_eff = 0.50, 0.91, 0.97, 0.99.

**Switch time (K2).** 54/54 cells within [0.79, 1.03]× of the closed form; slope 1.011, R² = 0.9997. Examples (n = 1, α = 1): d = 100 switches at 987 / 9855 / 98 544 updates for τ = 0.1 / 0.01 / 0.001 (predicted 1087 / 9967 / 98 767).

**Symmetry (K3).** τ = α = 0.01, d = 100, n = 1: 19 750 updates against 9855 with α = 1 — the fit rate doubles the latency exactly as τ would. 12/12.

**Hard = soft (K4).** K = 100 vs τ = 0.01, d = 100, n = 1: 9901 vs 9855. The hard front is a staircase (h = 1, 2, 3, … at t = 100, 200, 300, …), the soft front a line through the same points.

## 5. Discussion

The default τ = 0.005 means 200 updates per n-step segment plus the fit delay. For a decision 100 steps from its consequence that is ≥ 20 000 updates with one-step targets and ≥ 4000 with five-step — small against a million-step MuJoCo run, which is why dense-reward benchmarks never exposed it, and large against anything sparse or short. The practical reading is that τ should be chosen with the task horizon and the update budget in view, and that early in training a larger τ costs nothing in horizon terms that a smaller one would not also cost through α. The law also says where high update-to-data ratios get part of their early advantage: the horizon grows per update, not per environment step.

Limitations. The pilot is an exact linear system and confirms a derivation; the question the paper must answer is the size of α_eff in a replay-trained neural critic, which may dominate τ entirely. Synchronous updates; with replay the R12 second term adds. γ_eff is a horizon-matching definition.

## 6. [3080] Planned experiments

(a) TD3 and SAC on Hopper/Walker/HalfCheetah, τ ∈ {0.05, 0.005, 0.0005} × n ∈ {1, 5}: every 5k steps, regress the critic's V(s) on truncated Monte-Carlo returns at horizons {5, 10, 20, 50, 100, 200} from stored states to read h(t); fit the slope and extract α_eff. (b) A continuous-action patience gridworld: switch time against the law. (c) A τ schedule (0.05 → 0.005 over the first 100k updates) against fixed τ, pre-registered on early-return AUC. (d) UTD × τ: does h per environment step scale with UTD as the law says?

## 7. Conclusion

Polyak averaging delays value information by 1/τ per bootstrap segment, the critic's fit rate by 1/α more, and the two add. The critic's horizon therefore grows linearly in updates, which makes τ a discount schedule in disguise and completes the latency picture across the three target mechanisms of deep RL.

## References

Lillicrap et al. 2016; Fujimoto et al. 2018; Haarnoja et al. 2018; Kim et al. 2019; Zhang et al. 2021; Piché et al. 2022; Fedus et al. 2020; François-Lavet et al. 2015; OpenAI 2019 (Five); R12, R19 (this lab).
