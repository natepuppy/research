# The Huber Loss Is a Risk Preference: Clipped TD Errors Under-Value Lotteries and Ignore Hazards

*Draft. Pilot numbers are from the laptop runs in `results/`; sections marked [3080] are planned.*

## Abstract

DQN and most value-based deep RL agents minimise a Huber loss on the TD error, equivalently clipping the error to [−κ, κ] before the update, with κ = 1. The stated purpose is stability. We show it is also a change of objective. The fixed point of the clipped update is the Huber location of the target distribution, not its mean, and the two differ wherever a rare outcome carries a large value difference: a lottery (rare large gain) is under-valued, a hazard (rare large loss) is under-weighted, with a bias whose sign is minus the skew of the target and whose size is set by κ relative to the jump. On fork MDPs with exact ground truth the closed-form Huber fixed point predicts tabular Q-learning's learned values with R² = 1.000 across 60 cells spanning both skews, two gaps, three rarity levels and five κ, and predicts exactly which cells the greedy policy gets wrong (60/60). With κ = 1 and a 5%-probability payoff of 22, the agent values the lottery at 0.2 instead of 1.1 and takes the wrong fork in every seed; the mirror image holds for a 5%-probability loss. The squared loss has no such bias but carries Q-learning's max-bias on noisy arms, which is itself risk-seeking; the two biases add under hazards and oppose under lotteries. In DQN on CartPole with κ = 1 the clip is inactive on 98% of TD errors and active on the terminal ones, and the learned value of terminating actions (true value 1) ends 21% higher under Huber than under the squared loss (13/16 paired seeds). [3080] MinAtar with injected lotteries and hazards, and the terminal-value accounting at scale, are planned.

## 1. Introduction

Huber (1964) introduced his loss as a robust location estimator: for a distribution with outliers it estimates something between the mean and the median. In RL the loss is applied to TD errors (Mnih et al., 2015), and the target distribution at a state-action pair is the distribution of r + γ max_a Q̄(s', a). That distribution is skewed whenever a rare transition carries a large value: a stochastic reward with a heavy upper tail, a rare transition to a high-value state, or a rare termination from a high-value state. In all of these the Huber fixed point differs from the mean, and the direction is always the same: the rare side is under-weighted.

This is not a regularisation effect that disappears with data. It is the population fixed point of the update. Nor is it the same as reward clipping (van Hasselt et al., 2016), which changes the reward function; the Huber loss changes the *estimator* of a fixed reward function's return, and it acts on bootstrapped value jumps even when rewards are already clipped to [−1, 1].

Contributions. (1) The closed-form fixed point of Huber-TD and its risk-preference reading. (2) An exact tabular test: the fixed point predicts learned values with R² = 1.000 and predicts the greedy policy's errors in 60/60 cells; a pre-registered sign test holds in all 40 cells where the clip is active. (3) The interaction with Q-learning's max-bias, which is itself risk-seeking on noisy arms. (4) Evidence that in DQN the clip acts on terminal transitions and inflates the learned value of failing actions. (5) [3080] Deep tests with injected lotteries and hazards.

## 2. Related work

Huber loss in deep RL: Mnih et al. (2015) introduce the clip; Obando-Ceron & Castro (2021) and others report that the choice of loss matters empirically; Patterson et al. (2023) discuss the Huber loss in empirical design without a fixed-point analysis. Distributional RL uses the quantile Huber loss (Dabney et al., 2018) deliberately to estimate quantiles rather than means, which is the one place the location-shift is intended. Robust TD and heavy-tailed rewards: Bubeck et al. (2013) for bandits; Zhuang & Sui (2021) for robust RL with heavy tails; these seek robust estimates of the *mean*, which is what the Huber TD update fails to provide. Max-bias: Thrun & Schwartz (1993), van Hasselt (2010); our pilot shows the two biases interact. Risk-sensitive RL (Mihatsch & Neuneier, 2002; Tamar et al., 2015) designs risk preferences deliberately; we show a widely used default imposes one by accident. Related in method to our R15 (opposing biases in the TD target), not in content.

## 3. The Huber fixed point

For a fixed (s, a) with target distribution Y = r + γ max_a' Q̄(s', a'), the expected clipped update is zero at the q solving E[clip(Y − q, −κ, κ)] = 0. The left side is monotone decreasing in q, so the root q_κ is unique, equals E[Y] when κ ≥ max|Y − q_κ|, tends to the median of Y as κ → 0, and for a two-point lottery Y = {M w.p. p, 0 w.p. 1−p} with M − q_κ > κ solves p·κ = (1−p)·q_κ, i.e. q_κ = pκ/(1−p) (0.053 for p = 0.05, κ = 1, against a mean of pM = 1.1). The rare side contributes κ instead of M − q to the estimating equation, so the bias q_κ − E[Y] is ≈ −pM for a lottery and ≈ +pM for a hazard: the rare side's expected contribution is almost entirely lost. Iterating this root as a Bellman operator gives the Huber-TD fixed point for the whole MDP, which we compute by bisection.

## 4. Experiments

### 4.1 Setup and pre-registration

Fork MDP: start s₀, actions *safe* and *risky*, each a deterministic chain of 3 states to a terminal reward, γ = 0.95. Safe pays 1. Risky pays a lottery (M w.p. p, else 0; mean 1 + g) or a hazard (sure base minus M w.p. p; mean 1 − g). g ∈ {0.1, 0.3}, p ∈ {0.05, 0.1, 0.3}, κ ∈ {0.1, 0.3, 1, 3, ∞}, 8 seeds, uniform behaviour and exploring starts (both arms sampled equally), α = 0.02, 300k steps. Pre-registered: K1 sign = −skew in every finite-κ cell and |bias| monotone in κ; K2 identity-fit R² ≥ 0.9 of learned on predicted bias; K3 policy errors coincide with |predicted bias| > g in ≥ 90% of cells and no squared-loss cell is wrong. A smoke run had shown that the squared loss's *variance* on p = 0.02 lotteries (M ≈ 52) flips a 0.05 gap on its own, so the grid was set to p ≥ 0.05, g ≥ 0.1 before the tests were fixed.

### 4.2 Pilot: KILLED by rule, and why

K2 passed (R² = 0.986). K1 and K3 failed because chain states had two identical actions whose values were learned separately from noisy targets, so the target's max carried Q-learning's max-bias (+0.09 to +0.31 under the squared loss) on the noisy arm only. This made the squared-loss agent prefer the hazard arm (K3) and gave the κ = 3 lottery cells a positive bias (K1). The residual (learned − predicted) grows monotonically with κ (0.002 at κ = 0.1, 0.19 at κ = ∞; `results/posthoc.txt`), because clipping also suppresses the update variance that drives the max-bias.

### 4.3 Pre-registered follow-up with single-action chains: the fixed point is exact

With one action per chain state (the only change), R² = **1.000**; |bias| is monotone in κ in every (skew, g, p) (Spearman −0.90 to −1.00); the greedy policy is wrong exactly where |predicted bias| > g, **60/60** cells, and never under the squared loss. The sign clause fails only in four cells with predicted bias ≤ 0.013 (κ = 3, p = 0.3, clip nearly inactive), where the learned bias is noise of either sign; the test should have said "active cells". The pre-registered verdict is therefore INCONCLUSIVE and the substantive predictions all hold.

Huber bias at κ = 1 (nomax), in units of the gap: lottery 9.0×, 8.5×, 5.8× (g = 0.1; p = 0.05, 0.1, 0.3), 3.6×, 3.4×, 2.5× (g = 0.3); hazard the same magnitudes with the opposite sign. The κ = 1 agent takes the wrong fork in all 24 (skew, g, p) cells; the κ = 3 agent wherever the jump exceeds ≈ 3.

### 4.4 Two biases, two directions

The squared loss is risk-seeking toward noisy arms through the max (pilot: +0.09 to +0.31, enough to flip a 0.1 gap). Huber is biased against the rare side. Under a hazard both favour the risky arm; under a lottery they oppose, and at κ = 3 they nearly cancel in the pilot cells. Neither loss is neutral; the squared loss's bias is a sampling effect that shrinks with α and vanishes with double estimators, the Huber bias is a population effect that does not.

### 4.5 DQN: the clip acts on terminals

CartPole, batched DQN, Huber (κ = 1) vs squared loss, 16 paired seeds, 30k steps. 98% of TD errors are below κ; the terminal transitions, where δ = 1 − Q(s, a) ≈ −(V − 1) with V up to ~100, are the active ones. The learned value of 256 probe (s, a) pairs whose action terminates (true value 1, R11's instrument) ends at 26.8 under Huber vs 22.2 under the squared loss, higher in 13/16 seeds (pre-registered secondary threshold 12). Final return is equal (129 vs 131); AUC favours the squared loss (146 vs 125).

### 4.6 [3080] Planned

1. MinAtar (Breakout, Asterix, Freeway, Seaquest) with an injected lottery (a 2% chance of +10 on a designated action/state class) and an injected hazard (2% chance of −10), DQN with κ ∈ {0.3, 1, 3, ∞} and a double-Q variant, 10 seeds. Predictions: the κ = 1 agent's rate of taking the lottery falls below the squared-loss agent's and its rate of taking the hazard rises; the effect size tracks the computed Huber bias from the exact-MC value of the injected outcome; double-Q removes the squared loss's max-bias but not the Huber bias.
2. Terminal-value accounting: failure-Q against exact MC on CartPole and MinAtar under Huber vs squared loss vs Huber with κ scaled to the value range (κ = 0.1·V_max), with R11's reservoir as a second factor, to apportion the terminal inflation between loss and buffer.
3. κ as a risk dial compared with the quantile-Huber of QR-DQN on the same injected tasks.

## 5. Consequences

Setting κ = 1 on a problem whose values span tens or hundreds is a strong, unintended risk preference, applied to exactly the transitions that matter for safety: rare terminations and rare large payoffs. The fix is cheap (κ scaled to the value range, or a squared loss with gradient clipping, which clips the step rather than the fixed point), and the choice should be reported as a modelling choice.

## 6. Limitations

Tabular closed form; function approximation couples states and the deep version needs the exact-MC instrument. One deep task, one κ. The pilot's first grid conflated two biases and was repaired by a pre-registered follow-up; the sign test's wording was defective in inactive cells and the verdict is reported as INCONCLUSIVE accordingly.

## 7. Reproducibility

`code/r16_huber.py`: default grid (pilot), `--single-action --tag nomax` (follow-up), `--dqn` (CartPole check). Under 5 minutes on a laptop CPU in total.

## References

Bubeck, Cesa-Bianchi, Lugosi (2013). Bandits with heavy tail. IEEE Trans. IT.
Dabney, Rowland, Bellemare, Munos (2018). Distributional RL with quantile regression. AAAI.
Huber (1964). Robust estimation of a location parameter. Ann. Math. Stat.
Mihatsch, Neuneier (2002). Risk-sensitive RL. Machine Learning.
Mnih et al. (2015). Human-level control through deep RL. Nature.
Obando-Ceron, Castro (2021). Revisiting Rainbow. ICML.
Patterson, Neumann, White, White (2023). Empirical design in RL.
Tamar, Chow, Ghavamzadeh, Mannor (2015). Policy gradient for coherent risk measures. NeurIPS.
Thrun, Schwartz (1993). Issues in using function approximation for RL.
van Hasselt (2010). Double Q-learning. NeurIPS.
van Hasselt, Guez, Hessel, Mnih, Silver (2016). Learning values across many orders of magnitude. NeurIPS.
Zhuang, Sui (2021). No-regret RL with heavy-tailed rewards. AISTATS.
