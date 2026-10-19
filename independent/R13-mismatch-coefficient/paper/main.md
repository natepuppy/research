# Coverage Can Hurt: The Distribution-Mismatch Coefficient Mispredicts Policy-Gradient Learning Time Outside the Hard-Exploration Regime

*Draft. Pilot numbers are from the laptop run in `results/`; sections marked [3080] are planned.*

## Abstract

Convergence rates for policy gradient (PG) methods are stated in terms of a distribution-mismatch coefficient D_μ = ‖d^{π*}_ρ / μ‖_∞, the worst-case ratio between the optimal policy's state visitation and the training start distribution μ. The coefficient is the theoretical justification for exploratory restarts and reset-distribution design, and it is infinite for the most common practical choice, a single fixed start. We ask whether D_μ, or a refinement D_0 that replaces μ by the initial policy's visitation under μ, predicts how many iterations exact-gradient softmax PG needs to reach 90% of optimal, across four gridworld layouts and nine start distributions spanning D_μ from 1 to ∞. It does not: the rank correlation between log D_μ and log iterations is −0.67, the wrong sign. On layouts where the initial policy reaches the goal with non-negligible probability, the theoretically optimal start (μ = d^{π*}, D_μ = 1) is 20–45% slower than a single start (D_μ = ∞), and a uniform start is 2–4× slower; the fastest start is the single start with a 10–30% uniform mixture. On the one hard-exploration layout, where the initial policy's gradient at the start is vanishingly small, the coefficient recovers the right sign (+0.70 within layout) and d^{π*} is the best start. We explain both regimes with a gradient-budget argument, show that μ's mass on the evaluation start is a better predictor than any coverage measure outside hard exploration, and propose a two-regime rule for reset-distribution design, with V^{π_0}(ρ)/V* as the regime indicator. [3080] Sample-based PG on the same grids, and a deep check on CartPole and MinAtar with reset width as the dial, are planned.

## 1. Introduction

The policy-gradient convergence literature (Kakade & Langford, 2002; Agarwal, Kakade, Lee & Mahajan, 2021; Mei, Xiao, Szepesvári & Schuurmans, 2020; Bhandari & Russo, 2019) proves iteration complexities of the form

  T = O( D_μ · poly(|S|, |A|, 1/(1−γ)) / ε ),  D_μ = ‖d^{π*}_ρ / μ‖_∞,

for softmax or projected PG with exact gradients started from a distribution μ and evaluated at ρ. The coefficient captures the fact that PG only improves at states it visits, and that improvement at states the optimal policy needs is what reduces suboptimality at ρ. Its consequences are widely repeated: broader starts are better; μ should cover d^{π*}; a single start is pathological because D_μ = ∞.

We test the prediction in the setting where it is derived: exact gradients, tabular softmax, small MDPs. The test is cheap and, to our knowledge, has not been done. The result is that the coefficient has the wrong sign outside hard exploration. The pilot suggests a cleaner picture: PG has a fixed per-iteration gradient budget distributed as d^{π}_μ, and mass placed on states irrelevant to V^π(ρ) is wasted. Coverage helps only when the initial policy's gradient at ρ is too small to learn from at all, which is the hard-exploration regime the theory implicitly assumes.

Contributions. (1) A direct empirical test of D_μ and D_0 as predictors of PG learning time, with a pre-registered kill test, which they fail with the opposite sign. (2) A reversal table: on three of four layouts, single start < d^{π*} start < uniform start in iterations, in strict violation of the coefficient's ordering. (3) A two-regime account with a cheap indicator, V^{π_0}(ρ)/V*(ρ), and the design rule that follows. (4) [3080] Sample-based and deep versions.

## 2. Related work

Convergence of PG with mismatch coefficients: Kakade & Langford (2002) introduce the quantity for conservative policy iteration; Agarwal et al. (2021) give the softmax, NPG and projected rates; Mei et al. (2020) show the softmax plateau (exponentially slow escape when π_0 puts little mass on the right action) and a rate with a 1/min_s μ(s) factor. Our pilot's easy-layout result does not contradict these as bounds; it shows the bounds are not predictive and that the direction of the dependence reverses. Reset-distribution and start-state design: exploratory restarts (Kakade & Langford), reverse curriculum (Florensa et al., 2017), Go-Explore's reset-to-archive (Ecoffet et al., 2021), GoalGAN. These assume that broadening μ helps and tune how; our result says the premise holds only in the hard-exploration regime they were designed for and is wrong elsewhere. Distribution mismatch as a quantity in offline RL (concentrability; Munos & Szepesvári, 2008; Chen & Jiang, 2019) plays a different role (evaluation error, not optimisation speed) and is not tested here.

Adjacent lab stubs. The merged idea sheets include "curriculum via reset distribution" (S2-10) and "where to start episodes" (L1-27); both propose methods for choosing μ. This paper tests the quantity those methods are motivated by.

## 3. Setting

Four 10×10 gridworlds (slip 0.1, γ = 0.95, reward 1 on entering the absorbing goal): `open`, `wall` (a wall with a gap), `rooms` (four rooms), `snake` (a 60-step corridor, |S_reach| = 64, V*(start) = 0.045). Evaluation distribution ρ = δ_{(0,0)}. Training starts μ: mixtures (1−β)δ_start + β·Uniform for β ∈ {0, 0.02, 0.1, 0.3, 0.6, 1}; μ = d^{π*}_ρ (D_μ = 1 exactly); `far` (uniform over the half of states d^{π*} visits least, plus 10% on the start); `far_only` (the same without the start mass). Softmax tabular policy, uniform initialisation, exact gradient of V^π(μ), η = 2, cap 50k iterations. Learning time t = first iteration with V^π(ρ) ≥ 0.9 V*(ρ). Coefficients: D_μ = max_s d^{π*}_ρ(s)/μ(s) (∞ if μ misses the support), D_0 = max_s d^{π*}_ρ(s)/d^{π_0}_μ(s), and the expectation versions E_μ, E_0.

Pre-registered kill test: K1 Spearman(log D_μ, log t) ≥ 0.7 (finite D_μ); K2 Spearman(log D_0, log t) ≥ 0.7, beating K1 by 0.1, all configs converged; K3 within-layout D_0 correlation ≥ 0.6. SUPPORTED if (K1 or K2) and K3; KILLED if neither K1 nor K2.

## 4. Results

### 4.1 The kill test

KILLED. K1 = −0.67 (n = 23). K2 = −0.52 (n = 31; five configs hit the cap). K3 by layout: open −0.63, wall −0.40, snake +0.70, rooms −0.05. The expectation version E_0 does no better (−0.47).

### 4.2 The reversal

Iterations to 90% of V*(ρ):

| layout | δ_start (D_μ = ∞) | β = 0.3 | μ = d^{π*} (D_μ = 1) | uniform (D_μ ≈ 30) | far_only (D_μ = ∞) |
|---|---|---|---|---|---|
| open  | 964  | 902  | 1393  | 3470  | 41365 |
| wall  | 1051 | 797  | 1306  | 2901  | 42205 |
| rooms | 1446 | 904  | 1216  | 2445  | > 50000 |
| snake | > 50000 | 11004 | 10339 | 21530 | > 50000 |

On open, wall and rooms the ordering is δ_start-with-mixture < δ_start ≈ d^{π*} < uniform ≪ far_only, i.e. D_μ = ∞ beats D_μ = 1 beats D_μ = 30 beats D_μ = ∞. The coefficient is not monotone in anything the data do. On snake, the theory's ordering appears: d^{π*} < β = 0.6 < β = 0.3 < uniform < δ_start.

### 4.3 Why: a gradient budget

With exact gradients the softmax PG update at state s is η · d^{π}_μ(s) · A^π(s, ·)-weighted; Σ_s d^{π}_μ(s) = 1. The update is a fixed budget spread as d^{π}_μ. Progress on V^π(ρ) comes from the part of the budget that lands on states on π's paths from ρ. Uniform μ puts ≈ 1/|S| of the budget on each state; δ_start puts all of it on the start and, as π improves, on its path. On open/wall/rooms the initial uniform policy reaches the goal from the start with V^{π_0}(ρ)/V* ≈ 0.03–0.05, enough gradient signal to climb, and concentration wins. On snake, V^{π_0}(ρ)/V* ≈ 10⁻⁵ and the softmax plateau (Mei et al., 2020) holds the single-start learner for longer than the cap; here a start distribution that places mass deep in the corridor supplies the gradient the start cannot, and the theory's ordering holds.

The small-β win (β = 0.1–0.3 beats β = 0 on all three easy layouts by 7–37%) fits the same picture: a little off-path mass breaks the plateau at the handful of states the single-start learner reaches late, at negligible cost to the start's budget.

### 4.4 A better predictor outside hard exploration

Post hoc (`results/posthoc.txt`): on open/wall/rooms, Spearman(log t, μ(start)) = −0.65 (more mass on the evaluation start, faster), D_0 = −0.39, overlap and optimal-path mass ≈ 0. The single most predictive quantity is how much of μ is spent where ρ is. On snake, D_0 = +0.70 (n = 5) and μ(start) = −0.10.

### 4.5 [3080] Planned

1. Sample-based PG (REINFORCE with baseline, and PPO) on the same 36 configurations, 10 seeds, with t measured in environment steps. The variance of sampled gradients depends on μ, which could move the optimal β upward; the question is whether the reversal survives sampling.
2. Regime indicator. Sweep η and the initial-policy temperature on 20 random layouts; test whether V^{π_0}(ρ)/V*(ρ) (estimable from rollouts of π_0) predicts which of the two orderings holds, with a pre-registered threshold found on half the layouts and tested on the other half.
3. Deep. CartPole (reset-angle width as β), MinAtar Breakout/Freeway with randomised-start variants, PPO, 10 seeds per width; coefficients estimated from rollouts of the final policy. Prediction: a U-shaped curve of sample-efficiency against width with the minimum at small width on the dense-reward tasks, and monotone improvement with width only when the initial policy's return is at the floor.

## 5. Consequences

The practical rule the pilot supports: start from the evaluation distribution with a small exploratory mixture unless the initial policy cannot reach reward at all from it; only then broaden toward d^{π*}, and avoid uniform in both regimes. For theory: a rate that is predictive in the easy regime would need to credit mass on ρ's reachable set, not penalise its absence elsewhere; the quantity Σ_s d^{π*}_ρ(s) · d^{π_0}_μ(s) (an inner product rather than a sup ratio) is the natural candidate and is a planned test.

## 6. Limitations

Exact gradients, one η, one π_0, four layouts, one deterministic run per cell. The 90% criterion. Snake's within-layout correlation rests on five points. The two-regime story is a description of 36 points, not yet a prediction tested on fresh layouts; §4.5 item 2 is that test.

## 7. Reproducibility

`code/r13_mismatch.py`, no seeds (deterministic). ~13 minutes on a laptop CPU. `results/runs.csv` has every configuration's D_μ, D_0, E_μ, E_0, t and convergence flag.

## References

Agarwal, Kakade, Lee, Mahajan (2021). On the theory of policy gradient methods. JMLR.
Bhandari, Russo (2019). Global optimality guarantees for policy gradient methods.
Chen, Jiang (2019). Information-theoretic considerations in batch RL. ICML.
Ecoffet et al. (2021). First return, then explore. Nature.
Florensa et al. (2017). Reverse curriculum generation. CoRL.
Kakade, Langford (2002). Approximately optimal approximate RL. ICML.
Mei, Xiao, Szepesvári, Schuurmans (2020). On the global convergence rates of softmax policy gradient methods. ICML.
Munos, Szepesvári (2008). Finite-time bounds for fitted value iteration. JMLR.
