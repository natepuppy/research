# Coverage Can Hurt: The Distribution-Mismatch Coefficient Mispredicts Policy-Gradient Learning Time Outside Hard Exploration

*Conference-length version (RLC format). Every number here comes from this folder's results.*

## Abstract

Policy-gradient iteration complexity is stated with a distribution-mismatch coefficient \(D_\mu = \|d^{\pi^\ast}_\rho / \mu\|_\infty\), the worst-case ratio between the optimal policy's state visitation and the training start distribution \(\mu\). The coefficient is the usual justification for exploratory restarts, and it is infinite for a single fixed start. We test it where it is derived: exact-gradient tabular softmax policy gradient, four \(10\times 10\) gridworlds, nine start distributions, iterations to 90% of optimal. Verdict: KILLED. Spearman(\(\log D_\mu\), \(\log t\)) \(= -0.67\) on the 23 configurations with finite \(D_\mu\), and Spearman(\(\log D_0\), \(\log t\)) \(= -0.52\) on the 31 configurations that converged. The pre-registered bar was \(+0.7\). On open and wall, \(\mu = d^{\pi^\ast}\) (\(D_\mu = 1\)) is 20–45% slower than a single start (\(D_\mu = \infty\)), and a uniform start is 2–4× slower. A 10–30% uniform mixture on the single start is the fastest tabulated start on the three easy layouts. On the hard-exploration snake the within-layout correlation is \(+0.70\) (\(n = 5\)), the sign the theory predicts. Outside that regime, mass of \(\mu\) on the evaluation start predicts learning time better than any coverage measure (Spearman \(-0.65\)).

## 1 Introduction

The policy-gradient convergence literature (Kakade and Langford, 2002; Agarwal, Kakade, Lee and Mahajan, 2021; Mei, Xiao, Szepesvári and Schuurmans, 2020; Bhandari and Russo, 2019) proves iteration complexities of the form

\[
T = O\bigl(D_\mu \cdot \mathrm{poly}(|S|, |A|, 1/(1-\gamma)) / \varepsilon\bigr), \qquad D_\mu = \|d^{\pi^\ast}_\rho / \mu\|_\infty,
\]

for softmax or projected policy gradient with exact gradients, started from \(\mu\) and evaluated at \(\rho\). The coefficient encodes a real constraint: the update only improves states the learner visits, and suboptimality at \(\rho\) falls when those states are ones the optimal policy needs. The consequences that get repeated are stronger than that constraint: broader starts are better, \(\mu\) should cover \(d^{\pi^\ast}\), and a single start is pathological because \(D_\mu = \infty\).

Two gaps sit between the bound and that advice. \(D_\mu\) is infinite whenever \(\mu\) misses a state on the optimal path, including the ordinary choice of one fixed start, and policy gradient learns from a single start on many tasks. The coefficient also ignores where the initial policy goes. Early gradient signal is carried by \(d^{\pi_0}_\mu\), not by \(\mu\) alone. We therefore compute \(D_\mu\) exactly and, beside it,

\[
D_0 = \|d^{\pi^\ast}_\rho / d^{\pi_0}_\mu\|_\infty,
\]

and ask which one ranks configurations by iterations to 90% of \(V^\ast(\rho)\).

The pilot is the cheap test: exact gradients, tabular softmax, four layouts, 36 configurations, one deterministic run per cell. The rank correlations have the wrong sign. On the layouts where a uniform initial policy already reaches the goal with non-negligible probability, concentrating the start beats covering \(d^{\pi^\ast}\). On the one layout where that probability is about \(10^{-5}\), the coefficient recovers the predicted sign.

**Contributions.** (1) A pre-registered test of \(D_\mu\) and \(D_0\) as predictors of exact policy-gradient learning time. Both fail, with correlations \(-0.67\) and \(-0.52\). (2) The iteration table: on open and wall a single start beats \(\mu = d^{\pi^\ast}\), a small mixture beats the single start, and uniform is slower by a larger factor. (3) What ranks the easy layouts instead: \(\mu\)'s mass on the evaluation start (Spearman \(-0.65\)), and the regime split already visible in \(V^{\pi_0}(\rho)/V^\ast\).

## 2 Related work

Kakade and Langford (2002) introduce the mismatch coefficient for conservative policy iteration. Agarwal et al. (2021) give the softmax, natural, and projected rates. Mei et al. (2020) show the softmax plateau, an exponentially slow escape when \(\pi_0\) puts little mass on the right action, and a rate with a \(1/\min_s \mu(s)\) factor. The easy-layout result here is compatible with those statements as upper bounds. A bound can be loose by orders of magnitude, and the dependence that actually shows up in learning time can run the other way. The pilot is a test of prediction, not a counterexample to a proof.

Exploratory restarts (Kakade and Langford, 2002), reverse curricula (Florensa et al., 2017), and Go-Explore's reset-to-archive (Ecoffet et al., 2021) all widen \(\mu\) and then tune how. The quantity they are motivated by is the one tested here. Distribution mismatch in offline reinforcement learning (concentrability; Munos and Szepesvári, 2008; Chen and Jiang, 2019) controls evaluation error rather than optimisation speed, and it is not tested here.

## 3 Setting and the pre-registered test

Four \(10\times 10\) gridworlds, slip 0.1, \(\gamma = 0.95\), reward 1 on entering an absorbing goal. Layouts: `open`; `wall` (a wall with a gap); `rooms` (four rooms); `snake` (a 60-step corridor, 64 reachable states, \(V^\ast\) at the start equal to 0.045). The evaluation distribution is a point mass, \(\rho = \delta_{(0,0)}\).

Training starts: mixtures \((1-\beta)\delta_{\mathrm{start}} + \beta\cdot\mathrm{Uniform}\) for \(\beta \in \{0, 0.02, 0.1, 0.3, 0.6, 1\}\); \(\mu = d^{\pi^\ast}_\rho\), which has \(D_\mu = 1\) exactly; `far` (uniform over the half of the states \(d^{\pi^\ast}\) visits least, plus 10% on the start); `far_only` (the same support with no mass on the start). That is nine starts and 36 configurations.

The learner is a tabular softmax policy, uniform at initialization, updated by the exact gradient of \(V^\pi(\mu)\) at step size \(\eta = 2\), for at most 50,000 iterations. Learning time \(t\) is the first iteration with \(V^\pi(\rho) \ge 0.9\, V^\ast(\rho)\). Configurations that hit the cap are excluded from the rank correlations and reported as capped. \(D_\mu = \max_s d^{\pi^\ast}_\rho(s)/\mu(s)\), infinite when \(\mu\) misses the support. \(D_0\) replaces the denominator by the initial policy's visitation under \(\mu\). Expectation versions of both ratios were computed; the one correlation reported for them is for \(E_0\).

The kill test was written before the run.

- **K1.** Spearman(\(\log D_\mu\), \(\log t\)) \(\ge 0.7\) over configurations with finite \(D_\mu\).
- **K2.** Spearman(\(\log D_0\), \(\log t\)) \(\ge 0.7\) over all converged configurations, exceeding K1 by at least 0.1, and every configuration converges.
- **K3.** Within each layout, Spearman(\(\log D_0\), \(\log t\)) \(\ge 0.6\).

SUPPORTED if (K1 or K2) and K3. INCONCLUSIVE if (K1 or K2) only. KILLED if neither K1 nor K2.

## 4 Results

### 4.1 Kill test

KILLED. K1 \(= -0.67\) (\(n = 23\)). K2 \(= -0.52\) (\(n = 31\)); 5 of 36 configurations hit the cap, so the clause that every configuration converges fails as well. K3 by layout: open \(-0.63\), wall \(-0.40\), snake \(+0.70\), rooms \(-0.05\). The expectation ratio does no better: its Spearman is \(-0.47\). On the easy layouts a larger mismatch coefficient goes with faster learning.

### 4.2 Iterations to 90% of \(V^\ast(\rho)\)

Nine starts were run. The notes tabulate seven of them. Snake at \(\beta = 0.6\) is recorded in the text, at 9387 iterations, and is included below the table. "Cap" is the 50,000-iteration limit. The table marks four capped cells; five configurations in the full set of 36 hit the cap.

| layout | \(\beta = 0\) | \(\beta = 0.1\) | \(\beta = 0.3\) | \(\mu = d^{\pi^\ast}\) | uniform | far | far_only |
|---|---:|---:|---:|---:|---:|---:|---:|
| open | 964 | 892 | 902 | 1393 | 3470 | 1591 | 41365 |
| wall | 1051 | 855 | 797 | 1306 | 2901 | 2419 | 42205 |
| rooms | 1446 | 1037 | 904 | 1216 | 2445 | 1625 | cap |
| snake | cap | 19377 | 11004 | 10339 | 21530 | cap | cap |

\(D_\mu\) is infinite at a single start and at `far_only`, exactly 1 at \(\mu = d^{\pi^\ast}\), and about 30–35 at the uniform start on the easy layouts. On snake the uniform start has \(D_\mu = 3.7\).

On open and wall the theory's best case is 20–45% slower than a single start: 1393 against 964, and 1306 against 1051. Uniform takes 3470 and 2901 iterations, which the notes summarise as 2–4× slower still. On rooms the order of those two cells flips: \(d^{\pi^\ast}\) finishes in 1216 iterations and the single start in 1446, while uniform takes 2445 and `far_only` hits the cap. Across all three easy layouts the fastest tabulated cell is a small mixture, \(\beta = 0.1\) or \(\beta = 0.3\). The coefficient is not monotone in the table. Infinite, 1, about 30–35, and infinite again occur along that row of starts, and the times do not follow them.

On snake the single start, `far`, and `far_only` hit the cap. The finished times recorded in the notes are 9387 at \(\beta = 0.6\), 10339 at \(\mu = d^{\pi^\ast}\), 11004 at \(\beta = 0.3\), 19377 at \(\beta = 0.1\), and 21530 at uniform. Uniform is 2× \(d^{\pi^\ast}\), with \(D_\mu = 3.7\) against 1. The within-snake correlation on the five converged configurations is \(+0.70\).

`far_only` is the start that never sees the evaluation region: 41,365 and 42,205 iterations on open and wall, and the cap on rooms and snake. That is the infinite coefficient that matches the behaviour. The single start is also infinite, and on the easy layouts it is among the fastest cells. Infinity covers both.

### 4.3 A fixed gradient budget

With exact gradients the softmax update at state \(s\) is proportional to \(d^\pi_\mu(s)\), times an advantage weighting, and \(\sum_s d^\pi_\mu(s) = 1\). Each iteration spends a fixed budget, distributed as the current visitation. Progress on \(V^\pi(\rho)\) comes from the share that lands on states along paths from \(\rho\). Mass on states \(\rho\) never reaches is mass taken off those paths.

The uniform start spends about 99% of that budget off the states that matter for \(V^\pi(\rho)\). The single start spends all of it at the evaluation start and, as \(\pi\) improves, along the path the improving policy actually takes. On open, wall, and rooms the uniform initial policy already reaches the goal from \(\rho\): \(V^{\pi_0}(\rho)/V^\ast \approx 0.03\)–\(0.05\). There is enough signal to climb, and concentration wins. On snake the same ratio is about \(10^{-5}\). The softmax plateau (Mei et al., 2020) holds the single-start learner past the cap, because the initial policy almost never reaches the goal and the gradient at the start is exponentially small. A start that places mass in the corridor supplies gradient the start cannot. That is the regime the bound describes. The within-snake sign agrees, and it rests on five points.

The small-mixture win fits the same budget. \(\beta \in \{0.1, 0.3\}\) beats \(\beta = 0\) on every layout (open 892 against 964; wall 797 against 1051; rooms 904 against 1446), by 7–37% on the three easy layouts. A little mass off the path breaks the plateau at states the single-start learner would reach late, at a small cost to the start's share. The notes describe the curve of \(t\) against \(\beta\) as U-shaped on the easy layouts, with the minimum at \(\beta \approx 0.1\)–\(0.3\).

### 4.4 What predicts learning time

Post hoc, and not pre-registered. Over all 31 converged configurations, Spearman(\(\log t\), \(\cdot\)) is \(-0.52\) for \(D_0\), \(+0.07\) for an overlap proxy, \(-0.05\) for mass on the optimal path, and \(-0.55\) for \(\mu(\mathrm{start})\).

Restricted to open, wall, and rooms: \(D_0\) is \(-0.39\), overlap \(-0.06\), path mass \(-0.25\), and \(\mu(\mathrm{start})\) is \(-0.65\). More mass on the evaluation start, faster. Overlap and path mass rank almost nothing.

Snake, five converged configurations: \(D_0\) is \(+0.70\), overlap \(-0.10\), path mass \(+0.30\), and \(\mu(\mathrm{start})\) is \(-0.10\). The coverage ratio is the predictor only on that layout. Neither \(D_\mu\) nor \(D_0\) says which layout is which. \(V^{\pi_0}(\rho)/V^\ast\) does: about \(0.03\)–\(0.05\) where learning from \(\rho\) is fast, and about \(10^{-5}\) where the single start stalls.

## 5 Discussion and limitations

The reading that fits these 36 runs is two-regime. If the initial policy reaches the goal from \(\rho\) at all, start at \(\rho\) with a 10–30% uniform mixture. If it does not, start as close to \(d^{\pi^\ast}\) as you can, and avoid uniform in either regime. That is a description of this grid, not a prediction tested on fresh layouts. The inner product \(\sum_s d^{\pi^\ast}_\rho(s)\, d^{\pi_0}_\mu(s)\), which would credit mass on \(\rho\)'s reachable set instead of penalising its absence elsewhere, was not computed. Sample-based policy gradient on the same grids, a step-size and temperature sweep on new layouts, and a deep reset-width check were specified and not run. With sampled gradients the variance of the estimate also depends on \(\mu\), so the mixture that wins under exact gradients can move. This paper does not measure that shift.

The run is exact gradients, one step size, one initial policy (uniform softmax), four layouts, and one deterministic trajectory per cell, about 13 minutes on a laptop CPU. The 90% criterion ignores the tail, where escape from the plateau at distant states would count more. Snake's \(+0.70\) is five points, on a layout with 64 reachable states and \(V^\ast = 0.045\). Rank correlations drop the five capped runs. \(D_\mu\) remains a worst-case factor in a bound. On this grid it does not rank learning times, except on the layout where the initial policy's gradient at \(\rho\) is already gone.

## 6 Conclusion

Exact-gradient softmax policy gradient does not learn faster when the distribution-mismatch coefficient is smaller. Across 23 configurations with finite \(D_\mu\) the rank correlation with iterations to 90% of optimal is \(-0.67\); across 31 converged configurations the correlation for \(D_0\) is \(-0.52\). On open and wall the single start, \(D_\mu = \infty\), beats \(\mu = d^{\pi^\ast}\), \(D_\mu = 1\), by 20–45%, and uniform is slower by 2–4×. A 10–30% mixture is faster than the pure single start on every layout. Snake, where \(V^{\pi_0}(\rho)/V^\ast \approx 10^{-5}\), is the exception the bound describes: the within-layout correlation is \(+0.70\) on five converged runs, and the single start hits the cap. Outside that regime the quantity that ranks the runs is how much of \(\mu\) sits on the evaluation start.

## References

Agarwal, Kakade, Lee, Mahajan (2021). On the theory of policy gradient methods. JMLR. · Bhandari, Russo (2019). Global optimality guarantees for policy gradient methods. · Chen, Jiang (2019). Information-theoretic considerations in batch RL. ICML. · Ecoffet et al. (2021). First return, then explore. Nature. · Florensa et al. (2017). Reverse curriculum generation. CoRL. · Kakade, Langford (2002). Approximately optimal approximate RL. ICML. · Mei, Xiao, Szepesvári, Schuurmans (2020). On the global convergence rates of softmax policy gradient methods. ICML. · Munos, Szepesvári (2008). Finite-time bounds for fitted value iteration. JMLR.
