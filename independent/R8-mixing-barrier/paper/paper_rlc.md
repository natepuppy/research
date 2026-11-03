# After Reward Centering, \(\gamma \to 1\) Still Costs the Mixing Time

*Conference-length version (RLC format). Every number in this paper comes from this folder's results.*

## Abstract

Reward centering removes the \(\bar r/(1 - \gamma)\) offset from the values, and with it the most obvious reason that \(\gamma \to 1\) is expensive. What remains is the TD operator on the complement of the constant direction. Its slowest mode contracts at \(1 - \gamma\lambda_2\), where \(\lambda_2\) is the second eigenvalue of the transition kernel, so the cost of a long horizon should track the chain's mixing time: flat in \(\gamma\) on a fast chain, growing on a slow one, and unaffected by a rescaling of the targets. Tabular TD(0) on 40-state chains \(P = \alpha I + (1 - \alpha)P_{\mathrm{rand}}\), \(\alpha \in \{0, 0.5, 0.9, 0.97, 0.99\}\), \(\gamma \in \{0.9, 0.99, 0.999, 0.9999\}\), three trajectory seeds, and 3{,}000{,}000 steps, supports that account. With centering, the ratio of convergence times \(t(\gamma = 0.9999)/t(\gamma = 0.99)\) is 1.00 on the fast chain (\(\lambda_2 = 0.226\); both cells converge at the first check, 5{,}000 steps) and 5.72 on the slow chain (\(\lambda_2 = 0.992\)). The slow numerator is censored at 3{,}000{,}000 steps, so the true ratio is larger. Across all 20 centered cells, \(\log t_{\mathrm{conv}}\) has \(R^2 = 0.98\) against \(\log 1/(1 - \gamma\lambda_2)\) and \(R^2 = 0.08\) against \(\log 1/(1 - \gamma)\). Pop-Art-style target scaling does not remove the slow-chain growth: at \(\alpha = 0.99\) it converges at \(\gamma = 0.9\) and not at \(\gamma \in \{0.99, 0.999, 0.9999\}\), so the pre-registered ratio is 1.00 with both sides censored, and that test fails. The recorded verdict is **SUPPORTED** (K1 and K3; K2 false).

## 1 Introduction

Naik et al. (2024) subtract a running estimate of the average reward inside the TD update. The values no longer have to represent \(\bar r/(1 - \gamma)\), and their experiments find Q-learning much less sensitive to \(\gamma\) near 1. A follow-up on Bellman-error centering proves convergence and says the mechanism warrants further investigation. Residual degradation near \(\gamma = 1\) is still reported in deep continuing tasks. The claim tested here is about what centering does not remove.

Write the transition kernel's eigenvalues in decreasing modulus, with \(\lambda_1 = 1\) and second eigenvalue \(\lambda_2\). Centering kills the constant mode, the one that carries the diverging offset. On the orthogonal complement the discounted resolvent still sees a factor \(\gamma\lambda_2\), and the slowest remaining contraction per step is \(1 - \gamma\lambda_2\). The time to learn the centered value should therefore scale as \(1/(1 - \gamma\lambda_2)\). On a fast chain \(\gamma\lambda_2\) stays bounded away from 1 for every \(\gamma\) in the usual range, and centering makes \(\gamma \to 1\) free. On a slow chain \(\gamma\lambda_2 \to \lambda_2 \approx 1\), and the cost grows until it saturates at the mixing time. A rescaling of targets changes the magnitude of the update and does not change that conditioning.

**Contributions.** (1) A controlled grid: one family of reversible chains whose \(\lambda_2\) is set by a self-loop weight, tabular TD(0) along a single trajectory, and an oracle step size per cell. (2) Centering removes the horizon barrier on the fast chain (ratio 1.00, against a pre-registered ceiling of 1.5) and leaves a mixing barrier on the slow chain (ratio 5.72, against a floor of 3). (3) The spectral predictor beats the horizon: \(R^2 = 0.98\) against \(0.08\), and 0.99 with slope 1.23 on the 14 cells that are neither censored nor sitting on the measurement floor. (4) The negative result on scaling. K2 asked only that scaling preserve the slow-chain growth. On \(\alpha = 0.99\) the scaled \(\gamma = 0.9999\) and \(\gamma = 0.99\) cells are both at the budget, the recorded ratio is 1.00, and K2 is false. The verdict is SUPPORTED because the pre-registered rule requires K1 and K3.

## 2 Related work

Reward centering is Naik et al. (2024, arXiv:2405.09999). Bellman-error centering is the follow-up cited as arXiv:2502.03104, which establishes convergence and leaves the finite-time mechanism open. The scaling variant here is a tabular, Pop-Art-style normalisation of the TD target by a running standard deviation. It is one implementation; the pilot notes say a different normaliser could behave differently. The experiment is prediction, not control, and the chain is the behaviour chain.

## 3 Setup

Each chain has 40 states. \(P_{\mathrm{rand}}\) is drawn row-wise from a Dirichlet with concentration 0.5, and \(P = \alpha I + (1 - \alpha)P_{\mathrm{rand}}\). Rewards are uniform on \([1, 2]\), offset so that plain discounted TD has an \(\bar r/(1 - \gamma)\) term to pay for. \(\lambda_2\) is the second-largest absolute eigenvalue of \(P\). The five self-loop weights produce mean \(\lambda_2\) of 0.226, 0.605, 0.921, 0.976, and 0.992. The fast chain in the kill test is \(\alpha = 0\); the slow chain is \(\alpha = 0.99\).

Three methods walk the same trajectory. Plain TD(0) is the discounted update. Centered TD subtracts a running average-reward estimate from the reward, with centering step 0.1 times the TD step size. Centered+scaled does the same and divides the TD error by a running standard deviation of the bootstrap targets (smoothing constant 0.01). The step size is chosen from \(\{0.4, 0.2, 0.1, 0.05, 0.02, 0.01\}\), six values, and the best is kept per cell. That choice uses the true error, so it is an oracle.

The error is the stationary-weighted RMS of the centered value estimate against the true centered value \(v_c\), which solves \((I - \gamma P)v_c = r - \bar r\) and stays finite as \(\gamma \to 1\), divided by the stationary standard deviation of \(v_c\). Convergence time is the first check after which the error stays below 0.2 for the rest of the run. Checks are every 5{,}000 steps. The budget is 3{,}000{,}000 steps. A cell that never qualifies is entered at the budget. Reported times are geometric means over 3 seeds. The run is CPU tabular prediction, about 25 minutes.

- **K1.** With centering, \(t(0.9999)/t(0.99) < 1.5\) on the fast chain and \(> 3\) on the slow chain.
- **K2.** The same ratio with scaling stays \(> 2\) on the slow chain.
- **K3.** Over the centered grid, \(R^2\) of \(\log t_{\mathrm{conv}}\) on \(\log 1/(1 - \gamma\lambda_2)\) exceeds 0.8, and exceeds the \(R^2\) on \(\log 1/(1 - \gamma)\) by at least 0.1.

SUPPORTED if K1 and K3. INCONCLUSIVE if exactly one. KILLED if neither.

## 4 Results

The recorded line is `KILL TEST: SUPPORTED`, with K1 true (fast ratio 1.00, slow ratio 5.72), K2 false (scaled slow ratio 1.00), and K3 true (\(R^2 = 0.98\) against \(0.08\)).

| \(\alpha\) | \(\lambda_2\) | \(\gamma\) | plain | centered | centered+scaled | \(1/(1-\gamma\lambda_2)\) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.226 | 0.9 | 28231 | 5000 | 5000 | 1 |
| 0.00 | 0.226 | 0.99 | 451370 | 5000 | 5000 | 1 |
| 0.00 | 0.226 | 0.999 | 3000000 | 5000 | 5000 | 1 |
| 0.00 | 0.226 | 0.9999 | 3000000 | 5000 | 5000 | 1 |
| 0.50 | 0.605 | 0.9 | 54848 | 10000 | 7937 | 2 |
| 0.50 | 0.605 | 0.99 | 506515 | 10000 | 10000 | 2 |
| 0.50 | 0.605 | 0.999 | 3000000 | 10000 | 11447 | 3 |
| 0.50 | 0.605 | 0.9999 | 3000000 | 10000 | 11447 | 3 |
| 0.90 | 0.921 | 0.9 | 55464 | 25000 | 15000 | 6 |
| 0.90 | 0.921 | 0.99 | 503099 | 53133 | 64633 | 11 |
| 0.90 | 0.921 | 0.999 | 3000000 | 58285 | 88109 | 12 |
| 0.90 | 0.921 | 0.9999 | 3000000 | 58285 | 91458 | 13 |
| 0.97 | 0.976 | 0.9 | 38259 | 29720 | 34760 | 8 |
| 0.97 | 0.976 | 0.99 | 569868 | 171554 | 514342 | 30 |
| 0.97 | 0.976 | 0.999 | 3000000 | 337153 | 989005 | 40 |
| 0.97 | 0.976 | 0.9999 | 3000000 | 342084 | 1076686 | 42 |
| 0.99 | 0.992 | 0.9 | 38259 | 43089 | 51614 | 9 |
| 0.99 | 0.992 | 0.99 | 519621 | 524792 | 3000000 | 56 |
| 0.99 | 0.992 | 0.999 | 3000000 | 2290382 | 3000000 | 112 |
| 0.99 | 0.992 | 0.9999 | 3000000 | 3000000 | 3000000 | 125 |

*Geometric-mean convergence step, 3 seeds. A time of 3000000 is the budget: that cell had not converged. The centered cell at \(\alpha = 0.99\), \(\gamma = 0.999\) converged on 1 of 3 seeds. The rightmost column is the printed spectral predictor.*

**Fast chain.** Plain TD converges at 28{,}231 steps when \(\gamma = 0.9\) and at 451{,}370 when \(\gamma = 0.99\), and does not converge at 0.999 or 0.9999. Centered TD converges at 5{,}000 steps at every \(\gamma\), which is the first check, so the measurement is a floor. The K1 ratio is \(5000/5000 = 1.00\). The same floor appears for the scaled method. From \(\lambda_2 = 0.226\) the spectral predictor is 1 at every \(\gamma\) in the grid: after centering there is nothing left for \(\gamma\) to do, and the runs agree.

**Slow chain.** At \(\alpha = 0.99\), centered TD takes 524{,}792 steps at \(\gamma = 0.99\), 2{,}290{,}382 steps at \(\gamma = 0.999\) with one seed in three converged, and does not converge at \(\gamma = 0.9999\). The recorded ratio is 5.72, which clears the floor of 3. Because the numerator is the censoring point, the true ratio is larger. Plain TD is already at the budget for \(\gamma \ge 0.999\). Scaling on this chain converges at \(\gamma = 0.9\) (51{,}614 steps) and sits on the budget at \(\gamma \in \{0.99, 0.999, 0.9999\}\). The K2 ratio, \(\gamma = 0.9999\) over \(\gamma = 0.99\), is 1.00, both sides censored, which does not exceed 2. The test is uninformative in the form it was written: the two times it ratios are both the budget.

**The middle of the grid.** At \(\alpha = 0.9\) (\(\lambda_2 = 0.921\)) centered times go 25{,}000, 53{,}133, 58{,}285, 58{,}285 as \(\gamma\) goes from 0.9 to 0.9999, against a predictor that goes 6, 11, 12, 13. At \(\alpha = 0.97\) they go 29{,}720, 171{,}554, 337{,}153, 342{,}084 against 8, 30, 40, 42. The growth with \(\gamma\) flattens once \(\gamma\) is close enough to 1 that \(\gamma\lambda_2\) has stopped moving, which is the saturation-at-the-mixing-time half of the claim. Scaling is slower than centering alone on these chains: at \(\alpha = 0.97\), \(\gamma = 0.99\), 514{,}342 steps against 171{,}554.

**The regression.** On all 20 centered cells, \(\log t_{\mathrm{conv}}\) on \(\log 1/(1 - \gamma\lambda_2)\) has \(R^2 = 0.98\). On \(\log 1/(1 - \gamma)\) it has \(R^2 = 0.08\). The pre-registered bar is a gap of at least 0.1 in \(R^2\). Dropping censored cells and the floor cells (the 5{,}000-step entries, which are the first check rather than a measured time) leaves 14 cells, \(R^2 = 0.99\), slope 1.23. The horizon does not predict the time. The spectral gap does. The notes record that the slope may partly reflect the oracle step size shrinking as \(\lambda_2\) grows.

## 5 Reading

Centering removes the \(1/(1 - \gamma)\) barrier and leaves a \(1/(1 - \gamma\lambda_2)\) barrier. On the fast chain the second barrier is already saturated at \(\gamma = 0.9\), so moving \(\gamma\) from 0.99 to 0.9999 changes nothing that this experiment can see. On the slow chain the second barrier is the whole story, and a 3{,}000{,}000-step budget is not enough at \(\gamma = 0.9999\). The practical diagnostic the claim suggests is an estimate of \(\lambda_2\) of the on-policy chain: \(\gamma \approx 1\) is cheap when that eigenvalue is bounded away from 1, and it is the mixing time when it is not. This pilot computes \(\lambda_2\) from the known kernel. It does not estimate it from samples.

K2's failure is a result about the scaled method, not a rescue of the horizon story. The pre-registered sentence was that scaling would leave the slow-chain growth in place, with the ratio still above 2. The ratio it forms is 1.00, because neither \(\gamma = 0.99\) nor \(\gamma = 0.9999\) converges under scaling on \(\alpha = 0.99\). At \(\gamma = 0.9\) that same chain does converge under scaling, at 51{,}614 steps, so the failure is the high-\(\gamma\) pair the test named. Where both methods converge, scaling is slower (514{,}342 against 171{,}554 at \(\alpha = 0.97\), \(\gamma = 0.99\), the cell the notes single out). One implementation of Pop-Art-style normalisation on tabular TD is not a theorem about every normaliser.

## 6 Limitations

The chains are tabular, reversible, and nearly symmetric, so \(\lambda_2\) is a clean real number. A real task has a complex spectrum, and the relevant mixing measure may be different. There is one behaviour policy, no features, and no control. The step size is an oracle per cell; a learner that has to pick it would add a second \(\gamma\)-dependence, and the slope 1.23 may already contain some of that. The threshold 0.2 and the 5{,}000-step check quantise the times, and the fast centered row is a floor. Censoring at 3{,}000{,}000 steps makes the slow ratio a lower bound. The scaled method is one normaliser.

The follow-up that would put linear features on these chains, and that would run deep continuing control (MinAtar with resets removed, MuJoCo swimmer and hopper, DQN or SAC, \(\lambda_2\) from a learned feature covariance), was not run. No deep curve in this paper was produced.

## 7 Conclusion

On these 40-state chains, reward centering makes \(\gamma \to 1\) free exactly when the chain mixes quickly, and leaves a barrier that tracks \(1/(1 - \gamma\lambda_2)\) when it does not. The fast ratio is 1.00, the slow ratio is 5.72 and censored, and the spectral predictor has \(R^2 = 0.98\) against \(0.08\) for the horizon. Target scaling does not cut that barrier: its slow-chain ratio is the censored 1.00, and that pre-registered check fails. The kill test is SUPPORTED on the two claims it required.

## References

Naik et al. (2024). Reward centering. arXiv:2405.09999. · Bellman Error Centering. arXiv:2502.03104. · van Hasselt et al. (2016). Learning values across many orders of magnitude (Pop-Art), the normalisation this pilot's scaled variant imitates.
