# R8 pilot notes

Run: tabular TD(0) prediction on 40-state chains P = αI + (1−α)P_rand, α ∈ {0, 0.5, 0.9, 0.97, 0.99}, γ ∈ {0.9, 0.99, 0.999, 0.9999}, methods plain / reward-centered / centered + Pop-Art-style target scaling, step size swept over {0.4 … 0.01} and the best kept per cell, 3 trajectory seeds, 3M steps, convergence checked every 5k steps (error = weighted RMS to the true differential value, threshold 0.2, must stay below for the rest of the run). CPU, 25 min. Log: `logs/pilot.log`. Raw: `results/runs.csv`.

## Verdict

**SUPPORTED.** K1 true, K3 true, K2 false.

## Geometric-mean convergence step (3 seeds); `>3M` = never converged

| α | λ₂ | γ | plain | centered | centered+scaled | 1/(1−γλ₂) |
|---|---|---|---|---|---|---|
| 0 | 0.23 | 0.9 | 28k | 5k | 5k | 1 |
| 0 | 0.23 | 0.99 | 451k | 5k | 5k | 1 |
| 0 | 0.23 | 0.999 | >3M | 5k | 5k | 1 |
| 0 | 0.23 | 0.9999 | >3M | 5k | 5k | 1 |
| 0.9 | 0.92 | 0.9 | 55k | 25k | 15k | 6 |
| 0.9 | 0.92 | 0.99 | 503k | 53k | 65k | 11 |
| 0.9 | 0.92 | 0.999 | >3M | 58k | 88k | 12 |
| 0.9 | 0.92 | 0.9999 | >3M | 58k | 91k | 13 |
| 0.97 | 0.98 | 0.99 | 570k | 172k | 514k | 30 |
| 0.97 | 0.98 | 0.999 | >3M | 337k | 989k | 40 |
| 0.97 | 0.98 | 0.9999 | >3M | 342k | 1.08M | 42 |
| 0.99 | 0.99 | 0.99 | 520k | 525k | >3M | 56 |
| 0.99 | 0.99 | 0.999 | >3M | 2.29M (1/3 converged) | >3M | 112 |
| 0.99 | 0.99 | 0.9999 | >3M | >3M | >3M | 125 |

Full table in `results/summary.csv`.

## What this says

- **K1.** On the fast chain (λ₂ = 0.23) centering makes γ irrelevant: γ = 0.9999 converges in the same 5k steps as γ = 0.99 (ratio 1.00), while plain TD never converges past γ = 0.99. On the slowest chain (λ₂ = 0.99) centering does not save you: ratio ≥ 5.7 (the numerator is censored at 3M, so the true ratio is larger). Centering removes the 1/(1−γ) barrier but exposes a 1/(1−γλ₂) barrier underneath it.
- **K3.** Across all 20 centered cells, log t_conv regresses on log 1/(1−γλ₂) with R² = 0.98; on log 1/(1−γ) with R² = 0.08. Excluding the censored and floor cells (t = 5k is the first check, so it is a floor) leaves 14 cells with R² = 0.99 and slope 1.23. The spectral quantity predicts the convergence time; the horizon alone does not.
- **K2 (failed).** Pop-Art-style target scaling was supposed to leave the slow-chain growth in place. It did something worse: it never converged on the slow chain at any γ, so the ratio is 1.00 (both censored) and the test is uninformative. On the mid-speed chains scaling was slower than centering alone (α = 0.97: 514k vs 172k at γ = 0.99). So the honest reading is "scaling does not help and sometimes hurts", which is stronger than K2 asked for, but K2 as written is not met.

## Caveats

- Tabular prediction, one behaviour policy, synthetic chains. No function approximation, no control.
- The chains are reversible and nearly symmetric so λ₂ is a clean real number. Real tasks have complex spectra and the relevant quantity may be a different mixing measure.
- Step size was chosen by oracle sweep per cell. A practical learner would need to pick it without the truth; the slope 1.23 > 1 may partly reflect that the best step size shrinks with λ₂.
- Convergence threshold 0.2 and the 5k check interval both quantise the result; the first-row floor of 5k is a measurement floor, not a real time.
- The scaled variant is one implementation of Pop-Art-style normalisation on tabular TD; a different normaliser could behave differently.

## 3080 plan

- Linear features on the same chains (does λ₂ of the feature-space transition operator take over the prediction?).
- Deep continuing control: MinAtar with episode resets removed and MuJoCo swimmer/hopper as continuing tasks, DQN/SAC with reward centering (Naik et al. 2024) across γ ∈ {0.99, 0.999, 0.9999}. Estimate λ₂ from the empirical transition matrix on a learned discretisation and from the critic's feature covariance; test whether it predicts the γ at which learning stalls.
- Intervention test: since the barrier is mixing, exploration bonuses or options that cut the mixing time should move the barrier. Reward centering plus a count-based bonus on the slow chain is the clean first version.
- Compute: each deep run is minutes on a 3080; the whole grid (3 tasks × 3 γ × 2 methods × 5 seeds) is under a day.
