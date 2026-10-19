# R8: The mixing-time barrier. What γ → 1 still costs after reward centering

**Question.** Reward centering (Naik et al. 2024, arXiv 2405.09999) removes the r̄/(1−γ) offset from values and makes Q-learning robust to γ close to 1 in their experiments; follow-ups (Bellman Error Centering, 2502.03104) prove convergence but say the mechanism "warrants further investigation," and residual degradation near γ = 1 is still reported in deep continuing tasks. Our claim: what centering leaves behind is the TD operator on the complement of the constant direction, whose slowest mode contracts at 1 − γλ₂. So after centering the cost of γ → 1 should be set by the chain's spectral gap (mixing time), not by 1/(1−γ): flat in γ on fast-mixing chains, growing on slow ones until it saturates at the mixing time. Value scaling fixes magnitude, not conditioning, so it should not remove that growth. If this holds it gives a diagnostic (estimate λ₂ of the on-policy chain) that says when γ ≈ 1 is free and when it is not.

**Status: SUPPORTED.** Centering removes the γ dependence on the fast chain (ratio 1.00) but not on the slow chain (ratio ≥ 5.7); log t_conv is predicted by log 1/(1−γλ₂) with R² = 0.98 vs 0.08 for 1/(1−γ). Target scaling (K2) did not help and never converged on the slow chain. Details in `notes/pilot_notes.md`.

**Pilot.** Laptop pilot (CPU, tabular TD(0) along one Markov trajectory, chains with controlled λ₂). See `results/verdict.txt` and `notes/pilot_notes.md`. The paper version adds linear features and a deep continuing task (MinAtar / MuJoCo with resets removed) on the 3080, with λ₂ estimated from the critic's feature covariance.

## Pre-registered kill test (written after a 40k-step smoke run, before the pilot)

Chains: 40 states, P = αI + (1−α)P_rand, α ∈ {0, 0.5, 0.9, 0.97, 0.99}; rewards in [1, 2]. γ ∈ {0.9, 0.99, 0.999, 0.9999}. Methods: `plain`, `centered`, `centered+scaled`, each with the best of six step sizes. Error: RMS error of the centered value estimate under the stationary distribution, divided by the std of the true centered value. Convergence time: first step after which error stays below 0.2.

- **K1.** With centering, the ratio t_conv(γ = 0.9999) / t_conv(γ = 0.99) is < 1.5 on the fast chain (α = 0) and > 3 on the slow chain (α = 0.99).
- **K2.** The same ratio with scaling stays > 2 on the slow chain (scaling does not remove the growth).
- **K3.** Across the grid, log t_conv (centered) is predicted by log 1/(1−γλ₂) with R² > 0.8, and beats log 1/(1−γ) by at least 0.1 in R².

SUPPORTED if K1 and K3. INCONCLUSIVE if exactly one. KILLED if neither.

## Layout

| folder | contents |
|---|---|
| `code/` | `r8_mixing.py` (chains, vectorised TD variants, convergence times, verdict); `code/lib/io_utils.py` |
| `data/` | empty |
| `results/` | `runs.csv`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | empty |

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r8_mixing.py --smoke                         # ~15 s
.venv/bin/python code/r8_mixing.py --steps 1500000 --check-every 5000   # pilot, ~30 min
.venv/bin/python code/r8_mixing.py --summarize
```
