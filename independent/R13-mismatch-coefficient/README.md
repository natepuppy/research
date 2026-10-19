# R13: Does the distribution-mismatch coefficient predict how fast policy gradient learns?

**Question.** The convergence theory of policy gradient methods (Kakade & Langford 2002; Agarwal, Kakade, Lee & Mahajan 2021; Mei et al. 2020) is written in terms of a *distribution-mismatch coefficient*: the iteration complexity scales with D_μ = ‖d^{π*}_ρ / μ‖_∞, the worst-case ratio between where the optimal policy goes (started from the evaluation distribution ρ) and where the learner is started (μ). The theory is the standard justification for "exploratory restarts" and reset-distribution design. Nobody has checked whether the coefficient actually predicts learning time, even in the tabular exact-gradient setting where it is derived. Two things could go wrong: the coefficient is infinite whenever μ misses a state on the optimal path (e.g. a single fixed start), yet PG learns fine from a single start; and it ignores where the *initial policy* takes the learner, which is what determines early gradient signal. We compute D_μ exactly, alongside D_0 = ‖d^{π*}_ρ / d^{π_0}_μ‖_∞ (mismatch against the initial policy's visitation under μ), and ask which one predicts iterations-to-90%-optimal across four gridworld layouts and nine start distributions.

**Status: KILLED, with a reversal worth keeping.** Laptop pilot (CPU, exact-gradient softmax PG, 4 layouts × 9 start distributions, 36 configs). Spearman(log D_μ, log t) = −0.67 and (log D_0, log t) = −0.52: the wrong sign. On the three easy layouts the single start (D_μ = ∞) beats μ = d^{π*} (D_μ = 1) by 20–45% and beats uniform by 2–4×; a 10–30% uniform mixture on the single start is fastest. Only on the hard-exploration `snake` layout does the coefficient predict (+0.70, n = 5), and there d^{π*} is the best start. Post hoc: μ's mass on the evaluation start is the best predictor outside hard exploration (−0.65). See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`, `paper/main.md`. Paper version: sample-based PG (REINFORCE / PPO) on the same grids with 10 seeds; deep check on CartPole and MinAtar with reset-distribution width as the dial, coefficients estimated from rollouts of the final policy; the "coverage you can afford" rule that follows.

## Pre-registered kill test (written before the pilot ran)

Design: layouts `open`, `wall`, `snake`, `rooms` (10×10, slip 0.1, γ = 0.95, goal reward 1, absorbing goal); evaluation start ρ = δ_{(0,0)}; training starts μ ∈ {(1−β)δ + β·Uniform for β ∈ {0, 0.02, 0.1, 0.3, 0.6, 1}, d^{π*}_ρ itself, "far" (uniform over the half of states the optimal policy visits least, plus 10% on the start), "far_only"}. Softmax PG with exact gradient, η = 2, cap 50k iterations. t = iterations until V^π(ρ) ≥ 0.9 V*(ρ). Configurations that hit the cap are excluded from rank correlations and reported.

- **K1.** Spearman(log D_μ, log t) ≥ 0.7 over configurations with finite D_μ.
- **K2.** Spearman(log D_0, log t) ≥ 0.7 over all converged configurations, exceeds K1's correlation by ≥ 0.1, and every configuration converges (D_0 is finite wherever learning is possible).
- **K3.** Within each layout, Spearman(log D_0, log t) ≥ 0.6.

SUPPORTED if (K1 or K2) and K3. INCONCLUSIVE if (K1 or K2) only. KILLED if neither K1 nor K2.

## Layout

| folder | contents |
|---|---|
| `code/` | `r13_mismatch.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r13_mismatch.py --smoke     # ~10 s
.venv/bin/python code/r13_mismatch.py             # pilot, ~15 min CPU
.venv/bin/python code/r13_mismatch.py --summarize
```
