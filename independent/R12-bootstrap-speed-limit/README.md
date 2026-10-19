# R12: The speed limit of bootstrapping

**Question.** Deep value learning moves information backwards from where reward happens one bootstrap hop at a time. Two standard pieces of machinery set the speed of a hop. A target network updated every K gradient steps means a value can only propagate one n-step hop per K updates. Uniform replay means a transition at the current frontier of propagation is sampled only once every ~L/B updates (L transitions in an episode, batch B), and a hop cannot happen before the frontier is sampled. So for a reward L steps from the start,

  t_prop ≈ ⌈L/n⌉ · max(K, c·L/B),

with c an optimiser-dependent constant. The law has two regimes (target-limited and replay-limited) and a crossover at K ≈ c·L/B. Nobody has stated it, measured it, or used it to set K and n jointly, even though it predicts, for example, that Nature-DQN settings (K = 2.5k updates, B = 32, 1-step) sit squarely in the target-limited regime for any episode shorter than ~80 steps and in the replay-limited one for longer episodes, and that n-step returns buy an exact factor n in either regime while prioritised replay only helps in one of them.

**Status: pre-registered form KILLED; refined additive law CONFIRMED out of sample.** Target-limited regime is exactly (⌈L/n⌉−1)·K (median ratio 1.002); replay term adds c·⌈L/n⌉·L/B; the additive law, with c fixed from the pilot, fits a disjoint confirmation grid with identity R² = 0.997 (SGD) and 0.971 (Adam). Details in `notes/pilot_notes.md`, draft in `paper/main.md`.

**Pilot.** Laptop pilot (CPU, tabular TD on deterministic chains, 256 configurations × 3 seeds). See `results/verdict.txt` and `notes/pilot_notes.md`. Paper version: stochastic chains and gridworlds (frontier becomes a distribution), linear and deep function approximation (generalisation makes the effective hop longer than n; we measure n_eff), DQN on sparse MinAtar variants on the 3080, and the design rule that follows.

## Pre-registered kill test (written before the pilot ran)

Grid: L ∈ {8, 16, 32, 64}, n ∈ {1, 2, 4, 8}, K ∈ {1, 10, 100, 1000}, B ∈ {4, 32}, optimiser ∈ {per-hit SGD α = 0.3, per-state Adam lr = 0.05}; γ = 0.99; 3 seeds; t_prop = first update at which V(start) ≥ half its true value; cap 400k updates. Cells that do not converge within the cap are excluded from the fits. c is fitted by maximising R² over a log-grid.

- **K1 (the law).** Per optimiser, R² of log t_prop on log[⌈L/n⌉ · max(K, c·L/B)] ≥ 0.9.
- **K2 (two regimes).** Within groups of fixed (L, n, B): in target-limited cells (K ≥ 10·c·L/B) the slope of log t_prop on log K is ≥ 0.8; in replay-limited cells (K ≤ 0.1·c·L/B) it is ≤ 0.2.
- **K3 (n-step is exact).** In target-limited cells, within groups of fixed (L, K, B), the slope of log t_prop on log n is in [−1.3, −0.7].

SUPPORTED if K1 and K2 hold for both optimisers. INCONCLUSIVE if exactly one of K1, K2 holds for both. KILLED otherwise.

## Confirmation run (pre-registered after the pilot, before the run)

The pilot was KILLED by the rule above (SGD R² = 0.883; the Adam replay-limited slope could not be computed because the grid had only one K value in that regime). The pilot data pointed to a sharper, additive law: in the target-limited regime t_prop = (⌈L/n⌉ − 1)·K to within 0.2% (median), and the replay term adds rather than competes:

  t_prop = (⌈L/n⌉ − 1)·K + c·⌈L/n⌉·L/B,   c_SGD = 2.63, c_Adam = 4.21 (fitted on the pilot).

Post-hoc fit on the pilot grid: identity-fit R² in log space 0.996 (SGD), 0.924 (Adam); see `results/posthoc.txt`. Because this law was found in the data, it is tested on a fresh grid that shares no values with the pilot: L ∈ {12, 24, 48, 96}, n ∈ {1, 3, 6}, K ∈ {3, 30, 300, 3000}, B ∈ {8, 16}, 3 new seeds, c held fixed.

- **C1.** Out-of-sample identity-fit R² (log space, no free slope or intercept) ≥ 0.95 for SGD and ≥ 0.90 for Adam.

SUPPORTED if C1 holds for both; INCONCLUSIVE if for one; KILLED if neither. Results in `results/confirm/`.

## Layout

| folder | contents |
|---|---|
| `code/` | `r12_speed_limit.py`; `code/lib/io_utils.py` |
| `data/` | empty |
| `results/` | `runs.csv` (config × seed), `summary.csv` (config, with the fitted prediction), `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r12_speed_limit.py --smoke       # ~20 s
.venv/bin/python code/r12_speed_limit.py               # pilot, ~20 min on an M1 Pro (CPU)
.venv/bin/python code/r12_speed_limit.py --summarize
```
