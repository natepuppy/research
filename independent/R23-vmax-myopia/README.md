# R23: Rainbow is myopic by construction — the value support of C51 is a horizon

**Question.** C51 and every Rainbow descendant represent the return distribution on a fixed support [V_min, V_max] = [−10, 10] with 51 atoms (Bellemare et al. 2017; Hessel et al. 2018; also Dopamine's defaults). Atari rewards are clipped to {−1, 0, 1} and γ = 0.99, so the discounted return of a competent policy in a dense-reward game is tens of reward units, well above 10. The categorical projection clips every bootstrapped atom to V_max, so the learned value is the fixed point of a *clipped* Bellman operator, Q_c(s, a) = E[min(V_max, r + γ Z_c(s′))]. The consequence is exact and, as far as we can find, unstated beyond blog caveats: wherever the continuation values of two actions both exceed V_max, the clipped agent prefers the larger immediate reward and is otherwise indifferent — it is a one-step-greedy agent in the saturated region, and its planning horizon is "until V_max reward units are in view" rather than 1/(1 − γ). We test the claim with an exact tabular C51 and a scalar clipped-Q control on fork tasks where a patient action (0 now, large continuation) competes with a myopic one (1 now, smaller continuation), sweeping V_max, and compare the learned choice to the clipped fixed point computed by clipped value iteration.

**Status.** Laptop pilot run. **KILLED by rule** (K1 35/40, K3 28/33, K4 39/40) with **K2 80/80**: the C51 mean equals the clipped fixed point in every cell and action. Post-hoc (`results/posthoc.txt`): in all 18 both-saturate cells the clipped preference for the immediate reward is exactly (1 − γ)V_max, so the pre-registered 0.05 indifference threshold sat inside the band it was meant to split at V_max ≤ 5; all K1/K3 misses are those cells plus one atom-resolution cell at V_max = 10; 0/18 saturated cells chose `patient`; the patient boundary is at V_max = 1 + γV_M in all four forks as predicted. Mechanism confirmed, test mis-specified, not re-run. See `notes/pilot_notes.md`.

Laptop pilot (CPU, tabular categorical learner with the exact C51 projection, exact clipped value iteration as the prediction, no seeds — deterministic MDPs, full sweeps). Paper version on the 3080: Rainbow (Dopamine) on MinAtar and on four Atari games chosen by the predicted saturation (Atlantis, Enduro, Ms. Pac-Man, Q*bert) with V_max ∈ {10, 100}, 3 seeds, 10M frames each (~70 GPU-hours), plus a saturation audit of trained Rainbow agents: the fraction of visited states whose value sits on the top atom.

Prior statements: the C51 paper notes returns outside the support are projected to the boundary; rlbook.ai and similar notes observe that ±10 "does not bracket the achievable ±100" and that saturation is tolerable; QR-DQN removed the fixed support for flexibility. Nobody has written the clipped fixed point, derived the myopia, or tested the policy consequence. Related: reward clipping changes the objective (Pop-Art, van Hasselt et al. 2016); Huber as a risk preference (R16, this lab) — both are value-level distortions of the default Atari agent; this is a third, and it compounds with reward clipping (clipping makes rewards dense in reward units, which is what pushes values past V_max). Lab stubs: none.

## Pre-registered kill test (written before the pilot ran)

Design. Fork MDP, γ = 0.99, deterministic. Start s₀ with two actions. `myopic`: reward 1 now, then a corridor paying 1 per step for M steps (continuation value V_M = (1 − γ^M)/(1 − γ)), then terminal. `patient`: reward 0 now, then a corridor paying 1 per step for P steps (V_P), P > M + 1 so patient is optimal under the true objective: true Q(patient) = γV_P > 1 + γV_M = Q(myopic). Grid: (M, P) ∈ {(5, 12), (10, 25), (20, 50), (40, 100)} × V_max ∈ {3, 5, 8, 10, 15, 20, 30, 50, 100, 200}. Learners: (a) tabular C51 with 51 atoms on [−V_max, V_max], synchronous full sweeps (every (s, a) updated every step with its exact successor), learning rate 0.1 on the categorical cross-entropy (i.e. p ← p + 0.1(Π target − p)), run to convergence (5000 sweeps); (b) scalar Q-learning with the clipped target min(V_max, r + γ max Q′), same schedule. Prediction: clipped value iteration Q_c = fixed point of T_c Q = min(V_max, r + γ max_a′ Q(s′, ·)), computed exactly; the predicted choice at s₀ is argmax Q_c (ties → "indifferent").

- **K1 (the clipped fixed point predicts the choice).** The C51 agent's greedy choice at s₀ agrees with the clipped-VI prediction in ≥ 95% of the 40 cells (both the prediction and the learned choice call a cell "indifferent" when |ΔQ| < 0.05; an indifferent prediction agrees only with an indifferent learned choice). *Smoke-test change:* the prediction threshold was first written as 10⁻⁹, which labelled V_max = 3 cells "myopic" by 0.03 while the learner's 0.05 threshold called them indifferent; harmonised to 0.05 before the pilot. Sweeps raised from 5000 to 20 000 because the 100-state corridor had not converged at 300 smoke sweeps and 5000 looked marginal.
- **K2 (the values match).** Across all cells and both actions at s₀, the C51 mean value is within 5% of Q_c (relative to V_max), in ≥ 95% of (cell, action) pairs.
- **K3 (myopia appears exactly where the continuations saturate).** In every cell with V_max < 1 + γV_M (both continuations saturate) the C51 agent picks `myopic`; in every cell with V_max > γV_P (nothing saturates) it picks `patient`.
- **K4 (it is the clip, not the distribution).** The scalar clipped-Q control agrees with C51's choice in every cell.

SUPPORTED if K1, K2 and K3. INCONCLUSIVE if two. KILLED otherwise. K4 is reported. Also reported: the Atari reading — for V_max = 10, γ = 0.99 and clipped rewards, the saturation condition in reward units.

## Layout

| folder | contents |
|---|---|
| `code/` | `r23_vmax.py`; `code/lib/io_utils.py` |
| `results/` | `cells.csv` (per cell: prediction, C51 choice and values, scalar choice), `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r23_vmax.py --smoke
.venv/bin/python code/r23_vmax.py             # pilot, ~2 min CPU
```
