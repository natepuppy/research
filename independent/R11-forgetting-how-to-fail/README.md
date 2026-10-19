# R11: Forgetting how to fail

**Question.** Value-based agents on tasks with a time limit (CartPole, most MuJoCo locomotion, many MinAtar games) have a well-known habit: they reach the ceiling, then collapse. The folk explanation is that once the agent is good, failures stop happening, failure transitions leave the FIFO buffer, and the network forgets what failure looks like. Nobody has tested that explanation with a measurement that has ground truth. We give it one. In CartPole with reward 1 per step, a (state, action) pair whose action terminates the episode has Q*(s, a) = 1 exactly, for every policy. A fixed probe set of such pairs, collected once from a random policy, gives a scalar with a known true value that can be read off the network at any point in training: the *failure-Q inflation*. If the folk story is right, inflation should rise as terminals leave the buffer, should precede collapses, and pinning the terminals in place should prevent them.

**Status: INCONCLUSIVE by rule; the fix is strongly supported.** Terminal reservoir cuts collapses from 2.46 to 0.08 per run and lifts final score 119 → 154 (19/24 paired wins, p = 0.007); random old data does nothing (K2, K3 pass). Failure-Q drifts from 2 to 35 against a true value of 1 and tracks the loss of terminals (ρ = −0.53), but does not time individual collapses (AUROC 0.61), so K1 half-fails. Details in `notes/pilot_notes.md`, draft in `paper/main.md`.

**Pilot.** Laptop pilot (CPU, batched DQN on CartPole, 3 conditions × 24 seeds). See `results/verdict.txt` and `notes/pilot_notes.md`. Paper version: MinAtar and MuJoCo (with the Pardo et al. time-limit bootstrapping done correctly, so the effect is not confounded with truncation handling) on the 3080; forks from R7 to make the causal claim per run.

Adjacent stub in `research/ideas` (N1-11, "within-task forgetting of stepping stones") is about forgetting intermediate *skills*. This is about forgetting *terminals*, the only states whose values are anchored to a constant, and it comes with an exact ground truth and a one-line fix.

## Pre-registered kill test (written before the pilot ran)

Design: DQN, 40k steps, greedy evaluation every 1k steps (4 episodes). Conditions: `control`; `reservoir_term` (the first 256 terminal transitions the run sees are kept forever and 4 of every 32 minibatch samples come from them); `reservoir_rand` (256 early non-selected transitions kept the same way; controls for "any old data helps"). Same seeds across conditions. A collapse = an evaluation ≤ 100 after the run has had an evaluation ≥ 175. Final score = mean of the last 5 evaluations.

- **K1 (mechanism).** In control runs, failure-Q inflation at an evaluation where the run is good (≥ 175) predicts a collapse within the next two evaluations with AUROC ≥ 0.7, and failure-Q inflation is negatively rank-correlated with the buffer's terminal fraction (Spearman ≤ −0.5).
- **K2 (fix).** `reservoir_term` reduces collapses per run by ≥ 40% relative to control and does not lower the final score by more than 5 points.
- **K3 (specificity).** `reservoir_rand` achieves less than half of `reservoir_term`'s reduction.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither.

## Layout

| folder | contents |
|---|---|
| `code/` | `r11_forgetting.py`; `code/lib/dqn_batched.py` (vendored from R7); `code/lib/dqn_reservoir.py` (reservoir + probe set); `code/lib/io_utils.py` |
| `data/` | empty |
| `results/` | `runs.csv` (run × eval), `per_run.csv`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r11_forgetting.py --smoke        # ~10 s
.venv/bin/python code/r11_forgetting.py                # pilot, ~15 min on an M1 Pro (CPU)
.venv/bin/python code/r11_forgetting.py --summarize
```
