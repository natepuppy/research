# R9: The rescue window. Unlucky or doomed?

**Question.** Plasticity-loss work (primacy bias, Nikishin et al. 2022; dormant neurons; resets) shows that *periodic* resets help on average. The seed-variance literature shows some runs end badly for no visible reason. Nobody has asked the causal question in between: take a run that *will* end badly, intervene once at step t, and ask whether it can still be saved, and until when. The fork machinery from R7 makes this exact: the parent's fate is known (we ran it to the end), children share everything up to t, and the intervention is the only difference besides fresh randomness. If there is a window after which no cheap intervention rescues a run, that is a statement about when plasticity is lost in a single run, not on average, and it gives a budget rule: evaluate at t*, prune or reset, do not wait.

**Status: KILLED.** No intervention raised rescue probability by ≥ 0.2 at any fork ≤ 0.6T. The rescue criterion was set too high for CartPole (good runs themselves miss it 78–100% of the time), so the test had no power; the mean scores show a late window for a full reset (15k–20k) that closes by 25k. Rerun design in `notes/pilot_notes.md`.

**Pilot.** Laptop pilot (CPU, batched DQN on CartPole; parents are identical to R7's). See `results/verdict.txt` and `notes/pilot_notes.md`. Paper version: MinAtar / DMC on the 3080, with interventions that cover the plasticity toolbox (partial and full resets, LR drops, shrink-and-perturb, buffer re-weighting).

## Pre-registered kill test (written before the pilot ran)

Parents: 16 seeds × 30k steps, split into bad (bottom third by final score) and good (top third). Forks: 5k, 10k, 15k, 20k, 25k. Children per (parent, fork, condition): 3, all with fresh exploration and minibatch seeds. Conditions: `control`, `reset_last` (output layer re-initialised), `reset_all` (fresh network and optimiser, buffer kept), `lr_drop` (learning rate halved). Rescue = child final ≥ median final of the good parents.

- **K1.** Some intervention raises the rescue probability of bad parents by ≥ 0.2 over control at a fork ≤ 0.6T.
- **K2.** The window closes: for that intervention, rescue probability at the last fork is ≥ 0.2 below its best early value.
- **K3.** (secondary) The same intervention does not raise the harm rate on good parents by more than 0.15 at any fork.

SUPPORTED if K1 and K2. INCONCLUSIVE if only K1. KILLED if not K1.

## Layout

| folder | contents |
|---|---|
| `code/` | `r9_rescue.py`; `code/lib/dqn_batched.py` (vendored from R7); `code/lib/io_utils.py` |
| `data/` | empty |
| `results/` | `runs.csv`, `parents.json`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | empty |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r9_rescue.py --smoke        # ~30 s
.venv/bin/python code/r9_rescue.py                # pilot, ~40 min on an M1 Pro (CPU)
.venv/bin/python code/r9_rescue.py --summarize
```
