# R7: The fork test. When is a deep-RL run's fate sealed, and by which noise?

**Question.** Deep RL results vary a lot across seeds (Henderson 2018; Agarwal 2021), and identical initial weights still diverge (Appl. Sci. 2025), so the variance is created *during* training. Nobody has measured *when*. Fork a run at step t into children that share everything up to t and differ only in their future randomness: if the children end up alike, the fate was sealed by t. Doing this for several t gives a lock-in curve; reseeding only the exploration stream or only the minibatch stream says which noise seals it; comparing the lock-in to what the learning curve shows at t says whether the fate is visible when it is decided. This is Frankle et al.'s instability analysis (2020) brought to RL, with RL's extra noise source.

**Status: SUPPORTED (borderline).** Lock-in ICC reaches 0.63 at T/2 while the parent's visible score explains 0% of children's fate (K1, K3 pass; K2 passes by rule but the sign flips, so it is unresolved). Details in `notes/pilot_notes.md`.

**Pilot.** Laptop pilot (CPU, batched DQN on CartPole, 16 parents, 3 children per condition, 7 fork times). See `results/verdict.txt` and `notes/pilot_notes.md`. The paper version runs the same design on MinAtar (DQN) and DMC (SAC/TD3) on the 3080, where many runs fit in one batched process.

## Pre-registered kill test (written before the pilot ran)

Parents: 16 seeds × 30k steps. Fork times: 0, 2.5k, 5k, 10k, 15k, 20k, 25k. Children per (parent, fork, condition): 3. Conditions: `all` (exploration and minibatch streams reseeded), `explore` (only exploration reseeded), `batch` (only minibatch reseeded). Final score: mean of the last three greedy evaluations. ICC(t, c) = intraclass correlation of children's final scores grouped by parent.

- **K1.** Lock-in is early: ICC_all ≥ 0.5 with bootstrap 90% CI lower bound > 0.2 at some fork ≤ T/2.
- **K2.** The sources differ: |ICC_explore − ICC_batch| ≥ 0.15 at two or more forks, same sign.
- **K3.** Fate before visibility: at the first lock-in fork, R² of child final on the parent's *evaluation score at that fork* is ≤ ICC_all − 0.2.

SUPPORTED if K1 and (K2 or K3). INCONCLUSIVE if only K1. KILLED if not K1.

## Layout

| folder | contents |
|---|---|
| `code/` | `r7_fork.py` (parents, snapshots, children, ICC/R² analysis, verdict); `code/lib/dqn_batched.py` (R independent DQNs as one tensor program, per-run generators for exploration and minibatch noise, exact snapshot/fork); `code/lib/io_utils.py` |
| `data/` | empty |
| `results/` | `runs.csv` (one row per child), `parents.json`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | empty |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r7_fork.py --smoke          # ~30 s
.venv/bin/python code/r7_fork.py                  # pilot, ~40 min on an M1 Pro (CPU)
.venv/bin/python code/r7_fork.py --summarize
```
