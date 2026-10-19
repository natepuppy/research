# R17: Does decorrelation do anything? Minibatch correlation versus reuse in experience replay

**Question.** The textbook justification for experience replay (Lin 1992; Mnih et al. 2015) has two halves: replay *reuses* data, and sampling uniformly from a buffer *decorrelates* consecutive updates, which "reduces the variance of the updates" and avoids "feedback loops". The second half is repeated in every tutorial and has, as far as we can find, never been isolated: every comparison of replay against no-replay changes reuse and correlation together, and the replay-ratio and buffer-size literature (Fedus et al. 2020; D'Oro et al. 2023) varies how much data is reused, not how correlated a minibatch is. Closest prior work for the `recent` sampler: Combined Experience Replay (Zhang & Sutton 2017) adds the newest transition to each uniform minibatch; our `recent` sampler is the all-newest extreme of that fix, and the K = 100 result below is the regime it was designed for. We hold the buffer, the data, and the number of updates fixed and vary only *how the 32 transitions in a minibatch are chosen*: 32 independent draws (standard), 4 contiguous windows of 8, or one contiguous window of 32 (consecutive transitions from the same trajectory, maximal correlation). A fourth sampler, the last 32 transitions (no reuse, maximal correlation), is the classic no-replay baseline. We cross this with the target period, because our hypothesis is that correlated batches hurt through target-chasing instability, which a slow target should mask.

**Status: KILLED, with the reuse control coming out backwards.** Laptop pilot (CPU, batched DQN on CartPole, 4 samplers × 2 target periods × 16 seeds, 30k steps). Within-batch correlation costs 8 AUC points at either target period (4/16 wins for the fully contiguous batch, p = 0.08): the decorrelation effect is small and not significant, although the gradient's cosine to the buffer gradient does fall (0.66 → 0.59). With a fast target (K = 100) the no-replay sampler (last 32 transitions) beats uniform replay by 40 AUC and 69 final-return points (15/16, 14/16 wins); the replay learners peak at 5k steps and collapse, the on-policy learner does not. With K = 2500 replay wins on AUC by reuse and the two finish equal. Mechanism of the collapse (stale targets vs R11's forgotten failures) not separated here. See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`, `paper/main.md`. Paper version: MinAtar on the 3080 with the same dial plus gradient-noise measurements, Polyak targets, and a replay-ratio sweep to separate reuse, correlation and update count in one design.

Adjacent stubs in `research/ideas`: S3-4 ("critical batch size of TD") varies batch size, not correlation; S3-12 ("Durbin–Watson for TD") proposes measuring residual autocorrelation along trajectories as a diagnostic. Neither tests whether correlation in the minibatch matters.

## Pre-registered kill test (written before the pilot ran)

Design: batched DQN (2 × 64 MLP, Adam 1e-3, γ = 0.99, buffer 10k, batch 32, Huber loss, grad-norm clip 10, ε annealed 1 → 0.05 over 10k steps, learning from step 500), CartPole 200-cap, 30k steps, greedy evaluation every 1k (AUC = mean of the 30 evaluations). Samplers: `iid` (32 uniform draws), `win4` (4 uniform anchors, each extended to 8 consecutive transitions), `win1` (one anchor, 32 consecutive transitions), `recent` (the last 32 transitions written). Target period K ∈ {100, 2500}. 16 seeds, shared across cells. Every 1k steps, the cosine between the minibatch gradient and the gradient on a fixed 1024-transition uniform sample of the buffer is logged (gradient-estimate quality).

Hypothesis H: within-batch correlation hurts, and it hurts through the target: with a fast target (K = 100) correlated batches chase a moving target; with a slow target (K = 2500) the target is fixed for long enough that correlation is harmless.

- **K1 (correlation hurts at fast target).** At K = 100, AUC(iid) − AUC(win1) ≥ 20 points and iid beats win1 in ≥ 12 of 16 paired seeds.
- **K2 (the target masks it).** gap(K) = AUC(iid) − AUC(win1). gap(2500) ≤ 0.5 · gap(100).
- **K3 (reuse is a separate effect).** At both K, AUC(win1) − AUC(recent) ≥ 20 points (same correlation, reuse versus none).

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one of K1, K2. KILLED if neither. K3 is reported as the reuse control and does not enter the verdict. If K1 fails at both K (correlation does not matter at all at fixed reuse) the result is a negative one about the textbook justification and is written up as such.

## Layout

| folder | contents |
|---|---|
| `code/` | `r17_decorr.py`; `code/lib/dqn_batched.py` (vendored from R7); `code/lib/dqn_sampler.py` (per-run sampler and target period, gradient-cosine meter); `code/lib/io_utils.py` |
| `results/` | `runs.csv`, `per_run.csv`, `summary.csv`, `verdict.txt`, `posthoc.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r17_decorr.py --smoke       # ~1 min
.venv/bin/python code/r17_decorr.py               # pilot, ~20 min CPU
.venv/bin/python code/r17_decorr.py --summarize
```
