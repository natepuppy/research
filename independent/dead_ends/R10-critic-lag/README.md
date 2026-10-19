# R10: How far behind is the critic?

**Question.** Actor-critic theory needs the critic to be faster than the actor (two-timescale analysis: Konda & Tsitsiklis 2000; Borkar). Practice sets both learning rates to 3e-4 and moves on. Nobody has measured the quantity the theory is about: *which policy is the critic actually evaluating?* We define the critic's effective lag as the k ≥ 0 for which the critic's Q(s, π_t(s)) is closest to the true Q^{π_{t−k}}(s, π_t(s)) of the policy from k evaluation intervals ago, computed exactly by Monte-Carlo in a deterministic simulator. A lag of zero means the critic tracks the present policy; a lag of k means the actor is being improved against a value function of a policy it no longer is. This turns the two-timescale assumption into a number that can be read off a training run and related to the learning-rate ratio and to performance.

**Status: KILLED.** Pilot INCONCLUSIVE by rule, but K1 passed only through an argmin selection artefact; the pre-registered fine-resolution follow-up (100-step lags) was KILLED: the critic's error profile over lag is flat or increasing in every cell, so the critic is never closer to a past policy than to the current one. What survived: critic error against the current policy tracks policy drift (ρ = 0.90) and predicts final return (ρ = −0.78). Details in `notes/pilot_notes.md`; negative-result note in `paper/main.md`.

**Pilot.** Laptop pilot (CPU, batched DDPG on Pendulum, 3 actor LRs × 4 critic LRs × 8 seeds). See `results/verdict.txt` and `notes/pilot_notes.md`. Paper version: SAC/TD3 on DMC and MuJoCo on the 3080, MC ground truth from state resets (MuJoCo exposes `set_state`), plus the obvious intervention: set the critic LR from the measured lag.

Adjacent stub in `research/ideas` (S4-2, "anticipatory targets") proposes a *fix* (compute targets with the actor you will have). This project is the *diagnosis* that would justify such a fix; it does not implement one.

## Pre-registered kill test (written before the pilot ran)

Design: actor LR ∈ {1e-4, 3e-4, 1e-3} × critic LR ∈ {1e-4, 3e-4, 1e-3, 3e-3} × 8 seeds, 30k steps, evaluation every 1k steps. At each evaluation: 64 fixed probe states; critic Q(s, π_t(s)); exact MC Q^{π_{t−k}}(s, π_t(s)) for k = 0..6; k* = argmin_k mean |Q_critic − Q^{π_{t−k}}|. Analysis window: evaluations with ≥ 5 past policies available and step ≥ T/3.

- **K1 (a lag exists).** In cells with critic LR ≤ actor LR, k* > 0 in ≥ 60% of (run, eval) pairs, and the error against the best-matching past policy is ≥ 20% lower than against the current one.
- **K2 (lag scales with the ratio).** Spearman correlation between log(critic LR / actor LR) and mean k* across the 12 cells ≤ −0.6.
- **K3 (lag predicts performance).** Across cells, Spearman between mean k* and final return ≤ −0.5, or between current-policy critic error and final return ≤ −0.5.

SUPPORTED if K1 and K2. INCONCLUSIVE if K1 only. KILLED if not K1.

## Fine-resolution follow-up (pre-registered after the pilot, before the run)

The pilot came out INCONCLUSIVE (K1 and K3 true, K2 false), and the post-hoc check in `results/posthoc.txt` showed that K1's estimator is biased: k* = argmin over seven noisy, nearly flat errors lands on k > 0 by selection even when the *mean* profile is minimised at k = 0 (which it is, in every cell). So either there is no lag, or the lag is shorter than the 1k-step evaluation interval (the Polyak target has a ~200-update timescale). The follow-up runs the same grid with evaluations every 100 steps and lags 0..10 (0–1000 steps), and tests the mean profile, which has no selection bias:

- **F1.** In every cell with critic LR ≤ actor LR, the mean error profile over k is minimised at k > 0, with bootstrap-over-runs support ≥ 0.8.
- **F2.** Spearman of the profile's argmin on log(critic LR / actor LR) ≤ −0.6.

SUPPORTED if F1 and F2; INCONCLUSIVE if F1 only; KILLED if not F1. Results in `results/fine/`.

## Layout

| folder | contents |
|---|---|
| `code/` | `r10_critic_lag.py`; `code/lib/ac_batched.py` (batched DDPG, batched Pendulum, MC ground truth); `code/lib/io_utils.py` |
| `data/` | empty |
| `results/` | `runs.csv` (one row per run × evaluation), `summary.csv` (per LR cell), `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r10_critic_lag.py --smoke        # ~15 s
.venv/bin/python code/r10_critic_lag.py --T 30000      # pilot, ~15 min on an M1 Pro (CPU)
.venv/bin/python code/r10_critic_lag.py --summarize
```
