# R14: Is policy churn the exploration you did not ask for?

**Question.** Schaul et al. (2022) showed that in deep value-based agents the greedy action changes in a large fraction of states after every gradient update ("policy churn"), and conjectured that this churn is a substantial, unintended source of exploration. Tang & Berseth (2024) reduce churn and report gains. Nobody has asked what *controls* churn among the knobs every DQN already has, or tested the substitution that the conjecture implies: if churn explores, then the knob that damps it should make explicit exploration (ε) more necessary. The obvious damping knob is the target network period K, which is usually chosen for stability. We run a 4 × 3 factorial over K and a *constant* ε (no annealing, so ε is the only deliberate exploration), measure churn on every update, and test (i) whether K controls churn, (ii) whether the value of ε rises with K (substitution), and (iii) whether, at ε = 0, churn mediates performance.

**Status: KILLED (mechanism), mediation half survives.** Laptop pilot (CPU, batched DQN on CartPole, 4 target periods × 3 ε × 8 seeds, 30k steps). Churn *rises* with K (Spearman +0.88 across cells, the opposite of K1); ε's gain is not monotone in K (K2 failed); K3 passed (+0.72). K = 20 collapsed in all 24 runs and is excluded post hoc. Post hoc: among ε = 0 runs early churn predicts AUC within every K (+0.70 to +0.83, pooled +0.80, n = 24), decaying to +0.67 at ε = 0.05 and −0.02 at ε = 0.2: churn and ε substitute on the seed axis, causal direction open. See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`, `paper/main.md`. Paper version: MinAtar on the 3080 with the same factorial, plus direct churn manipulation (CHAIN-style regularisation) crossed with ε, and a per-step accounting of which exploratory actions came from churn versus ε.

**Smoke-test note (before pre-registration).** A 1-seed, 5k-step smoke run showed K = 1 (no target network) diverging to a constant-action policy, so the K grid starts at 20. The smoke also hinted that churn might *increase* with K, the opposite of the hypothesis; the tests below are written in the hypothesis's direction regardless.

## Pre-registered kill test (written before the pilot ran)

Design: K ∈ {20, 100, 500, 2500} × constant ε ∈ {0, 0.05, 0.2} × 8 seeds (same seeds across cells), 30k steps, greedy evaluation every 1k. Churn = fraction of 256 fixed probe states whose greedy action changes across one update, averaged per evaluation interval. AUC = mean evaluation score over training. Early churn = mean over the first third.

- **K1 (K controls churn).** Spearman of mean churn on log K over the 12 cells ≤ −0.8.
- **K2 (substitution).** gain(K) = AUC(ε = 0.2) − AUC(ε = 0). Spearman of gain on log K ≥ 0.8 and gain(K = 2500) − gain(K = 20) ≥ 20 points.
- **K3 (mediation).** Among ε = 0 runs, Spearman of AUC on early churn ≥ 0.4.

SUPPORTED if K1 and K2. INCONCLUSIVE if K1 and K3 only. KILLED otherwise.

## Layout

| folder | contents |
|---|---|
| `code/` | `r14_churn.py`; `code/lib/dqn_batched.py` (vendored from R7); `code/lib/dqn_churn.py` (per-run K and ε, churn meter); `code/lib/io_utils.py` |
| `results/` | `runs.csv`, `per_run.csv`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r14_churn.py --smoke     # ~20 s
.venv/bin/python code/r14_churn.py             # pilot, ~15 min CPU
.venv/bin/python code/r14_churn.py --summarize
```
