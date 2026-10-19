# R24: How to choose n — the n-step return as a latency–bias trade-off under a target network

**Question.** Every value-based agent sets n (DQN 1, Rainbow 3, R2D2 5, DER 20, BBF 10 → 3) by sweep. Two confirmed results from this lab make n a derived quantity. (i) Under a target network refreshed every K updates, the time for a reward L steps away to reach a decision is (⌈L/n⌉ − 1)·K (R12; exact for neural critics where data exists, R22): n divides the latency. (ii) Uncorrected n-step returns from ε-greedy replay carry a pessimistic bias: every behaviour step inside a segment can take the exploratory action, so the value of a path with L transitions bootstrapped in segments of n survives with factor (1 − f)^{(L+1) − ⌈(L+1)/n⌉}, where f is the per-step probability that the behaviour policy takes a path-breaking action (R15 measured the bias; this is its closed form on a path). The policy is wrong once the bias exceeds the value gap to the best alternative. So there is a largest safe n,

  n* = max{ n : γ^L (1 − f)^{(L+1) − ⌈(L+1)/n⌉} > g },

set by ε, the path length and the value gap and independent of K, and below n* the cost of a smaller n is pure latency, (⌈(L+1)/n⌉ − 1)·K. We test both halves. The practical output is a rule — raise n until the exploratory bias approaches the value gap, and the benefit is proportional to K — and a reading of the published choices: Nature DQN's ε = 0.1 makes n > 2 unsafe on 20-step paths with modest gaps, which is why n = 1 was right for it, while agents with ε = 0.01 or noisy nets can take n = 10–20, which is what the sample-efficient agents did.

**Status.** Laptop pilot run. **SUPPORTED**: K1 9/12 (largest safe n within one grid step of n* at K = 2000), K2 Spearman +1.00 (time to the correct policy on (⌈21/n⌉ − 1)K + 1 over 66 cells; median T/law 1.05, no fitted constant), K3 12/12 (same largest safe n at K = 200 and K = 2000). Post-hoc (`results/posthoc.txt`): learned Q(s₀, A) matches the closed-form bias with r = 0.985, RMSE 0.029 over 168 cells; the three K1 misses are the cells where n* has a predicted margin (+0.026, +0.005, +0.001) below the seed noise (std ≈ 0.06); seeds-correct is 0.96 above a 0.05 margin and ≤ 0.06 below zero. At K = 2000, ε = 0.01, gap ≥ 0.2 the correct policy arrives at 40 200 updates with n = 1 and 2 200 with n = 12, 8/8 seeds both. See `notes/pilot_notes.md`.

Laptop pilot (CPU, tabular Q-learning with uniform replay from a fixed ε-greedy dataset, uncorrected n-step, hard target table). Paper version on the 3080: DQN on MinAtar with n ∈ {1, 3, 5, 10, 20} × ε ∈ {0.01, 0.1} × K ∈ {500, 8000}, predicting the ordering of n from measured path lengths and gaps; and a reanalysis of the Rainbow / DER / R2D2 / BBF settings against the rule.

Prior work: Hessel et al. 2018 (n = 3 by sweep), Kapturowski et al. 2019 (n = 5), van Hasselt et al. 2019 DER (n = 20), Schwarzer et al. 2023 BBF (n annealed 10 → 3 and the claim that large n early, small n late is best — our rule says the same: early the gap is large relative to bias and latency dominates; late the gap shrinks), Fedus et al. 2020 (n-step makes large buffers work), Rowland et al. 2020 and Daley et al. 2023 (adaptive/off-policy-corrected n-step; the bias we quantify is what corrections remove). Lab stubs: none; R12/R15/R22 are the components.

## Pre-registered kill test (written before the pilot ran)

Design. States: s₀; corridor 1..L with L = 20; B terminal. At s₀: `A` → state 1 (reward 0), `B` → terminal with reward g. In the corridor: `forward` → next state (reward 0; from state L → terminal with reward 1), `fall` → terminal with reward 0. γ = 0.99, so Q(s₀, A) = γ^L = 0.818 and the gap is 0.818 − g with g ∈ {0.718, 0.618, 0.518} (gaps 0.1, 0.2, 0.3). Fixed dataset per (ε, seed): 20 000 transitions from a behaviour policy that picks A/B at random and in the corridor falls with probability f = ε/2 (the ε-greedy exploratory half that breaks the path), ε ∈ {0.01, 0.03, 0.1}. Learner: tabular Q, uniform replay from the fixed dataset, batch 32, α = 0.1, uncorrected n-step with episode-boundary truncation, hard target table every K updates, 100 000 updates. n ∈ {1, 2, 3, 5, 8, 12, 20}, K ∈ {200, 2000}, 8 seeds. Every 200 updates: exact greedy rollout from s₀; correct = reaches the goal. T = first correct update; final = correct at the end.

Prediction: n* from the formula above (computed in `code/`), per (ε, gap).

- **K1 (the bias threshold).** At K = 2000, for each (ε, gap) cell, let n̂ = the largest n in the grid with final-correct in ≥ 6/8 seeds (0 if none). n̂ equals the grid neighbour of n* (n* itself or the grid value adjacent to it) in ≥ 7/9 cells.
- **K2 (below the threshold, latency).** Over all (ε, gap, n, K) cells with final-correct ≥ 6/8, Spearman(median T, (⌈(L+1)/n⌉ − 1)·K + 1) ≥ 0.85.
- **K3 (the threshold does not depend on K).** n̂ at K = 200 equals n̂ at K = 2000 in ≥ 7/9 cells.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 reported.

**Smoke-test changes (made before the pilot, after a 2-seed × 8 000-update smoke run).** (a) ε grid extended to {0.01, 0.03, 0.05, 0.1}: with the original three values n* was 1 or ≥ 8 in eight of nine cells, so the test had almost no interior; ε = 0.05 gives n* ∈ {1, 2, 8}. K1 and K3 thresholds become ≥ 9/12. (b) "final-correct" is the greedy policy being correct in ≥ 80 % of the evaluations over the last 20 % of training, not at the single last evaluation: with α = 0.1 and sampled falls the learned Q(s₀, A) fluctuates by ≈ ±0.07, so a single-instant read is a coin flip when the bias sits within 0.07 of the gap. (c) Updates 100 000 → 60 000 (the longest predicted latency is 40 001; runtime).

## Layout

| folder | contents |
|---|---|
| `code/` | `r24_nstep.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per run), `summary.csv` (per cell), `nstar.csv` (n* and n̂ per (ε, gap)), `verdict.txt`, `posthoc.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r24_nstep.py --smoke
.venv/bin/python code/r24_nstep.py             # pilot, ~75 min CPU (1344 vectorised learners)
.venv/bin/python code/r24_nstep.py --summarize
```
