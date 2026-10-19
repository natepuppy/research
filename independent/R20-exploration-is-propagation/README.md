# R20: How much of "hard exploration" is slow value propagation?

**Question.** The sparse-reward problem is framed as an exploration problem: the agent must *find* the reward. A large literature (count-based bonuses, RND, Go-Explore, NGU) is built on that framing. But finding the reward once is not the end: the value of that one transition has to travel back to the start through bootstrapping, and R12 showed that with a hard target network refreshed every K updates and uniform replay, a reward L steps away cannot reach the start before (⌈L/n⌉ − 1)·K updates plus a replay term. For Nature-DQN settings (K = 2500 updates, n = 1) and a 30-step path that is 72 500 updates, which is longer than the random-walk discovery time of most tabular-scale corridors. Our hypothesis: under standard value-based settings, the time from first discovery to a working policy exceeds the time to discover, follows R12's propagation law in K and n, and is the part of "hard exploration" that cheap propagation fixes (n-step, fast targets) remove, whereas exploration bonuses mainly shorten the discovery half. If so, a large share of the sparse-reward difficulty attributed to exploration is a credit-assignment latency with a closed form.

**Status.** Laptop pilot done: **SUPPORTED**. In the Nature-DQN-like cell (K = 2000, n = 1) the post-discovery time exceeds the discovery time by 52× / 20× / 6× at D = 10 / 20 / 30 (post-discovery share 0.98 / 0.95 / 0.86) (K1); median T_p follows R12's target-sync law with Spearman 0.92 (K2), and a post-hoc two-term fit gives T_p = 0.96·(⌈D/n⌉ − 1)K + 59·D²/(nB) + 584, R² = 0.996. The bonus control (K3) failed at D = 30: the β = 0.1 count bonus was a weak explorer here and lengthened the propagation half in 16/18 cells, making total time-to-solve worse in 16/18 cells. Fast target + n = 5 cuts time-to-solve 7–31× with identical exploration. See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`. Paper version: DQN on MinAtar Freeway/Seaquest-sparse and on the "hard exploration" MiniGrid tasks on the 3080, with the discovery/propagation split measured by logging the first extrinsic reward and the first successful greedy episode, K × n × bonus factorial.

Closest prior work: RUDDER (Arjona-Medina et al. 2019) names three sub-problems (discover, keep, learn) but does not measure their shares; Go-Explore's two-phase structure (Ecoffet et al. 2021) treats post-discovery "robustification" as its own hard problem; Laidlaw et al. (2023, effective horizon) predict difficulty from a single scalar without separating the two phases. Lab stubs: the delayed-reward cluster is about reward *delay* within an episode, not about target-network latency across training; named to be safe. R12 is the law this project applies.

## Pre-registered kill test (written before the pilot ran)

Design: 1-D corridor of depth D ∈ {10, 20, 30}; start at 0, two actions (left/right), reward 1 and termination on reaching D, episode cap 4D, γ = 0.99, slip 0. Tabular Q-learning with uniform replay (buffer 20k, batch 32, one update per step after 200 steps), α = 0.1, uncorrected n-step targets, hard target table refreshed every K updates. ε = 0.1 constant; greedy ties (|Q range| < 10⁻⁴) broken uniformly at random, so the untrained agent is a random walk. Factorial: K ∈ {1, 200, 2000} × n ∈ {1, 5} × count bonus β/√N(s') with β ∈ {0, 0.1}; 8 seeds; 300k steps. Every 500 steps the greedy policy is rolled out exactly from the start for 4D steps (deterministic corridor): success = reaches the goal.

Measured per run: T_d = step of the first goal reach (discovery); T_p = step of the first greedy success minus T_d (propagation); share = T_p / (T_d + T_p). Runs that never discover, or discover but never succeed, are reported and excluded from medians.

- **K1 (propagation dominates in the standard setting).** In the (K = 2000, n = 1, β = 0) cells, median T_p > median T_d for every D.
- **K2 (R12's law).** Over the β = 0 cells with ≥ 4 discoveries, Spearman(median T_p, (⌈D/n⌉ − 1)·K + 1) ≥ 0.8.
- **K3 (bonus buys discovery).** At K = 2000, n = 1, the bonus reduces median T_d by ≥ 2× for every D. Its effect on T_p is reported without a threshold.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported as the exploration control.

## Layout

| folder | contents |
|---|---|
| `code/` | `r20_explore.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per run: T_d, T_p, share, final success), `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r20_explore.py --smoke
.venv/bin/python code/r20_explore.py            # pilot, ~10 min CPU
.venv/bin/python code/r20_explore.py --summarize
```
