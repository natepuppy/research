# R19: The credit-assignment speed limit of PPO

**Question.** PPO with GAE(λ) is the default policy-gradient method from MuJoCo to RLHF, and λ ∈ [0.95, 0.97] is a near-universal setting. With λ < 1 the advantage at a state is a geometric mixture of n-step returns and the critic's bootstrap, so a reward that is D steps away contributes to the advantage at the start state with weight ≈ (γλ)^D unless the critic has already absorbed it. The critic, in turn, is fit to λ-returns computed from itself, so new reward information moves back roughly 1/(1 − γλ) states per training iteration (fewer when the rollout segment T is shorter than that, because the segment boundary is a bootstrap). We propose and test a **latency law**: the number of PPO iterations before a distant reward first moves the policy at the start state is

  t_arrive ≈ D / h_eff,  h_eff = min(T, 1/(1 − γλ)),

so it scales linearly in D and in (1 − λ), and is ≈ 1 iteration for any D when λ = 1 and T ≥ D. The claim is a *latency*, not a *decay*: the reward is not lost, it arrives late. VC-PPO (Yuan et al. 2025, arXiv 2503.01491) is the closest prior work; it describes GAE at λ < 1 as "reward signal decay" in long-horizon RLHF and proposes a decoupled λ with λ_critic = 1. We make the mechanism quantitative, show it is an iteration count rather than a signal loss, and extend it to rollout length, which bounds propagation even at λ = 1. The consequence for practice is a closed-form default: set λ and the rollout length so that h_eff ≥ D / (budgeted iterations), and count iterations, not samples.

**Status.** Laptop pilot done: **SUPPORTED**. t_V ∝ D^0.95 (1 − λ)^0.88, R² = 0.999 (K1); λ = 1 gives t_V = 1 at every depth (K2); the start-state policy moves in the same iteration (Spearman 0.99, K4). The rollout-length extension (K3) scales correctly in D (slope 1.03) but the pre-registered min(T, 1/(1 − λ)) constant is wrong: the two slowdowns add as rates, 1/h_eff = (1 − λ) + 1/T_eff with T_eff ≈ 6 states per 16-step segment (post-hoc, R² 0.998). Every cell eventually solves the task (latency, not decay); with the critic frozen, λ = 0.95 never solves D = 128 (post-hoc P7). See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`. Paper version: PPO with neural actor/critic (CleanRL) on long-corridor MiniGrid / Sokoban-style delayed reward tasks on the 3080, sweeping λ and `num_steps`, measuring the iteration at which the start-state advantage first becomes informative.

Lab stubs named for adjacency: the delayed-reward cluster (reward delay inside an episode; we study iteration latency across training), RLVR stubs (λ = 1 critic-free is the regime where the law says latency vanishes — we note the implication, we do not study RLVR). Not overlapping with R12 (hard target network latency in value-based RL): same spirit, different mechanism (bootstrap-through-critic with geometric weights vs hard sync), and no shared code.

## Pre-registered kill test (written before the pilot ran)

Design: deterministic MDP with a choice at s₀ (two actions, go to corridor A or B). Each corridor has depth D; corridor states have two actions, `forward` and `stall` (stay). Reaching the end of corridor A gives reward 1 and terminates; corridor B's end gives 0 and terminates. Episode cap 3D. γ = 1. Each iteration collects N = 32 episodes from the current policy, computes λ-returns with the current critic as bootstrap at segment boundaries (segment length T; T = ∞ means no boundary within an episode), fits the tabular critic exactly to the per-state mean of λ-return targets (states not visited keep their value), computes GAE advantages, normalises them over the batch, and takes one SGD step on a tabular softmax policy (step size 0.5, no clipping — PPO's clipping is inactive at one step per batch). Policy initialised uniform; critic initialised at 0.

Grid: D ∈ {8, 16, 32, 64, 128}, λ ∈ {0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0}, T ∈ {∞, 16}, 8 seeds, 600 iterations max. Because t_V is an integer with a floor of 1, the K1 regression uses only cells whose predicted latency D·(1 − λ) is ≥ 4; the other λ < 1 cells are reported but not fit (decided before any run, when writing this grid).

Measured per run: t_V = first iteration at which V(s₀) ≥ 0.25 (half of the maximal start value under a policy that always picks A and always moves forward is 0.5; we use 0.25 so that the threshold is reached before the policy is near-deterministic); t_π = first iteration at which the exact probability that the *stochastic* policy reaches A's end within the cap exceeds 0.75. Runs that never reach a threshold are recorded at 600 and flagged.

*Smoke-test change (before the pilot):* the t_π threshold was first written as 0.5, but the untrained uniform policy already succeeds with probability ≈ 0.5 (it picks corridor A half the time and almost always reaches the end within the 3D cap), so 0.5 was hit at iteration 1 everywhere. Raised to 0.75 before any pilot run. Nothing else changed.

- **K1 (latency law in D and λ).** T = ∞, λ < 1, D·(1 − λ) ≥ 4: OLS of log median t_V on log D and log(1 − λ) has both slopes in [0.7, 1.3] and R² ≥ 0.9.
- **K2 (λ = 1 removes it).** T = ∞, λ = 1: slope of log median t_V on log D is < 0.3.
- **K3 (rollout length caps it).** T = 16, λ = 1: slope of log median t_V on log D over D ≥ 32 is in [0.7, 1.3], and for D ≥ 32 median t_V(T = 16, λ = 1) is within 1.5× of median t_V(T = ∞, λ = 0.95), whose h_eff is 20 (close to 16).
- **K4 (the policy inherits the latency).** Spearman(median t_π, median t_V) over all cells with both medians finite ≥ 0.8.

SUPPORTED if K1, K2 and K4. INCONCLUSIVE if exactly two of {K1, K2, K4}. KILLED otherwise. K3 is reported as the rollout-length extension. We also report, for every (λ, D), whether the policy eventually solves the task (the decay view predicts the start state never learns for large D at λ < 1; the latency view predicts it learns late).

## Layout

| folder | contents |
|---|---|
| `code/` | `r19_gae.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r19_gae.py --smoke
.venv/bin/python code/r19_gae.py             # pilot, ~15 min CPU
.venv/bin/python code/r19_gae.py --summarize
```
