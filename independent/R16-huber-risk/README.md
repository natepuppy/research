# R16: The Huber loss is a risk preference

**Question.** DQN and most of its descendants minimise a Huber loss on the TD error (equivalently: the TD error is clipped to [−κ, κ] before the update, κ = 1). The stated reason is robustness to large errors and stability. But a Huber M-estimator does not estimate the mean of a skewed distribution: its fixed point lies between the mean and the median. TD targets *are* skewed wherever a rare outcome carries a large value difference, which is exactly the situation of a lottery (rare large reward) or a hazard (rare large penalty, or a termination from a high-value state). Our hypothesis: Huber-TD systematically *under-values rare large gains* and *under-weights rare large losses*, with a bias whose sign is minus the skew and whose size is set by κ relative to the jump, so that a Huber agent is risk-averse toward lotteries and risk-seeking toward hazards, and picks the wrong action whenever the bias exceeds the action gap. This is a change of objective, not a regularisation, and it is computable exactly: the fixed point of the "Huber Bellman operator" E[clip(r + γ max Q(s') − q, −κ, κ)] = 0 predicts the learned Q.

**Status: pilot KILLED by rule (design let in a second bias); pre-registered single-action follow-up INCONCLUSIVE by rule with every substantive prediction confirmed.** The closed-form Huber fixed point predicts tabular Q-learning's learned values with R² = 0.986 (pilot) and **1.000** (follow-up); the greedy policy is wrong exactly where |predicted bias| > gap (60/60 cells) and never under the squared loss; the sign is −skew in all 40 active cells (the test's wording covered four inactive cells where the prediction is ≈ 0, hence INCONCLUSIVE). At κ = 1 a 5%-probability payoff of 22 is valued at 0.2 instead of 1.1 and the agent takes the wrong fork in every seed, for lotteries and hazards alike. The pilot's defect was informative: the squared loss carries a risk-seeking max-bias on noisy arms (+0.09–0.31), which Huber's clip suppresses. DQN check (K4) passed 13/16: Huber inflates the learned value of terminating actions (26.8 vs 22.2, true value 1). See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`, `paper/main.md`. Paper version: DQN on MinAtar with injected rare-reward lotteries and rare penalties on the 3080; CartPole/MinAtar failure-Q accounting (does the Huber loss contribute to the terminal-value inflation of R11?); κ as an explicit risk dial compared with distributional RL's quantile-Huber.

No adjacent stub: the merged idea sheets have nothing on the loss function's fixed point. Related to R15 (two biases) in method, not content.

**Smoke-test note (before pre-registration).** A 20k-step, 2-seed smoke with p = 0.02 (payoff M ≈ 52) showed the squared-loss estimate's *variance* alone (stationary sd ≈ 0.7 at α = 0.02) flipping the greedy action in a 0.05 gap, which would fail K3 for a reason unrelated to the hypothesis. The grid was narrowed to p ≥ 0.05 and g ≥ 0.1 and the run lengthened to 300k steps before the tests below were fixed. The variance-versus-bias trade-off that the smoke exposed is itself part of the story and is reported.

## Pre-registered kill test (written before the pilot ran)

**Tabular design.** Fork MDP: start s₀ with actions *safe* and *risky*, each leading down a deterministic chain of L = 3 states to a terminal reward; γ = 0.95. Safe pays m = 1. Risky pays a skewed lottery with mean m ± g:
- *lottery* (positive skew): M with probability p, else 0, with pM = m + g (risky is optimal by g);
- *hazard* (negative skew): a sure base b = m − g + pM, minus M with probability p, mean m − g (safe is optimal by g).
Grid: g ∈ {0.1, 0.3}, p ∈ {0.05, 0.1, 0.3}, κ ∈ {0.1, 0.3, 1, 3, ∞} (∞ = squared loss), both skews: 60 cells × 8 seeds. Behaviour: uniform random action at s₀ and exploring starts over all states, so both arms are sampled equally and exploration is not a confound. Tabular Q-learning with update Q ← Q + α·clip(δ, −κ, κ), α = 0.02, 300k steps. Measured: bias = Q̂(s₀, risky) − Q*(s₀, risky) averaged over the last quarter of training; the greedy action at s₀. Predicted: the Huber Bellman fixed point computed by bisection, iterated to convergence.

- **K1 (sign and monotonicity).** In every cell with κ < ∞, the bias is negative under lottery and positive under hazard (sign = −skew); within each (skew, g, p), |bias| is non-increasing in κ (Spearman(|bias|, κ) ≤ −0.8 over the five κ).
- **K2 (quantitative).** Identity-fit R² of learned bias on predicted bias ≥ 0.9 over all 60 cells.
- **K3 (policy).** The cells in which the greedy policy is wrong (majority of seeds) are those with |predicted bias| > g, with ≥ 90% cell-wise agreement, and no κ = ∞ cell is wrong.

SUPPORTED if K1, K2 and K3. INCONCLUSIVE if exactly two. KILLED otherwise.

### Pre-registered follow-up (written after the pilot, before it ran)

The pilot was KILLED by K1 and K3 while K2 passed at R² = 0.986. Both failures trace to a second bias the design let in: chain states have two identical actions whose Q-estimates are learned independently from noisy targets, so the target max_a Q(s', a) carries Q-learning's max-bias (Thrun & Schwartz 1993) on the *noisy* arm only. This is +0.09 to +0.31 under the squared loss, exceeds g = 0.1, and makes the squared-loss agent risk-seeking in the hazard cells on its own. Follow-up: `--single-action --tag nomax`, identical otherwise, chain states have one action, so the only bias left is Huber's. Same K1–K3, same thresholds; prediction: all three pass. If K1 or K3 still fails the hypothesis is wrong in a way the max-bias does not explain. The pilot's max-bias result is kept and reported: the squared loss is itself risk-seeking toward noisy arms through the max, and the two biases add under hazard and oppose under lottery.

*Outcome:* K2 R² = 1.000, K3 100% agreement and no squared-loss cell wrong, K1 monotonicity holds everywhere; K1's sign clause fails in four cells where the predicted bias is ≤ 0.013 (κ = 3, p = 0.3; the clip is nearly inactive) and the learned bias is noise. INCONCLUSIVE by rule; the sign holds in all 40 active cells.

**Secondary DQN check (reported, not in the verdict).** Batched DQN on CartPole (200-cap, reward 1 per step, γ = 0.99), Huber (κ = 1) vs squared loss, 16 paired seeds, 30k steps. Probe: 256 (s, a) pairs whose action terminates, so Q*(s, a) = 1 exactly (R11's instrument); the terminal TD error from a high-value state is a large negative number that Huber clips. Prediction K4: final failure-Q is higher under Huber than under squared loss in ≥ 12 of 16 paired seeds.

## Layout

| folder | contents |
|---|---|
| `code/` | `r16_huber.py` (tabular + exact fixed point; `--dqn` for the CartPole check); `code/lib/dqn_batched.py` (vendored); `code/lib/dqn_loss.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv`, `summary.csv`, `verdict.txt` (all runs), `posthoc.txt`; `nomax/` (follow-up); `dqn/` |
| `logs/` | `pilot.log`, `nomax.log`, `dqn.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r16_huber.py --smoke
.venv/bin/python code/r16_huber.py              # tabular pilot, ~30 s CPU
.venv/bin/python code/r16_huber.py --single-action --tag nomax   # pre-registered follow-up
.venv/bin/python code/r16_huber.py --dqn        # CartPole check, ~5 min CPU
```
