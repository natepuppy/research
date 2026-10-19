# R16 pilot notes

Three runs. **Pilot**: tabular Q-learning with clipped TD updates on fork MDPs, 2 skews × 2 gaps × 3 lottery probabilities × 5 κ = 60 cells × 8 seeds, 300k steps, exact Huber fixed points (CPU, 30 s). **nomax** (pre-registered follow-up): same with one action per chain state. **DQN check**: CartPole, Huber vs squared loss, 16 paired seeds, 30k steps (CPU, 2.5 min). Logs in `logs/`. Raw in `results/`, `results/nomax/`, `results/dqn/`. Post hoc: `results/posthoc.txt`.

## Verdicts

- **Pilot: KILLED by rule.** K2 (the Huber fixed point predicts the learned bias) passed at R² = 0.986. K1 and K3 failed, both for the same reason: the design gave chain states two identical actions whose Q-estimates are learned separately from noisy targets, so the max over them carries Q-learning's max-bias (+0.09 to +0.31 under the squared loss) on the noisy arm only. That made the squared-loss agent risk-seeking in the hazard cells on its own (K3) and gave the κ = 3 lottery cells a positive instead of negative bias (K1).
- **nomax: INCONCLUSIVE by rule, hypothesis confirmed in substance.** R² = **1.000**; policy flips agree with |predicted bias| > g in **60/60** cells and no squared-loss cell is wrong (K3 passes); |bias| is monotone in κ in every (skew, g, p) (Spearman −0.90 to −1.00). K1's sign clause fails in four cells where Huber is nearly inactive (κ = 3, p = 0.3, predicted bias ≤ 0.013) and the learned bias is noise of either sign; my wording said "every finite-κ cell" where it should have said "every active cell". In all 40 active cells the sign is −skew.
- **DQN K4 (secondary): passed, 13/16.** Failure-Q (true value 1) ends at 26.8 under Huber vs 22.2 under the squared loss. Only 2% of TD errors exceed κ = 1, and those are the terminal ones.

## Key table (nomax, κ = 1, the DQN default; V(risky) − V(safe) = ±g)

| skew | g | p | jump M | Huber bias | bias / gap | greedy wrong |
|---|---|---|---|---|---|---|
| lottery | 0.1 | 0.05 | 22 | −0.897 | 9.0× | 8/8 seeds |
| lottery | 0.1 | 0.30 | 3.7 | −0.576 | 5.8× | 8/8 |
| lottery | 0.3 | 0.30 | 4.3 | −0.747 | 2.5× | 8/8 |
| hazard | 0.1 | 0.05 | 22 | +0.899 | 9.0× | 8/8 |
| hazard | 0.3 | 0.30 | 4.3 | +0.747 | 2.5× | 8/8 |

At κ = 1 the Huber agent takes the wrong fork in every one of the 24 (skew, g, p) cells; at κ = 3 it is wrong wherever the jump exceeds ~3; with the squared loss it is right everywhere (nomax).

## What happened

1. **The Huber fixed point is exact.** The learned Q at the fork matches the root of E[clip(target − q, ±κ)] = 0 to three decimals in every cell once the duplicate-action max-bias is removed. The bias is −skew in sign, decreasing in κ, and vanishes once κ exceeds the jump (≈ M(1−p)).
2. **It is a large bias.** With κ = 1 and a lottery paying 22 with probability 5%, the Huber agent values the lottery at ≈ 0.2 instead of 1.1: the clip gives the rare win a weight of κ instead of M − q, so the fixed point is near the median outcome (zero). Mirror image for hazards: a rare −22 is valued as if it were −1.
3. **The squared loss has its own risk bias, through the max.** In the pilot design, the noisy arm's two duplicate action-values were both noisy estimates and the max over them was biased up by 0.09–0.31, which is the Thrun–Schwartz max-bias on the noisy arm only. So squared-loss Q-learning is risk-*seeking* toward any noisy arm; Huber adds a −skew bias on top, which compounds under hazard (both push toward the risky arm) and partly cancels under lottery. The pilot design was a mistake for isolating Huber but a useful accident for seeing the two together; the Huber clip also suppresses the max-bias (clipped updates have lower variance), which is the residual pattern in `posthoc.txt` §1.
4. **In DQN the clip bites on terminals.** CartPole rewards are 1 and Q ranges up to ~100, so κ = 1 is inactive on ordinary transitions (98% of TD errors are under 1) and active exactly on terminal ones, where δ = 1 − Q(s, a) ≈ −(V − 1). Huber's failure-Q is 21% higher than the squared loss's; final return is equal and AUC is lower under Huber (125 vs 146). This connects to R11: part of the terminal-value inflation there may be the loss, not just the buffer.

## What this says

The Huber loss on the TD error is a change of objective, computable in closed form: it replaces the mean of the target distribution by its Huber location, which under-weights rare outcomes on both sides. For an RL agent this is a risk preference (averse to lotteries, blind to hazards) whose size is set by κ relative to the value jump, and which flips the greedy action whenever it exceeds the gap. In reward-clipped Atari-style training the bias acts on value jumps, not rewards, i.e., on exactly the rare terminations and rare big-value transitions that safety cares about.

## Caveats

- Tabular, L = 3, two-outcome lotteries. Function approximation shares parameters across states, so the clipped update at one state moves others; the fixed-point argument is per-(s, a) and the deep version needs the exact-MC instrument.
- The DQN check is one task, 16 seeds, one κ; it confirms the direction on terminal values, not a performance consequence.
- Variance: the squared loss's estimate of a p = 0.05, M = 22 lottery has sd ≈ 0.5 at α = 0.02 (stationary), which is why the smoke grid was narrowed. Huber trades this variance for bias; the paper must show both sides.
