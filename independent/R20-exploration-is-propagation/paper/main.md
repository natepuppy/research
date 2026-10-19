# Exploration or Propagation? An Accounting of Time-to-Solve in Sparse-Reward Value Learning

**Abstract.** Sparse-reward tasks are called hard-exploration tasks, and the methods built for them (count bonuses, RND, Go-Explore, NGU) shorten the time to *find* the reward. We split the time-to-solve of a value-based learner into two halves, discovery (first reward) and propagation (first reward to first successful greedy episode), and measure both. Under Nature-DQN-like settings (hard target refreshed every 2000 updates, one-step targets) the propagation half dominates: on corridors of depth 10–30 it is 6–52× longer than discovery and accounts for 86–98% of the time-to-solve. It is not a diffuse "learning" cost: it equals the target-sync latency of R12 almost exactly (fitted coefficient 0.96 on (⌈D/n⌉ − 1)·K, R² = 0.996). A fast target with 5-step returns removes it and cuts time-to-solve 7–31× with no change to exploration. A count bonus, the standard remedy, shortened discovery in only 5 of 18 cells and lengthened propagation in 16 of 18, making total time-to-solve worse in 16 of 18. [3080] We replicate the accounting on MinAtar and MiniGrid hard-exploration tasks with DQN and RND/count bonuses. The sparse-reward problem has two halves; the half with a closed form is, in standard settings, the larger one, and the standard remedy targets the other.

## 1. Introduction

A sparse-reward task is "hard" when an ε-greedy agent takes a long time to solve it, and the field's diagnosis is that the agent takes a long time to *find* the reward. The remedies follow the diagnosis: intrinsic rewards for novelty (Bellemare et al. 2016; Burda et al. 2019), archives of visited states (Ecoffet et al. 2021), episodic memories (Badia et al. 2020). But finding the reward once is not solving the task. The value of that single transition must reach the start state through bootstrapping, and bootstrapping in a value-based learner with a hard target network moves one n-step hop per target refresh (R12). With K = 2500 updates per refresh and a 30-step path, that is 72 500 updates during which the agent already "knows" where the reward is and cannot act on it.

We measure the two halves separately. Discovery time T_d is the step at which the first reward is collected; propagation time T_p is the number of steps from then until the greedy policy first solves the task. The task family is chosen so that T_d is controlled (a random walk; we remove all optimism from the untrained agent by breaking greedy ties at random) and T_p can be read against a known law.

Contributions. (1) A pre-registered accounting showing that under standard target-network settings the propagation half is 6–52× the discovery half on depth-10–30 corridors. (2) The identification of that half with R12's target-sync latency (coefficient 0.96, R² = 0.996). (3) The observation that a stored count bonus shortens discovery weakly and lengthens propagation reliably, so that total time-to-solve gets worse. (4) [3080] The same accounting for DQN with RND/count bonuses on MinAtar and MiniGrid hard-exploration tasks.

## 2. Related work

Exploration: count-based (Bellemare et al. 2016; Tang et al. 2017), RND (Burda et al. 2019), Go-Explore (Ecoffet et al. 2021), NGU/Agent57 (Badia et al. 2020). Each evaluates by time-to-reward or final return; none separates post-discovery time. RUDDER (Arjona-Medina et al. 2019) names three sub-problems of delayed reward — discover, keep, learn — but does not measure their shares. Go-Explore's second phase ("robustification") is the closest acknowledgement that post-discovery work is substantial. Laidlaw et al. (2023) predict difficulty from an effective-horizon scalar that mixes both halves. Target-network latency: R12 (this lab) gives t_prop = (⌈L/n⌉ − 1)K + c·⌈L/n⌉L/B; Fedus et al. (2020) find n-step is what makes replay work; Lyle et al. and Kim et al. (2019) study target networks as a stability device, not as a latency. Stale intrinsic rewards in replay are discussed in the NGU and RND follow-ups as an implementation issue; we show where the cost lands. Lab stubs: the delayed-reward cluster concerns reward delay within an episode, not training-time latency; named for adjacency.

## 3. Setting

Corridor of depth D ∈ {10, 20, 30}, start at 0, actions left/right, reward 1 and termination at D, cap 4D, γ = 0.99. Tabular Q-learning with uniform replay (20k, batch 32, one update per step from step 200), α = 0.1, uncorrected n-step targets, hard target table refreshed every K updates. ε = 0.1; greedy ties broken uniformly at random so the untrained agent is a symmetric random walk. Count bonus β/√N(s′) added to the stored reward. Factorial K ∈ {1, 200, 2000} × n ∈ {1, 5} × β ∈ {0, 0.1}, 8 seeds, 300k steps. Every 500 steps the greedy policy is rolled out exactly from the start.

Pre-registered (README): K1 median T_p > median T_d at every D in the (K = 2000, n = 1, β = 0) cells; K2 Spearman(median T_p, (⌈D/n⌉ − 1)K + 1) ≥ 0.8 over no-bonus cells; K3 bonus halves median T_d at every D in the K = 2000, n = 1 cells. SUPPORTED if K1 and K2.

## 4. Results

All 288 runs discovered and solved.

**K1: propagation dominates.** (K = 2000, n = 1, β = 0): T_d = 346 / 1862 / 9324 steps and T_p = 18 154 / 37 716 / 57 648 at D = 10 / 20 / 30; ratios 52× / 20× / 6×; post-discovery share 0.98 / 0.95 / 0.86.

| D | K=1,n=1 | K=1,n=5 | K=200,n=1 | K=200,n=5 | K=2000,n=1 | K=2000,n=5 |
|---|---|---|---|---|---|---|
| 10 | 0.51 | 0.62 | 0.95 | 0.77 | 0.98 | 0.96 |
| 20 | 0.39 | 0.18 | 0.68 | 0.49 | 0.95 | 0.90 |
| 30 | 0.28 | 0.22 | 0.67 | 0.26 | 0.86 | 0.67 |

Median post-discovery share of time-to-solve, no bonus.

**K2: it is the target-sync latency.** Spearman(median T_p, (⌈D/n⌉ − 1)K + 1) = 0.92 over 18 cells. The post-hoc two-term R12 fit gives T_p = 0.96·(⌈D/n⌉ − 1)K + 59·D²/(nB) + 584 (R² = 0.996). The post-discovery time at K = 2000 is the number of target syncs the reward needs, times K, to within 4%.

**Removing it.** (K = 1, n = 5) against (K = 2000, n = 1): time-to-solve 500 vs 18 500 (D = 10), 2250 vs 39 500 (D = 20), 9250 vs 67 500 (D = 30): 31× / 18× / 7×, with the same ε and the same discovery process.

**K3 and the bonus.** The β = 0.1 count bonus halves median T_d at D = 10 and 20 but not at D = 30 (10 055 vs 9324), so K3 fails. Across all 18 cells the bonus shortens median T_d in 5, lengthens median T_p in 16, and makes total time-to-solve worse in 16 (e.g. D = 10, K = 1, n = 1: 500 → 6500 steps). The bonus is stored in the buffer at collection time; replayed stale, it injects value noise of order β at rarely-visited states that competes with the γ^D-attenuated true signal. This mechanism is not tested here.

## 5. Discussion

For a value-based learner with a slow target the sparse-reward problem is, in these tasks, mostly a propagation problem with a closed form, and the fastest fix is not a better explorer but n-step returns and a fast target. This is consistent with what works in practice (Rainbow's n = 3; R2D2's n = 5; Agent57's long n-step traces) and offers a reason: those choices shrink the half that was dominating. Exploration methods are evaluated by first-reward time and by final return; the accounting here suggests adding a third column, the propagation time, because a bonus that buys discovery and sells propagation can be a net loss, and the loss is invisible if only the end points are reported.

The share is not a universal constant. It is high when T_d is small relative to (D/n)K and falls as discovery gets genuinely hard (D = 30: 0.86; a Montezuma room: likely small). The claim is the decomposition and the law for one half, not a percentage.

Limitations. Tabular, deterministic, one-dimensional; the bonus control is weak (β = 0.1, stale) and must be redone with stronger and non-stale bonuses before anything is said about bonus methods in general; ε-greedy only; greedy success as the solve criterion.

## 6. [3080] Planned experiments

(a) DQN on MinAtar Freeway and a sparse-reward Seaquest variant, and on MiniGrid DoorKey-8x8 / KeyCorridor-S3R3, logging first extrinsic reward and first successful greedy episode; factorial K ∈ {100, 1000, 8000} × n ∈ {1, 3, 10} × bonus ∈ {none, count, RND}; pre-registered: post-discovery share under Nature-DQN defaults, and the sign of the bonus effect on T_p. (b) Non-stale bonus (recomputed at replay time) vs stored bonus to test the mechanism in §4. (c) Decoupled n for the first K updates after discovery (a "propagation burst") as a cheap intervention targeted at the larger half.

## 7. Conclusion

Discovery and propagation are different problems with different fixes. Measured separately, propagation is the larger half under standard value-based settings, it equals the target-sync latency, and the standard exploration remedy makes it worse.

## References

Bellemare et al. 2016; Tang et al. 2017; Burda et al. 2019 (RND); Ecoffet et al. 2021 (Go-Explore); Badia et al. 2020 (NGU, Agent57); Arjona-Medina et al. 2019 (RUDDER); Laidlaw et al. 2023 (effective horizon); Fedus et al. 2020; Hessel et al. 2018 (Rainbow); Kapturowski et al. 2019 (R2D2); Mnih et al. 2015; R12 (this lab).
