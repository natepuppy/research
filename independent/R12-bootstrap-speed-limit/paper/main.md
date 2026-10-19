# The Speed Limit of Bootstrapping: How Target Networks and Replay Set the Pace of Value Propagation

*Draft v0.1 (pilot scale). Sections marked [3080] are planned experiments; all numbers in this draft come from the CPU pilot and confirmation runs in `../results/`.*

## Abstract

Temporal-difference learning moves information about reward backwards one bootstrap hop at a time. Two pieces of standard deep-RL machinery set the pace of a hop. A target network refreshed every K gradient updates means a value can advance one n-step hop per refresh. Uniform experience replay means a transition at the current propagation frontier is sampled roughly once every L/B updates for an episode of L transitions and a batch of B. We state the resulting speed limit for a reward L steps from the start state,

t_prop = (⌈L/n⌉ − 1)·K + c·⌈L/n⌉·L/B,

and test it on tabular chains across 448 configurations spanning four orders of magnitude in t_prop. The target term is exact to within 0.2% (median) whenever it dominates; the replay term scales as predicted; the two add. The law was pre-registered in a sharper max() form, failed that test, was refined on the pilot data, and then confirmed out of sample on a disjoint grid with the constant held fixed (identity-fit R² = 0.997 for SGD, 0.971 for Adam). The law explains why target period and n-step returns interact, predicts that prioritised replay removes only the replay term, and gives a budget rule: in sparse-reward tasks the target period should not exceed the training budget divided by the number of hops. [3080] We then measure the effective hop length under function approximation and test the rule on DQN with sparsified MinAtar rewards.

## 1. Introduction

Deep value-based agents learn by bootstrapping: the target for Q(s, a) is built from Q at the next state (or n states on), and that estimate is itself built from estimates further along. Information about a reward therefore moves backwards through the state space one hop at a time, and the speed of that motion bounds how fast anything can be learned in a task where reward is far from where the agent starts.

Two design choices that every DQN-style agent makes constrain this speed, and they are usually chosen independently. The first is the target network (Mnih et al., 2015): targets are computed from a frozen copy of the network refreshed every K updates. Between refreshes, no new information can enter the targets, so the frontier of propagation can advance at most one hop per refresh. The second is uniform experience replay (Lin, 1992; Mnih et al., 2015): a transition is updated only when it is sampled, and the frontier transition of an episode of L transitions is one of L, so it waits about L/B updates for a minibatch of B to include it.

Both constraints are obvious once stated. Neither has been stated as a quantitative law, measured, or used to set hyperparameters jointly, even though Fedus et al. (2020) found empirically that the benefit of a large replay buffer depends on n-step returns and van Hasselt et al. (2018) and Lee & He (2019) analysed target networks for stability rather than speed. This paper does three things.

1. **States the law.** For a deterministic chain with a terminal reward L steps from the start, a hard target network refreshed every K updates, uniform replay over a stationary buffer, n-step targets and batch B, the first update at which the start state's value reaches a fixed fraction of its true value is t_prop ≈ (⌈L/n⌉ − 1)·K + c·⌈L/n⌉·L/B, with c a constant that depends on the optimiser and the step size but not on L, n, K or B (Section 3).
2. **Tests it honestly.** The pre-registered form, hops·max(K, c·L/B), was wrong in the crossover region. We report that failure, the refinement to the additive form, and an out-of-sample confirmation on a disjoint grid with c fixed (Section 4).
3. **Draws the design consequences** (Section 5): n-step returns are worth exactly a factor n in both terms; prioritised replay should remove the replay term but not the target term; and for sparse-reward tasks the target period should satisfy K ≤ budget/(L/n), which Nature-DQN settings violate for any episode longer than about 80 steps.

## 2. Related work

**Target networks.** Introduced for stability (Mnih et al., 2015); analysed as a damped fixed-point iteration (Lee & He, 2019; Fellows et al., 2023; Zhang et al., 2021) and as a source of the deadly-triad instability or its cure (van Hasselt et al., 2018). The *speed* cost of a long target period is folklore ("slower but more stable"). The closest statement is one sentence in Piché et al. (2022, functional regularisation): on an N-state chain with tabular values and a target refreshed every H steps, propagating the terminal reward "will require NH training steps"; they use it to motivate replacing the target network and do not test it, add the n-step or replay terms, or connect it to learning curves. We cite it as prior statement of the leading term.

**Replay.** Lin (1992); Mnih et al. (2015); Schaul et al. (2016) prioritise by TD error, which we predict removes the replay term of the law. Fedus et al. (2020) show that larger buffers help only with n-step returns and attribute it to off-policy-ness and staleness; our law gives a complementary mechanistic reading in terms of frontier sampling. Reverse and episodic backward replay (Lee et al., 2019; Rotinov, 2019) are explicitly designed to propagate faster, which the law explains: ordering updates from the terminal backwards makes every hop happen in one pass.

**n-step returns.** Sutton (1988); Hessel et al. (2018) and Kapturowski et al. (2019) use n = 3 and n = 5 as tuned constants; Hernandez-Garcia & Sutton (2019) study n in DQN empirically. The law says what n buys: a factor n in both terms, for as long as n < L.

**Propagation speed.** The horizon dependence of TD convergence is classical (Bertsekas & Tsitsiklis, 1996; Bhandari et al., 2018) and is about the discount factor. We hold γ fixed and vary the mechanics of the update instead; the quantity of interest is the time until the signal *arrives*, not the time until it is accurate.

## 3. Setting and the counting argument

**Environment.** A chain of states 0, 1, …, L−1 and a terminal state. The policy moves right deterministically. Reward 1 on the final transition, 0 elsewhere, discount γ. The true value is V(s) = γ^{L−1−s}.

**Learner.** A value table V and a target table V̄. At each update t: sample B transitions uniformly from a stationary buffer (uniform over the L transitions of the episode); for a sampled transition from state s, the n-step target is y(s) = γ^{L−1−s} if s + n ≥ L (the return is exact, no bootstrap) and y(s) = γ^n V̄(s+n) otherwise; apply the update V(s) ← V(s) + step(y − V(s)). The target table is refreshed, V̄ ← V, every K updates. Two optimisers: per-hit SGD with step α on every sampled copy, and per-state Adam on the batch-mean gradient (the realistic case: a tabular network with Adam).

**Propagation time.** t_prop = min{t : V_t(0) ≥ ½·V(0)}.

**Proposition (target term).** Let h = ⌈L/n⌉ be the number of hops from the start to the exact-return region. With a hard target network, V_t(0) = 0 for all t < (h − 1)·K.

*Proof sketch.* The states within n of the terminal receive exact targets and become non-zero at the first update that samples them. Any state s with s + n < L receives targets from V̄(s + n), which is zero until the first refresh after V(s + n) becomes non-zero. By induction, the j-th hop from the end cannot become non-zero before the (j − 1)-th refresh, so the start state (hop h) cannot move before (h − 1)·K updates have passed. ∎

The proposition is a lower bound. The observation of this paper is that it is attained almost exactly whenever K is large enough for the frontier to be fitted between refreshes.

**Replay term.** When K is small, the bottleneck is sampling. Each update samples B of L transitions uniformly, so the frontier transition appears in a batch with probability ≈ B/L and waits L/B updates in expectation; a hop needs some number of hits that depends on the optimiser and step, which we absorb into c. Over h hops this gives c·h·L/B. We do not have a proof that the two terms add rather than interleave; the pre-registered guess was that they compete (max), and the data say they add.

**The law.** t_prop ≈ (h − 1)·K + c·h·L/B.

## 4. Experiments

### 4.1 Pre-registered pilot

Grid: L ∈ {8, 16, 32, 64}, n ∈ {1, 2, 4, 8}, K ∈ {1, 10, 100, 1000}, B ∈ {4, 32}, optimiser ∈ {SGD α = 0.3, Adam lr = 0.05}, γ = 0.99, three seeds; 256 configurations, cap 400k updates; every cell converged. The pre-registered tests (README) were for the form h·max(K, c·L/B) with c fitted: K1 R² ≥ 0.9; K2 regime slopes on K (≥ 0.8 target-limited, ≤ 0.2 replay-limited); K3 slope on n in [−1.3, −0.7].

**Outcome: KILLED by the rule.** SGD R² = 0.883; Adam R² = 0.902 but its replay-limited slope was undefined because the grid had only one K value below the fitted crossover (a design flaw). K2 for SGD passed (0.91 / 0.03). K3 failed for both (−1.9, −1.5) because cells with n ≥ L were included, where no bootstrapping happens.

### 4.2 What the pilot data showed

Table 1 (n = 1; geometric mean over seeds) shows the two regimes. With K ≥ 100 the propagation time is (h − 1)·K + a few updates: for L = 64, K = 1000, t_prop = 63 003 under SGD and 63 005 under Adam against (h − 1)·K = 63 000. Across the 60 target-limited cells with h > 1, the median of t_prop/((h − 1)·K) is 1.002 (SGD) and 1.004 (Adam). With K = 1, t_prop scales as L²/(nB) (SGD: slope 0.99, R² 0.99 across 32 cells). In between, the two add: the additive form has identity-fit R² (log space, no free slope or intercept) 0.996 for SGD and 0.924 for Adam, against 0.860 and 0.881 for the pre-registered max form (`results/posthoc.txt`).

| L | K | B | t_prop (SGD) | t_prop (Adam) | (h−1)·K |
|---|---|---|---|---|---|
| 64 | 1 | 32 | 422 | 186 | 63 |
| 64 | 10 | 32 | 790 | 642 | 630 |
| 64 | 100 | 32 | 6 303 | 6 309 | 6 300 |
| 64 | 1000 | 32 | 63 003 | 63 005 | 63 000 |
| 64 | 1 | 4 | 3 422 | 1 328 | 63 |
| 64 | 100 | 4 | 7 445 | 6 426 | 6 300 |
| 16 | 1 | 32 | 26 | 35 | 15 |
| 16 | 1000 | 32 | 15 001 | 15 003 | 15 000 |

*Table 1. Propagation time (updates) at n = 1. The target term is attained almost exactly when it dominates; the replay term appears when K is small, and is larger for small B.*

### 4.3 Out-of-sample confirmation

Because the additive law was found in the data, we pre-registered a confirmation (README, "Confirmation") on a grid sharing no values with the pilot: L ∈ {12, 24, 48, 96}, n ∈ {1, 3, 6}, K ∈ {3, 30, 300, 3000}, B ∈ {8, 16}, three new seeds, with c fixed at the pilot values (c_SGD = 2.63, c_Adam = 4.21). Test C1: identity-fit R² in log space ≥ 0.95 (SGD) and ≥ 0.90 (Adam).

**Outcome: SUPPORTED.** R² = 0.997 (SGD) and 0.971 (Adam) over 96 cells each, all converged. Median t_prop/pred 0.97 and 0.96. By K: SGD medians 0.98, 0.87, 0.97, 1.00 at K = 3, 30, 300, 3000; Adam 0.62, 0.75, 0.96, 1.00. Adam's replay-limited cells are noisier (10–90% range 0.47–1.00) and faster than predicted at small K, consistent with its per-state normalisation taking large first steps on a freshly non-zero frontier.

### 4.4 n-step

Within target-limited cells where bootstrapping happens (h ≥ 2), the slope of log t_prop on log n is −1.21 (SGD) and −1.18 (Adam) over 16 groups each: slightly steeper than −1 because h = ⌈L/n⌉ − 1 rather than L/n. When n ≥ L the problem is Monte-Carlo and the law degenerates to the replay term alone.

### 4.5 [3080] Planned

(a) Stochastic chains and gridworlds: frontier as a distribution, law in expectation. (b) Linear and deep function approximation on the chain: generalisation makes a hop longer than n; we define n_eff by inverting the law and relate it to feature similarity of adjacent states. (c) DQN on MinAtar with rewards sparsified to episode success: propagation time against (K, n), additive law with fitted n_eff. (d) Polyak targets (prediction: K_eff ≈ 1/τ) and prioritised replay (prediction: replay term vanishes, target term unchanged).

## 5. Consequences

**Target period and n-step interact.** Both enter through h = ⌈L/n⌉. Doubling n halves both terms; doubling K doubles only the first. A practitioner who lengthens the target period for stability and does not raise n pays the full price in propagation time.

**A budget rule.** For a reward that first appears L steps from the start, the start state's value cannot move before (⌈L/n⌉ − 1)·K updates. Nature-DQN uses K = 2 500 updates, n = 1, B = 32: a reward 100 steps away needs ≈ 250 000 updates (1M environment steps) before it is *first* felt at the start; the replay term (c·100·100/32 ≈ 800) is negligible. Rainbow's n = 3 cuts this to ≈ 83 000 updates; R2D2's n = 5 to 50 000. Function approximation shortens the effective chain, and real tasks have shaping rewards, so these are lower bounds on a stylised quantity, but they are of the order of the gaps people observe between algorithms on sparse-reward games.

**Prioritised replay removes one term.** The frontier transition has the largest TD error, so PER samples it immediately: the replay term disappears and the target term is untouched. This predicts that PER helps most when K is small relative to c·L/B and not at all when K dominates. [3080] tests this.

**Why reverse replay works.** Updating an episode's transitions from the terminal backwards within one target period makes every hop happen in a single pass: t_prop ≈ K. The law says this is the best possible with a hard target network, and that the remaining cost is K itself.

## 6. Limitations

Everything here is tabular, deterministic and uniform-replay, with a single terminal reward and a stationary buffer. The constant c was fitted (one scalar per optimiser) and is not derived. The propagation criterion measures arrival of signal, not accuracy. Function approximation is the central open question: generalisation can move the frontier more than one state per hop, and the deep version of the law needs an effective hop length. Polyak averaging, prioritised replay and stochastic dynamics are untested predictions, listed as such.

## 7. Reproducibility

`code/r12_speed_limit.py` runs the pilot (`--seeds 3`, about 4 CPU-minutes) and the confirmation (`--confirm`, about 1 CPU-minute). The README records the pre-registration of both the original kill test and the confirmation test. All raw results are in `results/`.

## References

- Bertsekas, D. & Tsitsiklis, J. (1996). *Neuro-Dynamic Programming.*
- Bhandari, J., Russo, D., Singal, R. (2018). A finite time analysis of temporal difference learning with linear function approximation. COLT.
- Fedus, W. et al. (2020). Revisiting fundamentals of experience replay. ICML.
- Fellows, M. et al. (2023). Why target networks stabilise temporal difference methods. ICML.
- Hernandez-Garcia, J. F. & Sutton, R. S. (2019). Understanding multi-step deep reinforcement learning: a systematic study of the DQN target. arXiv:1901.07510.
- Hessel, M. et al. (2018). Rainbow: combining improvements in deep reinforcement learning. AAAI.
- Kapturowski, S. et al. (2019). Recurrent experience replay in distributed reinforcement learning. ICLR.
- Lee, D. & He, N. (2019). Target-based temporal-difference learning. ICML.
- Lee, S. Y., Choi, S., Chung, S.-Y. (2019). Sample-efficient deep reinforcement learning via episodic backward update. NeurIPS.
- Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning.*
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature.*
- Rotinov, E. (2019). Reverse experience replay. arXiv:1910.08780.
- Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized experience replay. ICLR.
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning.*
- van Hasselt, H. et al. (2018). Deep reinforcement learning and the deadly triad. arXiv:1812.02648.
- Zhang, S., Yao, H., Whiteson, S. (2021). Breaking the deadly triad with a target network. ICML.
