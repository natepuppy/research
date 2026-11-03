# Credit Assignment Is a Count of Iterations

*Combined draft. Every number is from a finished laptop pilot named in the text and listed in `../results/ledger.md`. Sections marked [3080] have not been run. No new experiment was run for this draft.*

## Abstract

A reward D steps from a decision cannot influence that decision until the value estimate has carried it there, and in every standard deep RL agent the carry is metered by the target. We show that the three target mechanisms in use are three ways of writing the same count. A hard target refreshed every K updates moves the value one n-step hop per refresh, so the decision waits (⌈D/n⌉ − 1)·K updates. A Polyak target of rate τ waits (⌈D/n⌉ − 1)/τ updates plus ⌈D/n⌉/α for the critic's own fit, and the critic's horizon therefore grows linearly in updates: at the default τ = 0.01 an agent is a γ ≈ 0.9 agent for its first thousand updates. GAE does not decay a distant reward; with a critic refit each iteration it delays the reward by ≈ D(1 − γλ) iterations, plus a second term D/T when the rollout is shorter than the task, and the policy moves in the same iteration the value arrives. The hard-target count is exact for a neural critic wherever data covers the path: under four encodings that generalise very differently, a sync carries 1.02–1.03 hops (R² = 1.000). The one knob that divides all three counts is n, and uncorrected n-step returns from ε-greedy data impose a ceiling on it that does not depend on the target; below that ceiling a smaller n is pure latency, and at ε = 0.01 raising n from 1 to 12 makes the correct policy arrive 18× sooner. Under Nature-DQN settings the count is the larger half of time-to-solve on sparse corridors (86–98%), and a count bonus, the usual remedy, made total time worse in 16 of 18 cells. The laws fit at R² ≥ 0.996 on tables. [3080] The same measurements in MinAtar DQN, CleanRL PPO and TD3 are what the paper still needs.

## 1. Introduction

Deep RL learns by bootstrapping. The target for a state is built from the estimate at a later state, and that estimate is itself built from something further on, so a reward moves backward one hop at a time. Three mechanisms decide how long a hop takes, and each is set as a stability knob rather than as a horizon.

A DQN target network is a frozen copy, refreshed every K gradient updates (Mnih et al., 2015). Between refreshes no new information enters the targets, so the front advances at most one hop per refresh. TD3 and SAC do not refresh; they average, Q̄ ← Q̄ + τ(Q − Q̄), with τ = 0.005 (Lillicrap et al., 2016; Fujimoto et al., 2018; Haarnoja et al., 2018). PPO refits a critic each iteration to λ-returns built from the previous critic (Schulman et al., 2016, 2017). All three move the front a fixed distance per update. The distance is a function of a hyperparameter that every configuration file contains and that nobody treats as one.

The closest prior statement is one sentence. Piché et al. (2022) note that on an N-state chain a hard target refreshed every H steps needs NH training steps to propagate a terminal reward, and that Polyak averaging also slows propagation. They do not write the n-step or replay terms, do not treat τ, α or λ, and do not measure any of it. VC-PPO (Yuan et al., 2025) observes that in long-horizon RLHF a reward barely reaches early tokens at λ = 0.95, calls this decay, and sets the critic's λ to 1. The observation is right and the name is wrong: with a critic the signal is delayed, not lost, and the delay has a closed form. Fedus et al. (2020) find that n-step returns are what make a large replay buffer usable. The latency account says why.

This paper states the three counts, shows they are one object, and draws the two consequences that follow before any new algorithm does.

1. **One count, three mechanisms.** Hard sync, Polyak averaging and GAE each move the value front a fixed number of states per update. The arrival time at a decision D steps from a reward is (⌈D/n⌉ − 1)·K, or (⌈D/n⌉ − 1)/τ + ⌈D/n⌉/α, or D·((1 − γλ) + 1/T). A hard sync every K = 1/τ updates reproduces the Polyak schedule as a staircase through the same points.
2. **The count survives function approximation.** A 2×64 network carries 1.02–1.03 hops per sync under one-hot, random, thermometer and coordinate inputs. Where every state has data, generalisation cannot invent a target the sync has not yet written.
3. **n is the knob, and it has a ceiling.** Below the largest n whose uncorrected-return bias stays inside the value gap, a smaller n costs only latency. The ceiling depends on ε and the gap and does not depend on K.
4. **The count is what sparse-reward time is made of.** On corridors of depth 10–30 with K = 2000 and n = 1, the updates after the reward has been found and before the greedy policy can use it are 86–98% of time-to-solve, and they match the hard-sync count with coefficient 0.96.

Every number below comes from a pre-registered tabular or small-network pilot. Two of the laws were refined after a pre-registered form failed; both refinements were then confirmed on data the fit had not seen, and both failures are reported in place. The deep replications are specified and unrun.

## 2. The count

**Setting.** A reward lies D transitions from a decision. The learner bootstraps n steps at a time, so the reward reaches the decision in h = ⌈D/n⌉ hops. The last hop is a sampled return and needs no bootstrap. What differs across agents is how long one hop waits.

**Hard target.** Let V be the online table and V̄ a copy refreshed every K updates. A state whose bootstrap target lies beyond the front receives V̄ = 0 until the refresh after the next state becomes non-zero. By induction the decision is identically zero until update (h − 1)·K. When the frontier transition must also be sampled from a buffer of L transitions in batches of B, a second term appears, proportional to h·L/B, and the two add:

t = (h − 1)·K + c·h·L/B.

The constant c depends on the optimiser and the step size. It does not depend on D, n, K or B. When K is large the second term is a rounding error.

**Polyak target.** The online value relaxes toward its target at rate α and the target relaxes toward the online value at rate τ. A reward D steps downstream passes through h filters of rate α and h − 1 filters of rate τ. The mean arrival time is

t = (h − 1)/τ + h/α.

Reading the horizon off the start-state value, it advances at n / (1/τ + 1/α) states per update. τ and α are interchangeable: a slow fit costs the same as a slow target. Setting K = 1/τ in the hard law recovers the same schedule, with a staircase in place of a line.

**GAE.** At iteration k the λ-return at a state is a geometric mixture of returns bootstrapped from V_{k−1}. If the previous critic already carries the reward within about 1/(1 − γλ) states, this one does too. The front moves ≈ 1/(1 − γλ) states per critic fit, and a rollout of T steps caps the move again, because the return bootstraps at the segment boundary. The two caps combine as rates:

t = D·((1 − γλ) + 1/T_eff)

critic fits. The policy gradient at the decision is driven by the same mixture, so the policy moves in the iteration the value arrives.

**What n does.** In all three expressions n appears only inside h = ⌈D/n⌉. Doubling n halves the count. Doubling K, halving τ, or moving λ from 1 to 0.95 changes only one mechanism's term. This is why n-step returns show up in every sample-efficient value-based agent and why lengthening the target period without raising n is the expensive way to buy stability.

| mechanism | who uses it | updates until a reward D steps away arrives |
|---|---|---|
| hard refresh every K | DQN, Rainbow | (⌈D/n⌉ − 1)·K |
| Polyak rate τ, fit rate α | DDPG, TD3, SAC | (⌈D/n⌉ − 1)/τ + ⌈D/n⌉/α |
| GAE, rollout T | PPO | D·((1 − γλ) + 1/T) iterations |

*Table 1. The replay term of the hard law and the O(1) fitting delay are omitted; both are measured below and both are small once the target term dominates.*

## 3. Hard targets

The pilot is a deterministic chain, tabular V, stationary uniform replay, n-step targets, SGD and Adam (R12). Propagation time is the first update at which the start state reaches half its true value.

The pre-registered form said the two terms compete, t = h·max(K, c·L/B). That form failed (SGD R² = 0.883). The data say the terms add. On the 60 target-limited cells the median of t / ((h − 1)·K) is 1.002 for SGD and 1.004 for Adam: for L = 64, n = 1, K = 1000 the measured times are 63 003 and 63 005 against a prediction of 63 000. With K = 1 the time scales as L²/(nB) (slope 0.99). Because the additive form was read off the pilot, it was re-tested on a disjoint grid — no shared value of L, n, K or B — with c frozen at the pilot values 2.63 (SGD) and 4.21 (Adam). Identity-fit R² in log space is 0.997 and 0.971 over 96 cells each.

| L | K | B | SGD | Adam | (h − 1)·K |
|---|---|---|---|---|---|
| 64 | 1 | 32 | 422 | 186 | 63 |
| 64 | 100 | 32 | 6 303 | 6 309 | 6 300 |
| 64 | 1000 | 32 | 63 003 | 63 005 | 63 000 |
| 64 | 1 | 4 | 3 422 | 1 328 | 63 |
| 16 | 1000 | 32 | 15 001 | 15 003 | 15 000 |

*Table 2. Propagation time in updates, n = 1. The target term is attained when it dominates. The replay term is what remains at K = 1, and it grows when the batch shrinks.*

Nature DQN refreshes every 2 500 updates with n = 1. A reward 100 steps from the start cannot move the start state's value before (100 − 1)·2500 ≈ 250 000 updates, about a million environment steps. The replay term at B = 32 is a few hundred updates and does not matter. Rainbow's n = 3 cuts the wait by about three; R2D2's n = 5 by about five. These are lower bounds on a stylised chain. They are the right order of magnitude for the gaps between these agents on sparse-reward games.

## 4. Polyak targets

The pilot is synchronous and tabular: every state is updated every step, the target is a geometric average, and the horizon is read by inverting V(s₀) = (1 − γ^h)/(1 − γ) on a chain that pays 1 per step (R21). A second task is a patience fork, one reward now against a larger reward d steps later, and the measurement is the update at which the patient action becomes preferred.

The horizon slope matches n / (1/τ + 1/α) in all 18 cells of τ ∈ {0.1, 0.01, 0.001}, α ∈ {1, 0.1, 0.01}, n ∈ {1, 5}, with R² = 1.000 and ratios between 0.98 and 1.10. The patience switch matches the closed form in all 54 cells, within a factor [0.79, 1.03], log–log slope 1.011, R² = 0.9997. At τ = α = 0.01 the wait is 19 750 updates against 9 855 with α = 1: the fit rate doubled the latency exactly as τ would have. A hard target with K = 100 matches τ = 0.01 (9 901 against 9 855). The hard front is the integers 1, 2, 3, … at updates 100, 200, 300; the soft front is the line through those points.

At τ = 0.01, α = 1 and n = 1 the horizon is 2, 11, 31 and 100 after 100, 1 000, 3 000 and 10 000 updates. For the first thousand updates this agent is a γ ≈ 0.9 agent. The default τ = 0.005 is 200 updates per hop before the fit delay is added. On a dense MuJoCo run of a million steps the horizon arrives quickly and the cost is invisible. On anything sparse, or short, it is the schedule.

## 5. GAE

The pilot is an actor–critic loop with an exact critic: each iteration collects 32 episodes, sets the critic to the per-state mean of its own λ-returns, and takes one tabular policy step (R19). Two corridors of depth D leave a choice state; one pays 1. Arrival is the iteration at which the start-state value reaches half its on-policy value, and separately the iteration at which the policy's success probability crosses 0.75.

Over the 30 cells with λ < 1 and no rollout cap, t_V = 1.25·D(1 − λ) + 1.9 with R² = 0.999. The pre-registered regression of log t on log D and log(1 − λ) has slopes 0.95 and 0.88. At λ = 1 the value arrives in one iteration at every depth from 8 to 128.

| λ | D = 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 0 | 11 | 22 | 42 | 82 | 160 |
| 0.8 | 3 | 6 | 10 | 20 | 38 |
| 0.95 | 1 | 2 | 4 | 6 | 11 |
| 0.99 | 1 | 1 | 1 | 2 | 3 |
| 1 | 1 | 1 | 1 | 1 | 1 |

*Table 3. Median iterations until the start-state value arrives. T = ∞.*

The policy inherits the count. Across 70 cells the Spearman correlation of the two arrival times is 0.99, and the policy arrives within [−2, +1] iterations of the value. All 70 cells reach success probability 0.98 by iteration 600. Freezing the critic at zero removes that guarantee: at λ = 0.95 the frozen learner solves D = 32, takes 218 iterations at D = 64, and fails at D = 128 with final success 0.16. Decay is the right description of a critic that is never updated. It is the wrong description of PPO.

A rollout of 16 steps reintroduces the delay at every λ, including λ = 1 (25 iterations at D = 128). The pre-registered claim that the two caps combine by a minimum fit worse than the rate sum 1/h = (1 − λ) + 1/T_eff, which fits the 35 short-rollout cells at R² = 0.998 with T_eff = 6.1 states. At T = 16, λ = 0.99 and λ = 1 are the same curve. Short segments dominate the target, which is why `num_steps` = 128 on a thousand-step episode puts the delay back into a run whose λ was set to 1 to remove it.

## 6. The same count in a network

The objection to all of the above is that a table cannot generalise and a network can, so one sync might carry the value several states and the counts would be loose upper bounds. We trained one 2×64 MLP, Adam, on one fixed dataset covering a corridor of length 40, and changed only the encoding: one-hot, random codes, thermometer, and a scalar coordinate (R22). The value front is the contiguous run of states, measured from the goal, whose predicted value has reached half the truth. Hops per sync are 40 divided by the slope of arrival time on K.

| encoding | K = 50 | K = 200 | K = 800 | hops per sync |
|---|---|---|---|---|
| one-hot | 2 100 | 7 850 | 31 250 | 1.03 |
| random | 1 850 | 7 850 | 31 250 | 1.02 |
| thermometer | 2 050 | 7 850 | 31 250 | 1.03 |
| coordinate | 2 050 | 7 850 | 31 250 | 1.03 |

*Table 4. Median updates until the value front reaches the start. Eight seeds. R² of each fit is at least 0.99998.*

Forty hops at K = 800 take 31 250 updates for every encoding: 39 syncs, plus about a hundred updates to fit the front. The hypothesis that a smooth encoding would carry at least twice as many hops is rejected at the third decimal. The reason is the target. Beyond the front the target network is zero, so the regression targets are zero, and a network trained where data exists is pushed toward zero there whatever its inductive bias. It can interpolate a target. It cannot invent one the sync has not written.

Representation matters in the two places the target does not pin. Across a block of ten states with no data, the coordinate network crosses at the full-coverage time with values within 1% of the truth; the one-hot network skips the gap with values 24% high. And the greedy policy can be right before any value arrives: with coordinate inputs the policy is correct at update 250 and the values arrive at 7 850, because the right action is a monotone function of the feature and the network's shared bias copies it. Return curves are the wrong instrument for these laws. The value front is the right one.

## 7. How large n can be

Since n divides every count, the useful question is how large it may be. Uncorrected n-step returns, which is what DQN-style agents use, answer it (R24). On a path where one wrong action loses the reward, each n-step segment is built from the behaviour policy's actions. If that policy falls off the path with probability f = ε/2 per step, then after h = ⌈(D+1)/n⌉ segments the value of the path is

γ^D (1 − ε/2)^{(D+1) − h},

and the greedy policy prefers the path only while this exceeds the best alternative. The largest such n depends on ε, the length and the gap. It does not depend on K. Below it, every reduction in n is paid entirely in latency.

The pilot is tabular Q-learning from fixed ε-greedy datasets, 1 344 learners, path length 20, ε ∈ {0.01, 0.03, 0.05, 0.1}, gaps 0.1–0.3, n up to 20, K ∈ {200, 2000}. Time to the correct greedy policy follows (⌈21/n⌉ − 1)·K + 1 with Spearman correlation 1.00 over the 66 cells that solve, median ratio 1.05, and no fitted constant. The learned values match the bias formula with correlation 0.985. The largest n that solves a cell is within one grid step of the predicted ceiling in 9 of 12 cells; the three misses are the cells whose predicted margin (0.026, 0.005, 0.001) sits inside the seed noise of the learned value (standard deviation 0.06), and with a margin of 0.05 all twelve agree. The ceiling is the same at K = 200 and at K = 2000 in all 12 cells.

At ε = 0.01, K = 2000 and a gap of at least 0.2, the correct policy arrives at update 40 200 with n = 1 and at update 2 200 with n = 12, in 8 of 8 seeds either way.

The published settings fall out of the same formula. At ε = 0.1, which is where Nature DQN finishes its anneal, any n > 1 is unsafe on a 20-step path with a gap of 0.1 to 0.3, so n = 1 was the right choice. At ε = 0.01 the ceiling is above 40, so Rainbow's n = 3 is safe and leaves most of the available speed-up unused, while DER's n = 20 and BBF's anneal from 10 to 3 sit inside the rule. The anneal is the corollary: early in training the gaps are large and latency dominates; late, the remaining decisions are the close ones and the ceiling falls.

## 8. Where the time goes

Sparse-reward tasks are described as hard exploration, and the methods built for them shorten the time to find the reward. Finding it is half of the problem. On a corridor, with ε-greedy tabular Q-learning, a hard target and one-step returns, we split time-to-solve into discovery (first reward) and propagation (first reward until the greedy policy solves the task) (R20).

At K = 2000 and n = 1 the two halves are 346 against 18 154 steps at depth 10, 1 862 against 37 716 at depth 20, and 9 324 against 57 648 at depth 30. Propagation is 98%, 95% and 86% of the total. A two-term fit of that half on the hard-sync law has coefficient 0.96 and R² = 0.996. Replacing (K = 2000, n = 1) by (K = 1, n = 5), with the same exploration, cuts time-to-solve by 31×, 18× and 7× at the three depths.

A stored count bonus, the standard remedy, shortened discovery in 5 of 18 cells, lengthened propagation in 16 of 18, and made the total worse in 16 of 18. The pre-registered claim that the bonus would halve discovery failed. The bonus result is the weak one in this paper: one coefficient, a stale bonus, a corridor. The accounting is not. The post-discovery half is the count in Table 1, and under the settings people actually run it is the larger half. Its share falls as discovery gets genuinely hard. The claim is the split and the law for one side of it.

## 9. What to set

The practical content is one rule per mechanism, all of them the same rule.

- **DQN-style.** Keep (⌈D/n⌉ − 1)·K well inside the update budget. If a reward a hundred steps away matters, K = 2500 and n = 1 spends the first million environment steps before the start state moves. Raise n up to the ceiling in Section 7; do not raise it past the point where the uncorrected bias exceeds the gap. Shorten K before adding an exploration bonus.
- **TD3 and SAC.** τ and the critic's fit rate are the same knob. τ = 0.005 is 200 updates per hop, and for the first several thousand updates the agent is solving a shorter-horizon problem than the one written in γ. If the early critic fits slowly, lowering τ further buys nothing.
- **PPO.** Count iterations, not samples. A reward D steps away needs about D(1 − γλ) critic fits to reach the decision, and D/T more if the rollout is shorter than the task. λ_critic = 1 removes the first term only when the rollout covers the distance. Raising the learning rate does not.

## 10. Related work

Target networks were introduced for stability (Mnih et al., 2015) and analysed as a damped fixed point (Lee and He, 2019; Zhang et al., 2021; Fellows et al., 2023). The speed cost is folklore. Piché et al. (2022) state the hard-sync leading term in one sentence and remark that Polyak averaging slows propagation; the n-step term, the replay term, the τ–α sum and the measurements are ours. GAE is presented as a bias–variance trade-off within one update (Schulman et al., 2016). VC-PPO (Yuan et al., 2025) is the nearest neighbour of the GAE result: it identifies the failure of λ = 0.95 on long sequences, frames it as decay, and decouples the critic's λ from the actor's. The iteration count and the rollout term are the account of their observation. Fedus et al. (2020) show empirically that n-step returns are what make large buffers work; Section 2 is the mechanism. Hernandez-Garcia and Sutton (2019) document that uncorrected n-step DQN needs careful tuning; Section 7 says at which ε the tuning becomes necessary. Panahi et al. (2024) vary the target period to control over-generalisation; Section 6 measures the quantity they hold fixed, hops per sync, and finds it invariant to the representation. Exploration methods (Bellemare et al., 2016; Burda et al., 2019; Ecoffet et al., 2021; Badia et al., 2020) are evaluated by first-reward time and final return. Go-Explore's robustification phase and RUDDER's three-way split (Arjona-Medina et al., 2019) are the closest acknowledgements that post-discovery work is real; neither identifies it with the target period.

Discount schedules are usually chosen. OpenAI Five annealed γ, and François-Lavet et al. (2015) study increasing it. Section 4 says a schedule is already present, set by τ.

## 11. What is not shown

The pilots are exact where they can be: tables with synchronous or uniform updates, one chain or one corridor, and in the network case a small MLP on a fixed dataset. Three limitations bound the claim.

The constants are partly fitted. The hard-sync replay coefficient c, the GAE factor 1.25 and the effective rollout length 6.1 are properties of the optimiser and the task, predicted in form and not in magnitude. The leading coefficients — 1 on K, 1 on 1/τ, 1 on D(1 − λ), 1 hop per sync — are not fitted, and they are the result.

Two pre-registered forms failed and were replaced. The hard-sync law was registered as a maximum of the two terms and the data forced a sum; the sum then held on a disjoint grid with the constant frozen. The GAE rollout law was registered as a minimum of the two caps and the data forced a sum of rates. Both are reported as refinements, not as the tests that were written down first. The network hypothesis, that a coordinate encoding would propagate at least twice as fast as one-hot, failed outright, and the failure is the result. The count-bonus prediction failed, and Section 8 does not rest on it.

Nothing here is a deep agent on a standard benchmark. A network could still fail the count online, where the policy shortcut of Section 6 decides which states receive data and the law applies only on the states that receive it. A MuJoCo critic could fit so slowly that α dominates τ and the Polyak term stops mattering. A PPO run could be limited by advantage normalisation or by early-termination bootstrap in a way this corridor does not contain.

## 12. [3080] The measurements that decide it

One RTX 3080, about a month, in this order. Each experiment measures the value front, not the return, because Section 6 shows a policy can be solved while the values are still thirty syncs away.

1. **DQN on MinAtar**, K ∈ {100, 1000, 8000}. The value front against truncated Monte Carlo returns. The claim survives if hops per sync stay near 1. This is the same code as the next item.
2. **The same DQN**, n ∈ {1, 3, 5, 10, 20} × ε ∈ {0.01, 0.1} × K ∈ {500, 8000}. Time to a return threshold tracks (⌈D/n⌉ − 1)·K. Asymptotic return falls once n exceeds the ceiling of Section 7, at a smaller n for ε = 0.1 than for ε = 0.01, and at the same n for both K. A truncated n-step arm should remove the fall.
3. **CleanRL PPO** on a long-corridor MiniGrid, depth 32–256, λ ∈ {0.9, 0.95, 0.99, 1}, `num_steps` ∈ {16, 64, 256, 1024}. The iteration at which the start-state advantage first agrees with the correct action tracks D((1 − γλ) + 1/T).
4. **TD3** on Hopper and Walker, τ ∈ {0.05, 0.005, 0.0005}, n ∈ {1, 5}. Every few thousand steps, regress the critic on truncated Monte Carlo returns at horizons 5 through 200 and read the horizon off the regression. The slope is n / (1/τ + 1/α_eff), which also identifies the fit rate the theory leaves free.
5. **The accounting** on MiniGrid DoorKey, logging first extrinsic reward and first successful greedy episode, at the same K and n as item 1.

## 13. Reproducibility

Each pilot is a single script with a pre-registered kill test written down before it ran.

| result | folder | run |
|---|---|---|
| Hard targets | `research/independent/R12-bootstrap-speed-limit/` | `code/r12_speed_limit.py`, then `--confirm` |
| Polyak targets | `research/independent/R21-soft-target-horizon/` | `code/r21_soft.py` |
| GAE | `research/independent/R19-gae-speed-limit/` | `code/r19_gae.py` |
| Neural critics | `research/independent/R22-generalisation-hops/` | `code/r22_hops.py`, then `--gap` |
| Choosing n | `research/independent/R24-nstep-rule/` | `code/r24_nstep.py` |
| Exploration share | `research/independent/R20-exploration-is-propagation/` | `code/r20_explore.py` |

The figures cited here are collected, with the file they came from, in `results/ledger.md`.

## References

Arjona-Medina et al. (2019). RUDDER. · Badia et al. (2020). Agent57. · Bellemare et al. (2016). Unifying count-based exploration. · Burda et al. (2019). RND. · Ecoffet et al. (2021). Go-Explore. · Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · Fellows et al. (2023). Why target networks stabilise TD methods. ICML. · François-Lavet et al. (2015). How to discount deep RL. · Fujimoto et al. (2018). TD3. · Haarnoja et al. (2018). SAC. · Hernandez-Garcia and Sutton (2019). Understanding multi-step deep RL. · Hessel et al. (2018). Rainbow. · Kapturowski et al. (2019). R2D2. · Lee and He (2019). Target-based TD learning. ICML. · Lillicrap et al. (2016). DDPG. · Mnih et al. (2015). Human-level control. · Panahi et al. (2024). RLC. · Piché et al. (2022). Beyond target networks. ICLR submission. · Schulman et al. (2016). GAE. · Schulman et al. (2017). PPO. · Schwarzer et al. (2023). BBF. · van Hasselt et al. (2019). DER. · Yuan et al. (2025). VC-PPO. arXiv:2503.01491. · Zhang et al. (2021). · The six pilots named in Section 13.
