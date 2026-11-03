# Credit Assignment Is a Count of Iterations

*RLC-length version. Tabular and fixed-dataset numbers are from the finished pilots named in the text and listed in `results/ledger.md`. The online-network numbers are from `results/gpu/runs.csv`, measured on an RTX 3080. No figure in this file was typed in by hand.*

## Abstract

A reward \(D\) steps from a decision cannot move that decision until a value estimate has carried it there, and in a standard deep RL agent the carry is metered by the target. A hard target refreshed every \(K\) updates takes \((\lceil D/n \rceil - 1)\,K\) updates. A Polyak target of rate \(\tau\) takes \((\lceil D/n \rceil - 1)/\tau\) updates plus a fit delay. GAE delays the same reward by about \(D(1 - \gamma\lambda)\) critic fits, plus a second term when the rollout is shorter than the task. On tables the leading coefficients are 1, not fitted: the hard-target ratio on 60 target-limited cells has median 1.002 (SGD) and 1.004 (Adam); the Polyak horizon matches \(n/(1/\tau + 1/\alpha)\) in 18/18 cells (\(R^2 = 1.000\)); at \(\lambda = 1\) the GAE value arrives in one iteration at every depth from 8 to 128. A 2×64 network on a fixed dataset carries 1.02–1.03 hops per sync under four encodings (\(R^2 \ge 0.99998\)). The same count is what remains after the reward has been found: under Nature-DQN settings it is 86–98% of time-to-solve on corridors of depth 10–30. On an RTX 3080, an online DQN with the same 2×64 network, collecting its own data, still takes 1.03–1.04 hops per sync (one-hot, \(R^2 = 1.000\)) whether episodes start at the decision or at a random state, and a Polyak target has slope 37.9–39.0 against a predicted 39. A coordinate encoding with episodes that always start at the decision is the failure mode the fixed-dataset study predicted: one of four seeds at \(K = 800\) never moved the front in 80,000 updates. A neural PPO on a length-16 corridor does not resolve the GAE formula. The median arrival is 12–16 iterations at every \(\lambda\) that was tried, and individual seeds take up to 22.

## 1 Introduction

Deep RL learns by bootstrapping. The target for a state is built from the estimate at a later state, so a reward moves backward one hop at a time. Three mechanisms decide how long a hop takes, and each is set as a stability knob.

A DQN target network is a frozen copy, refreshed every \(K\) updates (Mnih et al., 2015). Between refreshes no new information enters the targets, so the front advances at most one hop per refresh. TD3 and SAC average, \(\bar Q \leftarrow \bar Q + \tau(Q - \bar Q)\), with \(\tau = 0.005\) (Lillicrap et al., 2016; Fujimoto et al., 2018; Haarnoja et al., 2018). PPO refits a critic each iteration to \(\lambda\)-returns built from the previous critic (Schulman et al., 2016, 2017).

Piché et al. (2022) note that on an \(N\)-state chain a hard target refreshed every \(H\) steps needs \(NH\) training steps to propagate a terminal reward, and that Polyak averaging also slows propagation. They do not write the \(n\)-step or replay terms, do not treat \(\tau\), \(\alpha\) or \(\lambda\), and do not measure any of it. VC-PPO (Yuan et al., 2025) observes that in long-horizon RLHF a reward barely reaches early tokens at \(\lambda = 0.95\), calls this decay, and sets the critic's \(\lambda\) to 1. With a critic that is refit, the signal is delayed, not lost, and the delay has a closed form.

**Contributions.** (1) One count for hard sync, Polyak averaging and GAE, with the leading coefficient measured rather than fit. (2) The count survives a neural critic on a fixed dataset, at 1.02–1.03 hops per sync for every encoding, and it survives online data collection on a 3080 for one-hot features and for Polyak targets. (3) \(n\) is the knob that divides all three, and uncorrected \(n\)-step returns impose a ceiling that does not depend on \(K\). (4) Under ordinary DQN settings the post-discovery half of sparse-reward time is this count.

## 2 The count

A reward lies \(D\) transitions from a decision. The learner bootstraps \(n\) steps at a time, so the reward arrives in \(h = \lceil D/n \rceil\) hops. The last hop is a sampled return.

**Hard target.** The online value and a copy refreshed every \(K\) updates. By induction the decision is zero until update \((h - 1)\,K\). When the frontier transition must also be sampled from a buffer, a second term adds:

\[
t = (h - 1)\,K + c\,h\,L/B.
\]

The constant \(c\) depends on the optimiser. It does not depend on \(D\), \(n\), \(K\) or \(B\).

**Polyak target.** The online value relaxes toward its target at rate \(\alpha\) and the target relaxes toward the online value at rate \(\tau\):

\[
t = (h - 1)/\tau + h/\alpha.
\]

\(\tau\) and \(\alpha\) are interchangeable. A hard target with \(K = 1/\tau\) traces the same schedule as a staircase.

**GAE.** The front moves about \(1/(1 - \gamma\lambda)\) states per critic fit, and a rollout of \(T\) steps caps the move again. The two caps combine as rates, \(t = D\,((1 - \gamma\lambda) + 1/T_{\mathrm{eff}})\) critic fits. The policy gradient uses the same mixture, so the policy moves in the iteration the value arrives.

| mechanism | who uses it | updates until a reward \(D\) steps away arrives |
|---|---|---|
| hard refresh every \(K\) | DQN, Rainbow | \((\lceil D/n \rceil - 1)\,K\) |
| Polyak rate \(\tau\), fit rate \(\alpha\) | DDPG, TD3, SAC | \((\lceil D/n \rceil - 1)/\tau + \lceil D/n \rceil/\alpha\) |
| GAE, rollout \(T\) | PPO | \(D\,((1 - \gamma\lambda) + 1/T)\) iterations |

*Table 1. The replay term of the hard law and the \(O(1)\) fitting delay are omitted; both are measured below.*

## 3 Hard targets, on a table

The pilot is a deterministic chain, tabular \(V\), stationary uniform replay, \(n\)-step targets, SGD and Adam (R12). Propagation time is the first update at which the start state reaches half its true value.

The pre-registered form said the two terms compete, \(t = h \cdot \max(K, c\,L/B)\). That form failed (SGD \(R^2 = 0.883\)). The data say the terms add. On the 60 target-limited cells the median of \(t/((h-1)\,K)\) is 1.002 for SGD and 1.004 for Adam: for \(L = 64\), \(n = 1\), \(K = 1000\) the measured times are 63,003 and 63,005 against a prediction of 63,000. With \(K = 1\) the time scales as \(L^2/(nB)\) (slope 0.99). The additive form was then re-tested on a disjoint grid, with \(c\) frozen at 2.63 (SGD) and 4.21 (Adam). Identity-fit \(R^2\) in log space is 0.997 and 0.971 over 96 cells each.

Nature DQN refreshes every 2,500 updates with \(n = 1\). A reward 100 steps from the start cannot move the start state's value before \((100-1)\cdot 2500 \approx 250{,}000\) updates. Rainbow's \(n = 3\) cuts that wait by about three. These are lower bounds on a chain. They are the right order for the gaps between these agents on sparse-reward games.

## 4 Polyak targets, on a table

The pilot updates every state every step and reads the horizon by inverting \(V(s_0) = (1 - \gamma^h)/(1 - \gamma)\) on a chain that pays 1 per step (R21). A second task is a patience fork.

The horizon slope matches \(n/(1/\tau + 1/\alpha)\) in all 18 cells of \(\tau \in \{0.1, 0.01, 0.001\}\), \(\alpha \in \{1, 0.1, 0.01\}\), \(n \in \{1, 5\}\), with \(R^2 = 1.000\) and ratios between 0.98 and 1.10. The patience switch matches the closed form in all 54 cells, within a factor \([0.79, 1.03]\), log–log slope 1.011, \(R^2 = 0.9997\). At \(\tau = \alpha = 0.01\) the wait is 19,750 updates against 9,855 with \(\alpha = 1\). A hard target with \(K = 100\) matches \(\tau = 0.01\) (9,901 against 9,855).

At \(\tau = 0.01\), \(\alpha = 1\) and \(n = 1\) the horizon is 2, 11, 31 and 100 after 100, 1,000, 3,000 and 10,000 updates. For the first thousand updates this agent is a \(\gamma \approx 0.9\) agent. The default \(\tau = 0.005\) is 200 updates per hop before the fit delay.

## 5 GAE, with an exact critic

Each iteration collects 32 episodes, sets the critic to the per-state mean of its own \(\lambda\)-returns, and takes one tabular policy step (R19). Two corridors of depth \(D\) leave a choice state; one pays 1.

Over the 30 cells with \(\lambda < 1\) and no rollout cap, \(t_V = 1.25\,D(1-\lambda) + 1.9\) with \(R^2 = 0.999\). The pre-registered regression of \(\log t\) on \(\log D\) and \(\log(1-\lambda)\) has slopes 0.95 and 0.88. At \(\lambda = 1\) the value arrives in one iteration at every depth from 8 to 128. Across 70 cells the policy's arrival correlates with the value's at Spearman 0.99, and the policy arrives within \([-2, +1]\) iterations of the value. Freezing the critic at zero removes the guarantee: at \(\lambda = 0.95\) the frozen learner solves \(D = 32\), takes 218 iterations at \(D = 64\), and fails at \(D = 128\) with final success 0.16.

A rollout of 16 steps reintroduces the delay at every \(\lambda\), including \(\lambda = 1\) (25 iterations at \(D = 128\)). The pre-registered minimum of the two caps fit worse than the rate sum \(1/h = (1-\lambda) + 1/T_{\mathrm{eff}}\), which fits the 35 short-rollout cells at \(R^2 = 0.998\) with \(T_{\mathrm{eff}} = 6.1\).

## 6 A network, on a fixed dataset

One 2×64 MLP, Adam, on one fixed dataset covering a corridor of length 40. Only the encoding changes: one-hot, random codes, thermometer, and a scalar coordinate (R22). The value front is the contiguous run of states, from the goal, whose predicted value has reached half the truth. Hops per sync are 40 divided by the slope of arrival time on \(K\).

| encoding | \(K = 50\) | \(K = 200\) | \(K = 800\) | hops per sync |
|---|---:|---:|---:|---:|
| one-hot | 2,100 | 7,850 | 31,250 | 1.03 |
| random | 1,850 | 7,850 | 31,250 | 1.02 |
| thermometer | 2,050 | 7,850 | 31,250 | 1.03 |
| coordinate | 2,050 | 7,850 | 31,250 | 1.03 |

*Table 2. Median updates until the value front reaches the start. Eight seeds. \(R^2\) of each fit is at least 0.99998.*

Forty hops at \(K = 800\) take 31,250 updates for every encoding: 39 syncs, plus about a hundred updates to fit the front. The hypothesis that a coordinate encoding would carry at least twice as many hops is rejected. Beyond the front the target is zero, so a network trained where data exists is pushed toward zero there. It can interpolate a target. It cannot invent one the sync has not written.

Where the target does not pin, representation matters. Across a block of ten states with no data, the coordinate network crosses at the full-coverage time with values within 1% of the truth; the one-hot network skips the gap with values 24% high. With coordinate inputs the greedy policy is correct at update 250 and the values arrive at 7,850. Return curves are the wrong instrument. The value front is the right one.

## 7 Online, on a 3080

Section 6 trains on a dataset that already covers the chain. The objection that remains is that an agent collects its own data, and the shortcut of Section 6 can decide which states are visited. The measurement is the same corridor and the same front, with a 2×64 network trained by Adam on an RTX 3080 (PyTorch 1.8, CUDA). Length \(L = 40\), \(\gamma = 0.99\), \(V^\star(s) = \gamma^{L-1-s}\), \(\varepsilon = 0.2\), batch 64, replay capacity 50,000, learning starts after 1,000 transitions. The front is read every 100 updates. Four seeds. Code: `code/gpu_3080.py`. Rows: `results/gpu/runs.csv`.

**Hard target.** Median arrival, one-hot features:

| reset | \(K = 50\) | \(K = 200\) | \(K = 800\) | slope | hops per sync | \(R^2\) |
|---|---:|---:|---:|---:|---:|---:|
| random state | 3,100 | 8,900 | 32,100 | 38.67 | 1.034 | 1.000 |
| always the start | 3,150 | 8,900 | 32,100 | 38.62 | 1.036 | 1.000 |

*Table 3. Online DQN. The law's leading term is \((L-1)\,K = 39K\), which is 1,950, 7,800 and 31,200. The fitted intercept is about 1,200 updates, the warmup before any gradient step. All 24 one-hot runs solved.*

Coordinate features with random resets give slope 38.24, 1.046 hops per sync, \(R^2 = 0.985\), and one seed arrives far early (1,200 updates at \(K = 50\), 3,000 at \(K = 200\)), which is the shortcut of Section 6 showing up in a single run while the median still waits. Coordinate features with every episode starting at the decision are the miss: 11 of 12 runs solve, and one seed at \(K = 800\) has front 0 after 80,000 updates. The law describes the states the buffer actually covers. It does not create coverage.

**Polyak.** The target is a geometric average of the online network, updated every step, episodes reset to a random state. Arrival against \(1/\tau\):

| encoding | \(\tau = 0.05\) | \(0.01\) | \(0.002\) | slope on \(1/\tau\) | \(R^2\) |
|---|---:|---:|---:|---:|---:|
| one-hot | 1,850 | 4,900 | 20,050 | 37.90 | 1.000 |
| coordinate | 1,950 | 5,250 | 20,750 | 39.03 | 1.000 |

*Table 4. Median updates, four seeds. The predicted slope is \(L - 1 = 39\). The intercept, about 1,100–1,250, is the fit delay \(h/\alpha\).*

**Neural PPO.** Actor-critic, 64-unit tanh body, GAE, four epochs per iteration, corridor length 16, 24 episodes, \(\lambda \in \{1, 0.8, 0.95\}\) and segment lengths 16 and 64. All 24 runs put the front at the start. Median iterations: 12 at \(\lambda \in \{1, 0.95\}\) with segments of 64; 14 at those \(\lambda\) with segments of 16; 16 at \(\lambda = 0.8\) for both segment lengths. The ordering agrees with the law and the magnitude does not. \(D(1-\gamma\lambda)\) at this depth is a fraction of an iteration up to about four, and the critic's own fit, not the \(\lambda\)-horizon, is what the 12–16 iterations are. The GAE count stands on the exact critic in Section 5. This corridor is too short, and too quickly solved (every episode in the logged iteration returned the reward), to test it in a neural PPO. MiniGrid at depth 32–256 was not run.

## 8 How large \(n\) can be

Uncorrected \(n\)-step returns, which is what DQN-style agents use, set the ceiling (R24). On a path where one wrong action loses the reward, after \(h = \lceil (D+1)/n \rceil\) segments built from an \(\varepsilon\)-greedy behaviour the value of the path is \(\gamma^D (1-\varepsilon/2)^{(D+1)-h}\). The greedy policy prefers the path only while this exceeds the alternative. The largest such \(n\) depends on \(\varepsilon\), the length and the gap. It does not depend on \(K\).

The pilot is 1,344 tabular learners, path length 20, \(\varepsilon \in \{0.01, 0.03, 0.05, 0.1\}\), gaps 0.1–0.3, \(n\) up to 20, \(K \in \{200, 2000\}\). Time to the correct greedy policy follows \((\lceil 21/n \rceil - 1)\,K + 1\) with Spearman correlation 1.00 over the 66 cells that solve, median ratio 1.05, and no fitted constant. Learned values match the bias formula with correlation 0.985. The largest \(n\) that solves a cell is within one grid step of the predicted ceiling in 9 of 12 cells; the three misses have predicted margins 0.026, 0.005 and 0.001 inside the seed noise (standard deviation 0.06). The ceiling is the same at \(K = 200\) and at \(K = 2000\) in all 12 cells. At \(\varepsilon = 0.01\), \(K = 2000\) and a gap of at least 0.2, the correct policy arrives at update 40,200 with \(n = 1\) and at update 2,200 with \(n = 12\), in 8 of 8 seeds either way.

At \(\varepsilon = 0.1\), any \(n > 1\) is unsafe on a 20-step path with a gap of 0.1 to 0.3, so Nature DQN's \(n = 1\) was the right choice. At \(\varepsilon = 0.01\) the ceiling is above 40, so Rainbow's \(n = 3\) is safe and leaves most of the speed-up unused.

## 9 Where the time goes

On a corridor, with \(\varepsilon\)-greedy tabular Q-learning, a hard target and one-step returns, time-to-solve splits into discovery and propagation (R20). At \(K = 2000\) and \(n = 1\) the two halves are 346 against 18,154 steps at depth 10, 1,862 against 37,716 at depth 20, and 9,324 against 57,648 at depth 30. Propagation is 98%, 95% and 86% of the total. A two-term fit of that half on the hard-sync law has coefficient 0.96 and \(R^2 = 0.996\). Replacing \((K = 2000, n = 1)\) by \((K = 1, n = 5)\), with the same exploration, cuts time-to-solve by 31×, 18× and 7×.

A stored count bonus shortened discovery in 5 of 18 cells, lengthened propagation in 16 of 18, and made the total worse in 16 of 18. The pre-registered claim that the bonus would halve discovery failed. The accounting is the result. The bonus is not.

## 10 What to set

Keep \((\lceil D/n \rceil - 1)\,K\) inside the update budget. If a reward a hundred steps away matters, \(K = 2500\) and \(n = 1\) spends the first million environment steps before the start state moves. Raise \(n\) up to the ceiling in Section 8. Shorten \(K\) before adding an exploration bonus. For TD3 and SAC, \(\tau\) and the critic's fit rate are the same knob; \(\tau = 0.005\) is 200 updates per hop, and lowering it further buys nothing while the critic is still fitting. For PPO, count critic fits. \(\lambda = 1\) removes \(D(1-\gamma\lambda)\) only when the rollout covers the distance. Section 7 is why that sentence is not yet a neural measurement.

## 11 Related work

Target networks were introduced for stability (Mnih et al., 2015) and analysed as a damped fixed point (Lee and He, 2019; Zhang et al., 2021; Fellows et al., 2023). Piché et al. (2022) state the hard-sync leading term in one sentence. The \(n\)-step term, the replay term, the \(\tau\)–\(\alpha\) sum and the measurements are ours. Schulman et al. (2016) present GAE as a bias–variance trade-off within one update. Yuan et al. (2025) identify the failure of \(\lambda = 0.95\) on long sequences and frame it as decay. Fedus et al. (2020) show that \(n\)-step returns are what make large buffers usable. Hernandez-Garcia and Sutton (2019) document that uncorrected \(n\)-step DQN needs careful tuning. Panahi et al. (2024) vary the target period to control over-generalisation; Section 6 measures hops per sync and finds them invariant to the representation. Bellemare et al. (2016), Burda et al. (2019), Ecoffet et al. (2021) and Badia et al. (2020) evaluate exploration by first-reward time. Go-Explore's robustification phase and RUDDER (Arjona-Medina et al., 2019) acknowledge that post-discovery work is real; neither identifies it with the target period.

## 12 Limitations

Two pre-registered forms failed and were replaced. The hard-sync law was registered as a maximum and the data forced a sum, which then held on a disjoint grid with the constant frozen. The GAE rollout law was registered as a minimum and the data forced a sum of rates. The network hypothesis, that a coordinate encoding would propagate at least twice as fast, failed outright. The count-bonus prediction failed. The constants \(c\), the GAE factor 1.25 and \(T_{\mathrm{eff}} = 6.1\) are properties of the optimiser, predicted in form and not in magnitude. The leading coefficients are not fitted.

The 3080 measurement is one corridor, one network width, \(\varepsilon = 0.2\), and four seeds. It is not MinAtar, not MuJoCo, and not a long-horizon PPO. The neural PPO grid does not test \(D(1-\gamma\lambda)\). One coordinate run never covered the chain. Those are the boundaries of the online claim: the count survives online learning when the buffer covers the path, and a target sync still cannot invent a target for a state the data have not reached.

## 13 Conclusion

Hard sync, Polyak averaging and GAE are three ways of writing how many training iterations a reward needs to reach a decision. The leading term is a count, attained on tables, attained by a neural critic on a covered dataset, and attained by an online DQN and an online Polyak critic on a 3080. The knob that divides it is \(n\), up to a ceiling set by the behaviour policy and not by the target. Under the settings value-based agents actually run, most of the time charged to hard exploration is this count, and it is spent after the reward has already been found.

## References

Arjona-Medina et al. (2019). RUDDER. · Badia et al. (2020). Agent57. · Bellemare et al. (2016). Unifying count-based exploration. · Burda et al. (2019). RND. · Ecoffet et al. (2021). Go-Explore. · Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · Fellows et al. (2023). Why target networks stabilise TD methods. ICML. · Fujimoto et al. (2018). TD3. · Haarnoja et al. (2018). SAC. · Hernandez-Garcia and Sutton (2019). Understanding multi-step deep RL. · Lee and He (2019). Target-based TD learning. ICML. · Lillicrap et al. (2016). DDPG. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Panahi et al. (2024). RLC. · Piché et al. (2022). Beyond target networks. · Schulman et al. (2016). GAE. · Schulman et al. (2017). PPO. · van Hasselt et al. (2019). DER. · Yuan et al. (2025). VC-PPO. arXiv:2503.01491. · Zhang et al. (2021).

*The component pilots, each with its own RLC-length note, are `R12-bootstrap-speed-limit`, `R19-gae-speed-limit`, `R20-exploration-is-propagation`, `R21-soft-target-horizon`, `R22-generalisation-hops` and `R24-nstep-rule`.*
