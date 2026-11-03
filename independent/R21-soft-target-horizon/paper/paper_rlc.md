# Soft Target Updates Are an Implicit Discount Schedule

*RLC-length version; every number is from this folder's results.*

## Abstract

Polyak averaging \(\bar Q \leftarrow \bar Q + \tau(Q - \bar Q)\) is a horizon parameter. A reward \(k\) steps from a decision passes through \(\lceil k/n \rceil\) critic-fit stages of rate \(\alpha\) and \(\lceil k/n \rceil - 1\) target stages of rate \(\tau\), so the critic's effective horizon grows as \(h(t) \approx n\, t / (1/\tau + 1/\alpha)\). In a deterministic tabular system the measured slope matches that rate in 18 of 18 \((\tau, \alpha, n)\) cells, ratios 0.98 to 1.1, at \(R^2 = 1.000\). A policy that must choose between an immediate reward of 1 and a patient reward worth twice as much at full horizon switches at the closed-form arrival time: log–log slope 1.011, \(R^2 = 0.9997\), and 54 of 54 cells inside \([0.79, 1.03]\) of the prediction. Cells with \(\tau = \alpha\) take the predicted multiple of the \(\alpha = 1\) time in 12 of 12 cases. A hard refresh every \(K = 1/\tau\) updates matches the soft switch within a factor 1.08 in all 15 cells with at least six segments. At \(\tau = 0.01\), \(\alpha = 1\), \(n = 1\), the horizon-matched discount is 0.496 at 100 updates, 0.909 at 1,000, and 0.990 at 10,000.

## 1 Introduction

DDPG, TD3, and SAC update a target critic by Polyak averaging with \(\tau = 0.005\), a default that is tuned and not derived. The same default is a statement about when a reward becomes visible. With \(n\)-step targets, the online critic at a state relaxes toward the target critic \(n\) steps downstream at the fit rate \(\alpha\), and the target critic relaxes toward the online critic at rate \(\tau\). A reward that sits \(k\) steps downstream therefore crosses \(\lceil k/n \rceil\) filters of rate \(\alpha\) and \(\lceil k/n \rceil - 1\) filters of rate \(\tau\): the segment adjacent to the reward is a direct backup and does not wait on the target table. The arrival time is a sum of that many geometric waits. Its mean is \((m-1)/\tau + m/\alpha\) for \(m = \lceil k/n \rceil\), and the horizon of the critic — the truncation depth whose value equals the current estimate — advances at \(n/(1/\tau + 1/\alpha)\) states per update.

Until that horizon reaches the task, the greedy policy is the greedy policy of a shorter problem. Soft target updates are then an implicit discount schedule, running from a short horizon up to the task horizon on a clock set by \(\tau\), \(\alpha\), and \(n\). Two symmetries fall out of the same sum. The fit rate and the target rate trade one for one. A hard refresh every \(K\) updates is the same mean delay with the \(\tau\)-stage replaced by a deterministic wait of \(K = 1/\tau\) steps.

An off-by-one in the stage count was caught on a 3,000-step smoke run before the pilot: counting \(m\) target stages, rather than \(m-1\), over-predicted the \(n = 5\), \(d = 10\) cells (\(m = 2\)) by 2.4×, because only one \(\tau\)-stage lies on that path. The count was corrected by re-deriving the cascade. The horizon-rate prediction is unchanged, because on a chain of 200 states the missing stage is negligible beside \(m\).

**Contributions.** (1) The cascade and the horizon law \(h(t) \approx n\, t / (1/\tau + 1/\alpha)\), including the symmetry of \(\tau\) and \(\alpha\). (2) An exact pilot: 18/18 horizon slopes, 54/54 switch times, 12/12 on the symmetry, 15/15 on hard versus soft at \(K = 1/\tau\). (3) The discount reading of the same traces: at \(\tau = 0.01\) the agent is a \(\gamma_{\mathrm{eff}} \approx 0.9\) agent for its first thousand updates.

## 2 Related work

Lillicrap et al. (2016) introduce \(\tau\) without an analysis of propagation speed. Fujimoto et al. (2018) and Haarnoja et al. (2018) inherit the default. Target networks are analysed as stabilisers by Kim et al. (2019), Zhang et al. (2021), and Lee and He (2019). Piché et al. (2022) state that a hard sync needs on the order of \(N \cdot H\) steps to move a reward along an \(N\)-state chain, and that Polyak averaging also slows propagation. They do not give the \(\tau\) law, the fit-rate term, or the horizon reading. Fedus et al. (2020) find that \(n\)-step backups are what make a large replay buffer work; \(n\) divides the stage count here. Discount schedules have been chosen on purpose: OpenAI Five annealed \(\gamma\), and François-Lavet et al. (2015) study an increasing discount. The schedule below is already present at fixed \(\gamma\), set by \(\tau\). Hard-sync latency and GAE latency are the same kind of iteration count for the other two target mechanisms; this experiment is the Polyak case, run on its own system.

## 3 Setting

Every state is updated on every step. There is no replay and no exploration. The system is deterministic, so there are no seeds. The update is \(Q \leftarrow Q + \alpha(y - Q)\), with \(y\) the \(n\)-step target bootstrapped from the Polyak table \(\bar Q \leftarrow \bar Q + \tau(Q - \bar Q)\), at \(\gamma = 0.99\). The pilot took 17 CPU seconds.

**Chain.** \(D = 200\) states in a line, reward 1 per step, termination after the last state. The true start value is \(V^\ast(s_0) = (1 - \gamma^D)/(1 - \gamma)\). The effective horizon at update \(t\) is the truncation depth with the same value,

\(h(t) = \log\bigl(1 - (1-\gamma) V_t(s_0)\bigr) / \log \gamma\).

**Patience fork.** At the start state, `now` pays 1 and terminates. `wait` enters a corridor of \(d\) steps that pays \(R\) at the end and terminates, with \(R \gamma^d = 2\), so the patient action is optimal by a factor of two once the horizon covers the corridor. The switch time \(t^\ast\) is the first update with \(Q(s_0, \mathrm{wait}) > Q(s_0, \mathrm{now})\). Because the patient return is twice the impatient one, the switch sits at the median of the arrival distribution rather than at its mean. The Gamma-median approximation used as the prediction is

\(t^\ast_{\mathrm{pred}} = (m - 1)/\tau + m/\alpha - \max(1/\tau, 1/\alpha)/3\),

with \(m = \lceil d/n \rceil\).

The grid is \(\tau \in \{0.1, 0.01, 0.001\}\), \(\alpha \in \{1, 0.1, 0.01\}\), \(n \in \{1, 5\}\), and fork depths \(d \in \{10, 30, 100\}\), plus hard-sync controls at \(K = 1/\tau\) with \(\alpha = 1\).

- **K1.** For each of the 18 \((\tau, \alpha, n)\) cells, the OLS slope of \(h(t)\) on \(t\) over the range \(h \in [5, 100]\) lies in \([0.8, 1.2]\) times \(n/(1/\tau + 1/\alpha)\), and the linear \(R^2\) is at least 0.98, in at least 16 of the 18 cells.
- **K2.** Over the 54 fork cells, \(\log t^\ast\) on \(\log t^\ast_{\mathrm{pred}}\) has slope in \([0.9, 1.1]\) and \(R^2 \ge 0.98\), and the ratio \(t^\ast / t^\ast_{\mathrm{pred}}\) lies in \([0.7, 1.4]\) in at least 48 of 54 cells.
- **K3.** For the cells with \(\tau = \alpha\) (the pairs 0.1 and 0.01), the ratio of that switch time to the switch time at the same \(\tau\) with \(\alpha = 1\) lies within \([0.8, 1.25]\) of \((2m - 1 - 1/3)/((m - 1 - 1/3) + m\tau)\), at every depth and both backup lengths: 12 cells.
- **K4.** At \(\alpha = 1\), for every cell with \(m \ge 6\), the hard-sync switch time is within \([0.7, 1.3]\) of the soft \(t^\ast\).

Supported if K1, K2, and K3. K4 is the bridge from the soft cascade to a hard period and does not enter the verdict.

## 4 Results

The recorded verdict is **supported**. All four tests pass.

### 4.1 The horizon grows at the predicted rate

In all 18 cells the ratio of measured slope to \(n/(1/\tau + 1/\alpha)\) lies between 0.98 and 1.1, and the linear fit has \(R^2 = 1.000\). The recorded ratios are 1.1, 1.08, 1.05, 1.04, 1.0, 0.99, 1.0, 0.99, 1.0, 0.99, 1.0, 0.99, 1.0, 0.98, 1.0, 0.98, 1.0, 0.98. K1 asked for 16 of 18. It got 18 of 18.

The same traces, read as a discount, are an implicit schedule. Define \(\gamma_{\mathrm{eff}}(t) = 1 - 1/h(t)\), the discount whose mean horizon \(1/(1-\gamma)\) equals the current \(h(t)\). For \(n = 1\), \(\alpha = 1\):

| \(t\) | \(\tau = 0.1\): \(h\), \(\gamma_{\mathrm{eff}}\) | \(\tau = 0.01\) | \(\tau = 0.001\) |
|---:|---|---|---|
| 100 | 11, 0.908 | 2, 0.496 | 1, 0.090 |
| 1,000 | 100, 0.990 | 11, 0.909 | 2, 0.498 |
| 3,000 | 200, 0.995 | 31, 0.968 | 4, 0.749 |
| 10,000 | 200, 0.995 | 100, 0.990 | 11, 0.909 |
| 30,000 | 200, 0.995 | 200, 0.995 | 31, 0.968 |
| 100,000 | 200, 0.995 | 200, 0.995 | 100, 0.990 |

The chain's own horizon is 200, and \(h(t)\) stops there. At \(\tau = 0.01\) the agent is a \(\gamma_{\mathrm{eff}} = 0.909\) agent at update 1,000 and a \(\gamma_{\mathrm{eff}} = 0.990\) agent at update 10,000. The task discount \(\gamma = 0.99\) is reached on this schedule. At \(\tau = 0.001\) the same \(\gamma_{\mathrm{eff}} = 0.990\) is the value at update 100,000, where \(h = 100\).

### 4.2 The policy switches when the horizon arrives

Over the 54 fork cells, \(\log t^\ast\) on \(\log t^\ast_{\mathrm{pred}}\) has slope 1.011 and \(R^2 = 0.9997\). All 54 ratios \(t^\ast / t^\ast_{\mathrm{pred}}\) lie in \([0.7, 1.4]\); the realised range is 0.79 to 1.03. K2 asked for 48 of 54 inside that interval.

At \(n = 1\), \(\alpha = 1\), depth 100, the switch is at 987, 9,855, and 98,544 updates for \(\tau = 0.1\), 0.01, and 0.001. The predictions are 1,087, 9,967, and 98,767.

### 4.3 \(\tau\) and \(\alpha\) are the same delay

K3 compares cells whose two rates are equal with the same \(\tau\) at \(\alpha = 1\), where the target stages dominate and the fit is a single step. The predicted ratio of those times is \((2m - 1 - 1/3)/((m - 1 - 1/3) + m\tau)\), which for large \(m\) and small \(\tau\) is the \((2m-1)/(m-1)\) of two geometric stages against one. All 12 cells fall inside the pre-registered factor \([0.8, 1.25]\) of that prediction.

The depth-100, \(n = 1\) pair at rate 0.01 is the plain version of the same fact: 19,750 updates when \(\tau = \alpha = 0.01\), against 9,855 when \(\tau = 0.01\) and \(\alpha = 1\). Halving the fit rate doubles the latency, as halving \(\tau\) does. The default \(\tau = 0.005\) is the binding delay only when the critic fits a segment faster than the target averages it. Otherwise \(\alpha\) sets the horizon, and changing \(\tau\) does not.

### 4.4 A hard period \(K = 1/\tau\) is the same mean speed

At \(\alpha = 1\), for every cell with \(m \ge 6\), the hard-sync switch lies within a factor 1.08 of the soft switch. The 15 recorded ratios are 1.03, 1.01, 1.0, 1.06, 1.02, 1.04, 1.01, 1.0, 1.08, 1.02, 1.04, 1.01, 1.0, 1.08, 1.02. K4 asked for \([0.7, 1.3]\).

At depth 100, \(n = 1\), a hard refresh every \(K = 100\) updates switches at update 9,901; the soft table at \(\tau = 0.01\) switches at 9,855. The shapes differ and the means match. Sampled every 100 updates from \(t = 100\) to \(t = 1{,}000\) in that cell, the hard horizon is the staircase 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, one hop per refresh. The soft horizon is 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.9, 10.9, a line through the same mean speed.

## 5 Discussion

The default \(\tau = 0.005\) is a per-segment delay of 200 updates, plus \(1/\alpha_{\mathrm{eff}}\) for however fast the critic actually fits. With an ideal critic (\(\alpha = 1\)) a decision 100 steps from its consequence sees it after about 20,000 updates at \(n = 1\), and after about 4,000 updates at \(n = 5\). That is small beside a million-step dense-reward run, which is why the usual benchmarks do not expose it, and it is the whole of a short or sparse run. Early in training a larger \(\tau\) costs nothing, in horizon, that a slow fit would not also cost through \(\alpha\). The horizon grows per update, so a higher update-to-data ratio moves \(h\) further per environment step, which is a latency account of early sample efficiency.

Choosing \(\tau\) is choosing when the agent stops being greedy for a shorter horizon. The chain at \(\tau = 0.01\) spends its first thousand updates near \(\gamma_{\mathrm{eff}} = 0.9\) and reaches the task discount only around update 10,000. A schedule that starts \(\tau\) large and shrinks it is the law's own recommendation; this pilot does not run one.

## 6 Limitations

The pilot is an exact linear system. It confirms a derivation. The content that a neural critic still has to earn is the size of \(\alpha_{\mathrm{eff}}\): if a replay-trained network fits a segment more slowly than \(1/\tau\), the target rate is not the binding delay, and the symmetry in K3 is the reason. Updates here are synchronous and on-policy in the strong sense that every state is fit every step. Replay adds a second wait, the time until the frontier transition is sampled, which this grid does not contain. \(\gamma_{\mathrm{eff}}\) matches horizons; it is not a formal equivalence of the greedy policies. The stage count \(m-1\) was corrected before the pilot and is part of the pre-registration the results are scored against. TD3 and SAC on Hopper, Walker, and HalfCheetah, reading \(h(t)\) by regressing the critic on truncated Monte Carlo returns, have not been run.

## 7 Conclusion

A Polyak target is a cascade of geometric filters, one target stage and one fit stage per \(n\)-step segment between a decision and its consequence, with the segment next to the reward exempt from the target stage. The critic's horizon therefore grows linearly in the update count, at \(n/(1/\tau + 1/\alpha)\) states per update, in 18 of 18 cells of this system. The greedy policy switches from an impatient reward to a patient one at the median of that cascade, in 54 of 54 cells. Fit rate and target rate are interchangeable, and a hard period \(K = 1/\tau\) is the same mean with a staircase in place of a line. At the usual order of \(\tau\), the agent trains for a long prefix of learning as a shorter-horizon agent than the discount written in the backup.

## References

Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · François-Lavet et al. (2015). How to discount deep reinforcement networks: towards a new dynamic discounting deep RL agent. · Fujimoto, van Hoof, Meger (2018). Addressing function approximation error in actor-critic methods. ICML. · Haarnoja et al. (2018). Soft actor-critic. ICML. · Kim et al. (2019). DeepMellow: removing the need for a target network in deep Q-learning. · Lee and He (2019). Target-based temporal-difference learning. ICML. · Lillicrap et al. (2016). Continuous control with deep reinforcement learning. ICLR. · OpenAI (2019). Dota 2 with large scale deep reinforcement learning. · Piché et al. (2022). Beyond target networks: functional regularisation. · Zhang, Yao, Whiteson (2021). Breaking the deadly triad with a target network. ICML.
