# One Hop per Sync, Whatever the Encoding

*RLC-length version; every number is from this folder's results.*

## Abstract

A hard target network is supposed to move a value front one \(n\)-step hop per refresh in a table, and a neural critic is supposed to do better whenever the encoding says that neighbouring states are alike. On one corridor, one fixed dataset, one network, and one optimiser, four encodings — one-hot, random \(\pm 1\) codes, thermometer, and a scalar coordinate — give hops-per-sync multipliers 1.03, 1.02, 1.03, and 1.03. The regression of propagation time on the sync period has slopes 38.9, 39.1, 39.0, and 39.0 updates per unit period, and \(R^2 = 1.000\) on the one-hot line. The pre-registered test is **inconclusive**: the localist law survives function approximation (K1), and the claim that locality multiplies the speed fails outright (K2). With a transition at every state, a sync carries one hop because every state's target is pinned by its own data and the targets beyond the front are zero. A pre-registered coverage-gap follow-up does not come out as specified. The one-hot network was predicted to stall at a block of states with no data, and it does not: shared layers assign those states values 1.24 times \(V^\ast\) and the front skips the gap. The coordinate network crosses the same gap at the full-coverage time, 7,850 updates, with gap values 1.01 times \(V^\ast\).

## 1 Introduction

Closed-form latencies for value propagation under a target network are derived for tables: one hop per hard sync. The objection is that a network generalises. If raising the value at one state raises its neighbours, a single refresh might move the front several hops, and the latency would be an upper bound whose slack is a property of the encoding. That slack would be the propagation meaning of a good representation.

The measurement has to be made on values. In a corridor where the same action is correct everywhere, a network can be greedy-correct from the start through a shared output bias as soon as the goal transition is fit, with no value propagated at all. A smoke run of two seeds and 6,000 updates, before the pilot, showed coordinate runs "solved" at updates 200–250 at every target period, and one-hot runs solved before their values could have arrived. The front was therefore redefined on the values, and the multiplier was read from the slope of propagation time on \(K\) so that a fitting delay, which does not scale with \(K\), stays in the intercept. The thresholds were not changed.

**Contributions.** (1) Hops per sync under four encodings, with full data coverage: 1.02–1.03 for all four, \(R^2 = 1.000\). The pre-registered multiplier hypothesis is rejected, and that rejection is the result. (2) The coverage-gap follow-up: every encoding crosses a 10-state hole, the coordinate encoding at the full-coverage speed with values within 1% of \(V^\ast\), the one-hot encoding by skipping it with values 24% high. The pre-registered prediction that one-hot inputs would stall is false. (3) The policy shortcut that forced the redefinition: with coordinate inputs the greedy policy is correct at update 250 and the values arrive at update 7,850.

## 2 Related work

Piché et al. (2022) note that target networks slow propagation and propose functional regularisation so that targets stay current. Panahi et al. (RLC 2024) study prioritised replay with neural networks and vary the target period in order to control over-generalisation in prediction tasks. Generalisation together with bootstrapping is studied as a source of divergence (Achiam et al., 2019), of rank collapse (Kumar et al., 2021), and of capacity limits (Lyle et al., 2022); none of that work measures how many states one sync carries. Fedus et al. (2020) connect \(n\)-step returns to replay. A policy that improves while its value function predicts returns poorly is reported for PPO by Ilyas et al. (2020); the policy-front measurement below is the same split on a corridor, where the shortcut has a simple cause. The latency laws under test are the tabular hard-sync count and its Polyak and GAE relatives. This experiment asks only whether the hard-sync count changes with the encoding.

## 3 Setting

The corridor has \(L = 40\) states. Right moves toward the goal. Entering position 40 pays 1 and terminates. The discount is \(\gamma = 0.99\). The true value is \(V^\ast(s) = \gamma^{L-1-s}\).

The dataset is fixed: 20,000 one-step transitions from uniformly random state-action pairs, so every transition, including the goal transition, is present at equal density. The learner is a two-layer MLP with 64 ReLU units, Adam at \(10^{-3}\), batch 64, Huber loss, gradient-norm clip 10, one-step targets, and a hard target network synced every \(K\) updates. Training is offline, up to 60,000 updates. The encodings are one-hot (40-dimensional), a random \(\pm 1\) code per state (16-dimensional, fixed for the seed), a thermometer code (40-dimensional, bit \(k\) on when \(s \ge k\)), and the scalar \(s/L\). The grid is \(K \in \{50, 200, 800\}\) and eight seeds. A seed sets the network initialisation, the order of the dataset, and the random code. The pilot took about 9 CPU minutes.

Every 50 updates, one forward pass evaluates \(Q\) at all 40 states. The value front at time \(t\) is the largest \(d\) such that \(\hat V(s) = \max_a Q(s, a) \ge \tfrac12 V^\ast(s)\) for every \(s \in \{L-d, \ldots, L-1\}\). Propagation time \(T_p\) is the first \(t\) with the front equal to \(L\). A run that never gets there is recorded at 60,000. The greedy-policy front — the largest \(d\) from which the greedy policy reaches the goal — is logged and is not the measurement.

Per encoding, \(\eta = L / \mathrm{slope}\), where the slope is the OLS slope of median \(T_p\) on \(K\). The intercept absorbs the fitting delay.

- **K1.** One-hot: \(\eta \in [0.67, 1.5]\) and \(R^2 \ge 0.95\).
- **K2.** \(\eta(\mathrm{coord}) \ge 2\,\eta(\mathrm{one\text{-}hot})\), and \(\eta(\mathrm{coord}) \ge \eta(\mathrm{thermo}) \ge \eta(\mathrm{one\text{-}hot})\).
- **K3.** \(\eta(\mathrm{random}) \le 1.5\,\eta(\mathrm{one\text{-}hot})\).

Supported if K1 and K2. Inconclusive if exactly one of them holds. Killed if neither holds. K3 is reported and does not enter the verdict.

After the pilot, and before the follow-up, a second pre-registration: the same run at \(K = 200\) with a coverage gap. States 15–24 have no transitions. The other states keep the same 20,000 transitions, about 20,000/30 each. A table cannot propagate through the gap.

- **F1.** One-hot: the front is at most 16 at 60,000 updates, in 8 of 8 seeds.
- **F2.** Coordinate: solved, front \(= L\), within 60,000 updates, in at least 6 of 8 seeds.
- **F3.** Thermometer: solved in at least 6 of 8 seeds.

Random codes are reported with no prediction. The conjunction F1 and F2 was the outcome that would have turned the pilot's null into a statement about where the law applies. It is part of the recorded follow-up, and it did not happen.

## 4 Results

The recorded verdict on the pilot is **inconclusive**. K1 passes, K2 fails, K3 passes. On the follow-up, F2 and F3 pass and F1 fails, so the conjunction F1 and F2 does not hold.

### 4.1 One hop per sync, for every encoding

| Encoding | \(T_p\) at \(K=50\) | \(K=200\) | \(K=800\) | slope | intercept | \(\eta\) |
|---|---:|---:|---:|---:|---:|---:|
| one-hot | 2,100 | 7,850 | 31,250 | 38.9 | 117 | 1.03 |
| random | 1,850 | 7,850 | 31,250 | 39.1 | \(-50\) | 1.02 |
| thermometer | 2,050 | 7,850 | 31,250 | 39.0 | 83 | 1.03 |
| coordinate | 2,050 | 7,850 | 31,250 | 39.0 | 83 | 1.03 |

Medians over eight seeds. Every fit has \(R^2 \ge 0.99998\); the one-hot \(R^2\) recorded in the verdict is 1.000. Forty hops at \(K = 800\) take 31,250 updates for every encoding: 39 syncs, plus a fitting delay of about a hundred updates. The one-hot multiplier is \(\eta = 1.03\), inside \([0.67, 1.5]\), which is K1. The four slopes are 38.9, 39.1, 39.0, and 39.0. Coordinate is not twice one-hot, and the order coord \(\ge\) thermo \(\ge\) one-hot fails because the three multipliers are 1.03, 1.03, and 1.03. That is K2, false. Random codes sit at 1.02, inside the K3 bound of \(1.5\) times one-hot, which holds because nothing moved.

The mechanism is the target. The regression target at a visited state is \(r + \gamma \bar Q(s')\). Beyond the front, \(\bar Q\) is zero, so those targets are zero, and a network trained on data at every state is pushed toward zero there whatever its inductive bias. Generalisation can reshape the fit between pinned targets. It does not move a target that the sync has not yet updated. The multiplier hypothesis required the network to invent nonzero targets ahead of the front. The four lines say it does not.

### 4.2 Across missing data, the follow-up's localist prediction fails

At \(K = 200\), with states 15–24 absent from the dataset, F1 required the one-hot front to stay at or below 16 in every seed. Zero of eight seeds stalled. All eight solved, at median \(T_p = 3{,}450\), against 7,850 on the same encoding with no gap. F2 and F3 pass: the coordinate network solves in 8 of 8 seeds at median 7,850, and the thermometer network solves in 8 of 8 at median 6,250. Random codes solve in 7 of 8, median 4,850. The follow-up conjunction F1 and F2 is false.

A one-hot MLP is not a table. The hidden layer and the output layer are shared, so a state with no row of data still receives a value through the biases and through the random untrained row of the first weight matrix. End-of-run value ratios \(\hat V / V^\ast\), median over seeds and over the states in question:

| Encoding | gap states 15–24 | IQR in the gap | outside the gap |
|---|---:|---|---:|
| one-hot | 1.24 | 1.17–1.32 | 1.04 |
| random | 1.00 | 0.90–1.09 | 1.00 |
| thermometer | 1.32 | 1.00–1.67 | 1.01 |
| coordinate | 1.01 | 0.99–1.02 | 0.99 |

The one-hot gap values are 24% above \(V^\ast\), and that bias is what lets the front skip the hole. The coordinate network crosses at exactly the full-coverage propagation time, 7,850, which is the no-gap \(T_p\) at \(K = 200\), and its gap values sit within 1% of \(V^\ast\) (IQR 0.99 to 1.02). The thermometer network arrives sooner, at 6,250, with gap values 32% high (ratio 1.32, IQR 1.00 to 1.67). Random codes land near \(V^\ast\) in the median and spread from 0.90 to 1.09. Outside the gap, every encoding is close to \(V^\ast\): the ratios are 1.04, 1.00, 1.01, and 0.99.

What the gap changes is whether a front can cross unseen states, and how close to \(V^\ast\) the crossed values are. It does not raise the hops-per-sync multiplier on the states the dataset still pins. The coordinate front takes the same 7,850 updates it took with complete data.

### 4.3 The greedy policy can be right before the values arrive

On the full-coverage pilot at \(K = 200\), the median update at which each front first reaches \(L\):

| Encoding | policy front | value front |
|---|---:|---:|
| one-hot | 8,825 | 7,850 |
| random | 7,675 | 7,850 |
| thermometer | 4,250 | 7,850 |
| coordinate | 250 | 7,850 |

With a monotone feature, and a corridor in which one action is right at every state, the shared output bias makes the greedy policy correct long before the values arrive. For the coordinate encoding that gap is 250 updates against 7,850, a factor of 30. One-hot and random codes get no such shortcut: their policy fronts sit at or after the value front. The thermometer sits in between, at 4,250 against 7,850. This is encoding-dependent policy generalisation. It is why the value front is the measurement, and why a return, or a "solved" flag, can precede a learned value by the length of the latency the experiment was built to see.

## 5 Discussion

Where the dataset pins a target at every state, the hard-sync count is a statement about a network as much as about a table. One hop per sync holds to 3% (\(\eta = 1.03\) on one-hot inputs), and it holds at the same slope for an encoding with no similarity, an encoding with unstructured similarity, an encoding with monotone locality, and an encoding with smooth locality. "The network propagates faster because it generalises" is false on this corridor under a target network. The target pins every state that has data to the hop the sync has reached.

Representation enters in the two places the follow-up and the policy front measure, and neither of them speeds the pinned front. Across a hole in the data, the network is free: the coordinate encoding interpolates and lands within 1% of \(V^\ast\), while one-hot and thermometer values in the hole are 24% and 32% high. That overestimate is a property of the encoding and the shared layers. It showed up here with no max operator to blame, because the backup is one-step regression toward the target network on a deterministic corridor. And wherever the right action is a simple function of the features, the policy can be correct while the critic is still many syncs from the start. Any latency measured on returns inherits that confound. The value front does not.

The follow-up does not support the sharper pre-registered claim that localist inputs stall at a gap. F1 was a claim about a table, and the learner was a one-hot MLP. The failure is reported as a failed prediction. What remains, as a description of these runs rather than as a passed test, is that the four encodings all cross, and that the accuracy of the values in the gap depends on the encoding.

## 6 Limitations

The dataset is offline and uniformly covering. An online agent collects the states its policy visits, and the policy shortcut will shape that coverage. Only one-step targets were run; \(n\)-step backups divide the hop count and were not used to remeasure \(\eta\). The network is a small MLP, the domain is one corridor, and the periods are \(K \in \{50, 200, 800\}\). F1 was mis-specified, and the follow-up is inconclusive because of that miss: the correction is the measurement of gap values, not a replaced test that then passed. DQN on MinAtar and MiniGrid, reading a value front against truncated Monte Carlo returns at target periods 100, 1,000, and 8,000, has not been run.

## 7 Conclusion

Under a hard target, with data at every state, a neural critic moves the value front one hop per sync. The multiplier is 1.02–1.03 for one-hot, random, thermometer, and coordinate inputs, the four slopes agree at 38.9–39.1 updates per unit of \(K\), and the pre-registered claim that locality would at least double the speed is false. That failure is the pilot. Across a block of states with no data the picture changes, and not in the way the follow-up pre-registered: the one-hot network does not stall, it skips the gap with values 1.24 times \(V^\ast\), while the coordinate network crosses at the full-coverage time of 7,850 updates with values 1.01 times \(V^\ast\). The greedy policy, given a coordinate, is correct at update 250. The latency laws are statements about values on states the data pin. They are not statements about when a return first looks solved.

## References

Achiam, Knight, Abbeel (2019). Towards characterizing divergence in deep Q-learning. · Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · Ilyas et al. (2020). A closer look at deep policy gradients. ICLR. · Kumar, Agarwal, Ghosh, Levine (2021). Implicit under-parameterization inhibits data-efficient deep reinforcement learning. ICLR. · Lyle et al. (2022). Understanding and preventing capacity loss in reinforcement learning. ICLR. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Panahi et al. (2024). Prioritised replay, target period, and generalisation. RLC. · Piché et al. (2022). Beyond target networks: functional regularisation.
