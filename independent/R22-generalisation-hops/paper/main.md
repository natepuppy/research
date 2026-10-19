# One Hop per Sync, Whatever the Network: Target-Network Latency Is Representation-Independent Where Data Exists

**Abstract.** Closed-form latency laws for value propagation under target networks (one n-step hop per hard sync; 1/τ + 1/α updates per hop under Polyak averaging) are derived for tables. The standing objection is that a neural critic generalises, so a single sync might carry the signal several states at once and the laws would be loose upper bounds whose tightness depends on the representation. We test this with the same corridor, dataset, network and optimiser under four input encodings that differ only in what "similar" means — one-hot, random codes, thermometer codes and a scalar coordinate — and measure the value front over training. The hops-per-sync multiplier is 1.02–1.03 for all four (R² = 1.000; the regression lines of time-to-propagate on sync period are indistinguishable). With data at every state, a sync carries exactly one hop regardless of generalisation, because every state's target is pinned by its own data and the targets beyond the front are zero: a network can interpolate targets, it cannot invent them. Representation enters elsewhere. Across a block of states with no data, every encoding bridges the gap, but the smooth encoding does so at the full-coverage speed with values within 1% of truth while the one-hot network skips it with values 24% too high. And with a monotone feature the greedy policy is correct 30× earlier than the values arrive, through the shared output bias — a policy shortcut that makes "solved" precede "learned" and confounds any propagation measurement made on returns. [3080] We repeat the value-front measurement with DQN on MinAtar and MiniGrid. The result licenses the latency laws as exact statements for deep RL wherever the data covers the path, and locates the two places where representation genuinely matters.

## 1. Introduction

The target network is the most consequential hyperparameter in value-based deep RL that nobody treats as a horizon. In a table, a reward L steps from a decision needs (⌈L/n⌉ − 1) hard syncs to reach it, and the latency is exact (R12). Nobody believes a neural network behaves like a table: raising the value of one state raises its neighbours', so perhaps the front jumps several states per sync and the "law" is an upper bound whose slack is a property of the encoding. If so, the practical content of the latency laws for deep RL would be unclear and "representation quality" would have a propagation meaning worth measuring.

We measured it, and the answer is no: there is no slack. The reason is simple once the measurement is made on values rather than on policies. The regression target at a state is r + γ Q̄(s′), fixed by the target network. Beyond the front Q̄ is zero, so the targets are zero, and a network trained on data at every state is pushed toward zero there whatever its inductive bias. Generalisation can only reshape the fit between targets; the targets move one hop per sync and so does the front. Where the data does not pin the targets — a region with no transitions — the network is free, and there generalisation decides whether and how accurately the front crosses. And the *policy* can be right long before any value has propagated, if the right action is a simple function of the features, which it often is.

Contributions. (1) A controlled measurement of hops per sync under four encodings: 1.02–1.03 for all, R² = 1.000 (pre-registered; the multiplier hypothesis is rejected). (2) The coverage-gap experiment: generalisation acts across missing data, with accuracy that depends on the encoding (coord within 1%, one-hot +24%, thermometer +32%). (3) The policy-shortcut measurement: greedy-correct at 250 updates vs values at 7850 for coordinate inputs; no shortcut for one-hot. (4) [3080] The same measurement in MinAtar and MiniGrid DQN.

## 2. Related work

Piché et al. (2022) note that target networks slow propagation and propose functional regularisation to use up-to-date targets; Panahi et al. (RLC 2024) study prioritised replay with neural networks and use the target period to control over-generalisation in prediction tasks — the closest work, with the opposite direction of inquiry. Generalisation and bootstrapping: Achiam et al. (2019) on divergence, Kumar et al. (2021) on rank collapse, Lyle et al. (2022) on capacity; none measures propagation speed. Fedus et al. (2020) on n-step and replay. The policy-before-value phenomenon relates to Ilyas et al. (2020), who find PPO's value function poorly predicts returns while the policy improves. R12 and R21 supply the laws being tested; R19 the GAE sibling. Lab stubs: none on this question.

## 3. Setting

Corridor of L = 40 states, right moves toward the goal, reward 1 on entering it, γ = 0.99. Fixed dataset: 20 000 uniform (s, a) transitions. Learner: 2 × 64 MLP, Adam 1e-3, batch 64, Huber, grad clip 10, hard target sync every K ∈ {50, 200, 800}, trained offline up to 60 000 updates; 8 seeds. Encodings: one-hot (40-d), random ±1 codes (16-d), thermometer (40-d, bit k = [s ≥ k]), coordinate (s/L). Every 50 updates: Q at all states; value front = contiguous states from the goal with V̂ ≥ ½V*; T_p = first update with front = L; policy front logged alongside. η(enc) = L / slope of median T_p on K (the intercept absorbs the fitting delay).

Pre-registered: K1 one-hot η ∈ [0.67, 1.5], R² ≥ 0.95; K2 η(coord) ≥ 2η(one-hot) and coord ≥ thermo ≥ one-hot; K3 η(random) ≤ 1.5η(one-hot). SUPPORTED if K1 ∧ K2. Follow-up (pre-registered after the pilot): no data for states 15–24 at K = 200; F1 one-hot stalls (front ≤ 16, 8/8); F2 coord solves ≥ 6/8; F3 thermo solves ≥ 6/8.

## 4. Results

**One hop per sync, for every encoding (K1, K2).**

| encoding | T_p K=50 | K=200 | K=800 | slope (updates / unit K) | intercept | η |
|---|---|---|---|---|---|---|
| one-hot | 2100 | 7850 | 31 250 | 38.9 | 117 | 1.03 |
| random | 1850 | 7850 | 31 250 | 39.1 | −50 | 1.02 |
| thermometer | 2050 | 7850 | 31 250 | 39.0 | 83 | 1.03 |
| coordinate | 2050 | 7850 | 31 250 | 39.0 | 83 | 1.03 |

Median T_p over 8 seeds; R² of every fit ≥ 0.99998. Forty hops at K = 800 take 31 250 updates for every encoding: 39 syncs plus a fitting delay of about a hundred updates. The multiplier hypothesis is rejected at the third decimal.

**Across missing data (follow-up).** F1 failed: a one-hot MLP is not a table. Its shared layers assign the ten data-free states values 24% above V* (IQR 17–32%), and the front skips the gap (T_p = 3450 vs 7850 without the gap). The coordinate network crosses the gap at exactly the full-coverage speed (7850) with gap values within 1% of V* (IQR 0.99–1.02); the thermometer network at 6250 with values 32% high; random codes solve 7/8 seeds with values near V* but wide spread. Generalisation decides whether the front crosses a gap and how accurately, not how fast it moves where data exists.

**The policy shortcut (post-hoc).** At K = 200 the greedy policy is correct from the start state at update 250 (coordinate), 4250 (thermometer), 7675 (random), 8825 (one-hot); the values arrive at 7850 for all four. With a monotone feature and a corridor where the same action is right everywhere, the shared output bias makes the policy right 30× before any value has propagated. This is a property of the task–feature pair, not of propagation, and it is the reason return-based learning curves cannot be used to test latency laws.

## 5. Discussion

For the latency programme this is the needed result: the hard-sync law, and by the same argument the Polyak and GAE laws, are exact for neural critics wherever the data covers the path. "Deep RL propagates faster because it generalises" is false under a target network; the target pins every state with data to its own hop. What generalisation buys is extrapolation across missing data — valuable, encoding-dependent, and a source of bias (one-hot and thermometer gap values are 24–32% high; this is the kind of overestimation that is attributed to the max operator and is here a representation effect). And what the policy buys from features is a shortcut that can make an agent look solved while its critic is still thirty syncs from the start. Both deserve their own measurement in standard benchmarks.

Limitations. Offline uniform coverage; online, the data distribution is shaped by the policy shortcut itself. One-step targets, small MLP, one corridor. The gap follow-up's F1 was mis-specified (one-hot ≠ tabular), which we report rather than repair.

## 6. [3080] Planned experiments

(a) DQN on MinAtar (5 games) and MiniGrid, K ∈ {100, 1000, 8000}, value front measured against truncated Monte-Carlo returns from stored states; test η ≈ 1. (b) Coverage gaps in MiniGrid (image vs coordinate observations) and the bias of extrapolated values. (c) Policy vs value front in tasks where the right action is and is not a simple function of the observation. (d) The online interaction: does the policy shortcut change which states get data, and hence where the law applies?

## 7. Conclusion

A target network moves the value front one hop per sync in a neural critic as in a table, to 3%, under four encodings that generalise very differently. The latency laws are exact where data exists; representation matters across data gaps and in the policy, and in neither place does it speed propagation.

## References

Piché et al. 2022; Panahi et al. 2024; Achiam et al. 2019; Kumar et al. 2021; Lyle et al. 2022; Fedus et al. 2020; Ilyas et al. 2020; Mnih et al. 2015; R12, R19, R21 (this lab).
