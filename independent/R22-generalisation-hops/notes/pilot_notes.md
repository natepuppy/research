# R22 pilot notes

**Verdict: INCONCLUSIVE by rule (K1 pass, K2 fail), and the failure is the finding.** `results/verdict.txt`, `results/gap/verdict.txt`.

## What was run
Corridor L = 40, fixed offline dataset of 20k uniform (s, a) transitions, batched 2×64 MLP DQN (Adam 1e-3, batch 64, Huber, clip 10), hard target sync K ∈ {50, 200, 800}, four input encodings (one-hot, random ±1 codes, thermometer, scalar coordinate), 8 seeds. Value front = contiguous states from the goal with V̂ ≥ ½V*; T_p = updates until the front reaches the start. Hops per sync η = L / slope(T_p on K). ~9 min CPU. Follow-up: same at K = 200 with no data for states 15–24.

## Results
- **K1**: one-hot η = 1.03, R² = 1.000 (slope 38.9 updates per unit K for 40 hops, intercept 117). The hard-sync law holds under function approximation to 3%.
- **K2 (hypothesis: locality multiplies the speed)**: **false**. η = 1.02–1.03 for *every* encoding; the four regression lines are indistinguishable (slopes 38.9–39.2). With full data coverage a sync carries exactly one hop regardless of how the network generalises, because every state's target is pinned by its own data and the targets beyond the front are zero: generalisation can interpolate targets, it cannot invent them.
- **K3**: random codes η = 1.02, as predicted (trivially, given K2).
- **Gap follow-up**: F1 (one-hot stalls at the gap) **failed** — a one-hot MLP is not a table; shared layers give the gap states values (24% above V*, P1) that let the front skip the gap (T_p 3450 vs 7850 without a gap). F2/F3 passed: coord crosses the gap at exactly the full-coverage speed (7850) with gap values within 1% of V*; thermo crosses at 6250 with gap values 32% high. Generalisation decides *whether and how accurately* the front crosses missing data, not how fast it moves where data exists.
- **Policy shortcut (P2)**: with coord inputs the greedy policy is correct at update 250 while the values arrive at 7850; one-hot gets no shortcut. In tasks where the right action is a simple function of the features, "solved" precedes "learned" by a factor of 30 — a confound for any propagation measurement made on policies.

## What I believe now
The latency laws (R12 hard sync, R21 Polyak, R19 GAE) are representation-independent wherever the data pins the targets: one hop per sync is exact, not a tabular upper bound. This is better for the programme than the multiplier I hypothesised, and it is the answer to the reviewer objection the whole line faces. Representation enters in two other places: across data gaps (extrapolated values, accurate for smooth inputs and biased for others) and in the policy (feature-aligned shortcuts that make the policy right before any value has arrived).

## Caveats
- Offline fixed dataset with uniform coverage; online data is non-uniform and the policy shortcut will interact with data collection.
- One-step targets only; n-step divides the hop count and should not change η.
- Small MLP, one corridor, one K range. The deep check is MinAtar/MiniGrid with DQN, where the measurement must be made on values (regress against truncated MC returns) rather than on returns.
- The gap follow-up's F1 was mis-specified (one-hot ≠ tabular); the correction is in the notes and README, not silently.

## Next
3080: DQN on MinAtar and MiniGrid, hard sync K ∈ {100, 1000, 8000}, measure the value front against truncated MC returns and test η ≈ 1; a policy-vs-value front comparison in tasks where the action is and is not a simple function of the features; coverage-gap experiments with image vs coordinate observations.
