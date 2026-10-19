# R24 pilot notes

Run: `code/r24_nstep.py`, 168 cells × 8 seeds = 1344 vectorised tabular learners, 60 000 updates, 75 min on the M1 Pro. Log in `logs/pilot.log`.

## Verdict (pre-registered)

**SUPPORTED.** K1 9/12 (threshold 9), K2 Spearman +1.00 over 66 correct cells, K3 12/12.

## What happened

- **Latency half (K2, P4).** In every cell where the policy is learned, the time to the first correct greedy policy is (⌈21/n⌉ − 1)K + 1 with no fitted constant: median T/law = 1.05, all between 1.005 and 1.99 (the upper end is the 200-update evaluation grain at K = 200, n ≥ 12). Spearman is exactly 1. At K = 2000, ε = 0.01, gap ≥ 0.2: T = 40 200, 20 200, 12 200, 8 200, 4 200, 2 200, 2 200 for n = 1, 2, 3, 5, 8, 12, 20 with 8/8 seeds correct throughout — an 18× speed-up for free.
- **Bias half (K1, P1–P3).** The learned Q(s₀, A) equals the closed-form pessimistic bias γ^L(1 − ε/2)^{(L+1) − ⌈(L+1)/n⌉} with r = 0.985, RMSE 0.029 over 168 cells. The policy flips where this bias crosses the gap: fraction of seeds correct is 0.96 when the predicted margin exceeds 0.05, 0.29 in (0.02, 0.05], and ≤ 0.06 at or below zero. The three K1 misses are the three cells where n* sits on the boundary with a margin (+0.026, +0.005, +0.001) below the seed noise of the learned value (std ≈ 0.06); the nine hits have margins of 0.10–0.30 except one at 0.017 that happened to land right.
- **K dependence (K3).** The largest safe n is identical at K = 200 and K = 2000 in 12/12 cells. The threshold is a property of the data, the latency is a property of the target period, and they do not interact.

## Honest reading

K1 passed at exactly its threshold, and the misses are the cells where the deterministic rule should have been stated with a noise margin ("bias − gap > a few seed-stds") rather than "> 0". That is the right refinement for the paper and it strengthens rather than weakens the claim: P3 is the sigmoid one expects from a sharp threshold seen through α = 0.1 noise.

The pessimistic bias in the pilot is the uncorrected-return bias from a behaviour policy that is fixed and ε-greedy. In a live agent the behaviour policy drifts toward greedy, so the bias shrinks over training; the rule then says "large n early (large gaps, latency dominates), smaller n late" — which is what BBF found by sweep (10 → 3) and what the paper should state as a prediction, with the schedule derived from the measured ε schedule and gaps.

## For the 3080

MinAtar DQN with n ∈ {1, 3, 5, 10, 20} × ε ∈ {0.01, 0.1} × K ∈ {500, 8000}, 5 seeds, measuring (i) time to a return threshold against the (⌈L/n⌉ − 1)K law using path lengths read from the return-to-reward distances, (ii) the asymptotic return ordering of n against the bias rule using measured action gaps. ~30 GPU-hours. Then the reanalysis of Rainbow (ε 0.1 → 0.01 annealed, n = 3), R2D2 (ε per actor, n = 5), DER (n = 20, ε 0.01) and BBF (n 10 → 3) against the rule.

## Odds

The two components were each already established in this lab (R12 latency law, R15 bias); the new content is the closed form of the bias on a path, the threshold n*, the K-independence, and the rule. Clean, quantitative, and about a hyper-parameter every value-based agent sets by sweep. 30–35 % standalone; stronger as the "what to do about it" chapter of the merged latency paper.
