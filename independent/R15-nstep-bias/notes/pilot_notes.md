# R15 pilot notes

Three runs, all tabular Q-learning with replay (5000), batch 32, α = 0.1, target table every 100 updates, 8×8 slip-0.1 gridworld, exact Q*, 150k steps, 60 configs (5 n × 3–4 σ × 2 ε × 2 methods) × 4 seeds each. CPU, 15–20 min per run. Logs: `logs/pilot.log`, `logs/band.log`, `logs/band2.log`. Raw: `results/runs.csv`, `results/band/`, `results/band2/`. Post hoc: `results/posthoc.txt`.

## Verdicts

- **Pilot** (σ ∈ {0, 0.5, 1.5}, seeds 0–3): SUPPORTED by rule (K1, K2 pass; K3 fails), but K2 is degenerate: at σ > 0 the max-bias is 2–9 against V* = 0.468 and no agent's greedy return exceeds 0.4% of V*. "Best n" there ranks failed agents. My σ grid was set too high for α = 0.1. Recorded, not counted.
- **band** (σ ∈ {0, 0.1, 0.2, 0.3}, seeds 0–3; pre-registered follow-up with a working-agent condition at σ_max): K1 and K2 pass again, best n = 1 → 3 → 5–10 → 10 as σ rises; agents at σ_max = 0.3 reach 0.7% of V*, so the working-agent condition fails. Not SUPPORTED by my own rule. At σ = 0.1 agents reach 55% of V* and best n = 3.
- **band2** (σ ∈ {0, 0.05, 0.1}, **fresh seeds 4–7**; pre-registered replication with predictions P1–P4): P1 (sign) holds in all four noisy cells; P2 (best n larger at σ = 0.1 than σ = 0 for both ε, σ = 0.05 in between) holds: 1,1 / 3,2 / 3,3; P3 (plain Q at σ = 0.1 reaches ≥ 50% of V*) **fails at 45%**; P4 (best n is one of the two n with total overestimation nearest zero) holds. **INCONCLUSIVE by rule.** K3 (double Q halves the n-dependence) failed in all three grids.

Honest summary: every directional prediction about plain Q replicated out of sample; the one threshold I set on agent competence was missed by five points; the double-Q prediction was wrong.

## Key table (band2, plain Q, over_all = visitation-weighted overestimation, V* = 0.468)

| σ | ε | n=1 | n=2 | n=3 | n=5 | n=10 | best n (AUC) | best final / V* |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1 | +0.010 | −0.002 | −0.007 | −0.011 | −0.018 | 1 | 95% |
| 0 | 0.3 | +0.009 | −0.016 | −0.027 | −0.044 | −0.058 | 1 | 99% |
| 0.05 | 0.1 | +0.037 | +0.003 | −0.005 | −0.020 | −0.044 | 3 | 79% |
| 0.05 | 0.3 | +0.051 | −0.014 | −0.030 | −0.061 | −0.102 | 2 | 71% |
| 0.1 | 0.1 | +0.132 | +0.015 | +0.009 | −0.032 | −0.086 | 3 | 45% |
| 0.1 | 0.3 | +0.162 | +0.040 | −0.013 | −0.098 | −0.134 | 3 | 28% |

## What happened

1. **Uncorrected n-step is pessimistic, and the pessimism scales with ε and n.** At σ = 0 every n ≥ 2 undershoots Q*, by up to −0.06 (12% of V*) at n = 10, ε = 0.3, and costs 11–13% (ε = 0.1) to 22–23% (ε = 0.3) of final return against n = 1. This is the (T^μ)^{n−1} T* fixed-point effect, present identically for plain and double Q.
2. **Max-bias collapses with n much faster than pessimism grows.** Decomposing over_all(σ, n) = pess(n) + maxbias(σ, n) with pess from σ = 0: at σ = 0.1 the max-bias falls from +0.12–0.15 (n = 1) to +0.015–0.05 (n = 2) to ≈ 0.01 (n = 3). The pessimistic term moves from +0.01 to −0.03 over the same range. So the *reason* overestimation falls with n is mostly that a longer n-step target leans less on the bootstrapped max (weight γ^n, fewer compounding steps), with the pessimistic term as the smaller, opposite-signed correction. This is a refinement of the hypothesis, not the hypothesis.
3. **The best n sits where the two terms cancel.** In all 10 noisy plain-Q cells across band and band2, the best n by AUC is one of the two n whose total overestimation is nearest zero (chance 0.4^10 ≈ 10⁻⁴). The cancellation prediction holds in this form.
4. **Double Q halves max-bias, does not remove it, and shares the pessimistic term.** Slope on log n is 60–80% of plain Q's; double Q also benefits from n > 1 under noise (best n = 3–5 at σ = 0.1). K3 was written assuming double Q removes max-bias; in this setting (tabular, one target table, online argmax / target evaluation) it removes about half.
5. **Noise fragility.** With α = 0.1 the tabular agent fails outright by σ = 0.2 (reward 1, ~0.5 reward sd per step over 100-step episodes): the max-bias feedback drives the greedy policy to wander. The design mistake, repeated twice, was setting σ by the reward scale rather than by α·σ²/(2−α) against V*.

## What this says

Uncorrected n-step helps under reward noise because it shortens the bootstrapping chain the max-bias compounds through, and it costs under no noise because it blends in the behaviour policy's value. The best n is the one where the two biases net to zero, and that n rises with σ and falls with ε. The ε-dependence is the part that is new and testable in deep agents: with annealed ε, the pessimistic term shrinks over training, which predicts that the optimal n should *rise* over training (and that a fixed n is too large early and too small late), a specific prediction the age-dependent-n stub (S2-5) does not make.

## Caveats

- Tabular, one α, one grid, 4 seeds per set, two seed sets. Overestimation under noise interacts with α; a smaller α would shift the whole σ scale.
- Reward noise only; transition stochasticity (slip) was fixed at 0.1 and is a second source of max-bias not swept.
- The AUC criterion over 150k steps rewards early learning, where n-step's propagation speed (R12) also matters; the pilot does not separate speed from bias. A propagation-controlled comparison is in the 3080 plan.
- I set the σ grid three times. All three are reported; the predictions were fixed before the third and held on fresh seeds, but the competence threshold was my own and it was missed.
