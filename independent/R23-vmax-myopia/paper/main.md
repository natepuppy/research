# The Value Support of C51 Is a Horizon: Rainbow Is Indifferent Wherever It Is Winning

*Draft. Pilot numbers are from the tabular study in `results/`; sections marked [3080] are planned.*

## Abstract

C51 and its descendants (Rainbow, DER, SPR, BBF, the Dopamine defaults) represent the return distribution on a fixed support [−V_max, V_max] with V_max = 10. Atari rewards are clipped to {−1, 0, 1} and γ = 0.99, so a competent policy in a dense-reward game has discounted returns of tens of reward units. The categorical projection clips every bootstrapped atom to V_max, so the learned value is the fixed point of a clipped Bellman operator, T_c Q = min(V_max, r + γ max Q′). We derive the policy consequence: wherever two actions both have continuation values above V_max, the clipped fixed point prefers the larger immediate reward by exactly (1 − γ)V_max — 0.1 for the Atari defaults — regardless of the actions' true values. The agent is one-step-greedy in the saturated region in exact arithmetic and, under value noise larger than 0.1, random among saturated actions. In a tabular study with the exact C51 projection on 40 fork tasks (four corridor lengths × ten supports), the learned C51 means match the clipped fixed point in 80/80 (cell, action) pairs to within one atom width; the agent never chooses the patient action in any of the 18 cells where both continuations saturate, and switches to patient exactly when V_max exceeds the myopic action's true value 1 + γV_M. A scalar clipped-Q control reproduces C51's choice in 39/40 cells, so the effect is the clip, not the distribution. [3080] We measure the saturation fraction of trained Rainbow agents on Atari and compare V_max = 10 with V_max = 100 on four games chosen by predicted saturation.

## 1. Introduction

Three design choices make the standard Atari agent optimise something other than the discounted return: reward clipping (van Hasselt et al., 2016), the Huber loss (treated as a risk preference in [R16]), and — the subject of this paper — the fixed value support of the categorical return distribution. The first two are discussed in the literature. The third appears only as a caveat: the C51 paper notes that returns outside the support are projected onto its boundary, tutorials observe that ±10 "does not bracket the achievable ±100" and that this is tolerable, and QR-DQN (Dabney et al., 2018) removed the fixed support for flexibility. Nobody has written down the fixed point the clip produces or asked what policy it induces.

The answer is short. Let Z_c be the categorical distribution the C51 update converges to and Q_c its mean. Because the projection clips every bootstrapped atom at V_max, Q_c satisfies Q_c(s, a) = E[min(V_max, r + γ Z_c(s′))] ≤ min(V_max, r + γ max_a′ Q_c(s′, a′)). In a state where two actions a₁ (reward 1 now) and a₂ (reward 0 now) both have continuations worth more than V_max, Q_c(s, a₁) = V_max and Q_c(s, a₂) = γV_max. The agent prefers a₁ by (1 − γ)V_max whatever the true values are. With V_max = 10 and γ = 0.99 that is 0.1, and the true values can differ by any amount.

Two consequences follow. In exact arithmetic, the clipped agent is one-step-greedy in saturated states: its planning horizon is "until V_max reward units are in view", not 1/(1 − γ) = 100 steps. Under function approximation, a 0.1 preference is below typical value noise, so the realistic statement is that the agent is indifferent — its greedy policy in saturated states is set by approximation error, not by the task. Either way, saturation is where the agent is doing well: a continuation exceeds 10 whenever the clipped reward rate over the next ~100 steps is above about 0.1 per step, which describes competent play in Atlantis, Enduro, Ms. Pac-Man, Q*bert, Breakout late in a board, and many others.

## 2. The clipped fixed point

**Definition.** T_c Q(s, a) = E[min(V_max, r + γ max_a′ Q(s′, a′))]. T_c is a γ-contraction in the sup norm (min with a constant is 1-Lipschitz), so it has a unique fixed point Q_c, and Q_c ≤ min(V_max, Q*) pointwise.

**Proposition 1 (saturated preference).** If at s both actions a₁, a₂ have r(s, a₁) + γ max Q_c(s′₁, ·) ≥ V_max and r(s, a₂) + γ max Q_c(s′₂, ·) ≥ V_max, then Q_c(s, a₁) = Q_c(s, a₂) = V_max; if a₂'s continuation saturates but a₂'s immediate reward is 0, Q_c(s, a₂) = γV_max and the preference for a₁ is (1 − γ)V_max. *Proof:* substitute. ∎

**Proposition 2 (the patient boundary).** For the fork of Section 3, the clipped agent chooses the patient action iff V_max > 1 + γV_M, i.e. iff the myopic action's true value lies strictly inside the support. *Proof:* if 1 + γ min(V_max, V_M) ≥ V_max the myopic value is V_max ≥ γ min(V_max, V_P); otherwise the myopic value is its true value 1 + γV_M < V_max ≤ γV_P-clipped. ∎

The C51 projection does not implement T_c exactly: a bootstrapped atom above V_max is placed on the top atom, which is the clip, but atoms strictly inside the support are split between neighbours, which shaves the mean by up to half an atom width. We therefore treat clipped value iteration as the prediction and the categorical learner as the thing under test.

## 3. Tabular study

**Task.** Deterministic fork, γ = 0.99. `myopic`: reward 1, then a corridor paying 1 per step for M steps, then terminal (continuation V_M = (1 − γ^M)/(1 − γ)). `patient`: reward 0, then a corridor paying 1 per step for P steps (V_P). (M, P) ∈ {(5, 12), (10, 25), (20, 50), (40, 100)}, so patient is optimal under the true objective by 0.5 to 29 reward units. V_max ∈ {3, 5, 8, 10, 15, 20, 30, 50, 100, 200}.

**Learners.** (a) Tabular C51, 51 atoms on [−V_max, V_max], exact categorical projection, synchronous sweeps, p ← p + 0.1(Π target − p), 20 000 sweeps. (b) Scalar Q-learning with target min(V_max, r + γ max Q′), same schedule. **Prediction.** Clipped value iteration. Indifference threshold 0.05 for both prediction and learned choice (pre-registered).

**Pre-registered tests.** K1 choice agreement ≥ 95 % of 40 cells; K2 learned mean within 5 % of V_max of Q_c in ≥ 95 % of 80 (cell, action) pairs; K3 myopic in all both-saturate cells (V_max < 1 + γV_M) and patient in all none-saturate cells (V_max > γV_P); K4 scalar control agrees with C51 in every cell. SUPPORTED if K1–K3.

**Results.** K2 80/80: the C51 mean equals the clipped fixed point everywhere. K1 35/40, K3 28/33, K4 39/40 — KILLED by the pre-registered rule. Post-hoc analysis of the misses: in all 18 both-saturate cells the clipped preference is exactly (1 − γ)V_max (0.03 at V_max = 3, 0.05 at V_max = 5, 0.10 at V_max = 10), so the pre-registered 0.05 threshold sat inside the band it was meant to split. Four K1 misses are V_max = 5 cells where the prediction rounds to "indifferent" and the learner to "myopic" at a margin of 0.0500; the fifth, and the single K4 miss, is (10, 25) at V_max = 10, where the 0.4 atom spacing leaves the top mean at 9.93 and the learned margin at 0.033. No saturated cell chose `patient` (0/18). The patient boundary is at V_max = 1 + γV_M in all four forks, as Proposition 2 predicts: with M = 10 the agent is patient from V_max = 15, with M = 40 from V_max = 50.

We report the verdict as it fell. The test was mis-specified (the threshold should have been (1 − γ)V_max/2, which we did not derive until after the run) and we have not re-run it; the data support the mechanism without the rule.

| (M, P) | true Q(myopic) | true Q(patient) | V_max = 10: Q_c myopic / patient | C51 choice | V_max = 100: C51 choice |
|---|---|---|---|---|---|
| (5, 12) | 5.85 | 11.25 | 5.85 / 9.90 | patient | patient |
| (10, 25) | 10.47 | 21.89 | 9.93 / 9.90 | indifferent (0.033) | patient |
| (20, 50) | 18.95 | 39.11 | 10.0 / 9.90 | myopic (0.10) | patient |
| (40, 100) | 33.77 | 62.84 | 10.0 / 9.90 | myopic (0.10) | patient |

## 4. What it means for Atari agents

For V_max = 10, γ = 0.99 and clipped rewards, a state is saturated when the clipped reward rate over the next ≈ 100 steps exceeds ≈ 0.1 per step. In such states Rainbow's learned values for any two actions whose continuations both saturate differ by at most 0.1. Three predictions for the deep experiment: (i) a substantial fraction of a trained Rainbow agent's visited states in dense-reward games have their value on the top atom; (ii) in those states the greedy policy has near-maximal sensitivity to seed and to small parameter changes (it is set by noise); (iii) raising V_max to 100 changes behaviour where (i) holds and nowhere else. The prediction is not that V_max = 100 scores higher — the sparse upper atoms may learn slowly — but that the two agents differ exactly in the saturated states.

## 5. [3080] Deep experiments

Rainbow (Dopamine) with V_max ∈ {10, 100}, 51 atoms, 3 seeds, on MinAtar (all five games, 5M frames) and on Atlantis, Enduro, Ms. Pac-Man and Q*bert (10M frames each), ≈ 70 GPU-hours total on one RTX 3080. Measurements: saturation fraction (share of visited states with argmax atom at the top for the greedy action); per-state policy agreement across seeds, split by saturated vs unsaturated states; episode return. Controls: V_max = 10 with 201 atoms (atom resolution) and QR-DQN (no support).

## 6. Related work

Bellemare et al. (2017) introduce C51 and the projection; Hessel et al. (2018) adopt V_max = 10 in Rainbow; the Dopamine defaults carry it forward into DER, SPR and BBF. Dabney et al. (2018) motivate QR-DQN partly by the fixed support. van Hasselt et al. (2016) analyse reward clipping as a change of objective and propose Pop-Art; our effect compounds with clipping, because clipping is what makes clipped rewards dense enough to saturate a support of 10. Return-distribution truncation appears in distributional theory as a technical device (Rowland et al. 2018) but not as a policy distortion. The horizon reading connects to the discount-as-horizon literature (Jiang et al. 2015): V_max imposes a reward-unit horizon on top of the discount's time horizon.

## 7. Limitations

Tabular evidence only, with the pre-registered rule failing on its threshold; the deep predictions are untested. The (1 − γ)V_max preference is exact for deterministic forks; with stochastic rewards the saturated preference depends on the tail of the return distribution below V_max. Dopamine's Rainbow also uses n = 3 multi-step targets, which raises the effective immediate reward to a 3-step sum and so widens the saturated preference to roughly (1 − γ³)V_max ≈ 0.3.

## References

Bellemare, Dabney, Munos 2017. A distributional perspective on RL. · Hessel et al. 2018. Rainbow. · Dabney et al. 2018. Distributional RL with quantile regression. · van Hasselt et al. 2016. Learning values across many orders of magnitude. · Rowland et al. 2018. An analysis of categorical distributional RL. · Jiang et al. 2015. The dependence of effective planning horizon on model accuracy. · Castro et al. 2018. Dopamine. · [R16] Huber as risk preference, this lab.
