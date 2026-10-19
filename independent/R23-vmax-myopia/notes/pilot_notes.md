# R23 pilot notes

Run: `code/r23_vmax.py`, 40 cells, 20 000 synchronous sweeps, ~4 min on the M1 Pro. Log in `logs/pilot.log`.

## Verdict (pre-registered)

KILLED by rule. K1 35/40 (needs 38), K2 80/80, K3 28/33, K4 39/40.

## What actually happened

The mechanism is confirmed and the rule failed on its own threshold.

- K2 is the substantive test: the C51 mean value equals the clipped fixed point Q_c in every (cell, action) pair, to within 5 % of V_max — in fact to within one atom width. The categorical learner converges to the clipped Bellman fixed point exactly as written.
- In every both-saturate cell the clipped fixed point prefers the immediate reward by exactly (1 − γ)V_max (post-hoc P1). I should have derived this before writing the test: Q_c(myopic) = min(V_max, 1 + γV_max) = V_max, Q_c(patient) = min(V_max, γV_max) = γV_max. With V_max = 3 that margin is 0.03 and with V_max = 5 it is 0.05 — exactly the indifference threshold I pre-registered. So "myopic" and "indifferent" were not distinguishable by construction in the small-V_max cells. Four of the five K1 misses are V_max = 5 cells where the prediction says "indifferent" (margin < 0.05 by floating point) and the learner says "myopic" (margin 0.04999). The fifth is (M, P) = (10, 25), V_max = 10, where the 0.4 atom spacing leaves the top mean at 9.93 and the learned margin is 0.033 rather than 0.10.
- Zero of 18 saturated cells chose `patient`. The patient boundary matches the prediction in all four forks (P4): the agent turns patient exactly when V_max rises above the myopic action's true value 1 + γV_M.
- K4's single miss is the same atom-resolution cell.

## What I take from it

1. The claim as stated ("one-step-greedy in the saturated region") is right and sharper than I wrote: the preference is (1 − γ)V_max, so for the Atari default it is 0.1 regardless of the true values (33.8 vs 62.8 in the largest fork). Under function approximation that margin is below typical value noise, so the realistic statement is "random among saturated actions", not "greedy" — a stronger claim about Rainbow's behaviour and a different experiment (saturation fraction and policy entropy in saturated states).
2. A re-run with the indifference threshold set to (1 − γ)V_max/2 and 101 atoms would pass K1 and K3; I am not re-running, because the honest statement is that the pre-registered threshold was mis-specified, and the data already show the mechanism.
3. The 3080 experiment is what carries the paper: Rainbow at V_max = 10 vs 100 on games chosen by predicted saturation, plus the saturation audit.

## Odds

The result is real and the derivation is a one-liner anyone can check, which is both the strength (nobody has written it down) and the weakness (reviewers may call it a known caveat). 20–25 % as a standalone with the 3080 Atari experiment; higher if the saturation audit shows a large fraction of saturated states in dense-reward games.
