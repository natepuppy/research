# R9 pilot notes

Run: the same 16 CartPole DQN parents as R7 (30k steps). Forks at 5k, 10k, 15k, 20k, 25k; 3 children per (parent, fork, condition), all with fresh exploration and minibatch streams. Conditions: `control`, `reset_last` (output layer re-initialised), `reset_all` (fresh network and optimiser, buffer kept), `lr_drop` (learning rate halved). Parents split into terciles by final score: bad ≤ 119, good ≥ 154, good median 181. Rescue = child final ≥ 181. CPU, 26 min. Log: `logs/pilot.log`.

## Verdict

**KILLED.** No intervention raised the rescue probability of bad parents by ≥ 0.2 at any fork ≤ 0.6T (18k). The best gain was +0.11 (`reset_all`, 15k). The same summariser bug as R7 (T taken as the last fork rather than the training length) was fixed and the verdict re-run; it did not change.

## P(rescue | bad parent) and mean final of bad-parent children

| fork | control | reset_last | reset_all | lr_drop |
|---|---|---|---|---|
| 5k | 0.00 (107) | 0.06 (117) | 0.06 (114) | 0.00 (106) |
| 10k | 0.00 (95) | 0.00 (103) | 0.00 (112) | 0.00 (119) |
| 15k | 0.00 (109) | 0.00 (122) | 0.11 (155) | 0.00 (122) |
| 20k | 0.06 (118) | 0.06 (139) | **0.33 (169)** | 0.00 (116) |
| 25k | 0.00 (93) | 0.00 (114) | 0.00 (102) | 0.00 (100) |

## What this says

- The pre-registered rescue criterion was too hard for this environment. Control children of *good* parents fall below the good median 78–100% of the time, so "reach the good median" is a bar most good runs themselves do not clear after a re-fork. With that bar, rescue probabilities sit at the floor and the test has no power. This is a design error, not a finding about resets.
- The mean-score column tells a more interesting story that the kill test was not written to read. `reset_all` at 15k and 20k lifts bad-parent children from ~110–118 to 155–169, which is the level of good-parent children (158–165). At 25k the lift disappears (102 vs 93). So there is a window in which a full reset makes a doomed run look like a good one, and it closes with 5k steps left. That is exactly the shape the idea predicted, but at a later fork than the rule allowed (20k = 0.67T) and measured in means rather than rescue probability.
- `reset_last` and `lr_drop` do very little here. On CartPole the output layer is tiny, so a last-layer reset barely perturbs the function; a bigger network and a sparser task are where it is expected to matter.
- The harm metric (good-parent children below good median) is uninformative for the same reason the rescue metric is: the baseline is near 1.

## Caveats

- CartPole again: score saturation and spontaneous collapse make a single "final score" a poor fate label. Any paper version needs a smoother score (area under the last 20% of the curve) and an environment where good and bad runs are separated by more than noise.
- 18 bad-parent children per cell. A 0.33 rescue rate is 6 of 18.
- The one positive signal (reset_all at 20k) is post hoc and at one fork. It is a hypothesis for the next run, not a result.

## What a rerun should change (before any 3080 work)

1. Rescue defined relative to the control children of the same parent: rescue = child final exceeds the parent's control-child mean by more than one pooled SD. This removes the floor.
2. Score = mean of the last 20% of evaluations, not the last three.
3. Forks at 0.5T, 0.6T, 0.7T, 0.8T, 0.9T, because the window appears late.
4. Environment with a stable ceiling: MinAtar Breakout or Space Invaders.
5. Keep `reset_all`; add shrink-and-perturb and a buffer-reweighting intervention; drop `lr_drop` unless the network is larger.

If a rerun with (1)–(4) does not find a window, the idea is dead. If it does, R7 and R9 become one paper: forks establish when fate is set, resets establish until when it can be changed.
