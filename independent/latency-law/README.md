# Credit assignment is a count of iterations

**Question.** How many training updates does a reward D steps from a decision need before that decision can use it, and what sets the number?

**Claim.** The number is set by the target mechanism and by nothing else. A hard target refreshed every K updates takes (⌈D/n⌉ − 1)·K updates. A Polyak target of rate τ takes (⌈D/n⌉ − 1)/τ + ⌈D/n⌉/α. GAE takes about D·((1 − γλ) + 1/T) critic fits. The same hard-target count holds for a neural critic wherever the data covers the path. n is the one knob that divides all three, and it has a ceiling set by the exploratory bias of uncorrected n-step returns. Under ordinary DQN settings, most of the time charged to "hard exploration" is this count.

**Status.** RLC manuscript: `paper/paper_rlc.md`. The laptop pilots are unchanged. An RTX 3080 run (`code/gpu_3080.py`, `results/gpu/`) measured the same corridor online: one-hot DQN takes 1.03–1.04 hops per sync (\(R^2 = 1.000\)), Polyak slope 37.9–39.0 against a predicted 39. MinAtar, MuJoCo and long-horizon PPO were not run. The neural PPO grid on a length-16 corridor did not resolve the GAE formula.

## Where the numbers come from

| section of the draft | pilot | verdict |
|---|---|---|
| Hard targets | `../R12-bootstrap-speed-limit/` | first form killed; additive form confirmed out of sample, R² 0.997 / 0.971 |
| Polyak targets | `../R21-soft-target-horizon/` | SUPPORTED, all four tests |
| GAE | `../R19-gae-speed-limit/` | SUPPORTED (R² 0.999); the pre-registered rollout form was replaced by additive rates |
| Neural critics | `../R22-generalisation-hops/` | one hop per sync for every encoding; the "generalisation speeds it up" hypothesis failed |
| Choosing n | `../R24-nstep-rule/` | SUPPORTED |
| Exploration share | `../R20-exploration-is-propagation/` | propagation share and the latency identification passed; the count-bonus prediction failed |

The ledger of figures used in the draft is `results/ledger.md`.

## Layout

| folder | contents |
|---|---|
| `paper/main.md` | the combined draft |
| `results/ledger.md` | the numbers, each with its source file |
