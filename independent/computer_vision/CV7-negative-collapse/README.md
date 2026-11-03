# CV7: Negative exemplars collapse SAM 3, and where to put them instead

**Question.** SAM 3 takes a negative exemplar as a first-class prompt. Two papers this year find that the prompt does the opposite of what it is documented to do. On few-shot segmentation, frozen SAM 3 is strong with positive exemplars and drops sharply when negatives are added, up to the point of predicting all background. On a benchmark built to require negatives, the same frozen model falls from 30.6 to 18.6 mIoU, and the published fix is a trained LoRA. The claim here is a mechanism. The negative does not suppress the proposals that resemble it. It attenuates the whole concept: the presence score falls, and every proposal logit falls by about the same amount, whether the proposal matches the negative or not. If that is the mechanism, the exemplar is being used at the wrong layer. Run the positive-only pass and keep its presence score. Run the negative as its own positive, and reject proposals whose mask embedding matches that second set. No weight update.

**Status.** Not yet run. Laptop pilot written and smoke-tested (`code/cv7_collapse.py --smoke`). The pilot draws the attenuation from the rule above, so a pass checks that the logit drop is almost uncorrelated with similarity, that the presence score accounts for most of the lost mIoU, and that rejection on the positive-only pass recovers it. It does not show that real SAM 3 logits behave this way. That is the 3080 experiment.

Laptop pilot (CPU, numpy). Episodes of 8 proposals. Targets resemble the negative weakly, distractors resemble it strongly, with a small overlap. One negative pulls the presence score down by 0.26 on the CASE-like benchmark and by 0.23 on the COCO-like one, and shifts every logit by about the same additive amount plus a small similarity leak. The rejection threshold is a grid search on 400 validation episodes, then frozen. A label oracle, rejecting distractors with the true target mask and using the positive-only logits, stands in for the trained adapter.

Paper version on the 3080, about 15-25 GPU-hours. SAM 3 at 1008^2 in fp16 fits a 10 GB card. CASE, COCO-20i and PASCAL-5i, with the negative count swept through 0, 1, 3 and 5. The trained baseline is the published LoRA recipe on CASE. The diagnosis is logging: presence, per-proposal logits, and mask-embedding similarity, for one forward with the negative and one without.

Prior work: SAM 3, which specifies negative exemplars as a prompt type; "Few-Shot Semantic Segmentation Meets SAM 3" for the collapse under negatives in a training-free shared canvas; CASE for the positive-and-negative benchmark and the LoRA baseline; CAFE for SAM 3 accepting misleading concepts, a related failure of the concept channel; ActiveSAM for training-free use of the presence score, which this paper uses as a logged signal rather than as a class pruner.

## Pre-registered kill test (written before the pilot ran)

Design. 2,000 test episodes per benchmark, 400 validation episodes, 8 seeds. Collapse is the drop in mIoU from the positive-only pass to the pass that receives one negative. The presence share is (positive mIoU - presence-only mIoU) / (positive mIoU - collapsed mIoU), where the presence-only counterfactual keeps the original logits and applies the lowered presence score. Recovery is the fraction of that drop the rejection pass wins back.

- **K1 (global, not local).** The correlation between a proposal's logit drop and its similarity to the negative has absolute value < 0.20, and the presence share is ≥ 0.50, on both benchmarks. This must hold in ≥ 6 of 8 seeds.
- **K2 (the fix).** Rejection recovers ≥ 70% of the CASE gap and ≥ 60% of the COCO gap, in ≥ 6 of 8 seeds.
- **K3 (the ceiling, and the no-negative case).** The label oracle is within 3 mIoU of the fix on CASE, and with zero negatives the fix stays within 1 mIoU of the positive-only pass, in ≥ 6 of 8 seeds.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported. If K3 fails, the mechanism and the recovery are real and the training-free rule is leaving a gap a trained adapter would close, which changes the claim from "this replaces the LoRA" to "this is where the LoRA should look."

*Smoke-test changes (before the pilot, 2 seeds).* (a) The COCO presence drop was 0.20 per negative. Its presence share came back at 0.49, under the 0.50 bar, while CASE at 0.26 sat at 0.69. COCO is now 0.23 per negative, still below CASE. (b) The oracle ceiling added a logit nudge on top of perfect rejection and landed 3.3 mIoU above the fix, over a bar of 3. The ceiling is now label rejection on the positive-only logits, with no nudge. (c) The rejection pass ran even when the episode had no negative, so the zero-negative score moved. Rejection now runs only when a negative is present. Thresholds unchanged.

What would make the 3080 version a null: the logit drop tracks similarity to the negative, which would mean SAM 3 already excludes locally and the benchmarks are supplying the wrong negatives. Or the two-pass rejection recovers little of the gap, which would mean the collapse is not a presence-and-logit attenuation a second forward can route around.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv7_collapse.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per seed), `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` (after the pilot) |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cv7_collapse.py --smoke
.venv/bin/python code/cv7_collapse.py             # pilot, under a minute on CPU
.venv/bin/python code/cv7_collapse.py --summarize
```
