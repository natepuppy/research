# CV9: What does the box mean? Exemplars generalize by appearance, and negatives do not generalize

**Question.** SAM 3 says a positive exemplar on one dog finds all dogs, and that a negative exemplar on a false positive suppresses similar objects; its interactive evaluation adds one exemplar per click and reports faster gains than fixing instances by hand. The gains are not separated into what the positive clicks did and what the negative clicks did, and the evaluation is a sum over clicks, so three questions are open. First, what concept does a lone positive box define: the category, or things that look like the boxed object? Second, when a negative box is placed on a false positive of a text prompt, does that false positive go away, do the other false positives of the same prompt go away, and do true positives survive? Third, if negatives are local, can a second forward do what the negative was meant to do: run the negative box as its own positive concept, which by the first question returns everything that looks like it, and subtract those proposals from the text pass. The claims: a positive box generalizes by appearance and leaks across categories where text does not; a native negative removes its own target about half the time and other false positives rarely; subtraction removes other false positives at a materially higher rate at small cost to true positives.

**Status.** Measured on 500 COCO val2017 images on the 3080 (SAM 3 fp16, 194 negative episodes, 346 positive-generalization episodes, 6 minutes). **INCONCLUSIVE under the pre-registered test: K1 passes 8/8, K2 fails 0/8, K3 fails 0/8.** K1 (negatives are local): a native negative box removes its own target in 68.6 % of episodes, removes the other false positives of the same prompt in 34.3 %, and removes 16.3 % of true positives; presence drops by 1.35 logits on average. The anchored recipe (positive box on a true positive plus the negative) does better on every rate: target 62.4 %, others 43.2 %, true positives lost 7.6 %. K2 (subtraction): running the false-positive box as its own concept and removing what it explains deletes 82.2 % of the other false positives and 99.5 % of the target, and also 80.0 % of the true positives. The proposed fix is dead: SAM 3's false positives on COCO are, to its own exemplar encoder, the same thing as the true positives, which is consistent with most of them being unannotated instances (person 47 of 194 episodes, chair 19) rather than confusable categories. K3 (a box means appearance): box-only leak 0.112, text-only leak 0.112, text+box 0.109; box-only recall of the other instances 0.864 versus text 0.880. The 40-image probe's appearance-leak signal did not survive the 500-image measurement with this metric. Not run: LVIS or any dataset with exhaustive labels (which is where the subtraction question could be separated from COCO's missing annotations), SA-Co, cross-image negatives, any model other than SAM 3.

Prior work: SAM 3 (exemplar encoder; interactive PCS evaluation, Figure 7; stage-3 sampling of positives from missed ground truth and negatives from confident false positives). CASE and "Few-Shot Semantic Segmentation Meets SAM 3" find negatives harmful in a cross-image canvas; this paper is within-image, on the model's own false positives, with the per-effect decomposition. CV7 in this repository tested a different claim (a negative attenuating all proposals) and found no collapse on COCO; its negative-as-positive idea is the subtraction baseline here. EBOD builds a feature-matching pipeline to stop repeat false positives in SAM 3, which presumes native negatives are not enough.

## Pre-registered kill test (written before the measurement)

Design. COCO val2017, the first 500 images by id with a polygon instance of area ≥ 32² px, SAM 3 fp16, score threshold 0.5, vision features reused across prompts. A mask covers an object when IoU > 0.5. **Positive generalization**: for every present category with ≥ 2 instances of area ≥ 32² (up to two per image), the exemplar is the largest instance; a text-only pass, a box-only pass and a text+box pass are scored on recall of the *other* instances of that category and on **leak**, the fraction of kept masks whose area lies mostly (> 50 %) inside the ground truth of a different category. **Negatives**: for every present category, the text-only pass is run; an episode is any case with ≥ 1 true positive (covers a GT instance) and ≥ 1 false positive (area ≥ 32², covers no GT instance of the category). The target is the largest false positive; its box is its mask's bounding box. Three arms: **native** (text + negative box on the target), **anchored** (text + positive box on the largest true positive + negative box on the target, the SAM 3 interactive recipe), **subtraction** (box-only pass on the target box, then remove text-pass masks with IoU > 0.5 to any kept box-only mask; the text pass's own scores are unchanged). Per episode: target removed, fraction of the other false positives removed, fraction of true positives lost. Rates are means over episodes. 8 bootstrap resamples of images.

- **K1 (negatives are local).** Native removes the target in ≥ 50 % of episodes and removes other false positives at a rate at least 20 points lower than its target-removal rate, in ≥ 6 of 8.
- **K2 (subtraction generalizes).** Subtraction's other-false-positive removal rate exceeds native's by ≥ 20 points, with true-positive loss ≤ 10 %, in ≥ 6 of 8.
- **K3 (a box means appearance).** Box-only leak ≥ 0.20 and text-only leak ≤ 0.05, with box-only recall of other instances ≥ 0.80, in ≥ 6 of 8.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported; it is the premise of the subtraction arm, and if it fails the subtraction has no reason to generalize. The anchored arm is reported for all three rates.

What would make this a null: native negatives removing other false positives about as often as their target (the paper's "suppressing similar objects" holds), or subtraction removing true positives at the rate it removes false positives (appearance is not separable from the category on COCO's confusable pairs).

## Layout

| folder | contents |
|---|---|
| `code/` | `cv9_real.py`; `code/lib/sam3_utils.py`, `code/lib/io_utils.py` |
| `results/real/` | `positives.csv`, `negatives.csv`, `summary.csv`, `seeds.csv`, `verdict.txt` |

## Run

```powershell
.venv-gpu\Scripts\python.exe code\cv9_real.py --limit 5      # smoke
.venv-gpu\Scripts\python.exe code\cv9_real.py                # 500 images, ~30 min on a 3080
.venv-gpu\Scripts\python.exe code\cv9_real.py --summarize
```
