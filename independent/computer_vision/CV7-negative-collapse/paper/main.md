# Negative Exemplars Collapse SAM 3

*Draft skeleton, written before the pilot. Numbers in brackets are filled from `results/`. Sections marked [3080] are the paper-scale experiments.*

**Abstract.** SAM 3 documents a negative exemplar as a way to exclude. Frozen, it does the reverse: adding negatives drops few-shot segmentation by a wide margin and can erase the mask entirely. The published repair trains a LoRA. We ask where the prompt acts. In a controlled segmenter, the drop in a proposal's logit has correlation [0.xx] with how much that proposal resembles the negative, and the fall in the presence score accounts for [xx%] of the lost mIoU. Keeping the positive-only presence and rejecting proposals that match a second, negative-as-positive pass recovers [xx%] of the CASE gap and [xx%] of the COCO gap, within [x.x] mIoU of a label oracle. [3080] On SAM 3, over CASE, COCO-20i and PASCAL-5i, the same attenuation [does / does not] appear, and the two-pass rejection [does / does not] recover it without a weight update. If it does, negative exemplars belong in a second forward, not in the prompt that sets the concept.

## 1. Introduction

A negative exemplar is supposed to be local. It should remove the thing that looks like it and leave the rest of the concept alone. The measurements say the prompt is not local. One negative is enough to depress a frozen SAM 3, and more negatives depress it further, which is what a global gain would do and not what a similarity gate would do.

We separate the two stories with a counterfactual. Hold the logits fixed and apply only the new presence score. Hold the presence score fixed and apply only the logit shift. A global mechanism puts the loss in the presence score, and it shifts every logit together. A local mechanism shifts the logits in proportion to similarity with the negative. The fix follows whichever story is true. Under the global one, the positive-only pass is the one to keep, and the negative is only a filter on its proposals.

Contributions. (1) The decomposition of the collapse into presence and logit shift, and the correlation of that shift with similarity [pilot; 3080]. (2) The two-pass rejection, against the positive-only model and against a label oracle standing in for the trained adapter [pilot; 3080]. (3) [3080] The same decomposition on SAM 3, and the comparison with the published LoRA on CASE.

## 2. Related work

SAM 3. Few-Shot Semantic Segmentation Meets SAM 3. CASE and its LoRA baseline. CAFE. ActiveSAM, which spends the presence score on vocabulary pruning.

## 3. Setting

**Pilot.** 8 proposals. Target similarity to the negative is drawn from N(0.28, 0.09^2), distractor similarity from N(0.74, 0.09^2). Target logits from N(0.55, 0.30^2), distractor logits from N(-0.55, 0.35^2), so a uniform shift can push targets across the decision threshold. Presence starts uniform on [0.62, 0.92] and falls by 0.26 per negative on CASE and 0.23 on COCO. The logit shift is 0.50 + 0.03 per negative, plus 0.02 times centered similarity. A proposal is emitted when presence and its sigmoid both clear 0.5. The rejection threshold is fit on 400 episodes. The oracle rejects by the true target label and uses the positive-only logits. 2,000 test episodes, 8 seeds. The negative count is also swept through 0, 1, 3 and 5.

**[3080].** SAM 3, fp16, 1008^2. CASE, COCO-20i, PASCAL-5i. Two forwards per episode for the fix. The LoRA baseline follows the published CASE recipe. About 15-25 GPU-hours.

Pre-registered. K1: absolute correlation of logit drop with similarity < 0.20, and presence share of the collapse ≥ 0.50, on both benchmarks, in ≥ 6 of 8 seeds. K2: recovery ≥ 70% of the CASE gap and ≥ 60% of the COCO gap, in ≥ 6 of 8 seeds. K3, reported: oracle within 3 mIoU of the fix on CASE, and the zero-negative pass within 1 mIoU of positive-only. SUPPORTED if K1 and K2.

## 4. Results

[pilot] Correlation. Presence share. mIoU of the positive-only pass, the collapsed pass, the fix and the oracle. Recovery. The collapse curve against the number of negatives.

[3080] The same table on the three benchmarks. The correlation plotted by layer or by head if the logit shift is not uniform across the decoder. A split by CASE's exclusion families, where a negative is load-bearing, against families where it is not.

## 5. Discussion

The pilot generates the attenuation it then decomposes, so its job is to show that the correlation, the presence share and the recovery can come apart. The SAM 3 experiment is where the drop can turn out to be local after all. A correlation with similarity would reframe the result: the model excludes, and the benchmark's negatives are the wrong negatives. That is a different, smaller paper.

Limitations. The pilot's oracle sees the target labels, and a trained LoRA will not. K3 is the check on how far that gap can open before the training-free claim has to shrink. The pilot's similarities barely overlap, and real mask embeddings will overlap more; the 3080 threshold fit is where that overlap is paid for. Presence is a single scalar per episode here, and SAM 3's presence head is per concept, which the logging has to respect.

## 6. [3080] Planned experiments

(a) Positive-only and positive-plus-negative forwards on CASE, COCO-20i and PASCAL-5i, negative counts 0, 1, 3, 5. (b) The correlation and the presence counterfactual, from logged scores. (c) Two-pass rejection, threshold fit per benchmark on the validation episodes. (d) The published LoRA on CASE, as the trained ceiling. (e) CAFE's misleading prompts, as a check that the same presence path is involved when the concept itself is wrong.

## 7. Conclusion

[After the pilot.]

## References

SAM 3; Few-Shot Semantic Segmentation Meets SAM 3; CASE; CAFE; ActiveSAM.
