# One Patch Is the Resolution: A Transferable Failure Law for Vision-Language Models

*Draft skeleton, written before the pilot. Numbers in brackets are filled from `results/`. Sections marked [3080] are the paper-scale experiments.*

**Abstract.** Vision-language models fail on counting and on small objects, and the failure has been blamed on the vision tower's patch grid without a number attached. We measure an object's size and its gap to the nearest neighbor in units of the tower's own patch, and test whether the fail rate crosses one half at one patch for every tower. In a controlled scene generator the crossing is at [t₈, t₁₆, t₃₂] patches for towers of patch size 8, 16 and 32, and a score in patch units predicts failure with pooled AUROC [0.xx] against [0.xx] for the same length in raw pixels. On the objects the law flags, giving each object its own tile cuts the error to [0.xx], against [0.xx] for a global 2× upsample and [0.xx] for a zoom that spends its budget on the largest object. [3080] On CountBench, a small-object slice of RefCOCO and a stratified slice of TallyQA, with Qwen3-VL and InternVL, the same crossing [does / does not] transfer, and the retile [does / does not] beat a learned zoom model. If it transfers, the resolution of a vision-language model on instance tasks is one patch, and the fix is a function of that patch size rather than a trained policy.

## 1. Introduction

A workshop paper this year showed that counting accuracy collapses when objects, or the gaps between them, approach the patch size of the vision tower. The observation explains a failure. It does not say whether the boundary is the same boundary in every model, which is the difference between a diagnosis and a law. Raw pixels cannot be that law: a 16-pixel object is two patches of a fine tower and half a patch of a coarse one.

We state the law as a crossing. Plot the fail rate against the shorter of size and gap, divided by the tower's patch size. It crosses one half at one. We then spend the law. A global upsample and a learned zoom both leave some objects under the line. A retile that gives every flagged object two patches of room should not.

Contributions. (1) The crossing, and the demonstration that it lines up across patch sizes only when length is measured in patches [pilot; 3080]. (2) The retile, compared with a 2× upsample and with zoom-the-largest on the flagged slice [pilot; 3080]. (3) [3080] The same measurement on Qwen3-VL and InternVL, against DeepEyes as the learned-zoom baseline.

## 2. Related work

Counting to Four is still a Chore (the diagnosis). VLMCountBench. The binding problem in vision-language models (NeurIPS 2024). Learned zoom: DeepEyes, MGPO, ACTIVE-o3. Benchmarks that can be solved with little visual evidence: DatBench, Seeing without Looking.

## 3. Setting

**Pilot.** Four objects per scene, log-uniform size and gap from 3 to 96 pixels, three towers with patch size 8, 16 and 32 on shared scenes. Fail probability is a sigmoid of min(size, gap) / P centered at 1, plus a clutter term. Confidence does not see the grid. Interventions on objects with min(size, gap) < P: global 2×, zoom the largest object to two patches of clearance, retile every object to two patches of clearance. 4,000 scenes, 8 seeds.

**[3080].** Qwen3-VL-2B, Qwen3-VL-4B, InternVL-2B at 4-bit. A 7B model on a 12 GB card. CountBench, 3,000 small-object RefCOCO questions, 8,000 stratified TallyQA questions. Blob map from a connected-component pass or a detector. About 90 GPU-hours.

Pre-registered. K1: pooled patch-unit AUROC beats raw pixels by ≥ 0.05, and every tower's patch AUROC is ≥ 0.80 and beats clutter and confidence by ≥ 0.10. K2: the 0.5-crossing lies in [0.75, 1.25] for every tower and the crossings lie within 0.25 of each other. K3: on flagged objects, retile beats 2× and zoom-the-largest by ≥ 0.08 on every tower. SUPPORTED if all three.

## 4. Results

[pilot] Crossing per tower. Pooled AUROC against raw pixels, clutter and confidence. Error on flagged objects for the base tower, the 2× upsample, zoom-the-largest and the retile.

[3080] The same table on the three benchmarks. A plot of fail rate against min(size, gap) / P with the towers overlaid. Token cost of the retile against the accuracy it buys.

## 5. Discussion

The pilot generates errors from the law, so its job is to show that the comparisons can tell the law apart from raw pixels, from clutter, and from the two interventions that look cheaper. The vision-language experiment is where the law can fail. A crossing that moves by more than a quarter of a patch between model families, or a global resolution increase that matches the retile, ends the claim.

Limitations. The pilot's objects do not occlude, and its zoom is a rule rather than DeepEyes. The blob map on real images will mis-segment touching instances, which is part of the phenomenon and has to be reported as its own error, not folded into the model's.

## 6. [3080] Planned experiments

(a) Forwards at one resolution on the three benchmarks. (b) Resolution sweep on the 2B model. (c) Retile on the flagged slice. (d) DeepEyes on 2,000 questions. (e) The full TallyQA test set only if the stratified slice is ambiguous.

## 7. Conclusion

[After the pilot.]

## References

Counting to Four is still a Chore; VLMCountBench; the binding-problem paper, NeurIPS 2024; DeepEyes; MGPO; ACTIVE-o3; DatBench; Seeing without Looking; TallyQA; RefCOCO; CountBench.
