# CV8: A box switches off recognition — positive exemplars pin SAM 3's presence head

**Question.** SAM 3 separates recognition from localization: a presence token says whether the named concept is in the image, and every proposal's score is multiplied by it. The paper credits this head for the model's ability to say no, and it is why a prompt for an absent concept returns nothing. SAM 3 also takes image exemplars, a box with a positive or negative label, alone or together with text. The claim here is that a positive exemplar removes the veto. With a positive box in the prompt, the presence logit sits at its clamp of +10 whatever the text says and whatever is in the image, the boxed object is returned as an instance of any named concept, and when text and box disagree the box wins. The mechanism is the training protocol: positive exemplars are sampled from ground-truth boxes of concepts that are in the image, so a positive box was never seen with an absent concept and the head learned to read the exemplar label instead of the image. A negative exemplar is sampled from false positives, which do occur on negative images, so the same argument predicts that a lone negative box with an absent phrase does not pin presence high. The fix is one extra decoder pass. The vision backbone is shared, so the text-only presence can be computed for the same image and used to gate the exemplar pass. The test is whether that gate restores rejection of absent concepts without costing accuracy when text and box agree.

**Status.** Measured on 500 COCO val2017 images on the 3080 (SAM 3 fp16, 6,421 prompt episodes, 10 minutes). **INCONCLUSIVE under the pre-registered test: K2 and K3 pass, K1 fails on its AUROC threshold.** K1: presence AUROC present-vs-absent is 0.996 text-only and 0.673 with a positive box (0/8 resamples ≤ 0.60; the threshold was set from a 40-image probe that gave 0.31 and did not hold at 500). What the AUROC hides: with a positive box the presence logit is ≥ 0 for an absent phrase in 99.9 % of 1,500 episodes (mean 9.23, median 10.0, at the +10 clamp in 59 %), so the veto is gone and the residual AUROC comes from ordering inside the positive range. K2: the boxed object is returned as the absent concept in 89.5 % of episodes (text-only absent phrases return any mask in 1.3 %); in text/box conflicts the box object is covered in 86.5 % of 421 episodes and the named category reaches IoU > 0.5 in 10.5 %, 8/8. K3: gating the exemplar pass with text-only presence cuts absent-phrase false-positive episodes from 99.9 % to 1.9 % (relative drop 0.981) while consistent text+box mIoU moves 0.809 → 0.793 (−0.016), 6/8. Mechanism check: a lone negative box with an absent phrase sits at −10 in 95.4 % and returns zero masks in 100 %; a box with no text has mean presence 9.97. The 40-image probe numbers in the design section are not results. Not run: LVIS, SA-Co, other exemplar sources (detector boxes, cross-image supports), any model other than SAM 3.

Why it matters. Every pipeline that converts a box into a SAM 3 exemplar inherits the loss: few-shot segmentation that pastes a support box into the prompt, exemplar-based counting on images where the class may be absent, detector-to-SAM 3 pipelines where a wrong detector box becomes a confident wrong mask, and interactive annotation where a user adds a positive box to fix a miss. SAM 3's own exemplar evaluation is run only on positive data (the paper states this), so the failure is invisible in the published numbers.

Prior work: SAM 3 (presence head; exemplar encoder with a label embedding; exemplar evaluation restricted to positive data; stage-3 interactive sampling of positives from ground truth and negatives from false positives). CAFE measures misleading *text* prompts and finds accurate masks for wrong concepts; this paper is the exemplar channel. SAM3-LoRA reports the presence head collapsing to always-on after positive-only fine-tuning; here the frozen model already has that collapse whenever a positive box is present. "Few-Shot Semantic Segmentation Meets SAM 3" and CASE report negatives hurting in a shared canvas; neither measures presence. ComfyUI issue 17105 shows the presence gate matters for absent text prompts, in the text-only setting.

## Pre-registered kill test (written before the measurement)

Design. COCO val2017, the first 500 images by id that have at least one polygon instance of area ≥ 32² px. SAM 3 fp16 from `facebook/sam3`, score threshold 0.5, mask threshold 0, vision features computed once per image and reused for every prompt. The **anchor** is the largest instance in the image; its box is the exemplar. Per image: **T+** text-only for up to three present categories (largest by total area); **T−** text-only for three absent categories drawn with a fixed seed; **B+T−** the same three absent phrases with the anchor as a positive exemplar; **B+T+** the anchor's category with the anchor as a positive exemplar; **B+T×** the largest *other* present category's name with the anchor as a positive exemplar; **B−T−** two absent phrases with the anchor as a *negative* exemplar; **B** the anchor alone, no text. Recognition is the presence logit. A mask "covers" an object when its IoU with that object's ground-truth mask is above 0.5. The **gate** rescales every B+ score by the sigmoid of the text-only presence from T− or T+ for the same phrase, in place of the exemplar pass's own presence. 8 bootstrap resamples of images.

- **K1 (the veto is lost).** AUROC of the presence logit for present versus absent phrases is ≥ 0.95 text-only and ≤ 0.60 with a positive exemplar (T+ vs T− against B+T+ vs B+T−), in ≥ 6 of 8 resamples.
- **K2 (the box overrides the name).** In B+T−, the anchor is covered by a kept mask in ≥ 70 % of episodes. In B+T×, the anchor is covered in at least twice as many episodes as the text category's union reaches IoU > 0.5, in ≥ 6 of 8.
- **K3 (the gate restores the veto for free).** With the gate, the fraction of B+T− episodes with any kept mask falls by ≥ 80 % relative, and the B+T+ mean IoU against the anchor category's ground truth falls by ≤ 0.02 absolute, in ≥ 6 of 8.

SUPPORTED if K1, K2 and K3. INCONCLUSIVE if exactly two. KILLED otherwise. Reported, not scored: the fraction of B+T− and B− T− presence logits at the clamp (the mechanism prediction is +10 for positive, −10 for negative with an absent phrase), and B presence.

What would make this a null: presence staying informative under a positive box (the probe is wrong or does not generalize past 40 images), or the gate removing true positives in B+T+ (text-only presence is unreliable when the box is needed in the first place, which is exactly when users add boxes).

## Phase 2: pre-registered extensions (written after phase 1, before any phase-2 run)

Phase 1 left three gaps: "absent" relied on COCO's 80-class label set, the only exemplar source was a ground-truth box, and K1 used a rank metric that does not express the decision the presence head is for. Phase 2 addresses each on the same 500 images with the same model and thresholds, plus one control and one downstream pipeline.

**A. LVIS verified-absent (`code/cv8_lvis.py`).** LVIS v1 val covers 588 of the images on disk and lists for every image the categories annotators verified absent (`neg_category_ids`) and the categories annotated exhaustively. Vocabulary 1,203 categories with frequency tags r/c/f. Per image (first 500 by id with ≥ 1 verified-absent category and ≥ 1 polygon instance ≥ 32²): the anchor is the largest LVIS instance; **T−/B+T−** for up to 3 verified-absent categories (fixed seed per image); **T+** for up to 3 exhaustively-annotated present categories; **B+T+** the anchor's category with the anchor box, gated; **Bg+T−** control: the first verified-absent phrase with a positive box of the anchor's size placed at the image position whose overlap with every annotated LVIS instance is smallest (a box on background, as far as LVIS can certify it). Phrases are the category's first synonym with underscores replaced by spaces.

- **L1 (decision-level veto).** Fraction of verified-absent episodes with presence ≥ 0: ≤ 0.10 text-only and ≥ 0.90 with the anchor box, in ≥ 6 of 8 image resamples.
- **L2 (the box object is returned).** The anchor is covered (IoU > 0.5) in ≥ 70 % of B+T− episodes, ≥ 6 of 8.
- **L3 (gate restores the veto, cost bounded).** Gate cuts verified-absent false-positive episodes by ≥ 80 % relative; B+T+ mIoU cost ≤ 0.03 overall, ≥ 6 of 8. Reported, not scored: cost split by frequency r/c/f (the gate should cost most where text alone is weakest, the rare bin), and the Bg+T− presence distribution and mask rate (the mechanism claim predicts presence pins near +10 even on background; a presence that drops on background would mean the head is reading box content rather than the label).

**B. Detector-to-SAM 3 verification (`code/cv8_detector.py`).** Faster R-CNN ResNet-50 FPN v2 (torchvision COCO weights) on the same 500 COCO images, detections with score ≥ 0.30, up to 8 per image by score. A detection is **correct** if IoU ≥ 0.5 with a same-class COCO ground-truth box, **wrong** otherwise, and **class-absent** if its class has no ground truth in the image at all (the clean subset; "wrong" among present classes is confounded by missing COCO labels). Pipeline: SAM 3 with text = detector class and the detection as a positive exemplar, as a verification-and-mask step would use it; a detection is **accepted** when any mask survives the 0.5 score threshold. The gate replaces the exemplar pass's presence with the text-only presence for that class (one cached pass per class per image).

- **D1 (SAM 3 confirms the detector's errors).** Without the gate, ≥ 90 % of class-absent wrong detections are accepted, ≥ 6 of 8.
- **D2 (the gate fixes it at small cost).** With the gate, acceptance of class-absent wrong detections falls by ≥ 70 % relative while acceptance of correct detections falls by ≤ 5 points absolute, ≥ 6 of 8.
- Reported: AUROC for correct-vs-wrong of (i) the detector's own score, (ii) ungated SAM 3 max score, (iii) gated SAM 3 max score, on all wrong and on class-absent wrong; the detector's accept rates by class-absent / present-wrong / correct.

Phase 2 verdict: SUPPORTED if L1, L2, L3, D1, D2 all pass; INCONCLUSIVE if 3 or 4; KILLED otherwise. The phase 1 verdict is not revised by phase 2.

### Phase 2 results (500 images each, 3080, 16 minutes total)

**INCONCLUSIVE: L1, L2, D1 pass 8/8; L3 and D2 fail 0/8, both on the gate's cost or strength at the pre-registered τ = 0.5, not on the failure itself.**

- **L1 (pass).** On LVIS verified-absent phrases, presence ≥ 0 in 1.5 % of text-only episodes and **99.8 %** with the anchor box (60 % at the +10 clamp). The decision-level statement of the veto loss is clean; phase 1's AUROC failure was the metric.
- **L2 (pass).** The anchor is returned as the verified-absent concept in **96.6 %** of episodes (1,500 episodes, 1,203-class vocabulary).
- **L3 (fail on cost).** The gate cuts verified-absent false-positive episodes 99.8 % → 1.5 % (drop 0.985) but costs 0.056 mIoU on consistent prompts (0.866 → 0.810), above the 0.03 bound. The cost is where text alone is weak: text-only presence is ≥ 0 for only 92 % of present LVIS categories (frequent 93 %, common 77 %, rare 83 %, n = 479/20/1 anchors), and the 6.2 % of consistent prompts whose text presence is negative had ungated IoU 0.86, so the gate throws away good masks. Cost by anchor frequency: frequent 0.867 → 0.815, common 0.825 → 0.691.
- **Background-box control (reported).** A positive box of the anchor's size on the least-annotated region, with a verified-absent phrase: presence mean 8.79, ≥ 0 in **99.8 %**, a mask returned in 99.8 %, the box region itself covered in 37 %. The head pins on the exemplar *label*, not on there being an object in the box.
- **D1 (pass).** Faster R-CNN produced 3,122 detections on 499 images (2,008 correct, 578 wrong-but-class-present, 536 class-absent). SAM 3 with the detection as exemplar accepted **100 %** of the 536 class-absent wrong detections, and 100 % of everything else: as a verifier it is a constant.
- **D2 (fail on strength).** The gate drops class-absent acceptance 100 % → 39 % (relative 0.61, below 0.70) at a 1.1-point cost on correct detections (100 % → 98.9 %). As a correct-vs-class-absent scorer, gated SAM 3 reaches **AUROC 0.960** against the detector's own confidence at 0.919 and ungated SAM 3 at 0.665; on all wrong detections, 0.824 vs 0.890 vs 0.683. The residual 39 % have higher detector scores (0.72 vs 0.61) and are bottle/person/car/chair/book; checking them against LVIS labels on the same images (`lvis_check_posthoc.csv`) matches only 6.9 % to an LVIS instance of that class (0.9 % among rejected), so unlabeled-but-real objects account for part of the residual but most is unresolvable without re-annotation (LVIS is federated, so a missing LVIS label is not proof of absence).

**Post-hoc, not pre-registered: the gate's operating point.** τ = 0.5 was chosen before any run and is the wrong point for LVIS phrases. From the stored episodes (`results/real_lvis/gate_curve_posthoc.csv`), a hard gate that rejects only when text-only presence probability is below τ gives, on LVIS: τ = 0.05 → absent false positives 99.8 % → 9.8 % at cost 0.004 mIoU; τ = 0.10 → 5.7 % at 0.005; τ = 0.20 → 3.5 % at 0.018; τ = 0.30 → 2.5 % at 0.032. On the detector, τ = 0.10 keeps 99.9 % of correct detections and accepts 61 % of class-absent ones. Any future run should pre-register τ ≈ 0.1 and would have passed L3 at that point; this run did not, and its verdict stands. Not run at the end of phase 2: SA-Co, cross-image exemplars, multiple exemplars, any model other than SAM 3, a human check of the detector residual.

## Phase 3: pre-registered before any phase-3 run

Phase 2's gate was scored at the multiplicative threshold τ = 0.5, which the post-hoc curve showed was the wrong operating point, and every number above came from one pool of 500 images. Phase 3 does not touch those verdicts. It asks the same questions on images that appear in none of `results/real_lvis/episodes.csv`, with the operating point fixed in advance, and adds the ablations and the second detector that phase 2 did not have.

The hard gate rejects an exemplar pass when the text-only presence probability is below **0.10** (sigmoid). 0.10 is taken from the phase-2 curve, so it is a confirmation setting, not a discovery. The geometry gate keeps a mask that passes the 0.5 score iff the text-only probability is ≥ 0.10 or the mask's IoU with the exemplar's rectangle is < 0.5 (a low-confidence concept may keep masks that are not just the box). 8 image bootstraps, pass at ≥ 6 of 8. SAM 3 fp16, score threshold 0.5, same phrase rule as phase 2.

**Held-out LVIS (`code/cv8_heldout.py`), 500 images not in phase 2.** Anchor = largest LVIS instance. Up to 3 verified-absent phrases, up to 3 exhaustive present phrases, B+T+, background box, as in phase 2.

- **H1.** Verified-absent presence ≥ 0: ≤ 0.10 text-only, ≥ 0.90 with the anchor box.
- **H2.** Anchor covered in ≥ 0.70 of B+T− episodes.
- **H3.** Hard gate: B+T− episodes with any kept mask fall by ≥ 80% relative, and B+T+ mIoU falls by ≤ 0.02.
- **H4.** Geometry gate meets the same two bars as H3.
- **H5.** Background box: presence ≥ 0 in ≥ 0.90 of episodes.
- **H6.** Counting. B+T− has ≥ 1 kept mask in ≥ 0.90 of episodes; the hard gate drops that rate by ≥ 80% relative; mean absolute error of the count on B+T+ (kept masks versus the number of ground-truth instances of the anchor category) rises by ≤ 0.30.

Held-out tag: SUPPORTED if H1, H2, H3 and H5 pass. INCONCLUSIVE if exactly 3 of those 4. KILLED otherwise. H4 and H6 are scored in the same line and do not change the tag; they are the new-method and the counting claims.

**Ablations (`code/cv8_ablate.py`), 400 images from the phase-2 pool, one verified-absent phrase each.** Extra positive boxes are the next-largest other instances. Perturbations of the anchor box: scale 0.5 and 2 about its center (clipped to the image), and a shift right by half the box width (clipped).

- **A1.** Presence ≥ 0 in ≥ 0.90 of episodes with 1, with 2, and with 3 positive boxes (2 and 3 scored only on images that have that many other instances).
- **A2.** Presence ≥ 0 in ≥ 0.90 for the half-scale, double-scale, and shifted box.
- **A3.** The half-scale box covers the anchor in ≤ 0.50 of episodes while its presence ≥ 0 rate stays ≥ 0.90.

SUPPORTED if A1, A2 and A3. INCONCLUSIVE if exactly 2. KILLED otherwise.

**Second detector (`code/cv8_retinanet.py`).** RetinaNet ResNet-50 FPN v2, torchvision COCO weights, same 500 phase-2 images, score ≥ 0.30, max 8 detections. Correct = IoU ≥ 0.5 with a same-class COCO box. Class-absent = no ground-truth box of that class.

- **R1.** Ungated SAM 3 accepts ≥ 0.90 of class-absent detections.
- **R2.** Correct-versus-class-absent AUROC of the gated score (text-only presence in place of the exemplar presence) is ≥ 0.90 and higher than the detector's own score. 0.90 is a confirmation bar chosen after phase 2's Faster R-CNN gated AUROC of 0.960; it is not a discovery threshold.

SUPPORTED if R1 and R2. INCONCLUSIVE if one. KILLED if neither. Reported, not scored: hard-gate τ = 0.10 acceptance of class-absent and of correct detections.

**Held-out Faster R-CNN (`code/cv8_detector.py --dest results/real_detector_heldout --exclude results/real_detector/episodes.csv`).** Same definitions as R1 and R2, on 500 images in neither detector result file. Tag rule identical. This is the confirmation that phase 2's detector result was not a property of those 500 images.

**FSC-147 counting (`code/cv8_count.py`).** Test split, up to 400 images. Count = number of kept masks; ground truth = number of annotated points. Exemplar = the first exemplar box. Absent phrase = a different FSC-147 class, drawn with a fixed seed.

- **F1.** Absent class + exemplar box yields count ≥ 1 in ≥ 0.80 of images.
- **F2.** Hard gate drops that rate by ≥ 70% relative.
- **F3.** On the true class, hard-gated text+box MAE exceeds ungated text+box MAE by ≤ 2.0.

SUPPORTED if F1 and F2. INCONCLUSIVE if only F1. KILLED if F1 fails. F3 is scored in the line and does not change the tag.

Known before these runs, so not claimed as results: SA-Co/Gold ships annotations only (image paths point at MetaCLIP and SA-1B, which are not in the release), and `facebook/sam3.1` is a `Sam3VideoModel` checkpoint (`sam3.1_multiplex.pt`, 3.5 GB) that `Sam3Model` in this environment cannot load. Neither is run. The human review of the phase-2 detector residual is a viewer, not a measurement, until the boxes are labeled.

### Phase 3 results (3080, about 45 minutes)

Five of the six pre-registered tags are SUPPORTED. The geometry gate (H4) fails. Nothing here changes the phase 1 or phase 2 verdicts.

- **Held-out LVIS, SUPPORTED** (500 images not in phase 2). H1: verified-absent presence ≥ 0 in 1.6% of text-only episodes and 99.5% with the anchor box, 8/8. H2: the anchor is returned as that concept in 96.8%. H3: the hard gate at 0.10 cuts false-positive episodes 99.5% → 6.4% (relative drop 0.936) at a cost of 0.002 mIoU (0.864 → 0.862), 8/8. H5: a background box still has presence ≥ 0 in 99.6% (mean 8.74) and returns a mask in 99.6%. H6, scored and passing: false counts 99.5% → 6.4%, and the count MAE on the true category moves 0.708 → 0.710. **H4 fails 0/8.** The geometry gate only cuts false-positive episodes by 0.480 (to 51.8%) at zero cost, because the false mask often has IoU below 0.5 with the box rectangle and the rule keeps it. The hard gate is the one that works; the geometry rule does not.
- **Ablations, SUPPORTED** (400 of the phase-2 images, one verified-absent phrase). Presence ≥ 0 for 1, 2 and 3 positive boxes in 99.8%, 100% and 100% (n = 400, 305, 226). Presence ≥ 0 for a half-scale box, a double-scale box and a box shifted by half its width in 100%, 99.8% and 100%. Those perturbed boxes cover the anchor in 0.3%, 17.7% and 3.7%. The pin does not require the box to contain the object, and the mask follows the box rather than the object once the box is wrong.
- **RetinaNet, SUPPORTED** (500 phase-2 images, 2,767 detections, 326 class-absent). Ungated SAM 3 accepts 99.7% of class-absent detections. Gated AUROC correct-versus-class-absent is 0.958, against the detector's own score at 0.896 and ungated SAM 3 at 0.571. The hard gate, reported and not scored, still accepts 59.8% of class-absent detections and 99.9% of correct ones.
- **Held-out Faster R-CNN, SUPPORTED** (499 images in neither earlier detector file, 3,152 detections, 565 class-absent). Acceptance of class-absent detections is 99.8%. Gated AUROC is 0.961 against the detector at 0.932 and ungated SAM 3 at 0.646. Hard-gate acceptance of class-absent detections is 57.9%, of correct detections 99.9%. The same gap as RetinaNet: on LVIS-verified absence the hard gate reaches 6.4%, and on a detector's "class absent" (COCO, not verified) it reaches about 58%.
- **FSC-147 counting, SUPPORTED** (400 test images, mean ground-truth count 63.8). An absent class plus the exemplar box produces a count of at least 1 in 85.5% of images (presence ≥ 0 in 94.8%). The hard gate takes that to 4.3% (relative drop 0.950). True-class MAE is 34.1 text-only, 28.7 with the exemplar, 29.3 with the hard gate (cost 0.55, within the bound of 2.0). SAM 3 is not a strong counter on this benchmark at either setting; the exemplar is what creates the false counts, and the gate removes them.

Not run, as stated before the runs: SA-Co images, SAM 3.1. The labeling page for the 536 phase-2 class-absent boxes is `C:\Users\natha\dev\cv-data\smoke\cv8_review.html`. It is empty until someone labels it, and no label is reported here.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv8_real.py`, `cv8_lvis.py`, `cv8_detector.py`, `cv8_heldout.py`, `cv8_ablate.py`, `cv8_retinanet.py`, `cv8_count.py`; `code/lib/` |
| `results/real/` | phase 1 |
| `results/real_lvis/`, `results/real_detector/` | phase 2 |
| `results/real_heldout/`, `results/real_ablate/`, `results/real_retinanet/`, `results/real_detector_heldout/`, `results/real_count/` | phase 3, each with `episodes.csv`, `summary.csv`, `seeds.csv`, `verdict.txt` |

## Run

```powershell
.venv-gpu\Scripts\python.exe code\cv8_real.py --limit 5      # smoke
.venv-gpu\Scripts\python.exe code\cv8_real.py                # 500 images, ~20 min on a 3080
.venv-gpu\Scripts\python.exe code\cv8_real.py --summarize
.venv-gpu\Scripts\python.exe code\cv8_lvis.py                # needs lvis_v1_val.json; ~8 min
.venv-gpu\Scripts\python.exe code\cv8_detector.py            # downloads torchvision Faster R-CNN weights; ~8 min
.venv-gpu\Scripts\python.exe code\cv8_heldout.py             # phase 3, images not used in phase 2
.venv-gpu\Scripts\python.exe code\cv8_ablate.py
.venv-gpu\Scripts\python.exe code\cv8_retinanet.py
.venv-gpu\Scripts\python.exe code\cv8_count.py               # needs FSC-147 under cv-data\fsc147
```
