# Which Views the Reconstruction Knows It Misplaced

*Draft skeleton, written before the pilot. Numbers in brackets are filled from `results/`. Sections marked [3080] are the paper-scale experiments.*

**Abstract.** Feed-forward 3D models fail in two ways their confidence does not report. They assemble images from different scenes into one geometry, and they place a second copy of a scene that really is there. A probe on frozen pair features is already known to predict pose failure, and it has not been asked of a view set. In a controlled generator where the fused token carries both failures, the probe's held-out AUROC is [0.xx] on intruders against [0.xx] for anchor agreement, and [0.xx] on displaced duplicates where anchor agreement is [0.xx] and confidence is [0.xx]. Dropping the intruders and splitting the duplicates, one extra forward, cuts pose error by [xx%]. [3080] On VGGT and π³, with SysCON3D mixtures, held-out ScanNet and wide-span sets, the same split [holds / does not hold]. If it holds, a view set can be refused or cut apart before the camera head commits, using tokens the model already computed.

## 1. Introduction

VGGT's confidence is trained per point. It says which depths were hard to regress. It does not say whether the cameras were placed consistently, and on pairs a probe of the fused features is already much better at that second question. View sets add a failure pairs do not have. Some of the views may not belong, and some that do belong may be reconstructed twice, displaced, because the span was wide or the scene was symmetric. Anchor agreement catches the first, because an intruding view looks unlike the others. It misses the second, because the duplicate is a true view of the scene.

The claim is that both failures are in the fused tokens of the N-view model, on a dataset the probe was not fit on, and that reading them is enough to repair the set: drop what does not belong, split what was placed twice, run once more.

Contributions. (1) The intruder score against anchor agreement and confidence, transferred across scene families [pilot; 3080]. (2) The duplicate score, on views that appearance cannot reject [pilot; 3080]. (3) The drop-or-split repair and the pose error it returns [pilot; 3080].

## 2. Related work

VGGT and π³. "Does VGGT Know What It Does Not Know?" (the pair probe). "Can These Views Be One Scene?" (fabricated geometry; COLMAP as the remedy). "When Wider Views Fail" (displaced copies). ReVar3R (gauge-aware point variance). RobustVGGT (anchor agreement).

## 3. Setting

**Pilot.** 12 views. Families differ in fusion scale, fusion bias and appearance noise. Each seed holds one family out. Intruders: 2-4 views, low appearance agreement, high fused residual. Duplicates: 4 genuine views, high appearance agreement, high confidence, high fused residual. Probe: L2 logistic regression on the residual, the anchor agreement, the confidence and two noise features, with balanced sample weights. Flag: within-set z-score, threshold fit on the training families. Anchor agreement below 0.62 drops a flagged view; at or above 0.62 the flagged views form a second component. Pose error over genuine views is 3° in a pure component, 18° in a mixed one, and 40° for a view that was dropped. 80 sets of each kind per family, 8 seeds.

**[3080].** VGGT-1B and π³, 8-16 views. SysCON3D, ScanNet, DL3DV or Tanks and Temples, wide-span subsets. A view is failed when its rotation error exceeds 15° after similarity alignment. About 25-40 GPU-hours. The probe is fit on dumped tokens.

Pre-registered. K1: held-out intruder AUROC ≥ 0.85 and ≥ 0.10 above anchor agreement, in ≥ 6 of 8 seeds. K2: held-out duplicate AUROC ≥ 0.80, in ≥ 6 of 8 seeds. K3, reported: pose-error reduction ≥ 40% at ≤ 2 forwards. SUPPORTED if K1 and K2.

## 4. Results

[pilot] AUROC of the probe, anchor agreement, confidence and the raw fused residual, on the intruder slice and the duplicate slice. Reduction and forward count. The z threshold the training families chose.

[3080] The same table on real tokens, plus the layer at which the score appears, following the pair paper's finding that the signal is concentrated in a band of aggregator blocks. A split of wide-span duplicates against symmetric rooms, which the pilot treats as one signature.

## 5. Discussion

The pilot writes the residual it then reads, so it tests the comparison and the repair, not the existence of the residual. The reconstructor experiment is where the residual can be absent. A duplicate the model is confidently wrong about, with the same token it produces for a correct view, ends the claim. Intruder detection without duplicate detection is a smaller paper, and RobustVGGT already covers part of it.

Limitations. The pilot's families are noise settings, not datasets, and its second forward is a relabeling of the same draws rather than a second network evaluation. The 3080 version has to show that dropping a view and running again actually changes the remaining poses, which is true only if the bad views were pulling the alignment.

## 6. [3080] Planned experiments

(a) Token dumps on SysCON3D and one in-domain reconstruction set. (b) Held-out transfer to a second dataset. (c) Wide-span and symmetric subsets, scored separately. (d) Drop-or-split against anchor agreement and against native confidence, one extra forward. (e) π³ with the VGGT probe applied unchanged, and the reverse.

## 7. Conclusion

[After the pilot.]

## References

VGGT; π³; Does VGGT Know What It Does Not Know?; Can These Views Be One Scene?; When Wider Views Fail; ReVar3R; RobustVGGT.
