# CV6: Which views the reconstruction knows it misplaced

**Question.** Feed-forward reconstructors such as VGGT and π³ report a confidence per point and no confidence per camera. Two recent papers describe the failures that confidence misses. On hard image pairs, a linear probe on frozen mid-aggregator features predicts pose failure far better than VGGT's own confidence, but the probe is for pairs. On view sets, the same models fabricate a shared scene for images that were never one scene, and at wide camera spans they emit a second, displaced copy of geometry that really is there. The claim is that the N-view model already carries both failures in its fused tokens. A score read from those tokens, on a domain it was not fit on, flags intruder views and flags displaced-duplicate views. Anchor agreement, the score RobustVGGT uses, flags only the intruders, because a duplicate still looks like the scene. Dropping flagged intruders, and splitting flagged duplicates into their own component, then running the model once more, should remove most of the pose error. No retraining of the reconstructor.

**Status.** Not yet run. Laptop pilot written and smoke-tested (`code/cv6_scene.py --smoke`). The pilot draws the fused-token residual from the rule above, so a pass checks that the probe beats anchor agreement on the intruder slice, that anchor agreement fails on the duplicate slice, and that the drop-or-split rule recovers pose error. It does not show that a real VGGT or π³ token carries this residual. That is the 3080 experiment.

Laptop pilot (CPU, numpy). Sets of 12 views. Three families, named for ScanNet, DL3DV and Tanks and Temples, with different noise; each seed holds one family out and fits the probe on the other two. A third of sets are clean, a third contain 2-4 intruders, a third contain 4 genuine views placed a second time. Intruders disagree in appearance. Duplicates agree in appearance and have high confidence. Both have a high fused residual. The repair flags views by a within-set z-score of the probe, threshold fit on the training families. Flagged views below an anchor agreement of 0.62 are dropped. Flagged views at or above it are reconstructed as a second component. One extra forward.

Paper version on the 3080, about 25-40 GPU-hours. VGGT-1B and π³ at 8-16 views, which fits a 10 GB card. SysCON3D for mixed-scene inputs, ScanNet and DL3DV or Tanks and Temples for held-out transfer, and wide-span subsets for the displaced-copy failure. Ground truth is a relative rotation above 15° after a similarity alignment. The probe is fit on dumped tokens and evaluated on a dataset that supplied none of its training views.

Prior work: "Does VGGT Know What It Does Not Know?" for the pair probe this paper extends to view sets; "Can These Views Be One Scene?" for fabricated geometry on unrelated inputs, whose proposed remedy is COLMAP; "When Wider Views Fail" for displaced copies at wide spans; ReVar3R for per-point variance after gauge alignment, which is a different question; RobustVGGT for the anchor-agreement score used here as the baseline.

## Pre-registered kill test (written before the pilot ran)

Design. 80 sets of each kind in each family, 8 seeds. Intruder detection is scored on clean and intruder sets. Duplicate detection is scored on clean and duplicate sets. The two slices stay separate so a duplicate is not counted as a failed intruder call. Pose error is the mean over genuine views: 3° in a pure component, 18° in a mixed one, 40° if the view was dropped and never reconstructed. Reduction is measured on contaminated sets only.

- **K1 (intruders, held-out family).** Probe AUROC ≥ 0.85, and ≥ 0.10 above anchor agreement. This must hold in ≥ 6 of 8 seeds.
- **K2 (duplicates, held-out family).** Probe AUROC ≥ 0.80. Anchor agreement and confidence are reported beside it. This must hold in ≥ 6 of 8 seeds.
- **K3 (the repair).** Mean pose-error reduction ≥ 40% relative to the unrepaired set, at ≤ 2 forwards, in ≥ 6 of 8 seeds.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported. If K3 fails, the tokens know the failure and the repair rule is the wrong consumer of that knowledge.

What would make the 3080 version a null: the fused tokens do not beat anchor agreement on intruders, or the duplicate and wide-span failures are invisible in the tokens at every layer. Intruder detection alone is a short note on top of RobustVGGT. A duplicate the model is confidently wrong about, with nothing in the token to show it, ends the claim.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv6_scene.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per seed), `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` (after the pilot) |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cv6_scene.py --smoke
.venv/bin/python code/cv6_scene.py             # pilot, under a minute on CPU
.venv/bin/python code/cv6_scene.py --summarize
```
