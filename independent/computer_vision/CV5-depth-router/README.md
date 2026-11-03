# CV5: Let a 25M depth model answer the spatial questions

**Question.** CVPR 2026 kept rewarding a small specialized model that beats a large generalist on the slice the specialist was built for. Spatial questions are that slice, and it has not been measured. Depth Anything V2 Small is 25 million parameters. Paired with an off-the-shelf detector it can answer "which is closer," "is A left of B," and "how many" from boxes and a depth map. A vision-language model answers the same items from the image and the question. The claim is that the two fail on different items. Geometry is right when the depth gap is large, the objects are big enough to segment, and the detector has the category. The vision-language model is right when the question needs a category the detector missed, or when the depth map is flat and there is nothing for geometry to say. A router that is a threshold on depth gap, object size and detector confidence, fit on a few hundred labeled items and then frozen, should beat either system alone on the rest. No vision-language training.

**Status.** Not yet run. Laptop pilot written and smoke-tested (`code/cv5_router.py --smoke`). The pilot draws the two systems from the complementary rule above, so a pass checks that a threshold fit on 400 items recovers the split on held-out items. It does not show that a real depth model and a real vision-language model split this way. That is the 3080 experiment.

Laptop pilot (CPU, numpy). Each item has a depth gap, a size, an occlusion and a flag for "the detector has no category for this question." Detector confidence is high only when that flag is off. Geometry is usually right on the easy slice and usually wrong elsewhere. The vision-language model is usually right on semantic questions and on flat depth, and otherwise near 0.4. The router sees depth gap, size and detector confidence. It does not see the semantic flag except through confidence.

Paper version on the 3080, about 20 GPU-hours of successful forwards. Depth Anything V2 Small and a detector over the images is under two hours. The vision-language forwards are What'sUp (820 images), a spatial benchmark of a few thousand questions, and a counting slice, times two prompts, times Qwen3-VL-2B, Qwen3-VL-4B and one larger 4-bit model that fits the card. The router is fit on a few hundred labeled items. The depth features are logged once, so the bins (gap, size, occlusion) do not cost extra forwards.

Prior work: Depth Anything V2 for the frozen depth model; What'sUp (Kamath et al.) for the spatial failure of vision-language models; DrivePI and the CVPR 2026 pattern of a small specialist beating a generalist, which is the shape of the result rather than a method to extend; BenchDepth and FoundationGeo for depth evaluation, which this paper does not redo. SpatialVLM and its successors train spatial models. This paper trains nothing. It routes.

## Pre-registered kill test (written before the pilot ran)

Design. 4,000 test items, 400 validation items, 8 seeds. The router is a grid search over three thresholds, fit on the validation items to maximize accuracy, then frozen. An oracle router, fit on the test items, is computed only to measure how much the small validation fit leaves behind. It is not a system under test.

- **K1 (the errors are complementary, and they sort by depth gap).** The fraction of test items where exactly one system is correct is ≥ 0.30, and the median depth gap among items only geometry got right exceeds the median depth gap among items only the vision-language model got right by ≥ 0.15. This must hold in ≥ 6 of 8 seeds.
- **K2 (the frozen router wins).** Router accuracy exceeds the better of the two systems by ≥ 0.05, in ≥ 6 of 8 seeds.
- **K3 (a few hundred labels are enough).** The validation-fit router keeps ≥ 80% of the gain the test-set oracle threshold would have had, in ≥ 6 of 8 seeds.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported. If K3 fails, the split is real but the router needs more labels than the paper claims, and the paper changes shape.

*Smoke-test change (before the pilot, 2 seeds).* The first draft made the geometry-easy region narrow: large gap, large object, low occlusion. Geometry was then right on under a fifth of items, and the router's gain sat at +0.045 against a bar of 0.05, with the oracle threshold no higher. The easy region is now a depth gap above 0.30, size above 0.20 and occlusion below 0.80, still the same rule (geometry works when the gap is real, the object is segmentable and the detector saw it). Thresholds unchanged.

What would make the 3080 version a null: the depth model and the vision-language model are wrong on the same items, or the router ties the better of the two. There is then no complementary error to route, and "use a depth model" is a tool tip rather than a paper.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv5_router.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per seed), `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` (after the pilot) |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cv5_router.py --smoke
.venv/bin/python code/cv5_router.py             # pilot, under a minute on CPU
.venv/bin/python code/cv5_router.py --summarize
```
