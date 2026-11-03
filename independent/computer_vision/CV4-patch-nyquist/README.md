# CV4: One patch is the resolution — a transferable failure law for vision-language models

**Question.** Vision-language models fail at counting and at small-object grounding, and a 2026 workshop paper ("Counting to Four is still a Chore") traced a large part of that failure to the vision tower's patch grid: an object smaller than a patch, or a gap between objects smaller than a patch, is not resolved. That paper stops at the diagnosis. The claim here is a law. Measure an object's size and its gap to the nearest neighbor in units of the tower's own patch. The fail rate crosses one half at one patch, and that crossing is the same number for towers with different patch sizes. Raw pixels do not transfer, because a 16-pixel object is two patches of one tower and half a patch of another. If the law holds, the fix is determined: retile the image so every object is given at least two patches, answer per tile, and merge. A global 2× upsample and a zoom that spends its budget on the largest object both leave sub-patch objects unresolved, and the retile should beat both on that slice.

**Status.** Not yet run. Laptop pilot written and smoke-tested (`code/cv4_patch.py --smoke`). The pilot generates the errors from the law, so a pass checks the comparisons and the pipeline. It cannot falsify the law in a real vision-language model. That is the 3080 experiment.

Laptop pilot (CPU, numpy). Scenes of four objects with log-uniform size and gap. Three towers, patch size 8, 16 and 32, sharing the scenes. Fail probability is a steep sigmoid of the shorter of size and gap, in units of that tower's patch, plus a weak clutter term. Confidence is a clutter score with noise and does not see the grid.

Paper version on the 3080, about 90 GPU-hours of successful forwards. Qwen3-VL-2B, Qwen3-VL-4B and InternVL-2B at 4-bit, plus a 7B model if the card has 12 GB (a 10 GB card does not hold a 7B tower at the resolution this question needs). CountBench (540 images), a 3,000-question small-object slice of RefCOCO, and 8,000 stratified questions from the TallyQA test set, which has 38,589 and does not need to be run in full. One resolution for the main table, a resolution sweep on the 2B model, the retile on the flagged quarter of the 2B and 7B items, and one learned-zoom baseline (DeepEyes) on 2,000 questions. Short answers, image side capped near a thousand visual tokens.

Prior work: "Counting to Four is still a Chore" (the workshop diagnosis this paper turns into a threshold and a fix); VLMCountBench and the binding-problem account (NeurIPS 2024) for the broader counting failure; DeepEyes, MGPO and ACTIVE-o3 for learned zoom, which is the expensive baseline; DatBench and "Seeing without Looking" (CVPR 2026 workshop) for the related fact that benchmark scores move little when visual evidence is removed. The retile is not a learned zoom policy. It is a function of the patch size and a blob map.

## Pre-registered kill test (written before the pilot ran)

Design. 4,000 scenes, 8 seeds, patch sizes {8, 16, 32}, the same scenes for every tower. An object is flagged when the shorter of its size and its gap is below one patch. Interventions, scored only on flagged objects: a global 2× upsample; zoom-the-largest, which rescales only the biggest object in the scene; retile, which rescales every object. Both zoom and retile aim for two patches of clearance, past the decision boundary at one patch. A stop at exactly one patch sits on the sigmoid's midpoint and is not a fix.

- **K1 (the unit is the patch, and it is visible only across towers).** Pooled over towers, the AUROC of min(size, gap) / P for predicting a failure beats the AUROC of min(size, gap) in raw pixels by ≥ 0.05. Within a tower the two rankings are identical, so the gap has to come from pooling. Separately, every tower's patch-unit AUROC is ≥ 0.80 and beats clutter and the confidence score by ≥ 0.10.
- **K2 (one threshold).** The fail rate, as a function of min(size, gap) / P, crosses 0.5 inside [0.75, 1.25] for every tower, and the three crossings lie within 0.25 of each other.
- **K3 (retile beats the two cheaper-looking alternatives).** On flagged objects, retile error is ≤ the 2× upsample and ≤ zoom-the-largest by 0.08, on every tower.

SUPPORTED if K1, K2 and K3. INCONCLUSIVE if exactly two. KILLED otherwise.

*Smoke-test changes (before the pilot, 2 seeds × 600 scenes).* (a) The first draft compared patch units with raw pixels inside each tower. The gap was zero, because with one patch size the two are the same ranking. K1 is now the pooled comparison. (b) The crossing was estimated on objects whose gap was not binding (gap > 2P). At patch 32 that subset was empty and the crossing came back undefined. The crossing is now read from min(size, gap) / P directly. (c) Interventions were scored on every object, and retile stopped at exactly one patch, where the fail probability is still one half. A global 2× then beat the retile on the fine tower. Scoring is now restricted to sub-patch objects, and both retile and zoom aim for two patches of clearance. Thresholds unchanged.

What would make the 3080 version a null: the crossing moves by more than a quarter of a patch from one model family to another, or a global resolution increase captures the retile's gain. Either one means there is no law to publish, and the workshop diagnosis stands.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv4_patch.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per tower × seed), `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` (after the pilot) |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cv4_patch.py --smoke
.venv/bin/python code/cv4_patch.py             # pilot, under a minute on CPU
.venv/bin/python code/cv4_patch.py --summarize
```
