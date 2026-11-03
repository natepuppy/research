# CV1: A wrong first crop is a dead-end — irrecoverability in "thinking with images"

**Question.** Vision-language models trained to zoom (DeepEyes, ICLR 2026; MGPO; ACTIVE-o3; VIS-GRPO, ECCV 2026) issue a sequence of crops before answering. Every crop conditions all later reasoning on what it shows, and the budget of crops is small (DeepEyes allows six). If the first crop misses the relevant region, can the model recover? VIS-GRPO reports that GRPO-trained zoom policies collapse to zoom-averse behaviour because multi-zoom rollouts have lower mean reward early in training, and treats this as an advantage-normalisation problem. We propose a different reading: a bad crop is an *irrecoverable state* in the Fatemi & Killian (2021) sense — a prefix from which the probability of a correct answer under the policy's own continuation is near zero, while budget remains. If most zoom failures pass through such a state early, then (i) the low mean reward of multi-zoom rollouts is the dead-end rate, not an artefact of normalisation, (ii) a dead-end detector trained on prefix features can predict failure before the budget is spent, and (iii) the cheapest intervention — re-crop from the original image when the detector fires — should recover a large share of the oracle gain without retraining the VLM. The RL content is the irrecoverability estimate P(success | prefix) from many continuations, the detector, and the selective intervention; the vision content is that the prefix features are frozen vision features plus crop geometry, and the benchmarks are V* Bench, HR-Bench and MME-RealWorld.

**Status.** Not yet run. Laptop pilot designed and smoke-tested (`code/cv1_zoom.py --smoke`); the pre-registered kill test below is written and the thresholds are fixed. Pilot first, then `notes/pilot_notes.md`, then the 3080 version.

Laptop pilot (CPU, numpy, a synthetic zoom grid with a fixed stochastic base policy; the irrecoverability of every prefix is estimated from M = 32 Monte-Carlo continuations exactly as it would be with a VLM, and checked against a 256-continuation reference). Paper version on the 3080: DeepEyes-7B (or MGPO) at 4-bit as the fixed policy on V* Bench (191 items), HR-Bench 4K/8K and MME-RealWorld-Lite; 16 continuations per prefix for the irrecoverability label; detector on frozen DINOv3/SigLIP features of the current crop plus crop geometry and the question embedding; the gated re-crop intervention evaluated for accuracy and crop count; then, as the follow-up, the dead-end estimate as a pessimism/shaping term inside GRPO on Qwen3-VL-2B.

Prior work: Fatemi, Killian et al. 2021 (medical dead-ends; irrecoverability as a value of a fail-reward MDP), Killian et al. 2023 (dead-end discovery with uncertainty); DeepEyes (Zheng et al. 2025/ICLR 2026; end-to-end GRPO zoom with observation-token masking), MGPO (Huang et al. 2025; multi-turn grounding, cold-start problem for triggering zooms), ACTIVE-o3 (Zhu et al. 2025; active perception as region selection under GRPO), VIS-GRPO (Zhu et al. ECCV 2026; "visual laziness", stratified advantages by zoom depth and intent), TTRV (Singh et al. CVPR 2026; test-time RL, adjacent only). Early-exit / abstention literature for VLMs is adjacent but predicts the *answer* being wrong, not the *prefix* being unrecoverable. Lab stubs: the dead-ends cluster (`research-stubs/by-area.md`), especially "Which unseen actions are actually dangerous?" and "Backtracking from a crash to the last useful intervention", whose intervention-point question is exactly the re-crop point here.

## Pre-registered kill test (written before the pilot ran)

Design. Image = G × G grid of cells, G = 8; one target cell holds a symbol from C = 10 classes; the question is "what symbol is in the target cell". A view is a square sub-grid; the agent starts at the full image and may **zoom** into one of the four quadrants of the current view (side halves), **re-crop** (return to the full image, costs one action), or **answer**. The symbol can only be read once the view side is ≤ R = 2 and the target is inside the view (depth 2, after two zooms); it is then read correctly with probability 0.95. Total action budget B = 5, so exactly one re-crop-and-retry fits in the budget. Observation: for each quadrant of the current view a saliency cue = 1[target in quadrant] + N(0, σ_d²), with σ_d = σ · 0.7^d at depth d (closer views give clearer cues). Base policy: zoom into the quadrant with the largest cue (softmax at temperature 0.3 over cues); at a readable view, answer if the target is visible, otherwise re-crop with probability ρ else guess uniformly (chance = 0.10); guess uniformly when the budget runs out. Grid: σ ∈ {0.5, 1.0, 1.5} × ρ ∈ {0.0, 0.3}; 8 seeds; 1 000 training episodes, 500 held-out episodes for labels, 2 000 paired evaluation episodes per cell.

Irrecoverability. A *prefix* is a decision point at a non-readable view (depth 0 or 1) with budget left ≥ 3 — the points where a re-crop can still be followed by the two zooms needed to read, i.e. where an intervention is still possible. For every prefix, P̂(success | prefix) is the fraction of M = 32 Monte-Carlo continuations under the base policy that answer correctly (this is the estimator the 3080 version uses). A prefix is a **dead-end** if P̂ ≤ δ = 0.15 (chance is 0.10). Held-out prefixes also get a 256-continuation reference P_ref, which is used for labels in the evaluation.

Detector. Logistic regression on observable prefix features only: depth, budget left, number of re-crops, the max cue and the top-1 − top-2 cue margin at the current view, and the margin and chosen-quadrant cue at each previous zoom decision (padded). Trained on the training episodes' prefixes against the M = 32 label; evaluated on held-out prefixes against the P_ref label. Baseline detector: the current-step cue margin alone (what a one-step confidence heuristic would use).

Intervention. At a prefix at depth ≥ 1 (budget ≥ 3), if the detector's P(dead-end) > 0.5, re-crop instead of the base action; otherwise act as the base policy. The decision at a readable view where the target is not visible is left to the base policy in every arm, so the comparison isolates "predict the dead-end before you can see it". Comparators, on the same 2 000 initial states: base policy; a cue heuristic that re-crops when the current margin is below the training-set median; an oracle that re-crops whenever the target is not in the current view (the state is irrecoverable without a re-crop).

- **K1 (dead-ends are real and early).** Over the ρ = 0 cells: ≥ 50 % of failed base-policy episodes pass through a dead-end prefix, and the median depth of the first dead-end is ≤ 1 (the first crop). In the toy this is close to true by construction once ρ = 0 — the pilot's job for K1 is to confirm the estimator and pipeline, and the 3080 version is the real test of K1.
- **K2 (detectable from observables).** Held-out AUROC of the detector ≥ 0.80 in all three ρ = 0 cells, and ≥ 0.05 above the single-step margin baseline in those cells.
- **K3 (actionable).** Detector-gated re-crop raises success rate by ≥ 5 points absolute over the base policy, recovers ≥ 50 % of the oracle's gain, and uses ≤ 1 more action per episode than the base policy, in ≥ 2 of the 3 ρ = 0 cells.

SUPPORTED if K1 and K3. INCONCLUSIVE if exactly one. KILLED if neither. K2 is reported (it is the mechanism K3 depends on, and the one most likely to fail: if the current-step margin already carries all the signal, the dead-end framing adds nothing over a confidence heuristic). The ρ = 0.3 cells are reported as the robustness condition: a policy that already re-crops has fewer dead-ends by construction and the gain of the intervention must shrink accordingly.

What would make this a null: failures dominated by misreads at the final step (no early dead-end), or K2 failing on the margin comparison. Either way the paper is not worth writing in this form.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv1_zoom.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per cell × seed), `summary.csv` (per cell), `prefixes.csv` (held-out prefix sample with P̂, exact P, detector score), `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` (after the pilot) |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cv1_zoom.py --smoke
.venv/bin/python code/cv1_zoom.py             # pilot, ~10 min CPU
.venv/bin/python code/cv1_zoom.py --summarize
```
