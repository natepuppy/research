# A Wrong First Crop Is a Dead-End: Irrecoverability in Vision-Language Models That Think With Images

*Draft skeleton, written before the pilot. Numbers in [brackets] are to be filled from `results/`; sections marked [3080] are the paper-scale experiments.*

**Abstract.** Vision-language models trained to "think with images" issue a short sequence of crops before answering, and recent work (VIS-GRPO) reports that GRPO-trained zoom policies collapse to zoom-averse behaviour because multi-zoom rollouts earn lower mean reward. We give a different account of the same observation. Each crop conditions every later step on what it shows, the crop budget is small, and a crop that misses the relevant region leaves the model in a state from which, under its own continuation, the probability of a correct answer is near chance while budget remains — an irrecoverable state in the sense of medical dead-end discovery. We estimate P(success | prefix) from Monte-Carlo continuations of a fixed zoom policy and find that [X %] of failed trajectories pass through such a state, most at the first crop. A logistic detector on observable prefix features — crop geometry, saliency margins, and the history of previous decisions — identifies these states with AUROC [0.xx], above a single-step confidence heuristic ([0.xx]). Gating a single re-crop on the detector recovers [Y %] of the oracle intervention's gain at a cost of [Z] extra crops per question, without retraining the model. [3080] On V* Bench, HR-Bench and MME-RealWorld with DeepEyes-7B as the fixed policy, the same pipeline with frozen DINOv3 crop features yields [+A points] at [+B crops]. The result reframes visual laziness as a dead-end rate rather than a normalisation artefact, and suggests the irrecoverability estimate as a pessimism term for GRPO on zoom policies.

## 1. Introduction

Active perception in VLMs is now trained end-to-end with outcome rewards: DeepEyes, MGPO and ACTIVE-o3 let the model emit bounding boxes, receive the crops as new observation tokens, and optimise the whole trajectory with GRPO. Two facts about these trajectories matter here. The crop budget is small (six in DeepEyes), and every crop is appended to the context, so the model's subsequent reasoning is conditioned on it. A crop that misses the evidence is therefore not a neutral event: it consumes budget, it anchors the chain of thought on the wrong region, and recovery requires the model to both notice the miss and spend further budget undoing it.

Fatemi, Killian et al. (2021) formalise states like this as dead-ends: states from which, under the agent's own behaviour, failure is (nearly) certain. Their insight is that dead-ends are easier to identify than optimal actions and that the useful intervention is to avoid entering them. We apply the same lens to zoom trajectories. The pre-registered claims are that most failures pass through an early dead-end, that the dead-end is predictable from observable features before the budget is spent, and that a single detector-gated re-crop recovers a large share of what an oracle would.

Contributions. (1) The irrecoverability measurement on zoom trajectories, with the dead-end rate and the depth at which dead-ends occur [pilot; 3080]. (2) A dead-end detector on observable prefix features, compared against a single-step confidence heuristic [pilot; 3080]. (3) The gated re-crop intervention, with accuracy and crop-count costs against base, heuristic and oracle arms [pilot; 3080]. (4) [3080] The dead-end estimate as a shaping/pessimism term in GRPO for a 2B zoom policy, as a direct test of the VIS-GRPO account.

## 2. Related work

Dead-ends: Fatemi et al. 2021 (medical dead-ends, the fail-reward MDP and the two value functions), Killian et al. 2023 (dead-end discovery under uncertainty). Thinking with images: DeepEyes (Zheng et al.), MGPO (Huang et al.), ACTIVE-o3 (Zhu et al.), VIS-GRPO (Zhu et al., ECCV 2026) — the last identifies the collapse we reinterpret. Perception-oriented GRPO: Dr. Seg and DEVA (CVPR 2026). Test-time adaptation of VLMs: TTRV (CVPR 2026), adjacent. Abstention and early exit predict the *answer* being wrong; we predict the *prefix* being unrecoverable, which is earlier and actionable.

## 3. Setting

**Pilot.** A G × G grid with a target cell holding one of C symbols; views halve by quadrant; the symbol is readable at depth 2; budget B = 5; quadrant saliency cues with noise σ_d = σ·0.7^d; a fixed softmax base policy; re-crop probability ρ at an unreadable view. Prefix = a decision point at depth ≤ 1 with budget ≥ 3. P̂(success | prefix) from 32 continuations; dead-end if P̂ ≤ 0.15 (chance 0.10); a 256-continuation reference for evaluation. Detector: logistic regression on depth, budget, re-crop count, max cue, cue margin, and the margin and chosen-cue history. Arms: base, margin heuristic, detector-gated re-crop, oracle re-crop (target not in view). Grid σ ∈ {0.5, 1.0, 1.5} × ρ ∈ {0, 0.3}, 8 seeds.

**[3080] Paper setting.** DeepEyes-7B (4-bit) as the fixed policy. Per question, roll out N trajectories; at every prefix, 16 continuations give P̂; dead-end if P̂ ≤ δ with budget remaining. Detector features: DINOv3 CLS of the current crop, crop geometry (area ratio, centre offset, overlap with earlier crops), the question embedding, and the previous crops' features. Intervention: when the detector fires, discard the last crop from the context and re-prompt for a new region. Benchmarks: V* Bench, HR-Bench 4K/8K, MME-RealWorld-Lite.

Pre-registered (see README): K1 ≥ 50 % of failures through a dead-end with median first depth ≤ 1 (ρ = 0); K2 detector AUROC ≥ 0.80 and ≥ 0.05 above the margin heuristic in all ρ = 0 cells; K3 gated gain ≥ 5 points, ≥ half the oracle gain, ≤ 1 extra action, in ≥ 2 of 3 ρ = 0 cells. SUPPORTED if K1 ∧ K3.

## 4. Results

[pilot] Dead-end rate and first-dead-end depth per cell. Detector vs margin AUROC. Base / heuristic / gated / oracle success and action counts. The ρ = 0.3 cells as the robustness condition.

[3080] Same table on the three benchmarks; a figure of P̂ versus crop index for failed and successful trajectories; accuracy-vs-crops curves for the four arms.

## 5. Discussion

If K1 and K3 hold, "visual laziness" has a simple mechanism: multi-zoom rollouts have lower mean reward because a fixed fraction of them enter a dead-end at the first crop, and GRPO correctly learns that zooming is risky under the current crop-selection policy. The fix VIS-GRPO proposes (stratified advantages) treats the symptom; the dead-end estimate addresses the cause, either at inference (this paper) or as a pessimism term in training (follow-up). If K2 fails on the margin comparison, the honest conclusion is that a one-step confidence heuristic is enough and the dead-end framing is decoration.

Limitations. The pilot's dead-ends are partly by construction (ρ = 0); the real test of K1 is on the VLM. The continuation estimate is under the model's own policy, so a stronger model has fewer dead-ends and the intervention's headroom shrinks. Re-crop at inference costs tokens; the accounting must include the detector's false positives.

## 6. [3080] Planned experiments

(a) Rollout collection and P̂ on V*, HR-Bench, MME-RealWorld with DeepEyes-7B; (b) detector training on frozen features, with the single-step confidence and the model's own verbalised confidence as baselines; (c) the gated re-crop intervention; (d) the dead-end estimate as a shaping term in GRPO on Qwen3-VL-2B on a V*-style training set, against plain GRPO and VIS-GRPO.

## 7. Conclusion

[After the pilot.]

## References

Fatemi, Killian et al. 2021; Killian et al. 2023; Zheng et al. 2025 (DeepEyes); Huang et al. 2025 (MGPO); Zhu et al. 2025 (ACTIVE-o3); Zhu et al. 2026 (VIS-GRPO); Sun et al. 2026 (Dr. Seg); Das et al. 2026 (DEVA); Singh et al. 2026 (TTRV); Shao et al. 2024 (GRPO).
