# CV2: Does outcome-reward RL teach a vision-language model to see, or only to talk?

**Question.** GRPO on VLMs with a verifiable final-answer reward is now standard (Visual-RFT, DEVA and Dr. Seg at CVPR 2026, VisPlay, PeBR-R1). The reward is terminal: a scalar for the answer, after a chain of perception → reasoning → answer. Two recent papers hint that the gradient does not reach perception: Dr. Seg finds perception tasks need fine-grained, stable rewards, and PeBR-R1 finds that a separate perception stage with its own reward beats joint training. Nobody has asked the question directly with a controlled decomposition: of the accuracy gained by RL, how much comes from a better *reasoner* over the same visual features, and how much from better *features*? In RL terms the final-answer reward is maximally delayed relative to the perception that produced it (the "two ways a delayed reward misleads you" stub); the hypothesis is that under a terminal reward the gain is almost entirely absorbed by the module closest to the reward — the language model — and that perception probes barely move even when the vision tower is trainable. If so, "RL improves visual perception" claims need a perception-level reward or are measuring reasoning; and the staged recipes of PeBR-R1 and Dr. Seg have a mechanism. If not — if perception probes move substantially under a terminal reward — the delayed-credit account is wrong for this setting and that is also worth knowing.

**Status.** Laptop pilot run. Verdict: `KILL TEST: KILLED` (`results/verdict.txt`). Perception-only matched or beat reasoner-only in every cell. RLC note: `paper/paper_rlc.md`. No VLM was trained.

Laptop pilot (CPU, numpy, a two-module network — a perception MLP producing a latent and a reasoner MLP producing the answer — pretrained by supervised learning on a synthetic attribute-comparison task, then trained with group-relative policy gradient on the final-answer reward, with each module frozen or trainable; perception measured by a linear probe on the latent, reasoning by accuracy on noise-free inputs). Paper version on the 3080: Qwen3-VL-2B (QLoRA) with GRPO on a perception-heavy VQA set (counting, localisation, attribute binding; e.g. subsets of TallyQA, RefCOCO, CLEVR-style) under four LoRA placements — LM only, connector only, vision tower only, all — and probes for perception (linear probes on vision/connector features for counts, positions, attributes; grounding accuracy) separated from reasoning (accuracy given oracle captions or on text-only reformulations). Plus one arm with a Dr. Seg-style perception-level reward as the positive control.

Prior work: Liu et al. 2025 (Visual-RFT), Das et al. CVPR 2026 (DEVA; GRPO dynamics — diversity, exploration, reward aggregation — not credit), Sun et al. CVPR 2026 (Dr. Seg; perception needs fine-grained rewards), Chen et al. 2025 (PeBR-R1; perception-then-reasoning staging, joint training worse), He et al. CVPR 2026 (VisPlay), Shao et al. 2024 (GRPO). Delayed credit: Arjona-Medina et al. 2019 (RUDDER), Raposo et al. 2021 (synthetic returns), and the lab's delayed-feedback reading list. "RL mostly shapes high-level decision-making over SFT-acquired skills" is argued for terminal agents (Learning Generalizable Behaviors for Terminal Agents, 2026); this is the vision version of that claim with a controlled test. Lab stubs: "Two ways a delayed reward misleads you", "Does RLVR make models rigid?" (adjacent: the entropy measurements here feed it).

## Pre-registered kill test (written before the pilot ran)

Design. Input: three attributes with counts in {0, …, 4}, each encoded as a noisy one-hot block (one-hot + N(0, ν²)), plus 10 standard-normal distractor dimensions and a one-hot question in {largest, smallest, middle}; the answer is the attribute index satisfying the question (C = 3; ties resampled). Model: perception z = tanh(W₁x + b₁) with 32 units; reasoner h = tanh(W₂[z; q] + b₂), logits = W₃h + b₃. Pretraining ("SFT"): 300 Adam steps of cross-entropy, batch 64, on 2 000 labelled examples at ν = 0.3 — enough to be clearly above chance and clearly short of ceiling. RL: group-relative policy gradient (group of 8 sampled answers per example, reward 1[correct], advantage standardised within the group, no KL), 300 Adam steps of batch 64 at learning rate 1e-3, on fresh examples at the RL noise level ν_rl. Arms: **all** (both modules trainable), **reasoner-only** (perception frozen), **perception-only** (reasoner frozen); plus a **labelled-SFT** continuation of the same length as a reference ceiling. Cells: ν_rl ∈ {0.3, 0.6, 0.9} (matched noise, perception-stressed, perception-limited); 8 seeds.

Measurements on 4 000 held-out examples at ν_rl: **accuracy**; **perception probe** = mean held-out R² of a ridge regression from z to the three true counts; **reasoning accuracy** = accuracy on noise-free inputs (ν = 0), where perception is trivial; **answer entropy**. Δ_arm = accuracy(arm) − accuracy(SFT model).

- **K1 (RL helps under a terminal reward).** Median Δ_all ≥ 0.05 at ν_rl = 0.3 and at ν_rl = 0.6.
- **K2 (the gain lives in the reasoner).** In every cell with median Δ_all ≥ 0.03: median Δ_reasoner-only ≥ 0.7·Δ_all and median Δ_perception-only ≤ 0.3·Δ_all.
- **K3 (perception does not move).** In the **all** arm, the perception probe R² changes by less than 0.05 absolute from the SFT model in every cell (median over seeds), while the labelled-SFT reference raises it by ≥ 0.05 in at least the ν_rl = 0.6 cell (so the probe can move when the signal is there).

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported as the mechanism; if K2 holds and K3 fails (perception moves but does not help accuracy), the story becomes "perception moves but reasoning cannot use it", and the paper changes shape. We also report ν_rl = 0.9: the prediction is that Δ_all is small there — RL cannot repair perception it is not rewarding — and this is the condition the positive-control arm (perception-level reward, 3080) is meant to rescue.

What would make this a null: Δ_perception-only comparable to Δ_reasoner-only, or the probe moving as much under RL as under labelled SFT. Then a terminal reward does reach perception in this architecture and the delayed-credit framing does not apply; the 3080 run would be the place to see whether a 2B VLM behaves differently.

*Note on the toy.* A two-layer tanh network has clean, full gradients to the perception module; a 2B VLM with QLoRA and a frozen or lightly adapted vision tower does not. The toy is a best case for the "RL reaches perception" side, so a K2 pass here is informative and a K2 fail here is not decisive about the VLM.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv2_credit.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per cell × arm × seed), `summary.csv` (per cell × arm), `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` (after the pilot) |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cv2_credit.py --smoke
.venv/bin/python code/cv2_credit.py             # pilot, ~10 min CPU
.venv/bin/python code/cv2_credit.py --summarize
```
