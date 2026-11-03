# Sees or Talks? Where the Gain From Outcome-Reward RL on Vision-Language Models Lives

*Draft skeleton, written before the pilot. Numbers in [brackets] are to be filled from `results/`; sections marked [3080] are the paper-scale experiments.*

**Abstract.** Reinforcement learning with a verifiable final-answer reward improves vision-language models on perception-heavy benchmarks, and the improvement is routinely described as better visual perception. We ask where the gain actually lives. The reward is terminal — one scalar after perception, reasoning and answer — so from the perception module's point of view it is maximally delayed, and credit-assignment theory predicts that the module nearest the reward absorbs the improvement. We test this with a controlled decomposition: freeze perception and train the reasoner, freeze the reasoner and train perception, or train both, under the same group-relative policy gradient, and measure perception with linear probes on the visual features and reasoning with accuracy on inputs where perception is trivial. In a two-module network the gain from training both is [X points], of which [Y %] is reproduced by training the reasoner alone and [Z %] by training perception alone; the perception probe moves by [Δ] under RL against [Δ′] under labelled supervision of the same length. [3080] With Qwen3-VL-2B and LoRA placed on the language model, the connector, the vision tower or all three, GRPO on counting, localisation and attribute-binding data yields [results], with perception probes [moving/not moving], while a perception-level reward (Dr. Seg style) [does/does not] move them. The conclusion for practice: a terminal reward [does not] teach a VLM to see; it teaches the language model to use what the vision tower already provides, which is why staged recipes with perception-specific rewards outperform joint training.

## 1. Introduction

Visual-RFT, DEVA, Dr. Seg, VisPlay and PeBR-R1 all apply GRPO to VLMs with rewards on the final answer and report gains on detection, grounding, counting and VQA. Two of them notice something about perception specifically: Dr. Seg finds that segmentation needs fine-grained, stable rewards, and PeBR-R1 finds that a dedicated perception stage with its own reward beats joint training. Neither isolates the mechanism. We frame it as credit assignment. In the chain image → features → reasoning → answer, a terminal reward is a delayed reward for the features, and the standard failure of delayed rewards is that the gradient is absorbed by whatever is closest to the reward and can reduce the loss fastest. For a VLM that is the language model.

The controlled test is to freeze modules. If freezing perception costs almost none of the RL gain, and the perception probes do not move when perception is trainable, then "RL improves perception" is a misattribution and the perception-specific rewards of Dr. Seg and PeBR-R1 are not a refinement but a necessity. If perception moves and matters, the credit-assignment account is wrong here, which is also a result.

Contributions. (1) The decomposition protocol: freeze-and-train arms with perception and reasoning probes measured separately [pilot; 3080]. (2) The finding on the share of the RL gain attributable to each module and on probe movement [pilot; 3080]. (3) [3080] The positive control: a perception-level reward that moves the probes, placed against the terminal reward that does not. (4) The reading of PeBR-R1 and Dr. Seg as consequences.

## 2. Related work

GRPO on VLMs: Liu et al. 2025 (Visual-RFT), Das et al. 2026 (DEVA), Sun et al. 2026 (Dr. Seg), He et al. 2026 (VisPlay), Chen et al. 2025 (PeBR-R1). Credit assignment under delayed reward: Arjona-Medina et al. 2019, Raposo et al. 2021, Hung et al. 2019. "RL shapes high-level behaviour over SFT-acquired skills": Learning Generalizable Behaviors for Terminal Agents (2026). Probing vision features: Alain & Bengio 2017; probing VLM connectors for spatial and count information (recent). Lab stubs: two ways a delayed reward misleads you; does RLVR make models rigid.

## 3. Setting

**Pilot.** Three attributes with counts in {0..4}, each a noisy one-hot block plus distractors and a question token (largest / smallest / middle). Perception z = tanh(W₁x + b₁), 32 units; reasoner two layers over [z; q]. SFT: 300 Adam steps at ν = 0.3. RL: group-relative policy gradient (group 8, reward 1[correct], standardised advantages, no KL), 300 steps at ν_rl ∈ {0.3, 0.6, 0.9}. Arms: all, reasoner-only, perception-only, labelled-SFT reference. Probe: ridge regression z → counts, held-out R². Reasoning accuracy at ν = 0. 8 seeds.

**[3080] Paper setting.** Qwen3-VL-2B-Instruct, QLoRA, GRPO via TRL; training data: counting (TallyQA subset), referring localisation (RefCOCO with box IoU reward), attribute binding (CLEVR-style synthetic); LoRA placement arms: LM only, connector only, vision tower only, all. Probes: linear probes on connector outputs for count / position / colour; grounding accuracy; text-only reformulation accuracy (reasoning given oracle captions). Positive control: Dr. Seg-style fine-grained perception reward.

Pre-registered (see README): K1 Δ_all ≥ 0.05 at ν_rl ∈ {0.3, 0.6}; K2 Δ_reasoner-only ≥ 0.7·Δ_all and Δ_perception-only ≤ 0.3·Δ_all wherever Δ_all ≥ 0.03; K3 |Δ probe| < 0.05 under RL while labelled SFT moves it ≥ 0.05 at ν_rl = 0.6. SUPPORTED if K1 ∧ K2.

## 4. Results

[pilot] Table of accuracy, Δ accuracy, probe R², Δ probe, reasoning accuracy and entropy per cell and arm. The share of the gain reproduced by each single-module arm. The ν_rl = 0.9 cell where perception is the bottleneck.

[3080] The same table over LoRA placements and tasks; probe curves over training; the positive-control arm.

## 5. Discussion

If K2 and K3 hold: a terminal reward reaches the reasoner and stops there, in the best case for gradient flow (a small fully-differentiable network). The VLM case has worse gradient flow to perception (frozen or lightly adapted tower, QLoRA), so the effect should be at least as strong. The practical reading is that perception gains require perception rewards, which is what Dr. Seg and PeBR-R1 found empirically. If K2 fails in the toy — perception-only training recovers much of the gain — then a terminal reward can teach features when the gradient reaches them, and the question for the VLM is purely whether it does.

Limitations. The toy's reasoning is shallow; the probe is linear; the RL budget is matched by steps, not by samples. In the VLM the perception/reasoning boundary is blurred by the connector and the measurement relies on probes whose own capacity must be controlled.

## 6. [3080] Planned experiments

(a) Four LoRA placements × three tasks, GRPO with TRL on Qwen3-VL-2B; (b) perception probes before/after; (c) text-only reformulations as a reasoning measurement; (d) the perception-level reward as positive control; (e) entropy and diversity over training, feeding the RLVR-rigidity stub.

## 7. Conclusion

[After the pilot.]

## References

Liu et al. 2025; Das et al. 2026; Sun et al. 2026; He et al. 2026; Chen et al. 2025; Shao et al. 2024; Arjona-Medina et al. 2019; Raposo et al. 2021; Hung et al. 2019; Alain & Bengio 2017.
