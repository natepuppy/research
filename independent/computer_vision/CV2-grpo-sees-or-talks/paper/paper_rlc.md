# A Terminal Reward Reaches Both Modules in a Two-Layer Network

*RLC-length version of the synthetic pilot. Every number is from the summary printed by `code/cv2_credit.py` and from `results/verdict.txt`. Eight seeds. No vision-language model was run.*

## Abstract

Group-relative policy optimisation with a verifiable final answer is described as improving visual perception. The reward is terminal, so credit-assignment says the module nearest the reward should absorb the gain. A two-module network — a perception MLP and a reasoner MLP, pretrained by supervised learning, then trained with a group-relative policy gradient on the answer — was the pre-registered test. The rule is KILLED. At perception noise 0.3 the joint update gains +0.091 accuracy, the reasoner alone +0.042, and perception alone +0.053. At noise 0.6 the gains are +0.039, +0.019 and +0.030, so the joint gain misses 0.05 and perception-alone is the larger of the two frozen arms. At noise 0.9 the gains are +0.046, +0.026 and +0.032. The perception probe moves by at most +0.020 under the joint update. The labelled continuation, which was supposed to prove the probe can move, moves it by at most +0.019. In this architecture a terminal reward is not absorbed by the reasoner.

## 1 The question

Visual-RFT, DEVA, Dr. Seg, VisPlay and PeBR-R1 train vision-language models with a reward that arrives after perception, reasoning and the answer. Dr. Seg and PeBR-R1 both find that perception needs its own reward. The controlled version of that claim freezes one module at a time. If the joint gain is reproduced by training the reasoner and not by training perception, and if a probe on the visual features does not move, the terminal reward never reached perception.

The network used here is a best case for the opposite conclusion. Both modules are small tanh layers and the gradient to perception is dense. A failure of the claim in this network is evidence against the claim in the network. It is not evidence about a 2B model with a frozen vision tower. That model was not trained.

## 2 Design

Three attributes have counts in \(\{0,\ldots,4\}\), each a one-hot plus noise of scale \(\nu\), with 10 distractor dimensions and a question in \{largest, smallest, middle\}. Perception is a 32-unit tanh layer. The reasoner reads the latent and the question. Supervised pretraining is 300 Adam steps at \(\nu = 0.3\). Reinforcement learning is a group of 8 sampled answers, reward 1 if correct, advantages standardised inside the group, no KL penalty, 300 Adam steps at learning rate \(10^{-3}\). Arms: both modules, reasoner only, perception only, and a labelled continuation of the same length. Noise during RL is 0.3, 0.6 or 0.9. Eight seeds. Accuracy, a ridge probe from the latent to the three counts, and accuracy on noise-free inputs are measured on 4,000 held-out examples.

K1 requires the joint gain to be at least 0.05 at noise 0.3 and at noise 0.6. K2 requires, wherever the joint gain is at least 0.03, the reasoner-only gain to be at least 70% of it and the perception-only gain at most 30% of it. Supported if both. Killed if neither.

## 3 Result

| noise | SFT acc. | joint \(\Delta\) | reasoner \(\Delta\) | perception \(\Delta\) | probe \(\Delta\), joint | probe \(\Delta\), labelled |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.716 | +0.091 | +0.042 | +0.053 | +0.020 | +0.019 |
| 0.6 | 0.561 | +0.039 | +0.019 | +0.030 | +0.004 | +0.005 |
| 0.9 | 0.442 | +0.046 | +0.026 | +0.032 | +0.001 | +0.002 |

*Table 1. Medians over 8 seeds. \(\Delta\) is against the supervised model.*

K1 fails because the joint gain at noise 0.6 is +0.039. K2 fails in every cell: perception-only is larger than reasoner-only, and it is more than 30% of the joint gain. K3, the mechanism clause, also fails, for a reason that cuts both ways. The probe barely moves under the joint update, which was the prediction, and it also barely moves under labelled training, so the probe was not shown to be capable of moving. The recorded verdict is KILLED.

Answer entropy does drop under the joint update, from 0.934 to 0.565 at noise 0.3, and less so when only perception is trained (0.882). The policy becomes more peaked. That is not a perception gain. Noise-free reasoning accuracy rises under every arm, including perception-only (+0.077 at noise 0.3), so the frozen-module split does not isolate the two skills as cleanly as the design assumed.

## 4 Limitations

The positive control did not move the probe, so "perception did not move under RL" is not identified. The two frozen arms are close to each other and both are below the labelled continuation (+0.110, +0.069, +0.067). A terminal reward helps, and it helps whichever module is left trainable. That is the opposite of the delayed-credit prediction in this network, which is the network where the prediction was easiest to satisfy. Qwen3-VL with LoRA on the language model, the connector, or the vision tower was not run. The README's own note applies: a failure here is not a measurement of a 2B model, and a pass here would have been.

## References

Shao et al. (2024). GRPO. · Liu et al. (2025). Visual-RFT. · Das et al. (CVPR 2026). DEVA. · Sun et al. (CVPR 2026). Dr. Seg. · Chen et al. (2025). PeBR-R1. · He et al. (CVPR 2026). VisPlay. · Arjona-Medina et al. (2019). RUDDER.
