# Decorrelation Is Not Why Replay Works: Minibatch Correlation, Reuse, and the Target Period

*Draft. Pilot numbers are from the laptop run in `results/`; sections marked [3080] are planned.*

## Abstract

Experience replay is justified in two ways: it reuses data, and uniform sampling decorrelates consecutive updates, which is said to reduce update variance and break feedback loops. The two have never been separated, because every replay-versus-no-replay comparison changes both. We hold the buffer, the data and the number of updates fixed in DQN on CartPole and vary only how the 32 transitions of a minibatch are chosen: 32 independent draws, 4 contiguous windows of 8, one contiguous window of 32, or the last 32 transitions written (no reuse). Crossing this with the target period, we find that within-batch correlation costs under 10 AUC points at either period and is not significant (4/16 paired wins for the fully contiguous batch at both periods), even though it measurably degrades the minibatch gradient's alignment with the buffer gradient. The large effect is the opposite of the textbook's: with a fast target (period 100), training on the most recent 32 transitions beats uniform replay by 40 AUC points and 69 final-return points (15/16 and 14/16 wins). The replay learners reach the ceiling at 5k steps and collapse; the on-policy learner does not. With a slow target (period 2500) replay wins on AUC through reuse and the two finish equal. Replay's value here is reuse, it is contingent on the target period, and decorrelation contributes nothing detectable. [3080] MinAtar with the same dial, Polyak targets, a replay-ratio sweep and a terminal-reservoir control are planned to separate stale-target from forgotten-failure explanations of the collapse.

## 1. Introduction

Lin (1992) proposed replay for reuse. Mnih et al. (2015) added the decorrelation argument: consecutive samples are strongly correlated, so learning from them directly is inefficient, and sampling randomly from the buffer "breaks these correlations and therefore reduces the variance of the updates". The claim is intuitive, is repeated in every textbook treatment, and is testable: hold everything about the buffer fixed and change only the correlation structure inside the minibatch.

We ran that test and added the target period as a second factor, because our hypothesis was that correlated batches would hurt through target-chasing with a fast target and be harmless with a slow one. The hypothesis was wrong in an informative way: correlation does not hurt at either period, and the no-replay baseline, which the decorrelation argument says should be worst, is best at the fast period.

Contributions. (1) A controlled separation of within-batch correlation from reuse in DQN: correlation costs < 10 AUC points, n.s. (2) The finding that with a fast target, on-policy minibatches beat uniform replay by a wide margin because the replay learners collapse after reaching the ceiling and the on-policy learner does not. (3) The gradient-alignment measurement showing that correlation does degrade the gradient estimate of the buffer objective, and that this degradation is irrelevant to performance. (4) [3080] The designs that separate the two candidate mechanisms for the collapse.

## 2. Related work

Replay justification: Lin (1992); Mnih et al. (2015). Replay analysis: Zhang & Sutton (2017) on buffer size; Fedus et al. (2020) on replay ratio and the age of the oldest policy, who find that n-step is what makes large buffers useful; D'Oro et al. (2023) on replay ratio with resets. None varies within-batch correlation at fixed reuse. Closest to our `recent` finding: Zhang & Sutton (2017) also propose Combined Experience Replay, which adds the latest transition to every uniform minibatch and recovers most of the loss from large buffers; our `recent` sampler is the extreme form (the whole batch is the latest transitions), and the K = 100 collapse of uniform replay is the regime their fix targets. Correlated minibatches in supervised learning: the effect of non-iid batches on SGD is a variance effect that Adam's normalisation partly absorbs. Collapse after reaching the ceiling: our R11 (terminal transitions leaving the buffer), Nikishin et al. (2022) on primacy. Adjacent lab stubs: S3-4 (critical batch size of TD) and S3-12 (Durbin–Watson residual autocorrelation along trajectories) are about batch size and diagnostics, not about whether correlation in the batch matters.

## 3. Setting

Batched DQN, 2 × 64 MLP, Adam 1e-3, γ = 0.99, buffer 10k, batch 32, Huber loss, grad-norm clip 10, ε annealed 1 → 0.05 over 10k steps, one update per step from step 500, CartPole-v0 dynamics with a 200-step cap, 30k steps, greedy evaluation every 1k steps (AUC = mean of 30). Samplers: `iid`, `win4`, `win1`, `recent` (defined above). Target period K ∈ {100, 2500}. 16 seeds shared across cells. Every 1k steps the cosine between the applied minibatch gradient and the gradient on a fixed uniform 1024-transition sample of the buffer is recorded.

Pre-registered: K1 at K = 100, AUC(iid) − AUC(win1) ≥ 20 with ≥ 12/16 wins; K2 the gap at K = 2500 is ≤ half the gap at K = 100; K3 (reuse control) AUC(win1) − AUC(recent) ≥ 20 at both K. SUPPORTED if K1 and K2.

## 4. Results

### 4.1 Kill test

KILLED. K1: +8.3, 12/16 (p = 0.08). K2: +8.1 vs threshold 4.2. K3: −48.1 at K = 100, +23.9 at K = 2500.

### 4.2 Correlation does not matter at fixed reuse

win1 vs iid: −8.3 AUC (K = 100), −8.1 (K = 2500); 4/16 wins each (p = 0.08). win4 vs iid: +7.4 and −10.5, n.s. The gradient cosine to the buffer gradient falls with correlation (0.66 → 0.62 → 0.59 at K = 100; 0.51 → 0.50 → 0.45 at K = 2500), so the minibatch is a worse estimate of the buffer objective's gradient, and this has no detectable consequence: per-run Spearman(grad cos, AUC) = −0.02 overall and +0.08 within the replay samplers.

### 4.3 No replay beats replay with a fast target

| K | sampler | final (sd) | AUC (sd) |
|---|---|---|---|
| 100 | iid | 113.7 (38.1) | 119.1 (19.1) |
| 100 | win1 | 101.2 (50.8) | 110.8 (22.8) |
| 100 | recent | 183.0 (22.6) | 158.9 (14.2) |
| 2500 | iid | 146.2 (30.6) | 122.4 (16.0) |
| 2500 | win1 | 139.2 (23.0) | 114.3 (15.1) |
| 2500 | recent | 139.5 (35.5) | 90.4 (12.8) |

recent − iid at K = 100: +39.7 AUC, 15/16 (p = 0.001); +69.3 final, 14/16 (p = 0.004). At K = 2500: −31.9 AUC, 1/16; −6.7 final, 7/16 (n.s.). The curves explain the AUC: at K = 100 every replay sampler peaks at ≈ 165 at 5k steps and declines to 80–120 by 20k, while `recent` goes 143 → 171 → 164 → 181. At K = 2500 nothing moves before 10k steps (the target-period speed limit of R12), replay then climbs to 150–160 and `recent` to 130.

### 4.4 Two mechanisms for the collapse

(a) Stale targets: the replay buffer's actions and next-states come from older, worse policies; with a fast target the bootstrapped values track the current network, and the mismatch between on-policy targets and off-policy data could drive the decline. (b) Forgotten failures (R11): a good agent stops failing, its terminal transitions leave the FIFO buffer, and the network's values at failure states inflate until the policy breaks; the `recent` sampler cannot forget a failure because every failure is in its next minibatches. The pilot does not separate these. The decisive test is `iid` with R11's terminal reservoir at K = 100: if it matches `recent`, the mechanism is (b).

### 4.5 [3080] Planned

1. MinAtar (four games), the same four samplers × K ∈ {100, 500, 2500, 10000} × 10 seeds, 2M steps. Prediction: correlation stays a < 10% effect everywhere; `recent`'s advantage shrinks as the task's current-policy data becomes less informative and as K grows.
2. Mechanism split on CartPole and MinAtar: `iid` + terminal reservoir, `iid` with a policy-age cap on sampled transitions (only transitions from the last m policies), and `recent`, all at K = 100.
3. Polyak targets (τ ∈ {0.005, 0.05, 0.5}) in place of hard K, to test whether the crossover follows K_eff ≈ 1/τ.
4. Replay ratio × sampler, to see whether `recent` with 4 updates per step (reuse without staleness) dominates.

## 5. Consequences

If decorrelation is not what replay does, the design space for replay changes: samplers should be judged on what data they reuse and how stale it is, not on how independent the draws are. Prioritised replay's gains, for instance, have been read partly as a decorrelation effect; they are more plausibly a frontier-sampling effect (R12). And the result that on-policy minibatches beat replay with a fast target on a dense-reward task says the combination "small K, large buffer" is a bad default, independent of any decorrelation argument.

## 6. Limitations

CartPole, one architecture and learning rate, two target periods, 30k steps, 16 seeds. The `recent` sampler's win may be specific to tasks where the current policy's data is sufficient; the collapse it avoids is itself a CartPole-visible phenomenon (R11). AUC penalises slow starts. The mechanism split is planned, not done.

## 7. Reproducibility

`code/r17_decorr.py`, seeds 0–15, 14 minutes on a laptop CPU. `results/runs.csv` has every evaluation and gradient cosine; `results/per_run.csv` the per-run AUC, final and mean cosine.

## References

D'Oro et al. (2023). Sample-efficient RL by breaking the replay ratio barrier. ICLR.
Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML.
Lin (1992). Self-improving reactive agents based on RL, planning and teaching. Machine Learning.
Mnih et al. (2015). Human-level control through deep RL. Nature.
Nikishin et al. (2022). The primacy bias in deep RL. ICML.
Zhang, Sutton (2017). A deeper look at experience replay. NeurIPS DRL workshop.
