# Policy Churn Is Not Controlled by the Target Network, and Only Matters When Nothing Else Explores

*Draft. Pilot numbers are from the laptop run in `results/`; sections marked [3080] are planned.*

## Abstract

Schaul, Barreto, Quan and Ostrovski (2022) showed that deep value-based agents change their greedy action in a large fraction of states after every gradient update ("policy churn"), and conjectured that this churn is an unintended but substantial source of exploration. We test two consequences in a 4 × 3 factorial over the target-network period K and a constant ε on CartPole: that K damps churn, and that explicit exploration becomes more valuable as churn is removed. Both fail in the pre-registered form. Churn *increases* with K (rank correlation +0.88 across cells): a frequently refreshed target re-anchors the online network, and per-update greedy flips come from fitting the current target, not from drift. ε's value does not vary monotonically with K. What survives is the mediation half of the conjecture: among ε = 0 runs, early churn predicts area-under-curve with Spearman +0.80 (n = 24, within every K), and this correlation decays to +0.67 at ε = 0.05 and to −0.02 at ε = 0.2. Churn and ε are substitutes on the seed axis, not the K axis. We argue the causal direction is still open and [3080] propose a direct churn manipulation (a CHAIN-style regulariser crossed with ε on MinAtar) to settle it, together with an accounting of which exploratory actions originate from churn versus ε.

## 1. Introduction

Policy churn (Schaul et al., 2022) is the observation that, in DQN-style agents, the greedy action changes in roughly 10% of states after a single gradient step, far more than the value changes would suggest if actions flipped only near ties. Schaul et al. argued this is a hidden exploration mechanism; Tang & Berseth (2024) reduced churn with a regulariser (CHAIN) and reported gains, which cuts the other way. Neither line asks what controls churn among the knobs every DQN already has, nor tests the substitution a true exploration mechanism implies: remove it, and deliberate exploration should matter more.

The natural candidate knob is the target period K. A hard target refreshed every K updates sets how far the online network can drift between re-anchorings, and K is normally chosen for stability, not exploration. We ran the factorial and found that the natural guess about K's direction is wrong, and that the substitution appears on an axis we had not pre-registered.

Contributions. (1) A pre-registered factorial that kills "K damps churn": churn rises with K. (2) A within-cell mediation result: at ε = 0 early churn predicts learning in every K cell (+0.70 to +0.83), and the dependence vanishes as ε rises. (3) A statement of the causal question that remains and [3080] the manipulation that answers it.

## 2. Related work

Churn: Schaul et al. (2022) define and measure it, show it is not an artefact of ties, and conjecture an exploration role. Tang & Berseth (2024) regularise it away (CHAIN) with gains in several settings, suggesting churn is at least partly harmful. Target networks: Mnih et al. (2015); analyses of target period as a stability/propagation trade-off (Lee & He, 2019; Zhang et al., 2021), and our R12 result that the target period sets a hard propagation speed limit. Exploration via representation change: Ostrovski et al. (2021) on the tandem effect, Nikishin et al. (2022) on primacy. Adjacent lab stubs: "exploration from optimiser noise" (S2-15) proposes a method for exploiting incidental exploration; this paper tests whether the largest known source of it is controllable by a standard knob.

## 3. Setting

Batched DQN on CartPole-v1 (200-step cap), 2 × 64 MLP, Adam, replay 10k, batch 64, γ = 0.99, one update per step. K ∈ {20, 100, 500, 2500}, constant ε ∈ {0, 0.05, 0.2} (no annealing; ε is the only deliberate exploration), 8 seeds shared across cells, 30k steps. Greedy evaluation every 1k steps; AUC = mean of these. Churn = fraction of 256 fixed probe states whose greedy action changed across each update, averaged per 1k-step interval; early churn = mean over the first 10k steps.

Pre-registered test: K1 Spearman(mean churn, log K) over the 12 cells ≤ −0.8; K2 gain(K) = AUC(ε = 0.2) − AUC(ε = 0), Spearman on log K ≥ 0.8 and gain(2500) − gain(20) ≥ 20; K3 among ε = 0 runs Spearman(AUC, early churn) ≥ 0.4. SUPPORTED if K1 and K2; INCONCLUSIVE if K1 and K3 only; KILLED otherwise. The README records that a smoke run before pre-registration hinted churn might rise with K; the test was written in the hypothesis's direction regardless.

## 4. Results

### 4.1 Kill test

KILLED. K1 = +0.88. K2: gains +0, +79, +65, +73 (K = 20, 100, 500, 2500), Spearman +0.40. K3 = +0.72, passed.

### 4.2 K = 20 is degenerate

All 24 runs with K = 20 collapsed to a constant greedy action within the first 1k steps (final return 8.8–10.2, churn ≈ 0, at every ε). A near-on-policy target on CartPole lets the max-bias feedback run away before any return signal arrives. These cells carry no information about churn and are excluded below.

### 4.3 Churn rises with the target period

Mean churn per update (%), K = 100 / 500 / 2500: ε = 0, 0.34 / 1.11 / 0.81; ε = 0.05, 0.74 / 2.21 / 2.28; ε = 0.2, 1.83 / 3.40 / 3.30. Early churn is monotone: 0.25 / 0.46 / 1.03 at ε = 0, 1.52 / 1.71 / 3.38 at ε = 0.2. Per-run Spearman(early churn, log K) = +0.53. A frequently refreshed target keeps the online network close to a fixed point it has mostly fit, and small residual updates flip few actions; a stale target leaves a large residual that the network keeps fitting, and the fitting flips actions. Churn is a property of how far the online network is from its current target, and K sets that distance upward, not downward.

### 4.4 Mediation at ε = 0, and its disappearance with ε

Post hoc (`results/posthoc.txt`). Among ε = 0 runs, Spearman(AUC, early churn) = +0.78 (K = 100), +0.70 (K = 500), +0.83 (K = 2500); pooled +0.80, n = 24, p < 0.001. The ε = 0 outcome is bimodal within each cell (AUC range 9–163), and the seeds that learn are the seeds whose networks churned early. At ε = 0.05 the pooled correlation is +0.67; at ε = 0.2 it is −0.02 (p = 0.92). Final-return sd falls from 72–87 at ε = 0 to 11–50 at ε = 0.2.

This is the substitution the conjecture implies, on the seed axis: churn matters exactly when ε does not supply exploration. It is not, by itself, evidence that churn *causes* learning. A network whose values are moving changes its greedy actions; "learning causes churn" produces the same within-run correlation.

### 4.5 [3080] Planned

1. Direct manipulation. MinAtar (Breakout, Asterix, Freeway, Space Invaders), DQN with a CHAIN-style churn regulariser at four strengths × ε ∈ {0, 0.01, 0.05, 0.2} × 10 seeds. Prediction if churn is causal exploration: the regulariser hurts at ε = 0 and is neutral or helps at ε = 0.2, with the crossover at a strength that brings churn below ε's per-step action-change rate. Prediction if churn is a by-product: the regulariser's effect does not interact with ε.
2. Accounting. Log every environment action with a flag for whether it differed from the greedy action of the network one update earlier (churn-induced) or was an ε draw; relate the churn-induced fraction to state-coverage growth (count-based novelty on hashed observations) and to return.
3. K sweep on MinAtar with ε annealed as in the standard recipe, to check whether the K → churn direction found here (positive) holds with annealing and larger networks.

## 5. Consequences

The target period is not a churn knob, and the cheap story "stabilise the target to reduce churn" is wrong in direction. If churn is exploration, it is exploration that a stale target gives more of, which reframes the stability/exploration trade-off in choosing K. The mediation result means that in any ε = 0 or small-ε regime (including most actor-critic setups with deterministic policies), churn is a candidate explanation for seed-level bimodality and should be measured before being regularised away.

## 6. Limitations

CartPole, one architecture and learning rate, constant ε, 30k steps, 8 seeds. The K = 20 collapse is probably learning-rate specific. Probe states are fixed early in training. Early churn and AUC come from the same run; the mediation result is correlational and the causal direction is the planned experiment.

## 7. Reproducibility

`code/r14_churn.py`, seeds 0–7, 11 minutes on a laptop CPU. `results/runs.csv` holds every evaluation with its interval churn; `results/per_run.csv` the per-run summaries.

## References

Lee, He (2019). Target-based temporal-difference learning. ICML.
Mnih et al. (2015). Human-level control through deep RL. Nature.
Nikishin et al. (2022). The primacy bias in deep RL. ICML.
Ostrovski, Castro, Dabney (2021). The difficulty of passive learning in deep RL. NeurIPS.
Schaul, Barreto, Quan, Ostrovski (2022). The phenomenon of policy churn. NeurIPS.
Tang, Berseth (2024). Improving deep RL by reducing the chain effect of value and policy churn. NeurIPS.
Zhang, Yao, Whiteson (2021). Breaking the deadly triad with a target network. ICML.
