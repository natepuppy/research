# R22: Generalisation sets the speed of credit assignment under a target network

**Question.** R12 and R21 give closed-form latencies for value propagation under hard and soft target networks: one n-step hop per target refresh, exactly, in the tabular case. The objection every reviewer will raise is that a neural critic generalises: when the value of one state rises, the values of similar states rise with it, so a single refresh might move the front several hops. If that is true the laws are upper bounds whose tightness depends on the representation, and "how far does one sync carry the signal" becomes a measurable property of the input encoding. We test this directly: same corridor, same fixed dataset, same network and optimiser, four input encodings that differ only in what "similar" means — one-hot (no similarity), random ±1 codes (similarity without locality), thermometer codes (monotone locality), and a scalar coordinate (smooth locality). We measure hops per sync η = K·L / T_p, where T_p is the number of updates until the greedy policy first reaches the goal from the start, and the front position over time. The claim: the law holds with η ≈ 1 for one-hot inputs, η rises with locality, and unstructured similarity does not help. If so, the latency laws carry over to deep RL with a representation-dependent multiplier, and the multiplier is the propagation meaning of "a good representation".

**Status.** Laptop pilot done: **INCONCLUSIVE by rule, and the failed half is the result.** K1 passed: one-hot η = 1.03 (R² = 1.000), the hard-sync law holds under function approximation to 3%. K2 (locality multiplies the speed) failed outright: η = 1.02–1.03 for every encoding, the four regression lines are indistinguishable. With data at every state a target sync carries exactly one hop no matter how the network generalises — generalisation interpolates targets, it cannot invent them. The pre-registered gap follow-up (`results/gap/`) then showed where representation does enter: across missing data (every encoding crosses a 10-state gap; coord does so at the full-coverage speed with values within 1% of V*, one-hot skips it with values 24% high), and in the policy (with coordinate inputs the greedy policy is correct at update 250 while the values arrive at 7850 — a feature shortcut, not propagation). The latency laws are representation-independent wherever the data pins the targets. See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`. Paper version on the 3080: the same measurement with DQN on MinAtar (pixel-like binary planes, η expected near 1) and on MiniGrid with coordinate vs image observations, plus an online version where η is read from the TD-error front during ordinary training.

Adjacent prior work: Piché et al. 2022 (target networks slow propagation; propose functional regularisation), Panahi et al. RLC 2024 (prioritised replay × target period × generalisation, in prediction tasks — the closest; they vary target period to *control* generalisation, we measure how generalisation changes the propagation speed), Lyle et al. 2022 and Kumar et al. 2021 (representation rank/capacity, not propagation), Achiam et al. 2019 (divergence from generalisation + bootstrapping). Lab stubs: none on this; the plasticity/representation stubs (L1-9) are about post-training rigidity in LLMs and are named for adjacency only.

## Pre-registered kill test (written before the pilot ran)

Design. Corridor of L = 40 states, actions left/right, start 0, reaching position 40 pays 1 and terminates, γ = 0.99, deterministic. Fixed dataset: 20 000 one-step transitions from uniformly random (state, action) pairs, so every transition including the goal transition is present at equal density. Learner: 2 × 64 ReLU MLP, Adam 1e-3, batch 64, Huber loss, grad-norm clip 10, hard target network synced every K updates, one-step targets, trained offline on the fixed dataset for up to 60 000 updates. Encodings: `onehot` (40-d), `random` (16-d random ±1 code per state, fixed per seed), `thermo` (40-d thermometer, bit k = [s ≥ k]), `coord` (scalar s/L). K ∈ {50, 200, 800}; 8 seeds (seed sets network init, dataset order, random code). Every 50 updates: Q at all 40 states by one forward pass; the **value front** front(t) = the largest d such that V̂(s) = max_a Q(s, a) ≥ ½·V*(s) for every s ∈ {L − d, …, L − 1} (V*(s) = γ^{L−1−s}); T_p = first t with front = L. Runs that never reach front = L are recorded at 60 000 and flagged. The greedy-policy front (largest d from which the greedy policy reaches the goal) is logged alongside and reported, but is not the measurement.

Per encoding, the hops-per-sync multiplier is read from the slope: η(enc) = L / slope, where slope is the OLS slope of median T_p on K over K ∈ {50, 200, 800}. The intercept absorbs the fitting delay that does not scale with K (the 1/α term of R21), so the slope isolates what one sync carries.

- **K1 (the law survives function approximation for localist inputs).** `onehot`: η ∈ [0.67, 1.5] and R² of the linear fit ≥ 0.95.
- **K2 (locality multiplies the speed).** η(coord) ≥ 2·η(onehot), and η(coord) ≥ η(thermo) ≥ η(onehot).
- **K3 (similarity without locality does not help).** η(random) ≤ 1.5·η(onehot).

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported. We also report the front(t) traces: the law predicts a staircase with step K for `onehot` and larger, possibly irregular, steps for `coord`.

*Smoke-test change (before the pilot, 2 seeds × 6000 updates):* the first draft measured the front with the greedy policy. In this corridor "right" is the correct action at every state, so a network can be greedy-correct everywhere through the shared output bias the moment the goal transition is fit, with no value propagated at all: the `coord` runs were "solved" at update 200–250 at every K and `onehot` runs were solved well before their values could have arrived. That measures a policy shortcut, not propagation. The front was redefined on the values (above) and η redefined from the slope rather than from K·L/T_p, which the fitting intercept distorted. Thresholds unchanged.

## Pre-registered follow-up: coverage gap (written after the pilot, before the follow-up ran)

The pilot found η = 1.02–1.03 for every encoding (R² = 1.000): with full data coverage, generalisation does not move the front faster than one hop per sync, because every state's target is pinned by its own data and the targets beyond the front are zero. If generalisation matters for propagation it must be where the data does not pin the targets. Follow-up: the same experiment at K = 200 with a **coverage gap** — states 15–24 have no transitions in the dataset (the other states keep 20 000/30 each on average). A tabular learner cannot propagate through the gap at all.

- **F1 (localist inputs stall).** `onehot`: front ≤ 16 (the states between the gap and the goal, plus one) at 60 000 updates in 8/8 seeds.
- **F2 (smooth inputs cross the gap).** `coord`: solved (front = L) within 60 000 updates in ≥ 6/8 seeds.
- **F3.** `thermo`: solved in ≥ 6/8 seeds.
- `random`: reported, no prediction (unseen codes receive arbitrary values).

Results in `results/gap/`. F1 ∧ F2 is the outcome that turns the pilot's null into a statement: generalisation changes *where* the law applies (across data gaps), not *how fast* the front moves where data exists.

## Layout

| folder | contents |
|---|---|
| `code/` | `r22_hops.py`; `code/lib/dqn_batched.py` (Params/Adam/forward reused), `code/lib/io_utils.py` |
| `results/` | `runs.csv`, `front.csv`, `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r22_hops.py --smoke
.venv/bin/python code/r22_hops.py             # pilot, ~15 min CPU
.venv/bin/python code/r22_hops.py --summarize
```
