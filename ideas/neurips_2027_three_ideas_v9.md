# Two more ideas, round 9, chosen for acceptance odds, with pilots run and paper drafts written (18 Oct 2026)

The brief: "make them the highest likelihood of getting accepted; I want a paper in NeurIPS." I read that as an instruction to stop widening and start closing. The strongest material across eight rounds is the credit-assignment-latency programme — R12 (hard target sync), R19 (GAE/PPO, SUPPORTED at R² = 0.999), R20 (the exploration accounting). What stands between that material and an accepted paper is two things: the programme does not yet cover the third target mechanism (Polyak averaging, which is every continuous-control agent), and every reviewer will say "tables don't generalise, networks do, so your laws are loose bounds." R21 and R22 are those two pieces. Both were checked for prior work first: Piché et al. (ICLR 2022 submission, functional regularisation) state the hard-sync N·H chain latency in one sentence and note Polyak slows propagation, without quantifying either — now cited in R12, R21 and R22; Panahi et al. (RLC 2024) study prioritised replay × target period × generalisation in prediction tasks, the nearest neighbour of R22, with the opposite direction of inquiry. A PER × target-period idea was dropped because Panahi et al. occupy it.

Folders `R21-soft-target-horizon`, `R22-generalisation-hops` under `research/independent/`, standard layout, pre-registered kill tests, `results/posthoc.txt`, `paper/main.md` drafts.

| | idea | pilot verdict | honest odds (NeurIPS main) |
|---|---|---|---|
| R21 | Soft target updates are an implicit discount schedule: h(t) ≈ n·t/(1/τ + 1/α) | **SUPPORTED**, all four tests (horizon slope 18/18 at R² = 1.000; switch time 54/54, slope 1.011, R² = 0.9997; τ–α symmetry 12/12; hard = soft at K = 1/τ 15/15) | 25–30% alone; see below for the merged paper |
| R22 | Does generalisation speed up propagation under a target network? | **INCONCLUSIVE by rule; the failed half is the result.** η = 1.02–1.03 for all four encodings (R² = 1.000): one hop per sync exactly, representation-independent where data exists. Gap follow-up: representation acts across missing data and in the policy | 20–25% alone; decisive for the merged paper |
| **merged** | **Credit assignment is a count of iterations: one latency law for hard targets, soft targets and GAE, exact for tables and for networks** (R12 + R21 + R19 + R22, with R20 as the application) | all component pilots pass or confirm | **50–55%** after one 3080 month |

---

## R21. Soft target updates are an implicit discount schedule

**Gap.** τ = 0.005 is in every DDPG/TD3/SAC config and is tuned, never derived. Between a decision and its consequence k steps away the value passes through ⌈k/n⌉ segments; at each the target average (rate τ) and the critic fit (rate α) each add a geometric delay. So the critic's horizon grows linearly in updates, h(t) ≈ n·t/(1/τ + 1/α), and until h(t) reaches 1/(1 − γ) the agent is a shorter-horizon agent.

**Pilot.** Exact synchronous tabular updates, chain of 200 for the horizon and patience forks (now = 1 vs wait d steps for 2) for the switch time; τ ∈ {0.1, 0.01, 0.001} × α ∈ {1, 0.1, 0.01} × n ∈ {1, 5} × d ∈ {10, 30, 100}, plus hard-sync controls at K = 1/τ. Horizon slope within 0.98–1.10 of prediction in 18/18 cells (R² = 1.000). Switch time t* = (m − 1)/τ + m/α − max(1/τ, 1/α)/3: 54/54 within [0.79, 1.03]×, log–log slope 1.011, R² = 0.9997. τ and α interchangeable (12/12). Hard K = 1/τ matches soft within 8% (staircase vs line). At τ = 0.01 the agent is a γ_eff ≈ 0.9 agent for its first 1000 updates and a γ = 0.99 agent only after 10 000. One smoke-test correction (stage count m − 1, not m; re-derived, documented).

**Reading.** τ is a horizon parameter and α is its twin; the default imposes a discount schedule nobody chose. 17 seconds of CPU; a linear system, so exact by construction — the paper's weight is in the deep measurement of h(t) and of α_eff.

**What the paper needs.** TD3/SAC on Hopper/Walker: regress the critic on truncated MC returns at horizons 5–200 to read h(t) during training, for τ ∈ {0.05, 0.005, 0.0005} × n ∈ {1, 5}; extract α_eff; the continuous patience gridworld; a τ schedule against fixed τ. One 3080 week.

**Why 25–30% alone.** Clean, general, and it names something every practitioner has set blindly, but a reviewer can call the tabular part a derivation and the paper lives or dies on whether α_eff in a real critic leaves τ any role. Inside the merged paper it is the piece that makes the law cover continuous control.

---

## R22. Does generalisation speed up propagation under a target network?

**Gap.** The latency laws are derived for tables. The obvious objection is that networks generalise, so one sync might carry several hops. Nobody had measured hops per sync as a function of the representation.

**Pilot.** Corridor L = 40, fixed uniform dataset, batched 2 × 64 MLP DQN, hard sync K ∈ {50, 200, 800}, four encodings (one-hot, random codes, thermometer, scalar coordinate), 8 seeds, value front measured against ½V*. Hypothesis: η(coord) ≥ 2η(one-hot). Result: **η = 1.02–1.03 for all four**, regression lines of T_p on K indistinguishable (slopes 38.9–39.2 updates per unit K for 40 hops, R² = 1.000). With data at every state a sync carries exactly one hop regardless of generalisation: targets beyond the front are zero and the network cannot invent them. Pre-registered gap follow-up (no data at states 15–24): every encoding bridges the gap; coord at the full-coverage speed with gap values within 1% of V*, one-hot by skipping it with values 24% high, thermometer 32% high (my "one-hot stalls like a table" prediction failed — a one-hot MLP shares layers). Post-hoc: with coordinate inputs the greedy policy is correct at update 250 while the values arrive at 7850 — a feature shortcut, absent for one-hot.

**Reading.** The laws are exact for neural critics wherever the data covers the path. Representation matters in two other places: extrapolation across missing data (encoding-dependent accuracy, a source of overestimation that is not the max operator) and the policy (solved before learned). Both are reviewer-proofing for the programme and both are findings in their own right.

**What the paper needs.** DQN on MinAtar and MiniGrid with the value front measured against truncated MC returns at K ∈ {100, 1000, 8000}; coverage gaps with image vs coordinate observations; policy vs value front in tasks where the right action is and is not a simple function of the observation. One 3080 week.

**Why 20–25% alone.** A clean null with a mechanism and two positive side results, but as a standalone it reads as a check on someone else's law. Inside the merged paper it is the section that answers the objection before it is raised.

---

## The paper to write

**Title.** Credit assignment is a count of iterations: one latency law for the three target mechanisms of deep RL.

**Claim.** A reward D steps from a decision reaches it after a number of training iterations set by the target mechanism alone — (⌈D/n⌉ − 1)·K for hard sync, (⌈D/n⌉ − 1)/τ + ⌈D/n⌉/α for Polyak, ≈ D·((1 − γλ) + 1/T_eff) for GAE — exact for tables (R12, R21, R19) and exact for neural critics wherever data covers the path (R22). The policy follows in the same iteration (R19 K4). Consequences: the critic's horizon grows linearly in updates (an implicit discount schedule), most of "hard exploration" under standard value-based settings is this latency (R20), λ_critic = 1 is only a fix when the rollout covers the distance (R19), and n-step is the one knob that divides all three.

**Why 50–55%.** The components are each confirmed at R² ≥ 0.996 under pre-registration; the unification is clean; the practical rules are one-liners; the deep replications are cheap (MinAtar DQN, CleanRL PPO on MiniGrid, TD3 on Hopper — about one 3080 month in total); the related work is thin and known (Piché et al. one sentence; VC-PPO's decay framing; Panahi et al.). Risks: "each piece is simple" (true; the paper's job is to show that nobody put them together and that practice violates them), the deep constants may be loose (R22 says they will not be, for the hard-sync case), and VC-PPO's authors or the functional-regularisation line could publish the iteration-count framing first. Move now.

**Order of work on the 3080.** (1) R22's MinAtar value-front measurement (one week; it decides whether the word "exact" survives). (2) R19's CleanRL replication (one week; the RLHF-relevant piece and the biggest scoop risk). (3) R21's h(t) measurement in TD3 (one week). (4) R20's MiniGrid accounting (one week; the application section). Then write.

Live candidates across rounds, for the record: the merged latency paper; R16 (Huber as a risk preference) as the second-best standalone; R8, R11, R15, R3 as third-tier; R13, R17 short papers; R18 dropped.
