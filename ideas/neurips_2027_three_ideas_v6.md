# Three more NeurIPS-shaped ideas, round 6, with pilots run and paper drafts written (16 Oct 2026)

Same scope: NeurIPS main track, reinforcement learning, RTX 3080 as the ceiling, no overlap with the stub list, T1–T3, R1–R12 or the lab's areas. This round the brief was to discover the questions as well as run them. Theme: *theory's folk predictions, tested where they are cheapest to test exactly*. Three pieces of received wisdom, one from PG convergence theory, one from the policy-churn conjecture, one about why uncorrected n-step works, each with a quantity that can be computed exactly on a laptop. Adjacent stubs named in the READMEs: S2-10 / L1-27 (reset-distribution methods; R13 tests the quantity they rest on), S2-15 (exploiting optimiser noise; R14 asks whether the main source is controllable), S2-5 (n-step by transition age; R15 predicts a different schedule).

Folders `R13-mismatch-coefficient`, `R14-churn-exploration`, `R15-nstep-bias` under `research/independent/`, standard layout, pre-registered kill tests, `paper/main.md` drafts.

| | idea | pilot verdict | honest odds (NeurIPS main) |
|---|---|---|---|
| R13 | Does the distribution-mismatch coefficient predict PG learning time? | KILLED, with a clean reversal | 15–20% as a reversal paper |
| R14 | Is policy churn the exploration you did not ask for? (K × ε factorial) | KILLED (mechanism); mediation half survives | 10% |
| R15 | Two biases, one knob: why uncorrected n-step helps under noise | INCONCLUSIVE by rule; every directional prediction replicated on fresh seeds | 25–30% |

---

## R13. Coverage can hurt

**Gap.** PG convergence rates scale with D_μ = ‖d^{π*}_ρ/μ‖_∞ (Kakade & Langford 2002; Agarwal et al. 2021; Mei et al. 2020). The coefficient is the standard justification for exploratory restarts and is infinite for a single fixed start. Nobody had checked whether it predicts learning time even in the exact-gradient tabular setting where it is derived.

**Pilot.** Exact-gradient softmax PG on four 10×10 layouts × nine start distributions (single start, six mixtures with uniform, μ = d^{π*} with D_μ = 1 exactly, two "far" distributions). Spearman(log D_μ, log t) = **−0.67**; (log D_0, log t) = −0.52. Wrong sign. On the three layouts where the initial policy reaches the goal at all, the single start (D_μ = ∞) beats μ = d^{π*} (D_μ = 1) by 20–45% and beats uniform by 2–4×; a 10–30% uniform mixture on the single start is fastest everywhere. On the hard-exploration `snake` layout (V^{π_0}(ρ)/V* ≈ 10⁻⁵) the coefficient's ordering appears (+0.70 within layout) and d^{π*} is the best start.

**Reading.** Exact PG has a fixed gradient budget spread as d^{π}_μ; mass on states irrelevant to V^π(ρ) is wasted. Coverage pays only when the gradient at ρ is too small to climb, which is the regime the theory implicitly assumes. Two regimes, with V^{π_0}(ρ)/V* as the indicator. Post hoc the best single predictor outside hard exploration is μ's mass on the evaluation start (−0.65), not any coverage measure.

**What the paper needs.** Sample-based PG (does sampling variance move the optimum β?), a held-out test of the regime indicator on fresh random layouts, and a deep check (CartPole reset width; MinAtar randomised starts, PPO) predicting a U-shaped efficiency-vs-width curve on dense-reward tasks and monotone improvement only when the initial return is at the floor.

**Why 15–20%.** A sharp, cheap reversal of a result everyone cites, with a mechanism and a design rule. The risk is that reviewers read it as "bounds are loose, we knew that"; the answer is that the *sign* reverses and the two-regime indicator is a prediction. Needs the sample-based and deep checks to be more than a note. Not a method paper, which hurts at NeurIPS.

---

## R14. Churn is not controlled by the target network

**Gap.** Schaul et al. (2022) conjectured that policy churn is a hidden exploration source. Nobody had asked which standard knob controls churn, or tested the substitution a true exploration source implies (damp churn → ε matters more). The target period K was the obvious candidate.

**Pilot.** Batched DQN on CartPole, K ∈ {20, 100, 500, 2500} × constant ε ∈ {0, 0.05, 0.2} × 8 seeds. Churn **rises** with K (Spearman +0.88 across cells; +0.53 per run): a frequently refreshed target re-anchors the network, and per-update greedy flips come from fitting a stale target. K = 20 collapsed in all 24 runs. ε's AUC gain is flat in K (79, 65, 73). K1 and K2 fail; K3 (at ε = 0, early churn predicts AUC) passes at +0.72. Post hoc the mediation is strong and K-independent: within every K, Spearman(AUC, early churn) among ε = 0 runs is +0.70 to +0.83 (pooled +0.80, n = 24), decaying to +0.67 at ε = 0.05 and **−0.02** at ε = 0.2.

**Reading.** The mechanism as posed (K → churn → exploration) is dead in direction. The substitution the conjecture implies does exist, on the seed axis: churn predicts learning only when nothing else explores. Causal direction is open (learning causes churn is an equally good fit to a within-run correlation).

**What the paper needs.** A direct churn manipulation (CHAIN-style regulariser at four strengths × ε, MinAtar, 10 seeds) with the crossover prediction; an action-level accounting of churn-induced vs ε-induced exploratory actions against coverage growth; the K sweep with annealed ε to see if the positive K → churn direction holds at scale.

**Why 10%.** The surviving result is correlational and on CartPole; the 3080 manipulation would be the whole paper, and if it comes out "churn is a by-product" the result is a negative note. If the crossover appears it becomes a real contribution to a live debate (Schaul vs Tang & Berseth). Call it 10% now, 25% conditional on the manipulation showing the crossover.

---

## R15. Two biases, one knob

**Gap.** Uncorrected n-step is universal, theoretically wrong, empirically necessary (Fedus 2020), and the corrected alternatives do not help (Hernandez-Garcia & Sutton 2019). No published account of *why*.

**Hypothesis.** The uncorrected target carries a pessimistic term from blending the behaviour policy's value ((T^μ)^{n−1}T* fixed point, grows with n and ε) and Q-learning's optimistic max-bias under noise, which falls with n. Under noise n trades one against the other; without noise n > 1 is pure cost.

**Pilot.** Tabular Q with replay and a target table on an 8×8 slip gridworld, exact Q*, n ∈ {1, 2, 3, 5, 10} × σ × ε ∈ {0.1, 0.3} × {plain, double}, 150k steps. I set the σ grid too high twice (agents did not learn at σ ≥ 0.2 with α = 0.1); both runs are reported and discounted. Pre-registered replication on **fresh seeds** at σ ∈ {0, 0.05, 0.1}: overestimation falls with n in every noisy cell (Spearman −1.00); n ≥ 2 is pessimistic at σ = 0 and n = 10 costs 11–13% (ε = 0.1) to 22–23% (ε = 0.3) of return; best n rises 1 → 2–3 → 3 with σ; the best n is one of the two n at which the two biases net to zero in **10/10** noisy cells across the second and third grids (chance ≈ 10⁻⁴). My competence threshold (plain Q at σ = 0.1 ≥ 50% of V*) was missed at 45%, so the pre-registered verdict is INCONCLUSIVE. Double Q halves max-bias, shares the pessimistic term, and does not remove the n-dependence (K3 failed three times).

**Refinement from the data.** Decomposing overestimation into pess(n) (σ = 0) and maxbias(n, σ): the max-bias falls 7× from n = 1 to 2 and is gone by n = 3; the pessimistic term is smaller and grows slowly. So the active knob at small n is the max-bias shrinking (bootstrap weight γ^n, fewer compounding steps); the pessimistic term sets where the sum crosses zero.

**What the paper needs.** DQN/Double DQN on MinAtar with injected reward noise (n-step gain rises with σ, larger for DQN); the ε-schedule test (optimal n should *rise* as ε anneals; a schedule n(t) increasing over training beats any fixed n — the discriminating prediction against S2-5's age-based n); corrected vs uncorrected n-step at matched propagation speed (R12) as the direct test of "why uncorrected wins"; exact-MC overestimation in deep agents (R10/R11 instrument).

**Why 25–30%.** A simple accounting that explains a universal practice, with signed predictions that replicated out of sample, an exact tabular ground truth, and a cheap deep test with a specific schedule prediction. Risks: "tabular", the missed threshold (which I will report), and the chance the deep n-step gain is mostly propagation speed. The matched-speed control is what decides it.

---

## Recommendation

Across rounds 4–6 the live main-track candidates are **R8** (mixing barrier), **R11** (forgetting how to fail), **R12** (speed limit), **R15** (two biases) and **R3** (fictional MDP), with **R13** as a sharp short paper and **R7+R9** merged as second tier. R12 and R15 are natural companions: R12 is the *speed* story of n-step and target networks, R15 the *bias* story, and the matched-speed control in R15 uses R12's law directly. If one 3080 month is available, spend it on R15's deep test with R12's control built in, then R11's MinAtar replication. R10 and R14 are documented and stopped unless the churn manipulation can be run cheaply alongside R11 (same MinAtar DQN harness).
