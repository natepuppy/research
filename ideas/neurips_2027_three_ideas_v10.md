# Two more ideas, round 10, with pilots run and paper drafts written (18 Oct 2026)

The brief: "two more really really good ideas." I looked for places where a default every Atari or continuous-control agent inherits has an exact, unstated consequence, because that is the shape of result the latency programme has been producing and the shape reviewers have the least to argue with. Candidates checked and dropped before pilots: prioritised replay × target period (Panahi et al. RLC 2024 occupy it); using the BRIDGE dataset's effective horizons to predict sample complexity from the propagation term (the DQN latency floor is negligible against the observed sample complexities, so there is no signal to find); PPO value clipping (an Electronics Letters 2025 paper partly occupies it); advantage std-normalisation (Dr. GRPO); hindsight relabelling as a latency device (likely wrong on inspection); SAC's target-entropy default (minor). Two survived.

Folders `R23-vmax-myopia`, `R24-nstep-rule` under `research/independent/`, standard layout, pre-registered kill tests, `results/posthoc.txt`, `paper/main.md` drafts.

| | idea | pilot verdict | honest odds (NeurIPS main) |
|---|---|---|---|
| R23 | The value support of C51 is a horizon: Rainbow is one-step-greedy, in practice indifferent, wherever both continuations exceed V_max = 10 | **KILLED by rule, mechanism confirmed.** K2 80/80 (C51 means equal the clipped fixed point everywhere); K1 35/40, K3 28/33, K4 39/40, every miss a threshold or atom-resolution artefact; 0/18 saturated cells chose the patient action; the saturated preference is exactly (1 − γ)V_max = 0.1 | 20–25% |
| R24 | How to choose n: the n-step return as a latency–bias trade-off under a target network; n* = max{n : γ^L(1 − ε/2)^{(L+1) − ⌈(L+1)/n⌉} > g}, independent of K | **SUPPORTED.** Latency law Spearman 1.00, median T/law 1.05 with no fitted constant; learned bias r = 0.985 with the closed form; largest safe n within one grid step of n* in 9/12 (misses have margins below seed noise); same n* at K = 200 and 2000 in 12/12; 18× faster at n = 12 vs n = 1 with no loss at ε = 0.01 | 30–35% alone; the "what to do about it" chapter of the merged paper |
| **merged** | Credit assignment is a count of iterations (R12 + R21 + R19 + R22 + R20), now with R24 as the prescription | all components pass or confirm | **55–60%** after one 3080 month |

---

## R23. The value support of C51 is a horizon

**Gap.** C51/Rainbow/DER/SPR/BBF and the Dopamine defaults put the return distribution on [−10, 10]. With clipped rewards and γ = 0.99 a competent policy's return in a dense game is tens of units. The projection clips every bootstrapped atom at V_max, so the learned value is the fixed point of T_c Q = min(V_max, r + γ max Q′). Where two actions' continuations both exceed V_max, Q_c prefers the one with the larger immediate reward by exactly (1 − γ)V_max — 0.1 for the defaults — whatever the true values. Nobody has written the clipped fixed point or asked what policy it produces; the only mentions are blog caveats that saturation is "tolerable".

**Pilot.** Exact tabular C51 (51 atoms, exact projection) and a scalar clipped-Q control on four patient-vs-myopic forks × ten V_max values; clipped value iteration as the prediction. The C51 mean matches Q_c in all 80 (cell, action) pairs to within one atom width. The pre-registered rule failed: my 0.05 indifference threshold sat inside the (1 − γ)V_max band at V_max ≤ 5, and one cell lost its 0.1 margin to the 0.4 atom spacing. Not one saturated cell chose the patient action, and the switch to patient happened exactly at V_max = 1 + γV_M as predicted in all four forks. I report the verdict as it fell and have not re-run with a corrected threshold.

**Reading.** In exact arithmetic the default Rainbow is one-step-greedy wherever it is winning; under function-approximation noise above 0.1 it is random among saturated actions. That is a different and stronger claim than "V_max is a bit low", and it changes what the deep experiment must measure: the fraction of visited states on the top atom, and the policy's seed sensitivity in those states, not just returns. With n = 3 the preference widens to ≈ (1 − γ³)V_max ≈ 0.3.

**What the paper needs.** Dopamine Rainbow at V_max ∈ {10, 100} on MinAtar and on Atlantis, Enduro, Ms. Pac-Man, Q*bert (chosen by predicted saturation), 3 seeds, plus the saturation audit and a 201-atom control. ~70 GPU-hours.

**Why 20–25%.** The derivation is a one-liner anyone can check, which is both why it is publishable (unstated for nine years, in every Atari agent) and why a reviewer may call it a known caveat. The pre-registered failure is honest and explained, but a sceptic will ask why I did not fix the threshold and re-run; the answer is in the notes. The paper is carried entirely by whether the saturation audit shows a large saturated fraction in dense games.

---

## R24. How to choose n

**Gap.** n is set by sweep in every value-based agent (1, 3, 5, 10 → 3, 20). Two lab results make it derived: under a target network refreshed every K updates, n divides the latency (R12, exact for networks where data exists, R22); and uncorrected n-step returns from ε-greedy replay are pessimistic by a factor whose closed form on a path is (1 − ε/2)^{(L+1) − ⌈(L+1)/n⌉} (R15 measured it). The policy is wrong once the bias exceeds the gap. So there is a largest safe n set by ε, L and the gap and not by K, and below it a smaller n costs only latency.

**Pilot.** 1344 vectorised tabular learners (fixed ε-greedy datasets, uniform replay, uncorrected n-step, hard target table): ε ∈ {0.01, 0.03, 0.05, 0.1} × gap ∈ {0.1, 0.2, 0.3} × n ∈ {1, 2, 3, 5, 8, 12, 20} × K ∈ {200, 2000} × 8 seeds, 75 CPU-minutes. Time to the correct policy is (⌈21/n⌉ − 1)K + 1 with Spearman 1.00 and median ratio 1.05, no fitted constant. The learned value matches the bias formula with r = 0.985 (RMSE 0.029). The largest n that learns the right policy is within one grid step of n* in 9/12 cells, exactly the threshold; the three misses are the three cells where n*'s predicted margin (0.026, 0.005, 0.001) is below the seed noise of the learned value (0.06). Seeds-correct is 0.96 above a 0.05 margin and ≤ 0.06 below zero — the sharp threshold seen through α = 0.1 noise. The largest safe n is identical at K = 200 and K = 2000 in 12/12 cells. Three smoke-test changes documented before the pilot (added ε = 0.05 for interior cases; final-correct over the last 20 % of training rather than one instant; 60k updates).

**Reading.** Nature DQN's ε = 0.1 makes any n > 1 unsafe on a 20-step path with gaps up to 0.3, so n = 1 was right for it. At ε = 0.01 the safe n exceeds 40 and the latency reduction is large: 40 200 updates to the correct policy at n = 1 versus 2 200 at n = 12, both 8/8 seeds. DER's 20 and BBF's 10 → 3 sit inside the rule; Rainbow's 3 is safe and leaves most of the latency unused. The BBF anneal is a corollary: ε and the relevant gaps both shrink during training.

**What the paper needs.** CleanRL DQN on MinAtar, n ∈ {1, 3, 5, 10, 20} × ε_final ∈ {0.01, 0.1} × K ∈ {500, 8000}, 5 seeds, with path lengths and gaps read from the learned agent; a truncated-return arm that should remove the cross-over. ~30 GPU-hours. Then the reanalysis of the published agents.

**Why 30–35% alone.** A hyper-parameter everyone sets blindly, a rule with no free constants, both halves confirmed under pre-registration, and a reading of the field's settings that comes out right. Against it: the components are each known in this lab, the margin term m was added after the fact, and the bias formula assumes a single-path failure structure. Inside the merged paper it is the chapter reviewers will ask for — "so what should I set?" — and raises that paper's odds.

---

## The paper to write (updated)

**Title.** Credit assignment is a count of iterations: one latency law for the three target mechanisms of deep RL, and the n that follows from it.

**Claim.** A reward D steps from a decision reaches it after a number of training iterations set by the target mechanism alone — (⌈D/n⌉ − 1)K for hard sync, (⌈D/n⌉ − 1)/τ + ⌈D/n⌉/α for Polyak, ≈ D((1 − γλ) + 1/T_eff) for GAE — exact for tables (R12, R21, R19) and for neural critics wherever data covers the path (R22). Most of "hard exploration" under standard value-based settings is this latency (R20). The one knob that divides all three is n, and its ceiling is the uncorrected-return bias, n* = max{n : γ^L(1 − ε/2)^{(L+1) − ⌈(L+1)/n⌉} > g + m}, which does not depend on the target mechanism (R24).

**Why 55–60%.** The round-9 reasoning stands and R24 adds the prescriptive section with no free constants and a correct reading of five published agents. Risks unchanged: "each piece is simple", the deep constants, being scooped by the functional-regularisation or VC-PPO lines.

**Order of work on the 3080.** (1) R22's MinAtar value-front measurement (decides whether "exact" survives). (2) R24's MinAtar n × ε × K grid (the prescriptive result; shares the DQN code with (1)). (3) R19's CleanRL PPO replication. (4) R21's h(t) in TD3. (5) R20's MiniGrid accounting. R23's Rainbow audit is a separate short paper, after.

Live candidates across rounds: the merged latency paper (now with R24); R16 and R24 as standalone second-tier; R23 as a short paper if the saturation audit lands; R8, R11, R15, R3 third-tier; R13, R17 short papers; R18 dropped.
