# Three more NeurIPS-shaped ideas, round 5, with pilots run and paper drafts written (16 Oct 2026)

Same scope: NeurIPS main track, reinforcement learning, RTX 3080 as the ceiling, no overlap with the stub list, T1–T3, R1–R9 or the lab's areas. Two stubs are adjacent and are named in the READMEs: S4-2 "anticipatory targets" (a fix; R10 is the diagnosis that would justify it) and N1-11 "within-task forgetting of stepping stones" (skills; R11 is about terminals, with an exact ground truth). Theme this round: the *mechanics of the update* — what the critic is actually estimating, what the replay stream is missing, and how fast information can move through bootstrapping at all.

Folders `R10-critic-lag`, `R11-forgetting-how-to-fail`, `R12-bootstrap-speed-limit` under `research/independent/`, standard layout, pre-registered kill tests, `paper/main.md` drafts.

| | idea | pilot verdict | honest odds (NeurIPS main) |
|---|---|---|---|
| R10 | Is the critic behind? Which policy a deep critic evaluates | KILLED (fine-resolution follow-up) | 5% alone; workshop note |
| R11 | Forgetting how to fail: terminal transitions as the anchor of value learning | INCONCLUSIVE by rule; fix strongly supported | 30–35% |
| R12 | The speed limit of bootstrapping: target period × replay × n-step | pre-registered form KILLED; refined law CONFIRMED out of sample | 25–30% |

---

## R10. Is the critic behind?

**Gap.** Two-timescale theory needs the critic faster than the actor; practice uses equal LRs; the folk picture is that a slow critic evaluates a *stale* policy. Nobody has measured which policy the critic is evaluating. In a deterministic simulator you can: store every past actor, roll them out from probe states, and ask which one's exact Q the critic resembles.

**Pilot.** Batched DDPG on Pendulum, 3 actor LRs × 4 critic LRs × 8 seeds, lags 0–6k steps (pilot) and 0–1k steps at 100-step resolution (pre-registered follow-up). The critic is never closer to a past policy than to the present one: error profiles over lag are flat to within 2% or increase with lag. The pilot's K1 "passed" only because argmin over seven noisy, flat errors is biased toward k > 0; the mean profile has no such bias and sits at k = 0 in every cell. Fine-resolution test KILLED. Survivors: critic error against the current policy tracks policy drift (ρ = 0.90 across cells) and predicts final return (ρ = −0.78); slow critics cost 35–50 return points through *inaccuracy*, not staleness.

**Reading.** A TD-trained critic on replayed data is a bias field, not a delayed copy of Q^π. "Lag" is the wrong metaphor; "tracking error" is the right one. Methods motivated by lag (delayed actor updates, anticipatory targets) should be evaluated against tracking error. The instrument (exact MC against stored actors) is the reusable piece and goes into R11's next run.

**Odds.** ~5% as a main-track paper; a clean negative-result workshop note. Not worth 3080 time.

---

## R11. Forgetting how to fail

**Gap.** Reach-ceiling-then-collapse is endemic in DQN on time-limited tasks. The folk explanation (good agent → no failures → terminals leave the FIFO buffer → network forgets failure → collapse) has never been tested because "forgets failure" had no measurement. CartPole gives one: any (s, a) whose action terminates has Q*(s, a) = 1 exactly, for every policy. A fixed probe set of such pairs is a thermometer with a known true reading.

**Pilot.** DQN, 24 seeds × 3 conditions, 40k steps. Control: failure-Q drifts 2 → 35 (true value 1) as terminals fall below 1% of the buffer; 2.46 collapses per run, every run collapses. `reservoir_term` (first 256 terminals kept forever, 4 of 32 per minibatch): failure-Q 1.0 all run, 0.08 collapses per run (2 of 24 runs), final 154 vs 119, 19/24 paired wins (p = 0.007), half the variance. `reservoir_rand` (same amount of random early data): 2.17 collapses, no benefit. So K2 (fix) and K3 (specificity) pass. K1 (mechanism) half-passes: inflation tracks terminal loss (ρ = −0.53) but does not time individual collapses (AUROC 0.61). Inflation is chronic; collapses are episodic. Verdict INCONCLUSIVE by rule.

**What the paper needs.** The mechanism split: does the reservoir work because terminals *anchor the value scale* (then a flag-removed reservoir, same states with done = 0, should fail and Pop-Art-style anchors should substitute) or because it preserves *boundary knowledge* (then flag-removed should still work)? Plus overestimation on non-failure states against exact MC (the R10 instrument), forks for per-run causality (R7 machinery), and scale: MinAtar Breakout/Asterix (terminals) vs Freeway (none; predicted null), MuJoCo Hopper/Walker with SAC, buffer sizes 10⁴–10⁶ (prediction: effect shrinks as buffer grows).

**Why 30–35%.** A one-line, zero-cost fix with a 97% reduction in a failure mode that hits every run, an exact ground-truth measurement, and a specificity control is a strong package. The risk is "CartPole-only" and "already folklore"; the MinAtar/MuJoCo replication and the mechanism split answer both. Scoop risk moderate: the plasticity groups could run this in a week.

---

## R12. The speed limit of bootstrapping

**Gap.** A hard target network refreshed every K updates lets value advance one n-step hop per refresh; uniform replay samples the frontier transition once per ~L/B updates. Both are obvious; neither has been stated as a law, measured, or used to set K and n together.

**Pilot.** Tabular TD on deterministic chains, 256 configs × 3 seeds (L up to 64, n up to 8, K 1–1000, B 4/32, SGD and per-state Adam). The pre-registered max() form was KILLED (SGD R² 0.883; the Adam regime test was undefined by grid design). The data showed something sharper: in the target-limited regime t_prop = (⌈L/n⌉ − 1)·K to within 0.2% (median over 60 cells; e.g. 63 003 vs 63 000), the replay-limited regime scales as L²/(nB), and the two *add*. The additive law was then pre-registered and tested on a disjoint grid (L up to 96, n up to 6, K 3–3000, B 8/16) with c held fixed: identity-fit R² = 0.997 (SGD), 0.971 (Adam). CONFIRMED.

**Consequences.** n-step is worth exactly a factor n in both terms; PER should remove the replay term only; reverse replay reaches the bound t ≈ K; Nature-DQN settings (K = 2.5k updates, n = 1) need ~250k updates before a reward 100 steps away is first felt at the start, and Rainbow's n = 3 / R2D2's n = 5 cut that by exactly 3× / 5×.

**What the paper needs.** Stochastic chains (law in expectation); linear and deep FA on the chain (generalisation makes hops longer than n: measure n_eff); DQN on reward-sparsified MinAtar with the additive law and fitted n_eff; Polyak (K_eff ≈ 1/τ?) and PER (replay term gone?) as one-figure tests. A short finite-time proposition for the target term is already in the draft; the replay term's constant needs a derivation for the SGD case.

**Why 25–30%.** Exact, confirmed, cheap, and it explains two published empirical puzzles (Fedus 2020's n-step × buffer interaction; why reverse replay works). The reviewer risk is "obvious counting argument" and "tabular". The deep n_eff measurement and the PER prediction are what lift it.

---

## Recommendation

Across rounds 4 and 5 the live main-track candidates are **R8** (mixing barrier), **R11** (forgetting how to fail), **R12** (speed limit) and **R3** (fictional MDP), with **R7+R9** as a merged second-tier paper. R10, like R1, R2 and R9's pilot, is documented and stopped. R11 and R12 share a theme — what bootstrapping can and cannot learn from a replay stream — and could be sections of one larger paper if either comes up short alone.
