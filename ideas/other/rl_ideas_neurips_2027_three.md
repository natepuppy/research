# Three NeurIPS-shaped RL ideas (research as of 14 Oct 2026)

Target: NeurIPS 2027 main track (projected abstract deadline ~12–14 May 2027, full paper ~19–21 May 2027; CFP not yet posted). ICML 2027 is the fallback (~late Jan 2027).

Honest framing first. Nothing "definitely" gets into NeurIPS; NeurIPS 2026 accepted roughly a quarter of submissions and reviewer noise is real. What raises the odds is paper *shape*: (1) a live disagreement between several recent papers, (2) a controlled design whose outcome is informative either way, (3) a releasable artifact (code, a minimal method, or a theory with a prediction that was not fit after the fact), and (4) enough seeds/tasks that the statistics survive. Each idea below is built to that shape, with the 2026 papers it sits between, a kill test, and a 3080 compute plan. Estimated odds assume competent execution and are deliberately not inflated.

None of these overlaps the 250 in `rl_ideas_*.md` or the experiments in `research/independent/`.

---

## Idea 1. Why do flow/diffusion policies beat Gaussians in online RL? A controlled decomposition (and a minimal policy that matches them)

**Domain:** continuous control / robot learning. **Odds if executed well:** ~45–55%.

### The disagreement
Five 2026 papers show generative policies winning on high-dimensional control and each attributes it to something different:
- FPO (ICLR 2026, "Flow Matching Policy Gradients") and FPO++ (CoRL 2026, arXiv 2602.02481): gains attributed to *cross-dimension correlated exploration* (they show hip-joint cross-correlation heatmaps that diagonal Gaussians cannot represent).
- FFM (arXiv 2610.12848, Oct 2026): flow policies tie Gaussians on 20 low-dim Playground DMC tasks and only pull ahead on the 4 humanoid tasks; adding OU (temporally correlated) noise helps further, pointing at *exploration structure*, not expressivity.
- TruDi (trust-region diffusion, 2026): attributes the win to a *trust region on the whole denoising chain*; DPPO and FPO "make little progress" on Playground humanoid in their hands.
- GoRL (arXiv 2512.02581): attributes instability of FPO to the *surrogate ratio misaligning under distribution shift* and the lack of an entropy knob.
- FlowRL benchmark (arXiv 2603.27450): with aligned hyperparameters, *no* diffusion method dominates and SAC stays competitive on many tasks.
- QF3 (arXiv 2610.08789, off-policy flow RL from scratch on humanoids) adds a sixth candidate: the filtered critic gradient.

Meanwhile *Much Ado About Noising* (arXiv 2512.01809) showed that in behavior cloning the flow advantage is not multimodality or expressivity but *supervised iterative computation plus training-time stochasticity*; a two-step regression policy (MIP) matches flows. The authors explicitly say the finding may not carry to RL. Nobody has run the RL version.

### Hypothesis
On the humanoid tasks where generative policies win, most of the gap is recovered by a Gaussian policy given (a) a low-rank-plus-diagonal exploration covariance and (b) a two-step residual (MIP-style) action head. Multimodality of the learned action distribution is rarely used at decision time.

### Design
Fix one PPO-family trainer (REPPO or PPO on MuJoCo Playground) and compare at matched network size, update count, and tuning budget:
1. G-diag: standard diagonal Gaussian (baseline).
2. G-lowrank: Σ = D + UUᵀ, rank r ∈ {2, 4, 8} (correlated exploration only).
3. G-OU: diagonal Gaussian with temporally correlated noise (temporal correlation only).
4. G-MIP: deterministic two-step residual head trained with intermediate supervision + diagonal noise (iterative computation only).
5. G-GMM: mixture of K = 4 Gaussians (multimodality only).
6. Flow-K1: one-step pushforward policy trained with the FPO surrogate (surrogate without iteration).
7. Flow-K8: full FPO++/FFM (reference).
8. Flow-frozen-explore: a trained flow whose *actions* are taken as flow-mean + diagonal Gaussian noise (removes the flow's exploration structure while keeping its expressivity).

Diagnostics logged for every run: cross-dimension correlation of sampled actions at matched states; number of modes per state (sample 256 actions, cluster); the deterministic-evaluation gap (mean action vs sampled); the rank of the exploration covariance over training. Tasks: the 4 Playground humanoid tasks where the gap exists and 6 Playground DMC tasks where it does not; the contrast is the result. 10 seeds, rliable IQM.

### Kill test
Every matched Gaussian variant stays more than half the gap behind Flow-K8 on humanoid, and Flow-K8 is measurably multimodal at decision-relevant states. Then the paper becomes "expressivity is the mechanism," which is a weaker but still publishable result.

### Deliverable
A decomposition table attributing the gap to each mechanism, plus whichever variant closes it, released as a drop-in policy head. If G-lowrank + G-MIP closes it, that is a cheaper, likelihood-tractable replacement for flow policies in on-policy RL, with the entropy knob GoRL says flows lack.

### Compute (RTX 3080, 10 GB)
MJX on one GPU with 1024–2048 envs. Playground DMC runs are ~10–20 min; humanoid ~1–2 h. 8 variants × 10 tasks × 10 seeds is 2–4 weeks of GPU time; run 5 seeds on everything first, then 10 on the decisive contrasts. Use the released FPO++, TruDi, and FFM code as the flow reference so nobody can claim a weak baseline.

### Main risks
Hyperparameter fairness (publish the sweep budgets); a mixed answer that differs per task family; being scooped by the FPO or Simchowitz groups, who are the natural authors of this paper. Posting early to arXiv matters.

---

## Idea 2. Selection vs drift: a fixation theory of rare correct modes under finite-group RLVR, and what it predicts about the pass@k "kernel"

**Domain:** RL for LLM reasoning (theory + small-model experiments). **Odds if executed well:** ~35–50%; crowding is the risk.

### The disagreement
The "does RL expand capability" debate now has a stack of partially conflicting results:
- Yue et al. (2504.13837) and *Invisible Leash* (2507.14843): RLVR sharpens; base wins at large k; support cannot expand.
- ProRL (NeurIPS 2025) and ProRL v2: prolonged RL reaches problems the base never solves under sampling.
- Rewrite-grammar study (2607.07646) and f∘g (2509.25123): RL composes weakly supported primitives into procedures the base solves at 0% even at pass@1024, but cannot invent from zero mass.
- *Pass@k inversion* (2607.20543): a "boundary mode-commitment failure": on prompts where the correct mode is rare, finite-sample RLVR commits away from it; proposes PBA gating.
- *RLVR is a kernel, not a function* (2609.22547): prompts with the same base success p end at different post-RL success q, repeatably across generation halves; the map is a Markov kernel H(q|p), not a curve; most published crossovers are not statistically established.
- F-GRPO (2602.06717): one-step closed form for the probability a group misses a rare-correct mode, non-monotone in group size G; unsampled-correct mass can shrink while total correct mass grows.
- RLVεR / "Rate or Fate" (2601.04411) and G2RPO: mean-field replicator dynamics on the simplex; note in passing that finite G is "genetic drift" with O(G^-1/2) fluctuations, but do not analyze it.
- IsoCompute Playbook (2603.12151): compute-optimal rollouts per prompt rise then saturate; "sharpening on easy, coverage on hard."

What is missing is the *stochastic, dynamic* theory: not the expected one-step update, but the probability that a correct mode with initial mass p₀ fixes or goes extinct over a training run, as a function of G, step size η, and the drift injected by updates on *other* prompts through shared parameters. That is exactly Wright–Fisher / Kimura territory, and the machinery (diffusion approximation, backward Kolmogorov equation, fixation probability u(p₀), critical frequency) is standard.

### Hypothesis
Per-prompt success probability under GRPO follows, to a good approximation, a one-dimensional diffusion dp = μ(p; G, η) dt + σ(p; G, η, σ_int) dW, where μ is the selection term (nonzero only when the group is mixed, probability 1 − pᴳ − (1−p)ᴳ) and σ² sums within-group sampling noise and a cross-prompt interference term σ_int² (hitchhiking: updates on other prompts move p even when this prompt's groups are all-fail). This yields:
- P1. A critical p*(G, η, σ_int) below which extinction is more likely than fixation; the kernel H(q|p) is *bimodal* at boundary p (fixation or extinction), not a shifted unimodal bump. This is the shape "RLVR is a kernel" observed but did not explain.
- P2. p* falls roughly like 1/G until σ_int dominates, which gives the saturating n* of IsoCompute and the non-monotone miss probability of F-GRPO as special cases.
- P3. Reference resets and KL anchoring (ProRL) act by shrinking σ_int; PBA acts by setting μ = 0 on boundary prompts.
- P4. Prescription: per-prompt adaptive group size G_i ∝ 1/p̂_i (capped), and η scaled down on boundary prompts, should expand coverage at fixed compute. Compare against dynamic sampling (DAPO), PBA, and F-GRPO.

### Design
- E1 (cheap, decisive): a 20–50M transformer pretrained from scratch on the rewrite-grammar or f∘g synthetic family (both have released code), where p_i(0) is measurable with 10⁴–10⁵ samples per prompt. Run GRPO with G ∈ {2, 4, 8, 16, 32}, η sweep, 20+ seeds. Record fixation/extinction frequency vs (p₀, G, η); fit u(p₀) and compare to the theory's closed form. Measure σ_int directly: track p_i for held-out prompts never trained on and show their random-walk variance scales with η² and with gradient alignment.
- E2: Qwen2.5-0.5B or 1.5B with LoRA on Countdown / a MATH-lite subset, G ∈ {4, 8, 16, 32}; check P1 and P2 qualitatively.
- E3: re-analysis of public base/RL pairs used in the kernel paper (DeepScaleR, ProRL 1.5B): is H(q|p) bimodal at boundary p? 256 samples × ~500 prompts × 2 models at 1.5B is a few GPU-days on a 3080.
- E4: the adaptive-G rule vs baselines at matched rollout budget on E1 and E2.

### Kill test
Fixation/extinction frequencies in E1 do not depend on G and η the way the diffusion predicts, or H(q|p) at the boundary is unimodal. Then the theory is wrong; the σ_int measurement (which prompts move when they receive no gradient) is still a new, reportable object.

### Deliverable
Theory with a closed-form fixation probability, a measured interference term, a prediction of the published kernel's shape, and a one-line group-size rule.

### Compute
E1 is days on a 3080. E2 is 1–2 weeks (slow HF generation). E3 is a few days. All feasible; nothing needs a cluster.

### Main risks
Crowding: four groups have adjacent theory. Mitigate by posting the E1 result early and by being the paper that predicts the kernel. The theory must stay honest about where the diffusion approximation breaks (very small G, very large η).

---

## Idea 3. What does a reset reset? Trainability (plasticity) in prolonged RLVR, and a cross-regime test of 2026 plasticity theory

**Domain:** bridge between deep-RL plasticity theory and LLM RL. **Odds if executed well:** ~35–45%; the null outcome is the risk, and it must be designed to be informative.

### The disagreement
- ProRL (NeurIPS 2025) keeps 1.5B models improving for 2k+ steps by hard-resetting *both* the reference policy and the optimizer state when validation stalls; ProRL v2 resets the reference every 200–500 steps and explicitly does *not* clear optimizer state. Neither reports which component does the work, or why.
- SWD (ICLR 2026, arXiv 2604.01913) attributes plasticity loss in deep RL to NTK rank collapse plus Θ(1/k) gradient attenuation caused by *bootstrapping* and data non-stationarity. GRPO has no bootstrapping but extreme on-policy non-stationarity, and it has its own gradient-vanishing mechanism (group std → 0 as accuracy saturates). SWD's theory therefore makes a prediction about RLVR that has not been checked.
- *Predicting Plasticity* (arXiv 2605.09044, May 2026): dormant-neuron and rank diagnostics do not predict future trainability; proposes "optimization readiness" (gradient strength × reliability). OCP (2603.21173): plasticity loss is task-specific.
- RL's Razor (2509.04259) and *Retaining by Doing* (2510.18874): on-policy RL makes small, KL-minimal, sparse updates. If weights barely move, representational plasticity loss may not occur at all, and whatever saturates prolonged RLVR is coverage (entropy mechanism, 2505.22617; pass@k inversion) or optimizer-state staleness, not the network.
- Hussing et al. (2403.05996) implicate Adam's second moment in high-UTD failure; stale optimizer statistics are a plausible, cheap-to-fix culprit nobody has isolated in LLM RL.

Three competing accounts of why prolonged RLVR stalls, each predicting a different reset to work: (a) coverage exhaustion → nothing but entropy re-inflation or gating helps; (b) optimizer-state staleness → optimizer reset alone restores learning; (c) representational plasticity loss → weight-level resets (Shrink & Perturb / ReDo) are needed, and readiness/rank metrics degrade.

### Hypothesis
Prolonged RLVR does not lose representational plasticity (updates are too small and there is no bootstrapping), but it does accumulate stale Adam statistics; an optimizer-only reset recovers most of ProRL's reset benefit, and SWD's bootstrapping-driven gradient attenuation does not appear.

### Design
- Testbed A (cheap, long runs): 20–50M from-scratch transformer on synthetic verifiable task families (rewrite grammar / f∘g), GRPO for 5–10k steps with a curriculum that moves across task families so there is something new to learn late.
- Testbed B (real model anchor): Qwen2.5-0.5B full fine-tune with 8-bit Adam on Countdown, 1–2k steps; fits in 10 GB with gradient checkpointing and short sequences.
- At checkpoints t, run *fresh-task trainability probes*: fine-tune a copy for a fixed budget on (i) a held-out RL task family and (ii) a held-out supervised task; compare learning curves against the base checkpoint and against the same checkpoint after each intervention. The supervised probe isolates trainability from entropy/coverage, which is the confound every entropy paper leaves open.
- Interventions, factorial: none; optimizer-state reset only; reference reset only; both (ProRL); Shrink & Perturb on weights; temperature re-inflation (coverage control). Metrics: readiness, eNTK rank, GraMa, update sparsity and norm, entropy, coverage (pass@k on a fixed probe set).

### Kill test
Trainability probes are flat across RL steps and the optimizer reset does nothing: account (a). The paper then says "RLVR saturation is coverage, not plasticity; resets work through the KL anchor," which is less exciting but settles a question practitioners are already guessing at, and it is a clean negative test of applying SWD's theory outside bootstrapped RL. If optimizer reset alone restores learning, that is the headline and a free fix. If S&P is needed, plasticity loss has reached LLM RL and the plasticity toolkit transfers.

### Deliverable
A factorial table of which reset restores which probe, with a recommendation; a cross-regime test of a 2026 theory; a reusable trainability-probe protocol.

### Compute
Testbed A is days per configuration; Testbed B is 1–2 days per run. The factorial is a few weeks total on a 3080.

### Main risks
External validity of a 50M testbed (that is why Testbed B exists); the null. Preregister the three accounts and their predicted outcomes in the paper so the null reads as a result, not a failure.

---

## Why these three and not the others I checked

Verified as already taken in 2026 (do not pursue as framed):
- "Dense turn-level credit is reward shaping; GRPO z-scoring breaks invariance" — *The Dark Room in the Reward Channel* (2607.21273) and TIPS (2603.22293) cover it.
- "Entropy bonus fails as |A| grows; interpolate Atari → LLM" — AEnt (2509.03493) has the synthetic |A| = 10⁵ study; a 2026 comparative theory paper (2604.09676) covers the rest.
- "Plasticity metrics don't predict trainability" — *Predicting Plasticity* (2605.09044).
- "Controlled synthetic testbed for whether RL creates new skills" — Echo Chamber, f∘g, rewrite-grammar, and the compositional dependency-graph paper (2609.19465).
- Training-inference mismatch, async staleness, self-play collapse, reward hacking of verifiers: each has 5+ 2026 papers and needs cluster-scale compute to be credible.

The pattern across the 2026 literature: mechanism papers on LLM RL are a race against hundreds of groups; controlled decomposition papers in continuous control and cross-regime tests of theory are thinner and reward exactly the instrumentation already in each project's `code/lib/`.

## Reading list (the papers each idea must cite and beat)

Idea 1: 2602.02481 (FPO++), 2610.12848 (FFM), 2610.08789 (QF3), TruDi, 2606.21086 (ReFPO), 2512.02581 (GoRL), 2603.27450 (FlowRL), 2512.01809 (Much Ado About Noising), REPPO (Voelcker et al. 2026), PFPN (2003.06959).

Idea 2: 2504.13837, 2507.14843, 2505.24864 (ProRL), 2607.07646, 2509.25123, 2607.20543, 2609.22547, 2602.06717 (F-GRPO), 2601.04411 (RLVεR), G2RPO, 2603.12151 (IsoCompute), 2609.37868 (GRAFT), Kimura 1962 / Ewens for the math.

Idea 3: ProRL + ProRL v2 blog, 2604.01913 (SWD), 2605.09044, 2603.21173 (OCP), 2411.04832 v3 (survey), 2509.04259 (RL's Razor), 2510.18874, 2505.24061 (GraMa), 2403.05996 (Hussing et al.), 2505.22617 (entropy mechanism).
