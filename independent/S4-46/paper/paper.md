# Anytime Validity Is Nearly Free; Distribution-Free Validity Is What Costs Seeds: Decomposing the Price of Valid Sequential Comparison of RL Algorithms

*Anonymous authors. Full manuscript. A conference-length version is in `paper_rlc.md`; the earlier condensed draft is `paper_short.md`.*

**Reproducibility.** Every number in this paper is produced by the scripts in `../code/`: `s4_anytime.py` (the original beta and gridworld runs, Appendix F), `s4_extensions.py` (synthetic and BBF experiments; `results/extensions_summary.json`), `s4_real_data.py` (TD-MPC2 and fresh-run experiments; `results/real_data_summary.json`), `s4_fresh_runs.py` (the 480 DQN training runs; `data/fresh/fresh_runs.csv`), `make_figures.py` and `make_figures_real.py` (`figures/`). The released seed data are the per-run human-normalized Atari-100k scores of BBF (Schwarzer et al., 2023) in `data/bbf/` and the per-seed DMControl and Meta-World results of TD-MPC2 (Hansen et al., 2024) in `data/tdmpc2/`, both copied unchanged. Simulation uses seeds 0 and 1; standard errors are reported where they matter.

---

## Abstract

Comparisons of two reinforcement learning algorithms are typically run for a fixed seed budget, or grown until a t-test looks significant. The second habit rejects a true null 36–54% of the time in our simulations at nominal 5%. Anytime-valid tests fix this, and the betting test for bounded means of Waudby-Smith and Ramdas (2023) has been proposed for exactly this use. But at the 80%-power point it needs 1.8–5.4× the seeds of a one-look t-test on synthetic paired differences and 11–12× on real Atari-100k seed data. We ask which property is being paid for. The betting test bundles two guarantees: validity at every stopping time (*anytime*) and validity for every distribution on a declared interval (*distribution-free*). We separate them with tests that have one property and not the other: the anytime-valid t-test e-process of Pérez-Ortiz et al. (2024) and group-sequential designs (anytime, parametric), and the betting martingale read once at a pre-committed sample size (fixed-n, distribution-free). The decomposition is clean. Anytime validity is cheap: the betting martingale read once needs the same number of seeds as the version stopped at first crossing on every setting (44 vs 42, 32 vs 32, 27 vs 27, 111 vs 111), a fact we prove follows from Markov's and Ville's inequalities sharing the threshold \(1/\alpha\); and the parametric anytime e-process needs 1.6–2.0× the one-look t-test with no maximum sample size, about the same as a five-look group-sequential design. The 2–12× is the price of the bound, and it is set by the ratio of the declared half-width to the noise: 1.8× at a ratio of 4.3, 2.9× at 7.1, 5.4× at 13, 12× at 34, as it is for 55-game aggregate scores. We give a one-line heuristic for the betting test's seed count when its wager is clipped, \(n \approx h\log(1/\alpha)/\delta\), that predicts the observed counts to within 10–40%. Tightening the declared bound to a value the data happen to satisfy recovers the parametric cost (19 seeds vs 16), but on 83% of paths that bound was violated, so the recovered saving is not valid. On skewed differences the parametric tests inflate (t-test at \(n=5\): 12%; e-process: 11%) while the betting test stays at 0.3%. On noisy single-game comparisons, where the noise fills a sixth or more of the declared interval, the betting test matches or beats the parametric anytime test. Two further released benchmarks (TD-MPC2 and three baselines on 39 DMControl and 50 Meta-World tasks) show the other extreme: effects so large that a t-test needs two to four seeds, where the betting test still needs 9–64 because its count is set by the gap and the bound rather than by the noise, within 10–30% of the heuristic. Finally, 480 DQN runs we trained ourselves give paired streams with no resampling at all: on exact nulls every valid test holds \(\alpha\) while the peeking t-test rejects 17–27% within 30 looks; on the four comparisons with a detectable gap the genuine and bootstrap seed counts agree within 1–4 seeds; and the two regimes reappear, the betting test matching the parametric anytime test on bimodal Acrobot differences (28 vs 28 seeds) and failing to finish within 30 fresh pairs on tight Breakout differences where the t-test needs 8. The recommendation follows: if you can stand behind a Gaussian approximation, use an anytime-valid t-test and stop whenever you like; if you cannot, declare the tightest interval you can defend before you look, and plan on the multiplier.

---

## 1. Introduction

A claim that algorithm A is better than algorithm B on a benchmark is, operationally, a claim about the mean of paired differences of per-seed scores. The reinforcement learning community has become careful about how those means are summarized (Henderson et al., 2018; Colas et al., 2018; Agarwal et al., 2021; Patterson et al., 2023), but much less careful about *when the comparison stops*. Two protocols are common in practice. The first commits to \(n\) seeds in advance and runs a one-sample t-test on the differences. The second runs a few seeds, looks, and if the result is "almost significant," runs a few more and looks again. The second protocol is natural, is usually unreported, and does not have type I error \(\alpha\). In every null we simulated, looking after every seed and stopping at the first \(p < 0.05\) rejected a true null between 36% and 54% of the time (Section 5).

A test that permits this behavior without inflation is *anytime-valid*: the probability of ever rejecting a true null, at any data-dependent stopping time, is at most \(\alpha\) (Robbins, 1970; Howard et al., 2021; Ramdas et al., 2023). The betting test for bounded means of Waudby-Smith and Ramdas (2023) is the natural candidate for RL: human-normalized scores clipped to \([0,1]\) give paired differences in \([-1,1]\) by construction, and the test requires nothing more. An earlier version of this work measured that test against a t-test sized for the effect at hand and found that at 80% power it needed 2–5× as many seeds on synthetic differences, with the multiplier set by how loose the declared interval was relative to the noise. That is correct and it is also incomplete, because it does not say what the extra seeds buy.

The betting test bundles two guarantees. It is anytime-valid. It is also *distribution-free* on the declared interval: valid for every distribution of paired differences supported on \([L, H]\), including bimodal, skewed, and heavy-tailed ones. A practitioner who wants to look after every seed but is willing to assume approximate normality of seed differences pays for both properties and uses one. The question this paper answers is how the cost splits between them.

**Approach.** We fill in the two-by-two table of (anytime or fixed-\(n\)) × (parametric or distribution-free). The one-look t-test is fixed-\(n\) and parametric. The anytime betting test is anytime and distribution-free. We add tests on the off-diagonal: the anytime-valid t-test e-process of Pérez-Ortiz, Lardy, de Heide, and Grünwald (2024) and five-look group-sequential designs with Pocock and O'Brien–Fleming boundaries (anytime, parametric); and the betting martingale read once at a pre-committed \(n\), valid by Markov's inequality (fixed-\(n\), distribution-free). Differences along a row isolate the cost of the anytime property under a fixed distributional assumption; differences along a column isolate the cost of the distributional guarantee under a fixed stopping protocol. We run all of them on synthetic paired differences with controlled noise and on real per-run Atari-100k scores from BBF and its ablations.

**Contributions.**

1. *A decomposition of the seed cost.* Within the distribution-free family, the anytime property costs zero seeds at the 80%-power point on every setting (Section 7.1), and we show this is a consequence of a nonnegative martingale having the same rejection threshold under Markov's inequality at a fixed time and under Ville's inequality over all times (Proposition 1). Within the parametric family it costs 1.6–2.0× at 80% power and 1.1–1.7× in expected seeds, without a planned maximum sample size (Section 7.2).
2. *The bound is the cost.* The betting test's multiplier over the one-look t-test is a monotone function of the ratio of the declared half-width to the standard deviation of the differences, from 1.8× to 16× over the range we study (Sections 7.3, 8, and 10). We explain the mechanism (the clip on the betting fraction) and give a heuristic for the seed count when the clip binds, \(n \approx h\log(1/\alpha)/\delta\), which is linear in the declared half-width \(h\) and independent of the noise (Proposition 2, informal).
3. *A warning about tuning the bound.* Declaring a bound the data happen to satisfy recovers the parametric cost, but that bound was violated on 83% of 200-seed paths in our real-data example, so the saving is not a valid one (Section 8).
4. *Real-data evidence in both regimes.* On 55-game aggregate scores, whose differences are nearly Gaussian with standard deviation 3% of the interval, the parametric anytime test is 7× cheaper than the betting test. On single games, whose differences are heavy-tailed and bimodal with standard deviation up to 44% of the interval, the two cost about the same and the betting test is ahead on six of nine comparisons (Section 9).
5. *Two more benchmark suites, at the large-effect extreme.* On TD-MPC2's DMControl and Meta-World results (four algorithms, 39 and 50 tasks) the aggregate gaps are 0.05–0.55 and a one-look t-test needs 2–4 seeds; the betting test needs 9–64, ordered by the gap and within 10–30% of the heuristic on eight of ten comparisons, and the parametric anytime test has a floor of five (Section 10). Five-task slices land in the intermediate regime.
6. *Fresh seeds.* 480 DQN runs on four environments give paired streams with no resampling. Eight exact nulls hold \(\alpha\) for every valid test while the peeking t-test rejects 17–27% within 30 looks; on the four detectable comparisons, genuine and bootstrap seed counts agree within 1–4 seeds, which validates the resampled streams used elsewhere; and the regimes recur (betting equals the e-process at \(h/\mathrm{sd} = 2.3\), and does not finish in 30 pairs at \(h/\mathrm{sd} = 28\)) (Section 11).
7. *Concrete recommendations* keyed to what the practitioner is willing to assume (Section 12).

---

## 2. Related work

**Evaluation methodology in RL.** Henderson et al. (2018) documented that reported improvements in deep RL are often within seed-to-seed variation and that five seeds are not enough to distinguish typical effects. Colas et al. (2018) asked how many random seeds are needed and gave power calculations for the t-test and Welch's test under a fixed-\(n\) protocol. Agarwal et al. (2021) proposed interval estimates of aggregate metrics (IQM, optimality gap) with stratified bootstrap confidence intervals and released `rliable`; our aggregate seed is built on their optimality-gap transform. Jordan et al. (2020) and Patterson et al. (2023) address the broader design of RL experiments. None of these treats the stopping rule as part of the design, which is our subject.

**Anytime-valid inference.** Confidence sequences go back to Darling and Robbins (1967) and Robbins (1970) and were made practical by Howard, Ramdas, McAuliffe, and Sekhon (2021). The game-theoretic framing of e-values and test martingales is reviewed by Ramdas, Grünwald, Vovk, and Shafer (2023) and by Grünwald, de Heide, and Koolen (2024). The betting test for bounded means is Waudby-Smith and Ramdas (2023), whose predictable plug-in wager and empirical-Bernstein confidence sequence we use. The anytime-valid t-test as an e-process derived from scale invariance is Pérez-Ortiz, Lardy, de Heide, and Grünwald (2024). Johari, Koomen, Pekelis, and Walsh (2022) brought always-valid inference to industrial A/B testing and documented the same peeking inflation we measure.

**Group-sequential designs.** Clinical trials solved the peeking problem decades earlier with group-sequential boundaries: Pocock (1977), O'Brien and Fleming (1979), and the alpha-spending generalization of Lan and DeMets (1983). These designs are anytime-valid up to a planned maximum under a Gaussian model and are the right comparison point for an RL practitioner who is willing to assume normality and commit to a maximum budget. We are not aware of prior use in RL algorithm comparison.

**Position of this paper.** The individual tests are not new. The contribution is the decomposition: using tests that have exactly one of the two properties to measure what each property costs in seeds, on synthetic and real RL data, and the practical conclusion that falls out.

---

## 3. Setup

### 3.1 Observations

We observe a stream of paired differences \(x_1, x_2, \ldots\) on the raw score scale, where \(x_i\) is the difference between algorithm A's and algorithm B's score on seed \(i\). The hypothesis is \(H_0: \mathbb E[x] = 0\) against the two-sided alternative. All tests are run at \(\alpha = 0.05\).

Where a test requires observations on \([0,1]\), they are rescaled from a declared interval \([L, H]\) by \(u_i = (x_i - L)/(H - L)\), and the null mean 0 lands at \(m_0 = -L/(H - L)\). For the symmetric interval \([-1, 1]\), \(m_0 = 1/2\). If an observation falls outside \([L, H]\) we clip it for the purpose of running the test and record the violation; the test's guarantee does not hold on that path, and we report the violation rate alongside any result where it is nonzero.

### 3.2 Two properties

**Anytime validity.** A test with rejection indicator \(R_n\) after \(n\) observations is anytime-valid at level \(\alpha\) if \(\mathbb P_{H_0}(\exists n \ge 1: R_n = 1) \le \alpha\). Equivalently, for every stopping time \(\tau\), \(\mathbb P_{H_0}(R_\tau = 1) \le \alpha\). A fixed-\(n\) test satisfies only \(\mathbb P_{H_0}(R_n = 1) \le \alpha\) for the pre-committed \(n\).

**Distribution-free validity on an interval.** A test is distribution-free on \([L, H]\) if its validity guarantee holds for every distribution of \(x\) supported on \([L, H]\) with mean 0. A parametric test's guarantee holds under a model (here: approximate normality of the sample mean at the relevant \(n\)).

The two properties are logically independent, and the four cells of their product are occupied by the tests of Section 4:

| | **Parametric** | **Distribution-free on \([L,H]\)** |
|---|---|---|
| **Fixed \(n\)** | one-look t-test | betting martingale, one look (Markov) |
| **Anytime** | t-test e-process; group-sequential (to \(n_{\max}\)) | betting martingale, first crossing (Ville); EB confidence sequence |

### 3.3 Cost measures

For each setting and test we report \(n_{80}\), the smallest \(n\) such that the probability of having rejected by observation \(n\) is at least 0.8. For a fixed-\(n\) test this is the sample size one would commit to after a power analysis; for a sequential test it is the budget one must be prepared to spend. For sequential tests we also report the mean stopping time \(\mathbb E[n]\), with a path that never rejects counted at the cap. Power settings use 400 simulated paths and a cap of 200 seeds; the standard error of an \(n_{80}\) estimate is roughly 1–2 seeds at the values in our tables. Nulls use 1,000 paths, giving a standard error of about 0.007 on a rejection rate near 0.05.

---

## 4. Tests

### 4.1 Fixed-\(n\), parametric: the one-look t-test

Reject at the committed \(n\) if the one-sample t-test of mean 0 has two-sided \(p < \alpha\). We verified the implementation against `scipy.stats.ttest_1samp` and that its rejection rate on Gaussian nulls is 5.0% at every \(n\) from 2 to 200. For reference, under normality with effect \(\delta\) and standard deviation \(\sigma\), the sample size for 80% power is approximately \((z_{0.975} + z_{0.8})^2 \sigma^2/\delta^2 \approx 7.85\,\sigma^2/\delta^2\), with a small-\(n\) correction from the \(t\) quantiles.

### 4.2 The invalid baseline: the peeking t-test

Look after every seed from \(n = 2\) and stop at the first \(p < \alpha\). This is not a valid test and is included as the failure the other methods prevent.

### 4.3 Anytime, parametric: group-sequential designs

Five looks at \(n_k = \lceil k\, n_{\max}/5 \rceil\), \(k = 1, \ldots, 5\). At look \(k\) the t-test \(p\)-value is converted to a z-score \(z_k = \Phi^{-1}(1 - p_k/2)\). Pocock (1977) rejects when \(z_k \ge c_P\); O'Brien and Fleming (1979) reject when \(z_k \ge c_O\sqrt{5/k}\). The constants \(c_P, c_O\) are calibrated by 20,000 Monte Carlo Gaussian null paths so that the probability of crossing at any of the five looks is exactly \(\alpha\). The design needs a planned maximum; we set \(n_{\max}\) to twice the one-look t-test's \(n_{80}\) for the setting, which is a generous power analysis, and the test cannot reject after \(n_{\max}\). This is anytime-valid up to \(n_{\max}\) under the Gaussian model. In an earlier attempt we used a look after every seed, which yields absurd boundaries (\(|T| \ge 13.4\) from the heavy tails of \(t\) at one degree of freedom); five looks is the standard clinical choice and is what we report.

### 4.4 Anytime, parametric, no maximum: the t-test e-process

Let \(T_n\) be the one-sample t-statistic after \(n\) observations and put a prior \(\delta \sim \mathcal N(0, g^2)\) on the standardized effect \(\delta = \mu/\sigma\). Define

\[
E_n \;=\; \frac{\int f_{n-1,\,\delta\sqrt n}(T_n)\, \mathrm d\Pi(\delta)}{f_{n-1}(T_n)},
\]

where \(f_{\nu,\lambda}\) is the noncentral \(t\) density with \(\nu\) degrees of freedom and noncentrality \(\lambda\), and \(f_\nu\) the central one. Pérez-Ortiz et al. (2024) show that the density ratio of the scale-maximal invariant depends on the data only through \(T_n\), and that \((E_n)_{n \ge 2}\) is a nonnegative martingale under \(H_0\) with respect to the invariant filtration for *every* \(\sigma\); Ville's inequality then gives \(\mathbb P_{H_0}(\exists n: E_n \ge 1/\alpha) \le \alpha\). We reject at the first \(n\) with \(E_n \ge 1/\alpha\). The integral is a 40-node Gauss–Hermite sum. We use \(g = 1\) in the main tables and report \(g = 0.3\) as a sensitivity check (column "anytime t, \(g = 0.3\)"). The test has no maximum sample size and needs no power analysis.

### 4.5 Anytime, distribution-free: the betting test

The predictable plug-in martingale of Waudby-Smith and Ramdas (2023). On the unit scale with null \(m_0\), the wealth after \(t\) observations is

\[
M_t \;=\; \prod_{i \le t} \big(1 + \lambda_i (u_i - m_0)\big), \qquad
\lambda_i = \mathrm{clip}\!\left(\frac{\hat\mu_{i-1} - m_0}{\hat v_{i-1}},\ \pm\frac{0.99}{\max(m_0, 1 - m_0)}\right),
\]

where \(\hat\mu_{i-1}\) is the running mean and \(\hat v_{i-1} = \widehat{\mathbb E}[(u - m_0)^2]\) the running second moment about the null, both computed from observations before \(i\). The clip ensures \(1 + \lambda_i(u_i - m_0) > 0\) for every \(u_i \in [0,1]\), so \(M_t\) is a nonnegative martingale under any null distribution on \([0,1]\), and the anytime test rejects at the first \(M_t \ge 1/\alpha\) by Ville's inequality. For \(m_0 = 1/2\) the clip is \(|\lambda| \le 1.98\). This is a two-sided test because the sign of the wager follows the sign of \(\hat\mu - m_0\).

### 4.6 Fixed-\(n\), distribution-free: the betting martingale read once

The same \(M_n\), evaluated only at the committed \(n\); reject if \(M_n \ge 1/\alpha\). Validity is Markov's inequality: \(\mathbb P_{H_0}(M_n \ge 1/\alpha) \le \alpha\, \mathbb E_{H_0}[M_n] = \alpha\). The threshold is the same as in Section 4.5. This row exists so that the difference between it and Section 4.5 is exactly the cost of the anytime property within the distribution-free family.

### 4.7 A second anytime, distribution-free test: the empirical-Bernstein confidence sequence

The predictable plug-in EB confidence sequence of Waudby-Smith and Ramdas (2023, Theorem 2) with \(\lambda_t = \min\big(\sqrt{2\log(2/\alpha)/(\hat\sigma^2_{t-1}\, t\log(t+1))},\ 1/2\big)\), intersected over time; reject when the running intersection excludes \(m_0\). It is variance-adaptive and is included to check whether adaptivity to the noise escapes the cost we attribute to the bound.

### 4.8 Two facts about the thresholds

**Proposition 1 (anytime is free within a martingale family at the 80%-power point).** Let \((M_n)\) be a nonnegative martingale with \(M_0 = 1\) under \(H_0\). The fixed-\(n\) test "reject iff \(M_n \ge 1/\alpha\)" is valid at level \(\alpha\) for every fixed \(n\) (Markov), and the anytime test "reject at the first \(t\) with \(M_t \ge 1/\alpha\)" is valid at level \(\alpha\) (Ville). The anytime test rejects by time \(n\) on every path on which the fixed-\(n\) test rejects at \(n\). Hence for every alternative and every \(n\), \(\mathbb P(\text{anytime rejects by } n) \ge \mathbb P(M_n \ge 1/\alpha)\), and \(n_{80}(\text{anytime}) \le n_{80}(\text{fixed-}n)\).

*Proof.* Validity is Markov's inequality at fixed \(n\) and Ville's maximal inequality over all \(n\); both use the same threshold because \(\mathbb E_{H_0}[M_n] = 1\) and Ville's bound is \(\mathbb P(\sup_t M_t \ge 1/\alpha) \le \alpha \mathbb E[M_0]\). Dominance is immediate: \(M_n \ge 1/\alpha\) implies \(\sup_{t \le n} M_t \ge 1/\alpha\). \(\square\)

The proposition says the anytime version can only be cheaper, and that there is no analogue of the usual "price of peeking" within a test-martingale family. What it does not say is how much cheaper; Section 7.1 shows the answer is: not measurably, on our settings.

**Proposition 2 (informal; seed count of a clipped bet).** Suppose the unit-scale differences have mean \(m_0 + \delta_u\) and variance \(\sigma_u^2\), and the plug-in wager is clipped at \(c\). The log-optimal wager is of order \(\lambda^\ast \approx \delta_u/\sigma_u^2\). If \(\lambda^\ast \gg c\) the clip binds, the per-observation growth of \(\log M_t\) is approximately \(c\,\delta_u - c^2\sigma_u^2/2 \approx c\,\delta_u\), and the number of observations to reach \(\log(1/\alpha)\) is

\[
n_{\mathrm{bet}} \;\approx\; \frac{\log(1/\alpha)}{c\,\delta_u} \;=\; \frac{2h\,\log(1/\alpha)}{c\,\delta} \;\approx\; \frac{h \log(1/\alpha)}{\delta} \quad (c \approx 2,\ m_0 = 1/2),
\]

where \(h\) is the declared half-width and \(\delta\) the raw-scale gap. The count is linear in the declared half-width and does not depend on the noise, whereas the t-test's count is \(\approx 7.85\,\sigma^2/\delta^2\) and does not depend on the bound. The ratio of the two is of order \(0.4\,\delta\,h/(c\,\sigma^2)\), which grows as the noise shrinks relative to the interval.

*Derivation sketch.* \(\mathbb E\log(1 + \lambda(u - m_0)) \approx \lambda\delta_u - \lambda^2(\sigma_u^2 + \delta_u^2)/2\) by second-order expansion, maximized at \(\lambda^\ast = \delta_u/(\sigma_u^2 + \delta_u^2)\); with \(\lambda = c < \lambda^\ast\) the first term dominates. Appendix D compares the prediction to the measured \(n_{80}\): predicted 98 vs measured 111 on BBF−SPR aggregate, 22 vs 27, 23 vs 32, 26 vs 44 on the three beta settings. The gap is from the plug-in wager's warm-up and from the finite-sample slack in Ville's inequality; the trend is right.

---

## 5. Data

### 5.1 Synthetic paired differences

**Beta differences.** \(x = \mathrm{Beta}(a, b) - \mathrm{Beta}(b, a)\) with concentration \(a + b \in \{8, 24, 80\}\) and \((a - b)/(a + b) = 0.14\), so the mean gap is 0.14 and the standard deviation is 0.234, 0.141, or 0.077. Each term lies in \([0,1]\), so the difference lies in \([-1, 1]\) and that is the declared bound. These are the settings of the earlier draft, and they sweep the ratio of half-width to standard deviation from 4.3 to 13. The matching nulls set \(a = b\).

**Other nulls.** Gaussian with standard deviation 0.1; scaled \(t_3\) with standard deviation 0.1 (heavy tails); and a centered exponential with standard deviation 0.1 (skewed, with a hard lower edge at \(-0.1\) and a long right tail). All clipped to \([-1, 1]\); the clip was active on fewer than 0.1% of \(t_3\) draws and never on the others.

### 5.2 Real seeds: BBF on Atari-100k

Schwarzer et al. (2023) released per-run human-normalized scores for BBF and its ablations on 55 Atari games (the 26-game Atari-100k suite and 29 more), with 12–60 runs per game per configuration (12–14 for BBF RR2, 28–30 for BBF−WD, 16–60 for RR8). We use six configurations: BBF at replay ratio 2 (the main method), BBF without SPR, without periodic resets, without weight decay, with sticky actions, and BBF at replay ratio 8.

We clip each human-normalized score to \([0, 1]\). This is the quantity from which `rliable`'s optimality gap is built (Agarwal et al., 2021), it is bounded by construction, and a paired difference of two clipped scores lies in \([-1, 1]\). The declared bound is therefore not a guess about the data; it is a property of the metric. The cost is that the metric saturates on games where an agent exceeds human performance (ten games, among them Breakout and Boxing, have zero variance after clipping for BBF RR2), which is why we chose single games by a pre-stated rule (below).

Two kinds of seed stream are built from these pools.

- An **aggregate seed** draws one run per game for each algorithm, independently and uniformly from that game's pool, and takes the mean clipped score over the 55 games. The paired difference of two aggregate seeds is a mean of 55 bounded terms and is close to Gaussian; its standard deviation is 0.028–0.030 in every comparison we ran, so the ratio of half-width to standard deviation is about 34.
- A **single-game seed** is the paired difference of clipped scores on one game, one run drawn from each pool. These differences are heavy-tailed, often bimodal, and have standard deviations from 0.03 to 0.44.

A **true null with real seed noise** is BBF against itself, with the two sides drawn independently from the same pool. Seeds beyond the pool size are bootstrap replicates: a 200-seed path reuses runs. This is the right object for a sample-size calculation about the population those runs estimate, but it is not 200 fresh runs, and we return to this in Sections 11 and 14.

**Comparisons.** BBF (RR2) against: BBF−SPR (aggregate gap +0.031), BBF−Resets (+0.102), BBF−WD (+0.009), BBF+sticky (+0.027), and BBF at RR8 (−0.030). Single games were fixed before any test was run by one rule: BBF's clipped score on the game has mean in \((0.05, 0.95)\) and standard deviation above 0.02, so clipping has not removed the signal. Eight games pass: Alien, Asterix, ChopperCommand, Frostbite, Hero, Kangaroo, Qbert, Zaxxon. We run each against BBF−SPR and BBF RR8, giving 16 single-game comparisons with gaps from −0.36 to +0.23.

### 5.3 Real seeds, two more suites: TD-MPC2 on DMControl and Meta-World

Hansen, Su, and Wang (2024) released per-seed learning curves for TD-MPC2 and three baselines (DreamerV3, SAC, TD-MPC) on 39 DMControl tasks (episode reward in \([0, 1000]\)) and 50 Meta-World tasks (success rate in \([0, 1]\)), three seeds per task per algorithm (`data/tdmpc2/`; the 15 MyoSuite tasks are excluded). We score a run by its final evaluation, DMControl reward divided by 1,000 and Meta-World success as is, so every score is in \([0, 1]\) by construction and a paired difference is in \([-1, 1]\). Mean scores: DMControl 0.887 / 0.682 / 0.591 / 0.730 for TD-MPC2 / DreamerV3 / SAC / TD-MPC; Meta-World 0.957 / 0.407 / 0.755 / 0.905.

The same two constructions apply. An **aggregate seed** draws one run per task for each algorithm and averages over the 39 or 50 tasks; with three runs per task the per-task draw is thin, but the aggregate over 39–50 tasks has \(3^{39}\)–\(3^{50}\) distinct values and a standard deviation of 0.009–0.041, so these are the Gaussian, loose-bound regime of the BBF aggregates, with much larger gaps (0.05–0.55 against BBF's 0.01–0.10). A **five-task slice** draws a random subset of five tasks per path and aggregates over those, which is the "handful of tasks" benchmark common in papers; its standard deviation is 0.09–0.17 against the same bound. Nulls are TD-MPC2 against itself and SAC against itself. Comparisons: TD-MPC2 against each baseline, DreamerV3 against SAC, TD-MPC against SAC, on both suites, plus a declared-bound sweep on the smallest-gap comparison (TD-MPC2 vs TD-MPC).

### 5.4 Fresh seeds: 480 DQN runs on four environments

Every seed stream so far is a bootstrap from a finite pool of released runs. To have paired differences with no resampling at all we trained three DQN variants ourselves, 480 runs in total (`s4_fresh_runs.py`): **DQN** (two hidden layers of 64 or 128 SiLU units, Adam at \(10^{-3}\), Polyak target \(\tau = 0.005\), batch 64, replay 50,000, \(\varepsilon\) from 1 to 0.05 over the first 40% of training, Huber loss), **Double DQN** (the same with the double-Q target), and **DQN at learning rate \(3 \times 10^{-4}\)**, on CartPole-v1 (30,000 steps), Acrobot-v1 (50,000), and MinAtar Breakout and Space Invaders (100,000 steps, 1,000-step episode cap; Young and Tian, 2019). Seeds: 60 / 40 / 40 per agent on the classic-control tasks and 40 / 30 / 30 on MinAtar. A run's score is the mean return of ten evaluation episodes at the end of training, mapped to \([0, 1]\) by a rule fixed before training: CartPole \(R/500\), Acrobot \((R + 500)/500\), MinAtar \(\min(R/100, 1)\). The declared bound is therefore \([-1, 1]\) by construction, as for the released data.

| Environment | DQN | Double DQN | DQN, lr \(3 \times 10^{-4}\) |
|---|---|---|---|
| CartPole (\(n = 60/40/40\)) | 0.293 ± 0.225 | 0.287 ± 0.232 | 0.251 ± 0.274 |
| Acrobot (60/40/40) | 0.631 ± 0.293 | 0.705 ± 0.177 | 0.314 ± 0.327 |
| Breakout (40/30/30) | 0.111 ± 0.034 | 0.116 ± 0.032 | 0.070 ± 0.014 |
| Space Invaders (40/30/30) | 0.319 ± 0.104 | 0.336 ± 0.128 | 0.328 ± 0.097 |

Mean ± standard deviation of the normalized score over seeds. The scores span the regimes of the paper: CartPole and Acrobot are bimodal (a seed either solves the task or does not, so scores sit near 0.1–0.2 or near 0.8–1.0) with standard deviations of 0.2–0.3 against a half-width of 1; Breakout is tight (sd 0.014–0.034, \(h/\mathrm{sd} \approx 30\)); Space Invaders is in between. The gaps are small or zero on CartPole and Space Invaders (every pairwise gap within 0.05), and large on Acrobot (DQN and Double DQN beat the low learning rate by 0.32 and 0.39) and Breakout (by 0.041 and 0.046, with standardized effects \(\delta/\sigma\) of 1.1–1.3).

Two kinds of stream are built from these runs, both free of resampling. A **genuine paired stream** for A against B is a random disjoint pairing of A's seeds with B's: a permutation of each pool, paired in order, giving \(\min(|A|, |B|)\) i.i.d. paired differences in which no run appears twice; different permutations give different streams, so the 400 paths per setting are an exact permutation ensemble rather than a bootstrap. The cap is the pool size, 30 or 40. A **genuine null stream** pairs an agent's seeds with themselves after a random permutation, disjointly: an exact null with real seed noise, 30 (DQN) or 20 (Double DQN) pairs per path, 1,000 paths. For comparability with the released data we also run the same pools as bootstrap streams to the usual cap of 200.

---

## 6. Validity

![False-rejection rates on ten nulls](figures/fig3_nulls.png)

*Figure 1. False-rejection rate within 200 seeds on ten true nulls, 1,000 paths each. Dashed line at \(\alpha = 0.05\). The t-test with a look after every seed rejects 36–54% of true nulls. Every other method holds \(\alpha\), except the parametric ones on the skewed null.*

| Null | sd | peek t | Pocock | OBF | anytime t | anytime t (\(g{=}0.3\)) | betting, anytime | betting, one look (\(n{=}200\)) | EB-CS | one-look t (\(n{=}10\)) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| beta, conc. 8 | 0.239 | 0.434 | 0.051 | 0.049 | 0.034 | 0.027 | 0.030 | 0.004 | 0.000 | 0.046 |
| beta, conc. 24 | 0.141 | 0.446 | 0.055 | 0.061 | 0.036 | 0.026 | 0.014 | 0.007 | 0.000 | 0.039 |
| beta, conc. 80 | 0.078 | 0.414 | 0.052 | 0.034 | 0.031 | 0.023 | 0.001 | 0.001 | 0.000 | 0.049 |
| Gaussian | 0.100 | 0.420 | 0.051 | 0.047 | 0.030 | 0.025 | 0.003 | 0.003 | 0.000 | 0.046 |
| \(t_3\) | 0.093 | 0.418 | 0.050 | 0.055 | 0.028 | 0.020 | 0.006 | 0.004 | 0.000 | 0.048 |
| skewed exponential | 0.098 | 0.539 | 0.055 | 0.058 | **0.109** | 0.051 | 0.003 | 0.001 | 0.000 | **0.111** (at \(n{=}5\): 0.12) |
| BBF vs BBF, aggregate | 0.029 | 0.427 | 0.057 | 0.046 | 0.040 | 0.025 | 0.000 | 0.000 | 0.000 | 0.052 |
| BBF vs BBF, Pong | 0.186 | 0.393 | 0.036 | 0.044 | 0.017 | 0.016 | 0.022 | 0.009 | 0.000 | 0.034 |
| BBF vs BBF, Kangaroo | 0.359 | 0.361 | 0.047 | 0.054 | 0.014 | 0.018 | 0.023 | 0.002 | 0.000 | 0.035 |
| BBF vs BBF, Frostbite | 0.312 | 0.413 | 0.049 | 0.043 | 0.028 | 0.018 | 0.024 | 0.002 | 0.000 | 0.050 |

Standard error at 1,000 paths is about 0.007 near 0.05. Five observations.

1. **Peeking is the failure.** Looking after every seed and stopping at the first \(p < 0.05\) rejects 36–54% of true nulls within 200 seeds, on every distribution including the near-Gaussian aggregate. This is the practice the rest of the paper exists to replace.
2. **The group-sequential designs sit at \(\alpha\)** on every null, including the skewed one, because their first look is at \(n_{\max}/5 \ge 40\), by which point the t-statistic is close to Gaussian even on skewed data.
3. **The anytime t e-process and the one-look t-test fail in the same place.** On the skewed null the e-process rejects 10.9% and the one-look t-test at \(n = 5\) rejects 12%. The e-process is a likelihood ratio under the Gaussian model and inherits exactly the small-\(n\) non-normality the t-test suffers; it is not a new failure, it is the same one made visible at every \(n\). The narrower prior (\(g = 0.3\)) halves it to 5.1% because it bets less aggressively early. On the other nine nulls the e-process is conservative (1.4–4.0%).
4. **The betting test never exceeds 3.0%**, and is very conservative when the noise is far inside the bound (0.000–0.003 on the aggregate, Gaussian, and conc.-80 nulls). That conservatism is the same mechanism that costs it power in Section 7.
5. **The EB confidence sequence never rejected a true null in 10,000 paths**, which foreshadows its power.

The six TD-MPC2 nulls (Section 10; Appendix C) and the eight exact nulls on fresh DQN seeds (Section 11) repeat the pattern: peeking 39–44% and 17–27% (within 200 and within 15–30 looks respectively), the group-sequential designs and the one-look t-test at \(\alpha\), the e-process and the betting test below it.

---

## 7. Cost, decomposed

![Seeds to 80% power for seven tests on six settings](figures/fig1_n80_decomposition.png)

*Figure 2. Seeds to 80% power. Top row: beta differences with gap 0.14 at three concentrations, declared bound \([-1, 1]\). Bottom row: BBF aggregate comparisons, bound \([-1, 1]\) by construction. Blue and cyan: parametric fixed-\(n\) and group-sequential. Purple: parametric anytime e-process. Orange and brown: the same betting martingale read once or stopped at first crossing. Gray: EB confidence sequence.*

| Setting | gap | sd | \(h/\mathrm{sd}\) | one-look t | Pocock | OBF | anytime t | betting, one look | betting, anytime | EB-CS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| beta, conc. 8 | 0.140 | 0.234 | 4.3 | 23 | 35 | 35 | 43 | 44 | 42 | 131 |
| beta, conc. 24 | 0.142 | 0.141 | 7.1 | 11 | 16 | 16 | 18 | 32 | 32 | 118 |
| beta, conc. 80 | 0.139 | 0.077 | 13 | 5 | 8 | 8 | 9 | 27 | 27 | 112 |
| beta, conc. 80, bound \([-0.5, 0.5]\) | 0.141 | 0.078 | 6.4 | 5 | 8 | 8 | 9 | 16 | 16 | 57 |
| BBF vs BBF−SPR, aggregate | +0.031 | 0.029 | 34 | 9 | 12 | 16 | 16 | 111 | 111 | >200 |
| BBF vs BBF−Resets, aggregate | +0.102 | 0.029 | 34 | 3 | 4 | 6 | 6 | 34 | 34 | 149 |
| BBF vs BBF−WD, aggregate | +0.009 | 0.029 | 34 | 83 | 133 | 133 | 138 | >200 | >200 | >200 |
| BBF vs BBF+sticky, aggregate | +0.027 | 0.029 | 34 | 11 | 18 | 18 | 19 | 126 | 126 | >200 |
| BBF RR2 vs RR8, aggregate | −0.030 | 0.028 | 34 | 10 | 12 | 16 | 17 | 114 | 114 | >200 |

Entries are \(n_{80}\); ">200" means 80% power was not reached within the cap. Expected stopping times are in Appendix C.

### 7.1 Anytime is free inside the betting family

The betting martingale read once at a fixed \(n\) and the same martingale stopped at its first crossing have the same \(n_{80}\) on every row: 44/42, 32/32, 27/27, 16/16, 111/111, 34/34, 126/126, 114/114. Proposition 1 says the anytime version cannot be worse; the data say it is not measurably better either, because on the alternative the wealth process is close to monotone and the first crossing of \(1/\alpha\) happens within a seed or two of the time at which the fixed-\(n\) reader would have seen it. The earlier draft attributed the whole multiplier to "the price of validity." The price of *anytime* validity, holding the distributional assumption fixed, is zero at the 80%-power point.

### 7.2 Anytime costs 1.6–2.0× inside the parametric family, with no maximum \(n\)

The e-process needs 43/23, 18/11, 9/5 seeds on the beta settings (1.9×, 1.6×, 1.8×) and 16/9, 6/3, 138/83, 19/11, 17/10 on the real aggregates (1.8×, 2.0×, 1.7×, 1.7×, 1.7×). This is the cost of being allowed to stop at any time while controlling \(\alpha\) under the Gaussian model, and it comes with no planned maximum and no power analysis.

The five-look group-sequential designs are a little cheaper at \(n_{80}\) (Pocock 35, 16, 8; 12, 4, 133, 18, 12) because they were handed a planned \(n_{\max}\) of twice the t-test's \(n_{80}\), which is information the e-process does not get, and they cannot run past it. In expected stopping time the two are close: on BBF vs BBF−SPR, Pocock stops at a mean of 11 seeds, OBF at 13, the e-process at 12, against the one-look t-test's committed 9. On the hardest real comparison, BBF vs BBF−WD (gap 0.009, sd 0.029, 83 seeds for a one-look t-test), the e-process rejects on 96% of paths with a mean stopping time of 92 seeds. Someone willing to assume approximate normality who wants to stop whenever they like pays 1.1–1.7× in expected seeds and 1.6–2.0× at the 80%-power point. Either number is small next to the 36–54% false-rejection rate of the protocol it replaces.

### 7.3 The bound is the cost

The betting test's multiplier over the one-look t-test is 1.8× at sd 0.234, 2.9× at 0.141, 5.4× at 0.077, and 11–12× at 0.029. The ratio of declared half-width to standard deviation in those rows is 4.3, 7.1, 13, and 34. Halving the declared bound on the conc.-80 setting (fourth row), which is legitimate there because beta differences with concentration 80 never exceeded 0.45 in magnitude in 160,000 draws, cuts the betting cost from 27 to 16 seeds and leaves every parametric test unchanged. Nothing else in the row moved. The bound is the cost, and Section 8 confirms this with a sweep.

The absolute counts behave as Proposition 2 predicts. The betting \(n_{80}\) on the three beta settings is 42, 32, 27 while the noise falls by a factor of three and the t-test's count falls from 23 to 5; the betting count is governed by the gap on the unit scale (\(0.07\) in all three rows) and barely by the noise. Halving the half-width halves the predicted count (22 to 11) and the measured one falls from 27 to 16.

### 7.4 Adaptivity to the variance does not escape the bound

The empirical-Bernstein confidence sequence needs 112–131 seeds on the beta settings, 149 on the easiest real aggregate, and never reaches 80% power on the other four within 200 seeds. It is a two-sided interval (a one-sided test built from the same bound would be cheaper), its \(\lambda_t\) is also capped (at 1/2), and its width includes a \(\log(2/\alpha)\) term that the plug-in bet does not pay. Variance adaptivity does not help because the binding constraint is not an unknown variance; it is the reservation of capital for an observation at the edge of the declared interval, which is a property of the bound and not of the data.

---

## 8. The bound, swept

![Betting cost against declared bound](figures/fig2_bound_sweep.png)

*Figure 3. BBF vs BBF−SPR, aggregate seeds, sd 0.029. Seeds to 80% power for the anytime betting test as the declared half-width shrinks from 1 to 0.05. Red labels mark bounds that at least one observation in a path violated; the test's guarantee does not hold on those paths.*

| Declared half-width \(h\) | \(h/\mathrm{sd}\) | betting \(n_{80}\) | \(\mathbb E[n]\) | paths with a violation | predicted \(h\log(1/\alpha)/\delta\) | anytime t | one-look t |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 34 | 111 | 103 | 0% | 97 | 16 | 9 |
| 0.50 | 17 | 60 | 54 | 0% | 48 | 17 | 10 |
| 0.25 | 8.6 | 34 | 30 | 0% | 24 | 17 | 10 |
| 0.10 | 3.4 | 19 | 15 | **83%** | 10 | 17 | 9 |
| 0.05 | 1.7 | 16 | 12 | **100%** | 5 | 17 | 10 |

Declaring the true bound costs 111 seeds; 0.5 costs 60; 0.25 costs 34; 0.1 costs 19; 0.05 costs 16. The last two are within 3 seeds of the parametric e-process. But at half-width 0.1 some observation left the interval on 83% of 200-seed paths, and at 0.05 on 100% of them. On those paths the observation was clipped before the wager was applied, so the martingale ran on a different data stream than the one observed and the \(\alpha\) guarantee is void. A bound of 0.25 (8.6 standard deviations) held on every one of 400 paths and costs 34 seeds, 3.8× the t-test and 2.1× the e-process. A bound of 0.5 (17 standard deviations) held and costs 60.

**Mechanism.** Near \(m_0 = 1/2\) on the unit interval the betting fraction cannot exceed about 2, because a single observation at 0 or 1 would otherwise make the capital negative. The growth-optimal fraction for a shift \(\delta_u\) with variance \(\sigma_u^2\) is \(\delta_u/\sigma_u^2\). On the aggregate data after rescaling to \([0,1]\), the shift is 0.0155 and the variance is \(2 \times 10^{-4}\), so the uncapped fraction would be about 75 and the test is forced to bet at 2, forty times less than it would like. A t-test does not reserve capital for a jump to the end of the interval. That reservation is what the extra seeds buy, and on the skewed null in Section 6 it is what kept the betting test at 0.3% while the parametric tests were at 11%.

**What the sweep says about practice.** The temptation is to look at the first few seeds, note that nothing is within a tenth of the declared bound, and tighten it. On these data a half-width of 0.1 is 3.4 standard deviations, looks safe from ten seeds, and is violated on 83% of 200-seed paths. The only bounds that held on every path were the ones at 8.6 standard deviations or more, and they cost 3.8× and 6.7× the t-test. There is no free tightening.

---

## 9. When the noise fills the bound, the two families agree

![Single games](figures/fig4_games.png)

*Figure 4. Single-game comparisons, BBF against two ablations, declared bound \([-1,1]\). Each point is a game; horizontal is the anytime t e-process's \(n_{80}\), vertical the anytime betting test's. Color is the standard deviation of the paired difference. Comparisons where no test reached 80% power within 200 seeds are omitted.*

| Game, comparison | gap | sd | \(h/\mathrm{sd}\) | one-look t | Pocock | anytime t | betting | \(\mathbb E[n]\) anytime t | \(\mathbb E[n]\) betting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ChopperCommand, RR8 | −0.358 | 0.438 | 2.3 | 15 | 20 | 26 | **26** | 18 | 19 |
| ChopperCommand, −SPR | +0.227 | 0.342 | 2.9 | 18 | 27 | 36 | **30** | 27 | 24 |
| Frostbite, RR8 | −0.151 | 0.350 | 2.9 | 46 | 69 | 77 | **72** | 51 | 48 |
| Kangaroo, RR8 | −0.112 | 0.266 | 3.8 | 44 | 67 | 85 | **64** | 65 | 50 |
| Zaxxon, −SPR | +0.091 | 0.206 | 4.9 | 43 | 70 | 79 | **71** | 53 | 56 |
| Zaxxon, RR8 | −0.127 | 0.201 | 5.0 | 23 | 37 | **40** | 46 | 28 | 36 |
| Asterix, RR8 | −0.105 | 0.196 | 5.1 | 24 | 37 | 55 | **52** | 40 | 41 |
| Qbert, RR8 | −0.073 | 0.168 | 6.0 | 47 | 66 | 83 | **77** | 56 | 60 |
| Asterix, −SPR | −0.186 | 0.164 | 6.1 | 7 | 11 | **13** | 23 | 11 | 20 |
| Hero, −SPR | +0.028 | 0.098 | 10 | 95 | 160 | **160** | 184 | 105 | 139 |
| Hero, RR8 | −0.072 | 0.088 | 11 | 14 | 18 | **25** | 56 | 18 | 49 |
| Alien, −SPR | −0.015 | 0.041 | 24 | 55 | 90 | **102** | >200 (33%) | 77 | 192 |
| Alien, RR8 | −0.008 | 0.028 | 36 | 118 | 160 | **190** | >200 (0%) | 121 | 200 |

Rows are sorted by \(h/\mathrm{sd}\). Bold marks the cheaper of the two anytime tests. Three comparisons (Qbert −SPR, Kangaroo −SPR, Frostbite −SPR, gaps 0.02–0.06 against sd 0.17–0.38) did not reach 80% power with any test within 200 seeds and are omitted.

Single-game differences of clipped scores are heavy-tailed, often bimodal, and have standard deviations up to 0.44 against a declared half-width of 1. Where \(h/\mathrm{sd}\) is below about 6, the betting test and the parametric e-process need about the same number of seeds, and the betting test is ahead on six of eight such rows and tied on one (ChopperCommand RR8 26 vs 26; −SPR 30 vs 36; Frostbite 72 vs 77; Kangaroo 64 vs 85; Zaxxon −SPR 71 vs 79; Asterix RR8 52 vs 55; Qbert 77 vs 83). In expected stopping time the betting test is ahead or within three seeds on all of them. Where \(h/\mathrm{sd}\) is above 10, the parametric test is 2× cheaper (Hero RR8: 25 vs 56) or the betting test does not finish (Alien). That is the curve of Figure 3 read on real data. It is where the distribution-free test is a good deal: when a comparison is noisy enough that the noise is a real fraction of the interval it was proved for, the distributional guarantee comes for roughly nothing, and on the bimodal single-game distributions that guarantee is worth having.

The one-look t-test is cheapest on every row, as it must be: it has neither property.

---

## 10. Two more suites: when the effect is enormous, the bound still sets the count

The BBF aggregates have gaps of 0.01–0.10 at a standard deviation of 0.03. The TD-MPC2 benchmarks are the opposite case: a new method that beats its baselines by 0.05–0.55 on the aggregate, with standard deviations of 0.009–0.041, so the standardized effect \(\delta/\sigma\) is 2–24 and a one-look t-test needs two to four seeds. If the betting test's cost were a multiplier on the t-test's, it would need four to eight. Proposition 2 says instead that its count is \(\approx h\log(1/\alpha)/\delta\), independent of the noise, which at \(h = 1\) and \(\alpha = 0.05\) is \(3.0/\delta\): 15 seeds at a gap of 0.2, 58 at a gap of 0.05.

| Suite, comparison | gap | sd | \(h/\mathrm{sd}\) | one-look t | Pocock | OBF | anytime t | betting | EB-CS | Prop. 2: \(3.0/\delta\) | \(\mathbb E[n]\): anytime t / betting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DMC, TD-MPC2 vs DreamerV3 | +0.205 | 0.009 | 111 | 2 | 4 | 4 | 5 | 18 | 73 | 15 | 5 / 18 |
| DMC, TD-MPC2 vs SAC | +0.296 | 0.023 | 43 | 2 | 4 | 4 | 5 | 13 | 51 | 10 | 5 / 13 |
| DMC, TD-MPC2 vs TD-MPC | +0.157 | 0.023 | 43 | 3 | 4 | 6 | 5 | 23 | 96 | 19 | 5 / 22 |
| DMC, DreamerV3 vs SAC | +0.091 | 0.023 | 43 | 3 | 4 | 6 | 6 | 37 | 165 | 33 | 5 / 36 |
| DMC, TD-MPC vs SAC | +0.139 | 0.031 | 32 | 3 | 4 | 6 | 6 | 26 | 108 | 22 | 5 / 25 |
| MW, TD-MPC2 vs DreamerV3 | +0.551 | 0.032 | 31 | 2 | 4 | 4 | 5 | 9 | 28 | 5 | 5 / 8 |
| MW, TD-MPC2 vs SAC | +0.201 | 0.033 | 30 | 3 | 4 | 6 | 5 | 18 | 75 | 15 | 5 / 18 |
| MW, TD-MPC2 vs TD-MPC | +0.052 | 0.022 | 45 | 4 | 6 | 6 | 7 | 64 | >200 | 58 | 6 / 61 |
| MW, DreamerV3 vs SAC | −0.348 | 0.040 | 25 | 3 | 4 | 4 | 5 | 12 | 44 | 9 | 5 / 12 |
| MW, TD-MPC vs SAC | +0.151 | 0.033 | 30 | 3 | 4 | 6 | 6 | 24 | 101 | 20 | 5 / 23 |
| DMC, TD-MPC2 vs TD-MPC, 5-task slice | +0.162 | 0.105 | 9.5 | 5 | 6 | 8 | 8 | 42 | 186 | 19 | 7 / 34 |
| DMC, TD-MPC2 vs DreamerV3, 5-task slice | +0.221 | 0.130 | 7.7 | 3 | 4 | 6 | 5 | 32 | 140 | 14 | 5 / 23 |
| DMC, TD-MPC vs SAC, 5-task slice | +0.152 | 0.169 | 5.9 | 9 | 13 | 13 | 16 | 47 | 198 | 20 | 23 / 42 |
| MW, TD-MPC2 vs TD-MPC, 5-task slice | +0.066 | 0.089 | 11 | 34 | 59 | 59 | 71 | >200 | >200 | 45 | 44 / 95 |
| MW, TD-MPC2 vs DreamerV3, 5-task slice | +0.570 | 0.145 | 6.9 | 3 | 4 | 6 | 5 | 10 | 33 | 5 | 5 / 9 |
| MW, TD-MPC vs SAC, 5-task slice | +0.140 | 0.138 | 7.2 | 11 | 18 | 18 | 21 | 57 | >200 | 21 | 26 / 47 |

Betting one-look and betting anytime coincide on every row, as on every row of Section 7 (Appendix C). Declared bound \([-1, 1]\) throughout. Nulls on these data (TD-MPC2 vs itself and SAC vs itself on both suites, aggregate and five-task slice) behave as the BBF nulls did: the peeking t-test rejects 39–44%, Pocock and OBF 3.8–5.8%, the e-process 2.6–3.7%, the betting test 0.0–0.5%, EB 0.0% (Appendix C).

**The count follows the gap, not the noise.** Across the ten aggregate rows the t-test needs 2–4 seeds while the betting test needs 9–64, and the betting count is ordered by the gap: 9 at 0.55, 12–13 at 0.30–0.35, 18 at 0.20, 23–26 at 0.14–0.16, 37 at 0.09, 64 at 0.05. Proposition 2's \(3.0/\delta\) is 10–30% low on eight of the ten rows and 40–70% low on the two largest gaps (0.35, 0.55), where the predicted count is under ten and the plug-in wager's first few observations, spent estimating the mean, are a large fraction of it. The multiplier over the t-test is 4–16×, the largest in the paper, precisely because the t-test is so cheap here. The e-process has a floor: it needs five observations on every aggregate row regardless of the effect, because with one to three degrees of freedom the likelihood ratio of a \(t\)-statistic under the \(g = 1\) prior does not reach 20 however large the statistic is (an infinite \(T_n\) gives \(\log E_n = 0.55, 1.39, 2.41, 3.58\) at \(n = 2, 3, 4, 5\) against the threshold \(\log 20 = 3.0\)). Its multiplier over the t-test is therefore 1.7–2.5× on these rows, slightly above the 1.6–2.0× of Section 7.2, for the same reason.

**The bound sweep repeats.** On TD-MPC2 vs TD-MPC the betting count goes 23 → 13 → 8 on DMControl and 64 → 34 → 19 on Meta-World as the half-width goes 1 → 0.5 → 0.25, with no violations; at \(h = 0.1\) it is 6 and 10, with the bound violated on 100% and 85% of paths. Linear in \(h\), as in Section 8.

**Five-task slices.** Averaging over five tasks instead of 39–50 raises the standard deviation to 0.09–0.17 and moves \(h/\mathrm{sd}\) from 25–111 to 6–11. The t-test now needs 3–34 seeds, the e-process 5–71, and the betting test 10 to more than 200; the multiplier over the t-test falls to 3–11× and over the e-process to 2–6×. This is the intermediate regime between the aggregates and the single games of Section 9, and it lands where Figure 5 says it should: at \(h/\mathrm{sd} \approx 6\)–\(11\) the distribution-free guarantee still costs a factor of two to six over the parametric anytime test.

![Multiplier against h/sd](figures/fig5_multiplier_vs_ratio.png)

*Figure 5. Every real-data power setting in the paper (BBF aggregates and single games, DMControl and Meta-World aggregates and five-task slices, fresh DQN runs). Left: the anytime betting test's \(n_{80}\) as a multiple of the one-look t-test's, against the declared half-width over the standard deviation of the paired differences; black ticks are Proposition 2's prediction; open markers are lower bounds where 80% power was not reached in 200 seeds. Right: the same for the parametric anytime e-process.*

---

## 11. Fresh seeds: genuine paired streams without resampling

Section 5.4 describes the 480 runs. Here every path is a disjoint random pairing of real runs: no run is reused within a path, and the only randomness across paths is the pairing. The cap is the pool size (30 or 40 pairs for a comparison, 20 or 30 for a null), which is the honest horizon of a 30–60-seed experiment.

**Exact nulls on fresh seeds.** Eight genuine nulls (DQN and Double DQN against themselves on four environments, 1,000 permutations each, 20–30 pairs per path):

| Null (pairs per path) | sd | peek t | Pocock | OBF | anytime t | betting | EB-CS | one-look t (\(n = 10\)) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CartPole, DQN (30) | 0.316 | 0.204 | 0.022 | 0.041 | 0.002 | 0.006 | 0.000 | 0.042 |
| CartPole, Double DQN (20) | 0.331 | 0.167 | 0.013 | 0.046 | 0.002 | 0.000 | 0.000 | 0.021 |
| Acrobot, DQN (30) | 0.419 | 0.236 | 0.045 | 0.049 | 0.012 | 0.022 | 0.000 | 0.058 |
| Acrobot, Double DQN (20) | 0.242 | 0.184 | 0.024 | 0.046 | 0.003 | 0.000 | 0.000 | 0.035 |
| Breakout, DQN (20) | 0.048 | 0.268 | 0.051 | 0.061 | 0.024 | 0.000 | 0.000 | 0.060 |
| Breakout, Double DQN (15) | 0.047 | 0.233 | 0.059 | 0.046 | 0.015 | 0.000 | 0.000 | 0.041 |
| Space Invaders, DQN (20) | 0.144 | 0.258 | 0.067 | 0.054 | 0.024 | 0.000 | 0.000 | 0.053 |
| Space Invaders, Double DQN (15) | 0.186 | 0.226 | 0.048 | 0.045 | 0.014 | 0.000 | 0.000 | 0.047 |

The peeking t-test rejects 17–27% of these exact nulls within 15–30 looks (it reached 36–54% within 200 looks on the bootstrap streams; the inflation grows with the number of looks, as it must). Every valid test holds \(\alpha\) within sampling error (standard error 0.007; the largest entry is Pocock's 0.067 on Space Invaders), including on the bimodal CartPole and Acrobot differences, where a single seed difference can be \(\pm 0.8\). The anytime t e-process is at 0.2–2.4% and the betting test at 0.0–2.2%.

**Paired comparisons on fresh seeds.** Four of the twelve comparisons have a gap worth detecting within the pool size; the other eight (every CartPole and Space Invaders pair, and DQN vs Double DQN everywhere) have gaps of 0.002–0.077 against standard deviations of 0.05–0.35 and no test reaches 80% power at 30–40 pairs, or, on the bootstrap streams, within 200 (the one exception is Acrobot DQN vs Double DQN, gap −0.07, where the one-look t-test reaches 80% at 176 bootstrap seeds).

| Comparison | gap | sd | \(h/\mathrm{sd}\) | stream (cap) | one-look t | Pocock | OBF | anytime t | betting | EB-CS | \(\mathbb E[n]\): anytime t / betting |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| Acrobot, DQN vs lr \(3{\times}10^{-4}\) | +0.321 | 0.430 | 2.3 | genuine (40) | 18 | 27 | 27 | 28 | **28** | >40 | 19 / 20 |
| | +0.320 | 0.436 | 2.3 | bootstrap (200) | 17 | 23 | 30 | 31 | 29 | 63 | 21 / 21 |
| Acrobot, Double DQN vs lr \(3{\times}10^{-4}\) | +0.391 | 0.368 | 2.7 | genuine (40) | 9 | 12 | 16 | 16 | **16** | >40 | 12 / 13 |
| | +0.390 | 0.368 | 2.7 | bootstrap (200) | 10 | 14 | 14 | 17 | 16 | 46 | 12 / 13 |
| Breakout, DQN vs lr \(3{\times}10^{-4}\) | +0.041 | 0.036 | 28 | genuine (30) | 8 | 13 | 13 | 14 | **>30 (0%)** | >30 | 11 / 30 |
| | +0.041 | 0.036 | 28 | bootstrap (200) | 8 | 11 | 11 | 14 | 85 | >200 | 11 / 79 |
| Breakout, Double DQN vs lr \(3{\times}10^{-4}\) | +0.046 | 0.035 | 29 | genuine (30) | 7 | 11 | 11 | 11 | **>30 (0%)** | >30 | 9 / 30 |
| | +0.047 | 0.035 | 29 | bootstrap (200) | 7 | 11 | 11 | 11 | 75 | >200 | 9 / 70 |

Three things follow.

1. **The bootstrap streams were telling the truth.** Wherever both streams reach 80% power, the genuine and bootstrap \(n_{80}\) agree within 1–4 seeds for every test (18/17, 27/23, 27/30, 28/31, 28/29; 9/10, 12/14, 16/14, 16/17, 16/16; 8/8, 13/11, 14/14; 7/7, 11/11, 11/11), and the expected stopping times agree to within a seed. The resampled streams of Sections 7–10 are a sample-size calculation for the population the pools estimate, and on these four comparisons the calculation is right.

2. **The two regimes of Section 9 reappear on fresh data.** On Acrobot the paired difference is bimodal (a seed either learns to swing up or does not) with \(h/\mathrm{sd} = 2.3\)–\(2.7\), and the betting test needs exactly as many fresh pairs as the parametric anytime test (28 vs 28, 16 vs 16) while being valid for that bimodal distribution. On Breakout the difference is tight, \(h/\mathrm{sd} \approx 28\), the t-test needs 7–8 seeds, the e-process 11–14, and the betting test rejects on 0 of 400 genuine 30-pair streams (its wealth cannot reach 20 in 30 steps at a wager of 2 and a unit-scale gap of 0.02); the bootstrap extension puts it at 75–85, where Proposition 2 predicts \(3.0/0.046 = 65\) and \(3.0/0.041 = 73\) (13–14% low, as on the released data).

3. **The 30–40-seed horizon is short.** With an effect of \(\delta/\sigma \approx 0.75\) (Acrobot, DQN vs the low learning rate) the one-look t-test needs 18 seeds and every anytime test 27–28, which is most of a 40-seed budget; with \(\delta/\sigma \approx 0.1\)–\(0.2\) (the other eight comparisons) nothing is detectable at this scale by any test, valid or not, and the peeking protocol would still have declared a winner 17–27% of the time.

---

## 12. Recommendations

The results reduce to a decision keyed to what the experimenter is willing to assume.

1. **Do not look after every seed and stop at the first \(p < 0.05\).** It rejected a true null 36–54% of the time in every null we tried, including near-Gaussian real-data nulls.
2. **If you will commit to \(n\) in advance and look once,** a t-test is the cheapest option. Its false-positive rate was within sampling error of 5% on every null with standard deviation 0.08 or more, except the skewed one at small \(n\) (12% at \(n = 5\)). Use a power analysis to pick \(n\) and do not add seeds after looking.
3. **If you want to stop whenever you like and are willing to assume approximate normality of seed differences,** use the anytime t e-process (Section 4.4). It needs no planned maximum, costs 1.6–2.0× the one-look t-test at the 80%-power point and 1.1–1.7× in expected seeds, and controls \(\alpha\) on every null we tried except the skewed one, where it inflates together with the t-test. If you can commit to a maximum budget, a five-look group-sequential design is equivalent in cost and slightly more robust to skew because its first look is late.
4. **If you cannot assume normality,** or your differences are skewed and you will stop at small \(n\), use the betting test. Declare the smallest interval you can defend *before* you see any stopping decision. Clipped human-normalized scores give \([-1, 1]\) for free. Expect the seed count to be about \(h\log(1/\alpha)/\delta\), linear in the half-width and independent of the noise, and the multiplier over a one-look t-test to grow with \(h/\mathrm{sd}\): 1.8× at 4.3, 2.9× at 7.1, 5.4× at 13, 12× at 34. Expect the multiplier against the parametric anytime test to be near 1 when \(h/\mathrm{sd} \lesssim 6\). When the effect is large, do not expect the betting test to be as cheap as the t-test is then: at a gap of 0.05 on the unit interval it needs about 60 seeds no matter how small the noise (Section 10).
5. **Do not declare a bound by looking at the range of the data you have.** On the aggregate BBF data a half-width of 0.1 looked safe at 3.4 standard deviations and was violated on 83% of 200-seed paths. Only bounds at 8.6 standard deviations or more held on every path.
6. **Aggregate scores and single games call for different tests.** A 55-game aggregate is nearly Gaussian with \(h/\mathrm{sd} \approx 34\): use the parametric anytime test (16 seeds vs 111). A single game, or a single environment with bimodal outcomes such as Acrobot, has \(h/\mathrm{sd}\) of 2–6: use the betting test and get the distribution-free guarantee at no cost (28 vs 28 fresh seeds).
7. **A 30–40-seed budget resolves standardized effects of about 0.75 and larger with any valid anytime test, and nothing much smaller with any test.** On our fresh runs the eight comparisons with \(\delta/\sigma \le 0.2\) were undetectable at 40 seeds, and would have been "significant" 17–27% of the time under peeking.

---

## 13. Discussion

**What was actually bought.** The earlier framing, "valid sequential testing costs 2–5× the seeds," conflated two things. The seeds a practitioner spends on a betting test are buying insurance against distributions they have not seen, in an amount proportional to the declared interval. If they would be content to assume normality (which is what every t-test in the literature already does), the sequential property itself costs 1.6–2.0× at 80% power and nothing at all within the betting family. That reframes the engineering question from "can we afford anytime validity" to "how tight an interval can we honestly declare," which is a question about the metric, not the test.

**Why the bound binds.** The betting test's clip is not an implementation detail. Any test that is valid for every distribution on \([0,1]\) must survive an observation at the edge, and the capital it holds back for that possibility is capital that cannot compound. The t-test holds back nothing because it assumes the edge is unreachable. On data where the edge *is* reachable (single games, where a run that solves a level and one that does not can differ by most of the interval) the two tests make the same reservation, and the betting test is as cheap as the parametric one while being correct on the bimodal distributions where the parametric one is a model.

**Metric design.** The fact that the aggregate seed has \(h/\mathrm{sd} \approx 34\) is a consequence of averaging 55 bounded terms: the interval stays \([-1,1]\) while the standard deviation shrinks like \(1/\sqrt{55}\). A distribution-free test on an aggregate pays for a bound that averaging has made very loose. A tighter valid bound on an aggregate would have to come from a structural argument (for example, a per-game bound that is then averaged, which gives the same \([-1,1]\)) or from a different metric. We do not see a way around this for aggregates, and we read it as a reason to prefer the parametric anytime test there.

**Large effects do not rescue the bound.** A natural hope is that the betting test's overhead is a small-effect problem. The TD-MPC2 aggregates say otherwise: at standardized effects of 7–24, where a t-test needs two seeds, the betting test needs 9–64 because its wager is capped at 2 and each observation can move \(\log M_t\) by at most about \(2\delta_u\). The number of seeds a distribution-free test needs is a function of the gap measured in units of the declared interval, and a huge effect in standard-deviation units can be a small one in interval units. The practitioner who reports "TD-MPC2 beats SAC with \(p < 0.05\) after three seeds" is right under normality and would need thirteen seeds to say it without that assumption.

**What the fresh runs add.** They remove the one methodological objection we could not answer from released data, that bootstrap streams might misstate the seed count. On the four comparisons where both can be computed they do not, within 1–4 seeds. They also put the two regimes side by side in one training script: Acrobot, where a seed either solves the task or does not and the betting test is free, and Breakout, where seeds agree to within 0.03 and the betting test cannot finish in the pool.

**On the e-process's skew failure.** The anytime t e-process rejecting 10.9% of skewed nulls is not a defect peculiar to the sequential method; it is the one-look t-test's 12% at \(n = 5\) made visible at every \(n\). Any protocol that stops at small \(n\) on skewed data is in this regime. The group-sequential design avoids it by construction (first look at \(n \ge 40\)), and that is a legitimate reason to prefer it when a maximum is acceptable.

---

## 14. Limitations

**Bootstrap seed streams.** The released-data seed streams are resamples from pools of 12–60 runs per game (BBF) or three runs per task (TD-MPC2), so a 200-seed path reuses runs. The sample-size numbers are for the population those runs estimate, not for 200 fresh runs. Section 11 checks this on the one dataset where we have enough fresh runs to compare: genuine and bootstrap \(n_{80}\) agree within 1–4 seeds on four comparisons with \(n_{80} \le 31\). We have no such check for counts above the pool size (BBF−WD at 83, Alien at 118, the betting test at 75–126), where the bootstrap is the only evidence. The TD-MPC2 per-task pools of three runs are thin; the aggregate over 39–50 tasks has enough distinct values to behave as a continuous variable, but a per-task comparison would not, and we do not report one.

**Regimes.** The aggregate seed is close to Gaussian by construction, which is the regime most favorable to the parametric tests and least favorable to the bound. The single-game rows and the Acrobot runs are the opposite regime. The five-task slices of Section 10 and the Breakout runs of Section 11 are the intermediate and tight regimes respectively; together the real-data settings span \(h/\mathrm{sd}\) from 2.3 to 111.

**Fresh runs.** The 480 runs are three variants of one algorithm family on four small environments with a 30,000–100,000-step budget; the agents are not tuned, and the normalization \(\min(R/100, 1)\) on MinAtar is a choice (the largest run scored 0.595, so the cap never bound). The genuine paths have 30–40 pairs, so they cannot say anything about tests whose cost exceeds the pool size, which is exactly where the betting test lands on Breakout.

**Dependence.** Seeds within a configuration share a code path and hyperparameters; Henderson et al. (2018) and Agarwal et al. (2021) document that this induces dependence that no test here models. Our paired differences are i.i.d. by construction.

**Design choices for the baselines.** The group-sequential designs were given \(n_{\max} = 2 n_{80}(\text{t-test})\), which uses knowledge of the effect a practitioner would need from a power analysis. With a smaller \(n_{\max}\) they lose power; with a larger one they spend more \(\alpha\) early and their \(n_{80}\) rises toward the e-process's. The e-process depends on a prior scale \(g\); \(g = 0.3\) instead of 1 changed \(n_{80}\) by between −2 and +6 seeds across the beta and real aggregate settings (cheaper only when the effect is small enough that the narrow prior is better matched) and lowered the skewed-null false-rejection rate from 0.109 to 0.051.

**Betting variants.** The betting test is one predictable plug-in wager. Hedged and mixture variants in Waudby-Smith and Ramdas (2023) produce confidence sequences rather than tests and, in the earlier draft, cost more than the plug-in. We did not try bounds that adapt validly over time (for example, a bound that widens when violated and pays for it in \(\alpha\)); that is a different test and would be the natural next thing to build.

**Proposition 2 is a heuristic.** It predicts the betting count to within 10–40% and gets the trends right (linear in \(h\), flat in \(\sigma\)); it is not a theorem, and the slack is from the plug-in warm-up and from the gap between Ville's inequality and the actual crossing probability.

---

## 15. Conclusion

A valid sequential comparison of two RL algorithms costs more seeds than an invalid one, and the earlier literature (including our own earlier draft) reported that cost as a single multiplier. It is two multipliers. The one for stopping whenever you like is 1.6–2.0× under a Gaussian model and exactly 1× within the betting family. The one for being right about every distribution on a declared interval is set by how loose that interval is relative to the noise, and ranges from 1× when the noise fills the interval to 16× when it fills 1–3% of it, whatever the size of the effect. The second multiplier cannot be reduced by tightening the interval after looking, because the bounds that look safe are the ones that fail. The practical consequence is a short decision rule: parametric anytime for aggregates and for anyone willing to assume normality; distribution-free betting for single games, bimodal per-environment outcomes, skewed differences, and anyone who is not. Four hundred and eighty fresh runs say the resampled numbers behind that rule are the right ones, and that a 40-seed experiment resolves effects of three quarters of a standard deviation and little else.

---

## Reproducibility statement

`s4_extensions.py` runs every setting in Sections 6–9 with `--paths 400 --null-paths 1000 --cap 200` and writes `results/extensions_summary.json`; `s4_real_data.py` runs the TD-MPC2 and fresh-run settings of Sections 10–11 (random seed 1; genuine streams capped at the pool size) and writes `results/real_data_summary.json`; `s4_fresh_runs.py` trains the 480 DQN runs (seeds 0 to \(n - 1\) per agent and environment; evaluation environment seeded at seed + 10,000) and writes `data/fresh/fresh_runs.csv`. Every table entry is read from those files; `make_figures.py` produces Figures 1–4 and `make_figures_real.py` Figure 5. The random seed is 0. The BBF scores are the files `RR2_BBF.csv`, `RR2_BBF-SPR.csv`, `RR2_BBF-Resets.csv`, `RR2_BBF-WD.csv`, `RR2_BBF+sticky_100k.csv`, `RR8_BBF.csv` from `google-research/bigger_better_faster/scores`, unmodified. The TD-MPC2 results are the per-task CSVs under `results/{dreamerv3,sac,tdmpc,tdmpc2}/` of `nicklashansen/tdmpc2`, unmodified; a run's score is the last logged evaluation. The group-sequential constants are calibrated with a fixed seed (12345) and 20,000 paths and cached per \(n_{\max}\). The original kill test and gridworld runs of the earlier draft are in `s4_anytime.py` and summarized in Appendix F.

---

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., and Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS*.
- Colas, C., Sigaud, O., and Oudeyer, P.-Y. (2018). How many random seeds? Statistical power analysis in deep reinforcement learning experiments. arXiv:1806.08295.
- Darling, D. A. and Robbins, H. (1967). Confidence sequences for mean, variance, and median. *PNAS*.
- Grünwald, P., de Heide, R., and Koolen, W. (2024). Safe testing. *JRSS-B*.
- Hansen, N., Su, H., and Wang, X. (2024). TD-MPC2: Scalable, robust world models for continuous control. *ICLR*.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., and Meger, D. (2018). Deep reinforcement learning that matters. *AAAI*.
- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*.
- Johari, R., Koomen, P., Pekelis, L., and Walsh, D. (2022). Always valid inference: Continuous monitoring of A/B tests. *Operations Research*.
- Jordan, S. M., Chandak, Y., Cohen, D., Zhang, M., and Thomas, P. S. (2020). Evaluating the performance of reinforcement learning algorithms. *ICML*.
- Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials. *Biometrika*.
- O'Brien, P. C. and Fleming, T. R. (1979). A multiple testing procedure for clinical trials. *Biometrics*.
- Patterson, A., Neumann, S., White, M., and White, A. (2023). Empirical design in reinforcement learning. arXiv:2304.01315.
- Pérez-Ortiz, M. F., Lardy, T., de Heide, R., and Grünwald, P. (2024). E-statistics, group invariance and anytime-valid testing. *Annals of Statistics*.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical trials. *Biometrika*.
- Ramdas, A., Grünwald, P., Vovk, V., and Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science*.
- Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm. *Annals of Mathematical Statistics*.
- Schwarzer, M., Obando-Ceron, J., Courville, A., Bellemare, M. G., Agarwal, R., and Castro, P. S. (2023). Bigger, better, faster: Human-level Atari with human-level efficiency. *ICML*.
- Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
- Waudby-Smith, I. and Ramdas, A. (2023). Estimating means of bounded random variables by betting. *JRSS-B*.
- Young, K. and Tian, T. (2019). MinAtar: An Atari-inspired testbed for thorough and reproducible reinforcement learning experiments. arXiv:1903.03176.

---

## Appendix A. Test definitions in full

**A.1 t-statistics along a path.** For a path \(x_1, \ldots, x_N\), \(T_n = \bar x_n \sqrt n / s_n\) with \(s_n\) the unbiased sample standard deviation, computed for every \(n \ge 2\) by running sums; \(p_n = 2(1 - F_{t_{n-1}}(|T_n|))\).

**A.2 Group-sequential calibration.** For \(n_{\max}\) and five looks at \(n_k = \lceil k n_{\max}/5\rceil\), draw 20,000 Gaussian null paths of length \(n_{\max}\), compute \(z_k = \Phi^{-1}(1 - p_{n_k}/2)\) at each look, and set \(c_P\) to the \(1-\alpha\) quantile of \(\max_k z_k\) and \(c_O\) to the \(1 - \alpha\) quantile of \(\max_k z_k \sqrt{k/5}\). This gives \(c_P \approx 2.42\) and \(c_O \approx 2.05\) at every \(n_{\max}\) we used, within 0.01 of the classical Gaussian five-look constants 2.413 and 2.040; the constants are cached per \(n_{\max}\). In the power settings \(n_{\max} = \min(200, \max(10, 2 n_{80}(\text{t})))\); in null settings \(n_{\max} = 200\).

**A.3 The e-process integral.** \(E_n = \sum_{q=1}^{40} w_q\, f_{n-1,\, g z_q \sqrt n}(T_n) / f_{n-1}(T_n)\) with \((z_q, w_q)\) the probabilists' Gauss–Hermite nodes and weights normalized to sum to 1. \(E_1 = 1\). Steps with zero sample variance (possible on clipped single-game data when every observation so far is identical) give \(|T_n| = \infty\); we cap \(\log E_n\) at 50 there, which only affects paths that reject anyway.

**A.4 Betting wager.** \(\hat\mu_{i-1}\) and \(\hat v_{i-1} = \widehat{\mathbb E}[u^2] - 2 m_0 \hat\mu + m_0^2\) are running estimates from observations \(1, \ldots, i-1\); \(\lambda_1 = 0\). The clip is \(\pm 0.99/\max(m_0, 1 - m_0)\). Wealth is accumulated in log space with a floor of \(10^{-12}\) on each factor, which is never active when the bound holds.

**A.5 EB confidence sequence.** As in Waudby-Smith and Ramdas (2023, Theorem 2) with \(\psi_E(\lambda) = (-\log(1 - \lambda) - \lambda)/4\), \(\hat\mu_t\) and \(\hat\sigma_t^2\) with prior pseudo-counts \((1/2, 1/4)\), \(\lambda_t\) capped at \(1/2\), the interval \(\big(\sum \lambda_i u_i \pm (\log(2/\alpha) + \sum v_i \psi_E(\lambda_i))\big)/\sum \lambda_i\), intersected over \(t\).

## Appendix B. Data construction

**B.1 Beta differences.** For concentration \(\kappa \in \{8, 24, 80\}\) and relative gap \(r = 0.14\): \(a = \kappa(1 + r)/2\), \(b = \kappa(1 - r)/2\); \(x = \mathrm{Beta}(a,b) - \mathrm{Beta}(b,a)\). Mean \(r = 0.14\), standard deviation \(\sqrt{2\, ab/(\kappa^2(\kappa+1))}\): 0.234, 0.141, 0.077. Nulls use \(a = b = \kappa/2\).

**B.2 BBF pools.** Each CSV has one row per (game, run) with the human-normalized score. The 55 games are the union across configurations; every configuration has every game. Pool sizes are 12–60 runs. Clipping to \([0,1]\) leaves 45 games with nonzero variance for BBF RR2; ten saturate at 1 (Assault, Boxing, Breakout, DemonAttack, JamesBond, Krull, MontezumaRevenge, RoadRunner, Robotank, Venture).

**B.3 Aggregate seed.** For each of the 55 games, one run is drawn uniformly with replacement from the left pool and one from the right pool; the aggregate seed difference is the mean over games of the left clipped score minus the mean of the right clipped score. 400 such differences form one path; the test reads the first 200.

**B.4 Single-game selection.** Rule fixed before any test was run: BBF RR2's clipped score on the game has mean in \((0.05, 0.95)\) and standard deviation above 0.02. Passing games: alien, asterix, choppercommand, frostbite, hero, kangaroo, qbert, zaxxon.

**B.5 TD-MPC2 pools.** One CSV per (algorithm, task) with columns `step, reward|success, seed`; seeds 1–3 throughout. DMControl tasks are the 39 with a `reward` column that are not Meta-World (`mw-`), MyoSuite (`myo-`), or the four ManiSkill-style tasks (`lift-cube`, `pick-cube`, `stack-cube`, `turn-faucet`); score = final reward / 1000, clipped to \([0,1]\). Meta-World tasks are the 50 `mw-` files; score = final success. Aggregate and five-task-slice seeds are built as in B.3 with tasks in place of games; a slice draws its five tasks once per path without replacement.

**B.6 Fresh DQN runs.** `s4_fresh_runs.py`. Network: observation → 64 → 64 → actions (classic control) or 128 → 128 (MinAtar), SiLU. Adam (\(10^{-3}\), or \(3 \times 10^{-4}\) for the third agent), batch 64, replay 50,000, 1,000 warm-up transitions, one gradient step per environment step, Huber loss, gradient-norm clip 10, Polyak target \(\tau = 0.005\), \(\gamma = 0.99\), \(\varepsilon\)-greedy from 1.0 to 0.05 over the first 40% of steps. Observations scaled by \((2.4, 3.0, 0.21, 3.0)\) on CartPole and \((1,1,1,1,12.57,28.27)\) on Acrobot; MinAtar sticky actions off, 1,000-step cap, reward clipped at 1 per step. Evaluation: 10 episodes at \(\varepsilon = 0.05\) on a separately seeded environment. Genuine streams: `genuine_pair_path` permutes each pool and pairs in order; `genuine_null_path` permutes one pool and pairs its first half with its second half.

## Appendix C. Full results for the power settings

| Setting | one-look t | Pocock \(n_{80}\) / rej / \(\mathbb E n\) | OBF | anytime t | anytime t \(g{=}0.3\) | betting one-look | betting anytime \(n_{80}\) / rej / \(\mathbb E n\) | EB-CS |
|---|---:|---|---|---|---|---:|---|---|
| beta conc. 8 | 23 | 35 / 0.95 / 25 | 35 / 0.97 / 30 | 43 / 1.00 / 29 | 42 / 1.00 / 33 | 44 | 42 / 1.00 / 33 | 131 / 1.00 / 115 |
| beta conc. 24 | 11 | 16 / 0.96 / 11 | 16 / 0.99 / 14 | 18 / 1.00 / 14 | 23 / 1.00 / 19 | 32 | 32 / 1.00 / 27 | 118 / 1.00 / 109 |
| beta conc. 80 | 5 | 8 / 0.99 / 6 | 8 / 1.00 / 7 | 9 / 1.00 / 8 | 14 / 1.00 / 13 | 27 | 27 / 1.00 / 25 | 112 / 1.00 / 107 |
| beta conc. 80, \(h = 0.5\) | 5 | 8 / 0.99 / 6 | 8 / 1.00 / 7 | 9 / 1.00 / 7 | 14 / 1.00 / 13 | 16 | 16 / 1.00 / 14 | 57 / 1.00 / 54 |
| BBF−SPR agg. | 9 | 12 / 0.98 / 11 | 16 / 0.99 / 13 | 16 / 1.00 / 12 | 21 / 1.00 / 18 | 111 | 111 / 1.00 / 103 | — / 0.00 / 200 |
| BBF−Resets agg. | 3 | 4 / 1.00 / 4 | 6 / 1.00 / 6 | 6 / 1.00 / 5 | 12 / 1.00 / 11 | 34 | 34 / 1.00 / 33 | 149 / 1.00 / 146 |
| BBF−WD agg. | 83 | 133 / 0.97 / 82 | 133 / 0.98 / 105 | 138 / 0.96 / 92 | 136 / 0.96 / 89 | — | — / 0.00 / 200 | — / 0.00 / 200 |
| BBF+sticky agg. | 11 | 18 / 0.98 / 13 | 18 / 0.99 / 15 | 19 / 1.00 / 14 | 23 / 1.00 / 20 | 126 | 126 / 1.00 / 117 | — / 0.00 / 200 |
| RR2 vs RR8 agg. | 10 | 12 / 0.97 / 11 | 16 / 0.99 / 13 | 17 / 1.00 / 13 | 21 / 1.00 / 18 | 114 | 114 / 1.00 / 106 | — / 0.00 / 200 |
| BBF−SPR agg., \(h = 0.5\) | 10 | 16 / 0.96 / 11 | 16 / 0.98 / 13 | 17 / 1.00 / 13 | 22 / 1.00 / 19 | 61 | 60 / 1.00 / 54 | — / 0.00 / 200 |
| BBF−SPR agg., \(h = 0.25\) | 10 | 14 / 0.96 / 10 | 14 / 0.98 / 12 | 17 / 1.00 / 13 | 22 / 1.00 / 19 | 34 | 34 / 1.00 / 30 | 131 / 1.00 / 122 |
| BBF−SPR agg., \(h = 0.1\) (83% violated) | 9 | 14 / 0.97 / 10 | 14 / 0.98 / 12 | 17 / 1.00 / 13 | 22 / 1.00 / 18 | 19 | 19 / 1.00 / 15 | 59 / 1.00 / 52 |
| BBF−SPR agg., \(h = 0.05\) (100% violated) | 10 | 14 / 0.96 / 10 | 14 / 0.99 / 13 | 17 / 1.00 / 13 | 22 / 1.00 / 18 | 18 | 16 / 1.00 / 12 | 36 / 1.00 / 32 |

"rej" is the fraction of 400 paths that rejected within 200 seeds; \(\mathbb E n\) counts non-rejecting paths at 200. Group-sequential rejection rates below 1.00 are paths that did not cross by \(n_{\max}\) and can never reject.

**TD-MPC2 nulls** (1,000 paths, cap 200). False-rejection rates, columns as in Section 6:

| Null | sd | peek t | Pocock | OBF | anytime t | anytime t (\(g{=}0.3\)) | betting, anytime | EB-CS | one-look t (\(n{=}10\)) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DMC, TD-MPC2 vs TD-MPC2, aggregate | 0.007 | 0.412 | 0.041 | 0.044 | 0.026 | 0.023 | 0.000 | 0.000 | 0.037 |
| DMC, SAC vs SAC, aggregate | 0.031 | 0.435 | 0.057 | 0.050 | 0.036 | 0.025 | 0.000 | 0.000 | 0.037 |
| DMC, TD-MPC2 vs TD-MPC2, 5-task slice | 0.022 | 0.424 | 0.058 | 0.051 | 0.037 | 0.032 | 0.000 | 0.000 | 0.054 |
| MW, TD-MPC2 vs TD-MPC2, aggregate | 0.022 | 0.434 | 0.040 | 0.038 | 0.037 | 0.023 | 0.000 | 0.000 | 0.062 |
| MW, SAC vs SAC, aggregate | 0.041 | 0.410 | 0.039 | 0.054 | 0.033 | 0.020 | 0.000 | 0.000 | 0.048 |
| MW, TD-MPC2 vs TD-MPC2, 5-task slice | 0.069 | 0.394 | 0.038 | 0.041 | 0.031 | 0.020 | 0.005 | 0.000 | 0.052 |

Bootstrap nulls on the fresh pools (DQN vs itself, cap 200): peek t 0.377–0.437, Pocock 0.041–0.049, OBF 0.037–0.051, anytime t 0.017–0.038, betting 0.000–0.035, EB 0.000–0.002. The TD-MPC2 power rows are in Section 10 and the fresh-run rows in Section 11; every test's rejection rate is 1.00 on every aggregate row; on the five-task slices the group-sequential designs, which cannot reject after \(n_{\max} = 2 n_{80}(\text{t})\), finish at 0.84–0.98 (DMC TD-MPC2 vs TD-MPC 0.96/0.98 by 10; DMC TD-MPC vs SAC 0.84/0.86 by 16; MW TD-MPC2 vs TD-MPC 0.87/0.88 by 74; MW TD-MPC vs SAC 0.85/0.86 by 18), and the betting test at 0.78 on MW TD-MPC2 vs TD-MPC. On the large-effect aggregates \(n_{\max} = \max(10, 2 n_{80}(\text{t})) = 10\), so the five looks are at 2, 4, 6, 8, 10 and the group-sequential \(n_{80}\) of 4 is the second look.

Single-game comparisons that no test resolved within 200 seeds: Qbert −SPR (gap −0.020, sd 0.165: one-look t rejected 45% at 200, betting 25%), Kangaroo −SPR (+0.041, 0.379: 43%, 18%), Frostbite −SPR (−0.058, 0.326: 65%, 38%).

## Appendix D. Proposition 2 against the data

| Setting | \(\delta\) | \(h\) | \(\delta_u = \delta/2h\) | \(\sigma_u^2\) | \(\lambda^\ast = \delta_u/\sigma_u^2\) | clip \(c\) | predicted \(\log(20)/(c\delta_u - c^2\sigma_u^2/2)\) | measured betting \(n_{80}\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| beta conc. 8 | 0.140 | 1 | 0.070 | 0.0137 | 5.1 | 1.98 | 27 | 42 |
| beta conc. 24 | 0.142 | 1 | 0.071 | 0.0050 | 14 | 1.98 | 23 | 32 |
| beta conc. 80 | 0.139 | 1 | 0.070 | 0.0015 | 47 | 1.98 | 22 | 27 |
| beta conc. 80, \(h = 0.5\) | 0.141 | 0.5 | 0.141 | 0.0061 | 23 | 1.98 | 11 | 16 |
| BBF−SPR agg. | 0.031 | 1 | 0.0155 | 0.00021 | 74 | 1.98 | 98 | 111 |
| BBF−SPR agg., \(h = 0.5\) | 0.031 | 0.5 | 0.031 | 0.0009 | 34 | 1.98 | 49 | 60 |
| BBF−SPR agg., \(h = 0.25\) | 0.031 | 0.25 | 0.062 | 0.0034 | 18 | 1.98 | 25 | 34 |
| BBF−Resets agg. | 0.102 | 1 | 0.051 | 0.00021 | 243 | 1.98 | 30 | 34 |
| BBF+sticky agg. | 0.027 | 1 | 0.0135 | 0.00021 | 64 | 1.98 | 114 | 126 |
| RR2 vs RR8 agg. | 0.030 | 1 | 0.015 | 0.00020 | 75 | 1.98 | 102 | 114 |

The prediction is 10–26% low on the aggregates and 19–36% low on the beta settings, where \(\lambda^\ast\) is only 2.6× the clip and the second-order term matters. The two trends it predicts hold: the count scales with \(h\) (98, 49, 25 predicted; 111, 60, 34 measured) and is nearly flat in the noise at fixed \(\delta_u\) (22–27 predicted; 27–42 measured).

## Appendix E. Sensitivity to the e-process prior

| Setting | anytime t, \(g = 1\) | anytime t, \(g = 0.3\) | difference |
|---|---:|---:|---:|
| beta conc. 8 | 43 | 42 | −1 |
| beta conc. 24 | 18 | 23 | +5 |
| beta conc. 80 | 9 | 14 | +5 |
| BBF−SPR agg. | 16 | 21 | +5 |
| BBF−Resets agg. | 6 | 12 | +6 |
| BBF−WD agg. | 138 | 136 | −2 |
| BBF+sticky agg. | 19 | 23 | +4 |
| RR2 vs RR8 agg. | 17 | 21 | +4 |
| skewed-null false rejection | 0.109 | 0.051 | |

The narrow prior is cheaper only when the standardized effect is small (BBF−WD, \(\delta/\sigma = 0.3\)) and costs 4–6 seeds otherwise. It halves the skewed-null inflation because it wagers less on early, non-Gaussian \(T_n\).

## Appendix F. The earlier draft's kill test and gridworld runs

The original experiment (`s4_anytime.py`, `results/summary.csv`) compared three protocols at \(\alpha = 0.05\) on 300 simulated "papers": a fixed-\(n\) t-test, the optional-stopping t-test, and the anytime betting test. On the beta null (concentration 8, 80 seeds), the fixed test rejected 8.3%, optional stopping 39.0%, and the betting test 0.0%. On the beta gap, all three reached power 1.00, with the betting test stopping at a mean of 27.5 seeds against the fixed budget of 80. A tabular gridworld with two agents differing in step size gave the same pattern (null: 0.7%, 14%, 0.0%; gap: mean stopping 8.1 seeds against a budget of 30). That experiment established that the betting test is valid and that it beats an over-provisioned fixed budget; it did not compare against a budget sized to the effect, which is what the present paper does.
