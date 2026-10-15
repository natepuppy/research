# Off-Policy TD Diverges Along Its Symmetric Spectrum: A Necessary Condition, Two Mechanisms of Deep Divergence, and the Price of Damping

*Anonymous authors. Full manuscript. A conference-length version is in `paper_rlc.md`; the earlier condensed draft is `paper_short.md`.*

**Reproducibility.** Every number in this paper is produced by the code in `../code/`. `n1_td_curl.py`, `probes/push_further.py`, and `probes/push_again.py` produce the expected-update sweeps of Section 5 and Appendix G. `n1_extensions.py` produces the theory checks (`--part theory`), the sampled linear comparison against GTD2, TDC, and ETD (`--part linear`), the nonlinear Baird sweep (`--part nonlinear`), and the CartPole runs (`--part cartpole`); `n1_deep_envs.py` produces the Acrobot, MountainCar, and MinAtar runs; `analyze_deep_envs.py` produces the mechanism statistics of Section 6 and Figure 6; `n1_anchor.py` produces the anchored-variant comparison of Section 7.4; `make_figures.py` produces `figures/fig1–5`. All systems are small: linear value functions on at most eight states, a 321-parameter network on Baird's features, and 4,600–43,000-parameter Q-networks on classic control and MinAtar. Nothing here is Atari or MuJoCo.

---

## Abstract

Semi-gradient temporal-difference (TD) learning is not gradient descent, and the Jacobian \(K\) of its expected update splits into a symmetric part \(S\) and a skew part \(A\). We ask which part is responsible for divergence under the deadly triad, whether the answer gives a usable diagnostic, and what one should change in the update. The first answer is an elementary inequality (Bendixson): the largest real part of any eigenvalue of \(K\) is at most \(\lambda_{\max}(S)\), so a diverging field must have an unstable symmetric part, and rotation can only hide instability, never create it. We confirm this on 900 random off-policy linear systems with zero violations, and on Baird's counterexample, where deleting the skew part makes the weights grow *faster*. The second answer is negative in a precise way: of 440 linear systems with \(\lambda_{\max}(S) > 0\) only 254 diverge, and across 210 deep Q-learning runs on five environments an online estimate of \(\lambda_{\max}(S)\) is positive in every run that could be measured, stable and unstable alike. The sign of \(S\) locates divergence; it does not forecast it. For the deep runs we instead measure the learning rate times the sharpness of the TD loss at the moment \(|Q|\) starts to grow, and find two different mechanisms behind "divergence without a target network": on CartPole the product is \(0.002\)–\(0.009\) at growth onset and the onset time scales as \(1/\mathrm{lr}\), an outward field; on MinAtar Breakout it is \(1.9\) (range \(1.2\)–\(2.7\)) at onset, a step-size overshoot; Space Invaders sits between, and on Acrobot and MountainCar nothing diverges in 10,000 steps although \(\lambda_{\max}(S) > 0\) throughout and the product sits at \(2.5\)–\(2.7\). Three short propositions say what damping the positive modes of \(S\) buys for the explicit and implicit step and what it costs: the damped fixed point differs from the TD fixed point by \(K_\varepsilon^{-1} D w^\ast\). The constructive part is SD-TD, which tracks the top eigenmodes of \(S\) from single transitions with an online Oja rule and damps them at \(O(dk)\) cost. It stops the divergence of TD(0) and ETD(0) on every off-policy linear family we tried and is inactive on-policy, but on Baird-type systems its fixed point is essentially the zero predictor. An anchored variant that damps toward a TDC reference copy preserves the fixed point and is stable wherever TDC is, and its error tracks TDC's to two decimals at every step size: the fixed-point-preserving spectral correction exists but buys no speed. The same tracking rule stabilizes 20/20 seeds of a nonlinear Baird network, 19/20 CartPole DQN runs without a target network, and reduces MinAtar divergence from 7/8 to 1/8 (Breakout) and 8/8 to 4/8 (Space Invaders), at the cost of suppressing the value scale where no divergence was coming.

---

## 1. Introduction

The deadly triad is standard: bootstrapping, off-policy sampling, and function approximation can make TD learning diverge even though the Bellman operator is a contraction (Baird, 1995; Tsitsiklis and Van Roy, 1997; Sutton and Barto, 2018, §11.2). The classical theory says *when* this can happen. What is less clear is *which part of the update* is responsible once one is staring at a Jacobian, whether that part can be watched during training, and what one should change in the update when one sees it.

The semi-gradient field \(f(w)\) is not the gradient of any objective, so its Jacobian \(K = \partial f/\partial w\) has a symmetric part \(S = (K + K^\top)/2\) and a skew part \(A = (K - K^\top)/2\). The squared Frobenius share of \(A\), \(\|A\|_F^2/\|K\|_F^2\), is a *curl fraction* between 0 and 1; a gradient field has curl fraction 0. A natural guess, and the one that motivated this project, is that the rotation is the problem: semi-gradient TD diverges because it is "not a gradient," so cancelling or shrinking \(A\) should be the correction, and the size of \(A\) should predict divergence. On Baird's counterexample the curl fraction is 0.324 and it separates the diverging run from a frozen-target run at the first checkpoint, before any weight has moved. The guess looks right.

It is wrong in a specific and useful way. Deleting \(A\) on Baird's problem makes the weights grow faster, not slower. Across five families of random off-policy linear systems, every system that diverged had an unstable symmetric part, and none needed its skew part to diverge. The reason is an inequality that is over a century old (Bendixson, 1902): the real part of every eigenvalue of \(K\) is bounded above by \(\lambda_{\max}(S)\). Rotation can only reduce the real part below the symmetric bound. On Baird it reduces it from 1.021 to 0.239, and the iteration diverges anyway.

That inequality reorganizes the question in two directions. One is diagnostic: if divergence lives in \(S\), then \(\lambda_{\max}(S) > 0\) is a necessary condition, and one can ask whether it is also a useful warning sign during training. We find it is not, in the linear case (42% false positives on our distribution of systems) and in deep Q-learning (positive in every one of 210 runs across five environments, including the ones that never diverge). What *does* discriminate, in the deep case, is a different quantity: the learning rate times the sharpness of the TD loss at the moment \(|Q|\) begins to grow. Measured across CartPole, Acrobot, MountainCar, and two MinAtar games, it shows that "divergence without a target network" is not one phenomenon. On CartPole the growth starts at a product of 0.002–0.009, two to three orders of magnitude below the step-size threshold of 2, and its onset scales as \(1/\mathrm{lr}\): the field points outward and a smaller step only delays the blow-up. On MinAtar Breakout the growth starts at a product of 1.9, the step overshoots a stiff direction, and a smaller step prevents it. Space Invaders lies between, and on Acrobot and MountainCar nothing diverges in 10,000 steps even though the symmetric part is unstable throughout and the product hovers at 2.5. The practitioner's rules of thumb ("lower the learning rate," "add a target network") treat these as one problem; the measurement says they are two, with different remedies.

The other direction is constructive: if divergence lives in \(S\), the correction is an operation on \(S\), damp its positive eigenmodes. We ask what that buys (Propositions 1–2), what it costs (Proposition 3), whether it can be run from single transitions at the cost of TD (SD-TD, Section 7), and whether the cost can be avoided. The last question has a clean answer now: an anchored variant that damps toward a TDC reference preserves the fixed point and is stable wherever TDC is, and its error tracks TDC's to two decimals, so a fixed-point-preserving spectral correction exists and is no faster than the convergent method it borrows its reference from.

**Contributions.**

1. *A theorem-level answer to "which half diverges."* Lemma 1 (Bendixson) and three propositions on the explicit step, the implicit step, and the fixed-point shift under damping, with proofs, verified numerically on 900 systems with zero violations (Section 4). The lemma is classical; its application to the TD Jacobian, and the consequence that editing the skew part can never stabilize an unstable field, appear to be new in this literature.
2. *The diagnostic is necessary and not sufficient, in linear and deep settings.* \(\lambda_{\max}(S) > 0\) has no false negatives and a 42% false-positive rate on 900 linear systems, and an online estimate of it is positive in all 208 deep runs with a finite measurement, stable and unstable alike (Sections 4.3, 6.4).
3. *Two mechanisms of deep divergence, separated by one measurement.* Across five environments (210 runs, 5–8 seeds per configuration), the learning rate times the sharpness of the TD loss at growth onset is 0.002–0.009 on CartPole (lr \(\le 3 \times 10^{-3}\)) and 1.2–2.7 on MinAtar Breakout. The first is an outward field (onset \(\propto 1/\mathrm{lr}\), a smaller step delays but never prevents); the second is overshoot (a smaller step prevents it). Acrobot and MountainCar do not diverge at all within the budget and sit at the edge-of-stability value of the product (Section 6).
4. *The constructive consequence and its price.* SD-TD tracks the top eigenmodes of \(S\) with an online Oja update from single transitions and damps them. Across 16,632 sampled runs on five linear families it stops the divergence of TD(0) and ETD(0), is inactive on-policy, and has the same per-step cost as GTD2/TDC. Proposition 3 gives its fixed-point shift in closed form; on Baird-type systems the shifted fixed point has RMSVE 0.99 times that of predicting zero (Section 7).
5. *The fixed-point-preserving variant exists and buys nothing.* Damping toward a TDC reference copy is stable wherever TDC is, preserves the fixed point, is inactive on-policy, and its RMSVE matches TDC's within 0.02 at every step size on every family (Section 7.4). Two other candidates (an oblique flip and a slow reference copy) fail from samples.
6. *Transfer to networks.* The tracked damping stabilizes 20/20 seeds of a nonlinear Baird network, 19/20 CartPole DQN runs without a target network, and reduces MinAtar divergence from 7/8 to 1/8 (Breakout) and 8/8 to 4/8 (Space Invaders). Where no divergence was coming (Acrobot, MountainCar) it suppresses the value scale by a factor of 10–20 (Section 7.6).

---

## 2. Related work

**Divergence of off-policy TD.** Baird (1995) gave the counterexample; Tsitsiklis and Van Roy (1997) proved on-policy convergence via the negative definiteness of the symmetric part of the TD Jacobian in the \(d_\mu\) inner product, and showed off-policy divergence is possible. Kolter (2011) characterized the set of off-policy fixed points and showed the TD fixed point can be arbitrarily bad. Bhandari, Russo, and Singal (2018) give finite-time bounds that rest on the same negative-definiteness condition. We take this classical picture and ask about the Jacobian's two halves.

**Algorithms designed for the triad.** Gradient-TD methods (Sutton et al., 2009: GTD2 and TDC) replace the semi-gradient by the gradient of the projected Bellman error, whose Jacobian is symmetric and negative definite, at the cost of a second weight vector and a second step size. Emphatic TD (Sutton, Mahmood, and White, 2016) reweights updates with a followon trace so that the expected Jacobian is negative definite under the behavior distribution, at the cost of high-variance emphasis weights. These are our baselines. SD-TD differs in leaving the semi-gradient and its fixed-point structure alone except along the damped modes; the anchored variant of Section 7.4 borrows TDC's reference and inherits its speed.

**Target networks and representations.** Zhang, Yao, and Whiteson (2021) prove that a target network with a projection step breaks the deadly triad for linear Q-learning. Ghosh and Bellemare (2020) characterize representations under which off-policy TD is stable. Van Hasselt et al. (2018) studied the triad empirically in DQN and introduced the notion of soft divergence we observe on CartPole. Achiam, Knight, and Abbeel (2019) analyze the Jacobian of deep Q-learning and propose a preconditioner; their analysis treats the full Jacobian, whereas we isolate its symmetric part.

**Optimization dynamics.** The "edge of stability" literature (Cohen et al., 2021) studies gradient descent at lr × sharpness ≈ 2. Our deep diagnostic measures the same quantity for TD and finds divergence far below it on CartPole, at it on MinAtar Breakout, and stable training hovering at it on Acrobot and MountainCar. Asadi et al. (2023) view TD as an optimization method and analyze when it behaves like one. Lyle et al. (2022, 2023) study capacity loss and plasticity in value-based deep RL, where growth of the value scale plays a role; our measurements of \(|Q|\) relative to \(1/(1-\gamma)\) are in the same spirit.

**Online eigenvector tracking.** The Oja (1982) rule and its deflated variants are the standard way to track a few top eigenvectors of a streaming covariance; SD-TD applies them to an unbiased rank-two estimate of \(S\).

**Benchmarks.** MinAtar (Young and Tian, 2019) provides small Atari-like games on a \(10 \times 10 \times c\) grid; we use Breakout and Space Invaders as the two environments where plain-SGD DQN diverges fast enough to be studied in 10,000 steps.

**Bendixson's inequality** (Bendixson, 1902) is textbook numerical linear algebra. Its use to separate "which half of the Jacobian diverges" is, as far as we know, new in the TD literature.

---

## 3. Preliminaries

### 3.1 Linear off-policy TD

A linear system has states \(s \in \{1, \ldots, n\}\), features \(x(s) \in \mathbb R^d\), rewards \(r(s, a)\), a target policy \(\pi\), a behavior policy \(\mu\) with stationary distribution \(d_\mu\), transitions \(P(s' | s, a)\), and discount \(\gamma\). With \(\rho(s, a) = \pi(a|s)/\mu(a|s)\), the expected off-policy TD(0) update is \(w \leftarrow w + \alpha(Kw + b)\) with

\[
K = \mathbb E_{s \sim d_\mu,\, a \sim \mu,\, s' \sim P}\big[\rho\, x(s)\,(\gamma x(s') - x(s))^\top\big], \qquad
b = \mathbb E\big[\rho\, r(s,a)\, x(s)\big].
\]

The TD fixed point is \(w^\ast = -K^{-1} b\) when \(K\) is invertible; with full-rank features it is exact. The continuous-time field \(\dot w = Kw + b\) is stable iff every eigenvalue of \(K\) has negative real part. The explicit step is stable iff the spectral radius of \(I + \alpha K\) is below 1. On-policy (\(\mu = \pi\)), Tsitsiklis and Van Roy (1997) show \(S\) is negative definite and TD converges; off-policy, neither statement need hold.

### 3.2 Symmetric and skew parts

Write \(K = S + A\) with \(S = (K + K^\top)/2\) and \(A = (K - K^\top)/2\). For a transition \((x, \rho, x')\) with \(\delta x = \gamma x' - x\), the rank-one term \(\rho\, x\, \delta x^\top\) has symmetric part \(\tfrac12 \rho(x\,\delta x^\top + \delta x\, x^\top)\), which is what SD-TD estimates (Section 7). The curl fraction is \(\|A\|_F^2/\|K\|_F^2\).

### 3.3 Deep Q-learning and its two matrices

For a Q-network \(Q_\theta\) trained by semi-gradient on the TD loss \(\ell(\theta) = \tfrac12 \mathbb E[(r + \gamma \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a))^2]\) with the bootstrap value treated as a constant, the update direction is \(-g(\theta)\) with \(g = \nabla_\theta \ell\) evaluated at \(\bar\theta = \theta\). The *field* is \(f = -g\) and its Jacobian is \(K = -Dg\). Two scalars of \(\operatorname{sym}(Dg)\) matter below. Its largest eigenvalue is the *sharpness* of the TD loss, the quantity whose product with the learning rate governs step-size stability of gradient descent (threshold 2 on a quadratic). Its most negative eigenvalue, negated, is \(\lambda_{\max}(S)\) of the field: the quantity Lemma 1 says must be positive for the field to point outward. Both are estimated by power iteration with Jacobian-vector products on a replay batch (Section 6.1).

### 3.4 Error metric

For linear systems we report the root-mean-square value error under the behavior distribution, \(\mathrm{RMSVE}(w) = \sqrt{\sum_s d_\mu(s)\,(x(s)^\top w - v_\pi(s))^2}\), and compare to the RMSVE of predicting zero everywhere, \(\sqrt{\sum_s d_\mu(s)\, v_\pi(s)^2}\), which is the natural reference for a method whose fixed point turns out to lie near zero.

---

## 4. The symmetric part bounds the field

### 4.1 Lemma

**Lemma 1 (Bendixson).** For any real square \(K\) with \(S = (K + K^\top)/2\),
\[
\max_i \operatorname{Re}\lambda_i(K) \;\le\; \lambda_{\max}(S).
\]

*Proof.* Let \(Kv = \lambda v\) with \(\|v\|_2 = 1\), \(v\) possibly complex. Then \(\lambda = v^* K v = v^* S v + v^* A v\). For skew-symmetric real \(A\), \((v^* A v)^* = v^* A^\top v = -v^* A v\), so \(v^* A v\) is purely imaginary, and \(\operatorname{Re}\lambda = v^* S v \le \lambda_{\max}(S)\) by the Rayleigh quotient bound for the Hermitian form of a real symmetric matrix. \(\square\)

The lemma says an unstable field has an unstable symmetric part, and the skew part can only reduce the real part below \(\lambda_{\max}(S)\). On Baird's problem \(\lambda_{\max}(S) = 1.021\) and the largest real part of \(K\) is \(0.239\): the rotation is slowing the instability by a factor of four and is not its source. Deleting \(A\) removes that slowing, which is why the weights grow faster without it (Section 5.1). The lemma also says there is no correction that acts on \(A\) alone and stabilizes an unstable field: if \(\lambda_{\max}(S) > 0\), then for any skew \(A'\), \(S + A'\) has a Rayleigh quotient \(u^\top(S + A')u = u^\top S u > 0\) at the top eigenvector \(u\) of \(S\), and the numerical range of \(S + A'\) contains a point with positive real part. (This does not by itself mean an eigenvalue has positive real part, which is exactly the gap between the diagnostic and divergence in Section 4.3.)

### 4.2 Damping and three propositions

**Definition (damped field).** Let \((\lambda_i, u_i)_{i=1}^d\) be the eigenpairs of \(S\) and \(\varepsilon > 0\). Define
\[
D = \sum_{i:\,\lambda_i > 0} (\lambda_i + \varepsilon)\, u_i u_i^\top, \qquad K_\varepsilon = K - D.
\]
Then \(S_\varepsilon := \operatorname{sym}(K_\varepsilon) = S - D\) has eigenvalues \(-\varepsilon\) in place of each positive \(\lambda_i\) and the nonpositive ones unchanged; the skew part is untouched. Write \(\varepsilon' = -\lambda_{\max}(S_\varepsilon) = \min(\varepsilon,\ -\max\{\lambda_i : \lambda_i \le 0\}) > 0\) for the realized margin.

**Proposition 1 (explicit step).** For every \(\alpha > 0\),
\[
\|I + \alpha K_\varepsilon\|_2^2 \;\le\; 1 - 2\alpha\varepsilon' + \alpha^2\|K_\varepsilon\|_2^2 .
\]
In particular the explicit damped step is a strict contraction in \(\|\cdot\|_2\) whenever \(\alpha < 2\varepsilon'/\|K_\varepsilon\|_2^2\).

*Proof.* For any unit \(v\), \(\|(I + \alpha K_\varepsilon)v\|^2 = 1 + 2\alpha\, v^\top K_\varepsilon v + \alpha^2\|K_\varepsilon v\|^2 = 1 + 2\alpha\, v^\top S_\varepsilon v + \alpha^2 \|K_\varepsilon v\|^2 \le 1 - 2\alpha\varepsilon' + \alpha^2\|K_\varepsilon\|_2^2\), using \(v^\top A v = 0\) and \(v^\top S_\varepsilon v \le -\varepsilon'\). Take the supremum over \(v\). \(\square\)

**Proposition 2 (implicit step).** For every \(\alpha > 0\), \(I - \alpha K_\varepsilon\) is invertible and
\[
\|(I - \alpha K_\varepsilon)^{-1}\|_2 \;\le\; (1 + 2\alpha\varepsilon')^{-1/2} < 1 .
\]

*Proof.* For unit \(v\), \(\|(I - \alpha K_\varepsilon)v\|^2 = 1 - 2\alpha\, v^\top S_\varepsilon v + \alpha^2\|K_\varepsilon v\|^2 \ge 1 + 2\alpha\varepsilon'\). So \(I - \alpha K_\varepsilon\) is bounded below and hence invertible, and its inverse has norm at most \((1 + 2\alpha\varepsilon')^{-1/2}\). \(\square\)

Both propositions are statements about \(S\) alone; the skew part enters only through \(\|K_\varepsilon\|_2\) in Proposition 1, which sets the step size at which the explicit step overshoots. The propositions are elementary. Their use is to say precisely what "damp the positive modes of \(S\)" buys, and to separate two failure modes of the explicit step, an outward field and overshoot of a stiff stable direction, that look the same from the outside. Section 5.3 exhibits both on exact matrices and Section 6 exhibits both in deep Q-learning.

**Proposition 3 (fixed-point shift).** Suppose \(K\) and \(K_\varepsilon\) are invertible, with \(w^\ast = -K^{-1} b\) and \(w_\varepsilon = -K_\varepsilon^{-1} b\). Then
\[
w_\varepsilon - w^\ast \;=\; K_\varepsilon^{-1} D\, w^\ast .
\]
In particular \(w_\varepsilon = w^\ast\) iff \(D w^\ast = 0\), i.e., iff \(w^\ast\) is orthogonal to every damped mode \(u_i\).

*Proof.* \(K_\varepsilon w_\varepsilon + b = 0\) and \(K w^\ast + b = 0\). Subtracting, \(K_\varepsilon(w_\varepsilon - w^\ast) = (K - K_\varepsilon) w^\ast = D w^\ast\). \(\square\)

Proposition 3 is the cost of the correction. Off-policy, the damped modes are not in general orthogonal to the fixed point, and Section 7.3 measures how far from it they are: on Baird-type systems the shift takes the exact fixed point almost to zero. On-policy \(D = 0\) and there is no shift, consistent with Tsitsiklis and Van Roy. The proposition also says what a fixed-point-preserving correction must do: damp \(w - w_{\mathrm{ref}}\) for some \(w_{\mathrm{ref}}\) that itself converges to \(w^\ast\), so that the damping term vanishes at the fixed point. Section 7.4 builds that variant and measures what it buys.

### 4.3 Numerical check, and the diagnostic is necessary but not sufficient

We checked Lemma 1 and Propositions 1–2 on 900 systems from the five families of Section 5.2 (150 Baird-structure, 300 random softmax, 150 cycle, 150 self-loop, 150 aliased): zero Bendixson violations (largest \(\operatorname{Re}\lambda(K) - \lambda_{\max}(S)\) over the 900 was \(4 \times 10^{-16}\)); zero violations of Proposition 1 at \(\alpha \in \{0.05, 0.5\}\); zero violations of Proposition 2 at \(\alpha \in \{0.05, 0.5, 5\}\) (`results/theory_checks.json`).

Of those 900 systems, 440 have \(\lambda_{\max}(S) > 0\) and 254 have an eigenvalue of \(K\) with positive real part. Lemma 1 guarantees the 254 are among the 440; the other 186 are stable fields whose symmetric part is nonetheless unstable. The sign of \(\lambda_{\max}(S)\) therefore has no false negatives and a 42% false-positive rate on this distribution. Among the 20 random-softmax systems of Section 7, eight have \(\lambda_{\max}(S) > 0\), five have a positive real eigenvalue of \(K\) (all below \(10^{-3}\)), and none diverges in 20,000 sampled steps at \(\alpha \le 0.005\). Section 6.4 repeats the exercise for deep Q-learning on five environments, with the same outcome: the tracked estimate is positive in every run that could be measured. A positive symmetric mode is where divergence lives when it happens. It is not a forecast that it will.

---

## 5. Exact linear fields: the curl marks divergence and does not cause it

### 5.1 Baird's counterexample

We use the construction of Sutton and Barto (2018, §11.2). Seven states; for upper state \(i \in \{0, \ldots, 5\}\) the feature vector has a 2 in coordinate \(i\) and a 1 in coordinate 7; the lower state has a 1 in coordinate 6 and a 2 in coordinate 7. The target policy always takes the solid action, which moves to the lower state; the behavior policy takes it with probability \(1/7\) and otherwise jumps uniformly to an upper state. Reward 0, \(\gamma = 0.99\), initial weights \((1, 1, 1, 1, 1, 1, 10, 1)\). The curl fraction of \(K\) is 0.324; a direct check of the Jacobian estimator in the code matches \(-2\) times the explicit Jacobian to \(10^{-16}\), so the curl number is not an artifact of stopping the gradient on the target.

With \(\alpha = 0.05\) and four seeds, frozen-target TD stays at weight norm 12.7 with curl fraction 0; semi-gradient TD diverges on every seed (final norms 47, 60, 59, 50). At step 0 the two conditions have identical TD error (7.629) and identical \(|Q|\) (4.286) and curl fractions 0 and 0.324, so curl "predicts" divergence with correlation 1 and the TD error cannot compete. That is a weak test, because the competing predictors have zero variance. The intervention is the real one. Subtracting \(A\) at every step, so the field is exactly \(Sw\), produces curl 0 and still diverges, ending at mean norm 70.7 against 54 for ordinary TD. Scaling the skew part in \(S + \beta A\) from \(\beta = 0\) to \(\beta = 1.5\) reduces the largest real part from 1.021 to 0.131 and never makes it negative. An adaptive rule that freezes the target the first time the curl exceeds 0.2 stays at norm 12.7, but it works because a frozen target is gradient descent on a quadratic, not because the curl was cancelled. Lemma 1 says this is the only thing \(A\) can do.

An on-policy deep network on an eight-state chain did not diverge under any of these conditions, including learning rates up to 0.5. That chain is representable by a table; it is a poor contrast and we do not use it as evidence.

### 5.2 Five families of linear systems

The same expected update defines \(K\) for any features, transitions, and pair of policies. We drew five families (Appendix B gives the exact generators):

- **Baird structure, random features.** Solid action to the last state, dashed uniform on the others, target always solid, behavior solid with probability \(1/n\), standard normal features, \(n \in \{4, \ldots, 7\}\), dimension \(n + 1\), \(\gamma = 0.99\).
- **Random kernels.** 3–7 states, dimension 2 to \(n + 1\), softmax transitions, target probability 0.95 on action 0, behavior 0.05, \(\gamma \in \{0.9, 0.99, 0.999\}\).
- **Cycles.** Target walks a directed cycle, behavior walks the other way, random features.
- **Self-loops.** Target jumps to one state, behavior stays put.
- **Aliased features.** Binary features of dimension about half the number of states, Baird or softmax backups.

Divergence of the expected explicit step means the spectral radius of \(I + \alpha K\) exceeds \(1 + 10^{-8}\).

| Family | Systems | Diverged at \(\alpha = 0.05\) | Fixed by damping (\(\varepsilon = 0.05\)) | Fixed by dropping \(A\) | Needed \(A\) to diverge |
|---|---:|---:|---:|---:|---:|
| Baird structure | 200 | 52 | 52 | 0 | 0 |
| Random kernels | 400 | 15 | 15 | 0 | 0 |
| Aliased features | 200 | 23 | 23 | 0 | 0 |
| Cycles | 200 | 0 | — | — | 0 |
| Self-loops | 200 | 0 | — | — | 0 |

"Needed \(A\) to diverge" counts systems where \(I + \alpha K\) is unstable but \(I + \alpha S\) is not. Damping did not destabilize anything that was stable: on an earlier draw of 150 Baird-structure systems, 29 diverged, damping fixed all 29 and broke 0 of the 121 stable ones.

The same holds for a *sampled* matrix. For 12 diverging Baird-structure systems we estimated \(K\) from importance-sampled TD updates (2,500 samples per standard-basis column; relative Frobenius error 0.10–0.18) and damped the eigenmodes of the estimate. All 12 explicit updates became stable at \(\alpha = 0.05\). The estimate of \(K\) is rough; the estimate of its top unstable mode is not (cosine with the true mode 0.996–0.999). This is the observation that motivates tracking the mode online rather than estimating the matrix.

### 5.3 A large step overshoots a stiff eigenvalue

At \(\alpha = 0.5\) damping alone is not enough, and Proposition 1 says why: the explicit step is only guaranteed below \(2\varepsilon'/\|K_\varepsilon\|_2^2\).

| Family | Diverged at \(\alpha = 0.5\) | Fixed by damping | Still diverges with \(A\) removed | Fixed by implicit step on \(K_\varepsilon\) |
|---|---:|---:|---:|---:|
| Baird structure | 49 | 31 | 18 | 49 |
| Random kernels | 25 | 11 | 14 | 25 |
| Cycles | 12 | 0 | 12 | 12 |
| Aliased features | 6 | 6 | 0 | 6 |

Wherever damping already worked, the largest stable scale on the skew part was 1: the rotation did not have to be reduced. Wherever damping failed, deleting \(A\) left the explicit step unstable. On every failure we inspected, the smallest eigenvalue of \(S_\varepsilon\) was below \(-2/\alpha = -4\): median \(-6.16\) on cycles (4 of 5 failures), \(-4.95\) on Baird structure (9 of 9), \(-5.04\) on random kernels (10 of 10). An explicit step of 0.5 on an eigenvalue of \(-5\) lands at \(1 + 0.5 \times (-5) = -1.5\), outside the unit interval: overshoot along a stiff *stable* direction after the unstable ones are gone. Clipping those eigenvalues into \((-2/\alpha + \varepsilon, -\varepsilon)\) and taking an explicit step fixed 9/9 Baird and 10/10 random failures and 3/5 cycle failures. The implicit step, as Proposition 2 promises for every \(\alpha\), fixed every diverging system in the table, and at \(\alpha = 0.05\) it left the damped systems stable.

So the explicit update fails in two symmetric ways. A positive eigenvalue of \(S\) makes the continuous-time field point outward, and damping removes it. A large negative eigenvalue makes the explicit step overshoot, and a smaller step, a clip on that eigenvalue, or an implicit step removes that. Neither sweep produced a system saved by editing \(A\) and not \(S\). Section 6 finds the same two failure modes, in different environments, in deep Q-learning.

---

## 6. Deep Q-learning without a target network: two mechanisms, five environments

The exact-matrix picture has two failure modes with two different remedies. In deep Q-learning they cannot be told apart by looking at \(|Q|\) blow up, and the usual remedies (a smaller learning rate, a target network) are applied to both. This section measures which one is happening, in each of five environments, with the same instrument.

### 6.1 Setup

A SiLU Q-network is trained by plain SGD on the semi-gradient TD loss from a replay buffer (batch 64), \(\varepsilon\)-greedy from 1.0 to 0.05 over the first half of 10,000 environment steps. "No target network" bootstraps from the online weights. The controls are a Polyak target network (\(\tau = 0.005\)) and the same no-target run at \(\gamma = 0.9\). SD-DQN (Section 7.6) is the no-target run with the tracked damping, \(k = 2\). Environments and architectures:

| Environment | Observation | Network | Params | lr (no target) | Reward scale | Seeds |
|---|---|---|---:|---|---|---:|
| CartPole-v1 | 4 (scaled) | 64–64 | 4,610 | \(10^{-3}, 3 \times 10^{-3}, 10^{-2}\) | +1/step, cap 500 | 5 |
| Acrobot-v1 | 6 (scaled) | 64–64 | 4,803 | \(10^{-3}, 3 \times 10^{-3}\) | −1/step, cap 500 | 8 |
| MountainCar-v0 | 2 (scaled) | 64–64 | 4,547 | \(10^{-3}, 3 \times 10^{-3}\) | −1/step, cap 200 | 8 |
| MinAtar Breakout | \(10 \times 10 \times 4\) | 64–64 | 30,019 | \(10^{-2}, 3 \times 10^{-2}\) | +1 per brick, cap 1,000 | 8 |
| MinAtar Space Invaders | \(10 \times 10 \times 6\) | 64–64 | 42,884 | \(10^{-2}, 3 \times 10^{-2}\) | +1 per alien, cap 1,000 | 8 |

MinAtar rewards are sparse and the network sees a 400–600-dimensional binary input; plain SGD at \(10^{-3}\) does not move \(|Q|\) off zero in 10,000 steps, so the MinAtar grid is ten times larger. CartPole has 10 configurations (50 runs, from the earlier draft); the other four environments have five each (160 runs). In total 210 runs.

Every 100 steps, on a 256-transition batch, we record: mean \(|Q|\); the learning rate times the *sharpness* of the TD loss (dominant eigenvalue of \(\operatorname{sym}(Dg)\), by warm-started power iteration); the tracked \(\lambda_{\max}(S)\) of the online TD field (most negative eigenvalue of \(\operatorname{sym}(Dg)\), negated, by a shifted deflated power iteration, Section 7.5); and the curl fraction of the field Jacobian estimated with three random probes. A run has diverged when mean \(|Q|\) exceeds \(10^4\) or the weights are non-finite.

### 6.2 The diagnostic question

Two stories explain divergence without a target network. **Story A** is step-size instability: the loss becomes too sharp for the learning rate and the update overshoots, the \(\mathrm{lr} \times \mathrm{sharpness} > 2\) threshold of gradient descent on a quadratic (Cohen et al., 2021). **Story B** is an outward field: the symmetric Jacobian of the TD field has a positive eigenvalue and \(Q\) grows along it at any learning rate. Lemma 1 says B requires \(\lambda_{\max}(S) > 0\); A does not. The two stories predict different scalings: under A, halving the learning rate should prevent divergence; under B, it should delay the onset in proportion. And they predict different values of \(\mathrm{lr} \times \mathrm{sharpness}\) *at the moment \(|Q|\) starts to grow*: near 2 under A, far below it under B. We define growth onset as the first measurement at which mean \(|Q|\) exceeds three times its value at the first measurement, and read off the product there.

### 6.3 Results

| Environment, configuration | Diverged | Onset step | lr × sharp. at growth onset | lr × sharp. (run median) | Final mean \(|Q|\) | Median final return |
|---|---:|---|---|---|---:|---:|
| CartPole, lr \(10^{-3}\) | 5 / 5 | 4,600–5,300 | 0.002 (0.002–0.003) | 0.05 | — | — |
| CartPole, lr \(3 \times 10^{-3}\) | 5 / 5 | 1,500–1,700 | 0.008 (0.007–0.009) | 0.17 | — | — |
| CartPole, lr \(10^{-2}\) | 5 / 5 | 500–700 | 0.64 (0.19–0.82) | 0.42 | — | — |
| CartPole, target net, lr \(3 \times 10^{-3}\) | 0 / 5 | — | — | 0.13 | 35 | 390 |
| CartPole, \(\gamma = 0.9\), lr \(3 \times 10^{-3}\) | 0 / 5 | — | — | 0.06 | 10 | 478 |
| Acrobot, lr \(10^{-3}\) | 0 / 8 | — | — | 0.78 | 88 | −485 |
| Acrobot, lr \(3 \times 10^{-3}\) | 0 / 8 | — | — | 2.69 | 82 | −500 |
| Acrobot, target net, lr \(3 \times 10^{-3}\) | 0 / 8 | — | — | 0.31 | 31 | −212 |
| Acrobot, \(\gamma = 0.9\), lr \(3 \times 10^{-3}\) | 0 / 8 | — | — | 0.15 | 10 | −498 |
| MountainCar, lr \(10^{-3}\) | 0 / 8 | — | — | 0.42 | 86 | −200 |
| MountainCar, lr \(3 \times 10^{-3}\) | 0 / 8 | — | — | 2.50 | 86 | −200 |
| MountainCar, target net, lr \(3 \times 10^{-3}\) | 0 / 8 | — | — | 0.31 | 34 | −200 |
| MountainCar, \(\gamma = 0.9\), lr \(3 \times 10^{-3}\) | 0 / 8 | — | — | 0.13 | 10 | −200 |
| Breakout, lr \(10^{-2}\) | 0 / 8 | — | — | 0.57 | 1.8 | 2.5 |
| Breakout, lr \(3 \times 10^{-2}\) | 7 / 8 | 1,600–9,800 | 1.93 (1.16–2.71) | 1.11 | — | 0.9 |
| Breakout, target net, lr \(3 \times 10^{-2}\) | 0 / 8 | — | — | 0.61 | 1.5 | 2.4 |
| Breakout, \(\gamma = 0.9\), lr \(3 \times 10^{-2}\) | 0 / 8 | — | — | 0.92 | 0.6 | 2.2 |
| Space Invaders, lr \(10^{-2}\) | 8 / 8 | 500–2,000 | 0.52 (0.08–2.17) | 0.52 | — | 4.8 |
| Space Invaders, lr \(3 \times 10^{-2}\) | 8 / 8 | 300–400 | (first measurement) | 1.20 | — | 4.8 |
| Space Invaders, target net, lr \(3 \times 10^{-2}\) | 0 / 8 | — | — | 0.46 | 3.3 | 5.8 |
| Space Invaders, \(\gamma = 0.9\), lr \(3 \times 10^{-2}\) | 0 / 8 | — | — | 0.69 | 1.1 | 6.2 |

All no-target rows are at \(\gamma = 0.99\) unless marked. "lr × sharp. at growth onset" is the median (range) over diverging runs. The run median is the median over measurements and runs. `results/deep_envs_summary.json` has every cell; SD-DQN rows are in Section 7.6.

![Figure 6](figures/fig6_deep_envs.png)

*Figure 6. Left: learning rate times sharpness, per run, in the no-target \(\gamma = 0.99\) configurations of each environment. Red: the value at growth onset for runs that diverged. Gray: the median over the run for runs that did not. Dashed: the step-size threshold 2. Right: fraction of runs diverged per configuration.*

![Figure 4](figures/fig4_cartpole.png)

*Figure 4. CartPole, five seeds per group. Left to right: mean \(|Q|\); learning rate times sharpness of the TD loss (dashed line at 2); tracked \(\lambda_{\max}(S)\) of the online TD field; curl fraction. Red: no target network, \(\gamma = 0.99\), diverges. Blue: target network. Green: no target network, \(\gamma = 0.9\). Black: SD-DQN with \(k = 2\).*

**CartPole: an outward field.** In every uncorrected \(\gamma = 0.99\) run, \(|Q|\) grows steadily from the first measurement. At lr \(10^{-3}\) it grows by a factor of 100 (0.1 to 14) between steps 300 and 2,400 while \(\mathrm{lr} \times \mathrm{sharpness}\) stays between 0.002 and 0.06, and the product crosses 1 only within the last 300 steps before the weights become NaN; at growth onset it is 0.002–0.003. At lr \(3 \times 10^{-3}\), \(|Q|\) goes from 0.27 to 20 by step 1,200 with the product at most 0.35 (Appendix D gives the trajectories). The growth is the slow "soft divergence" of Van Hasselt et al. (2018), and the sharpness catches up only at the end, as a consequence of large weights rather than a cause. Reducing the learning rate delays the blow-up roughly in proportion (500–700 steps at \(10^{-2}\), 1,500–1,700 at \(3 \times 10^{-3}\), 4,600–5,300 at \(10^{-3}\)) and never prevents it, which is what an outward field with growth rate proportional to the learning rate predicts and what a step-size threshold does not. At lr \(10^{-2}\) the two stages are compressed and the product at onset is 0.2–0.8, still below 2.

**Breakout: overshoot.** At lr \(3 \times 10^{-2}\) seven of eight runs diverge, between steps 1,600 and 9,800, with no steady growth beforehand: \(|Q|\) sits at 1–3 for thousands of steps with \(\mathrm{lr} \times \mathrm{sharpness}\) oscillating between 1.1 and 2.7 (median 1.1 over the run), and the blow-up happens within 100–300 steps of a measurement at which the product is 1.2–2.7 (median 1.9). At the last finite measurement before divergence the median product is 2.1. At lr \(10^{-2}\), one third the step, zero of eight runs diverge in the same budget and the product sits at 0.57. This is Story A: a smaller step prevents it, and the onset is a threshold crossing rather than a growth process. The target network and \(\gamma = 0.9\) controls at lr \(3 \times 10^{-2}\) also do not diverge; they lower the sharpness (product 0.6–0.9) rather than the step.

**Space Invaders: a field that inflates the sharpness.** All eight runs diverge at both learning rates. At lr \(10^{-2}\) the sequence is visible: \(|Q|\) rises from 0.5 to 4–11 over 200–400 steps while the product climbs from 0.1 to 0.5–2.2, and the weights are non-finite within a further 100–200 steps (Appendix D). The growth begins below the threshold (median product 0.52 at onset, range 0.08–2.17) and ends above it. The onset step moves from 300–400 at lr \(3 \times 10^{-2}\) to 500–2,000 at lr \(10^{-2}\): a factor of 1.5–5 for a factor of 3 in the step, between the \(1/\mathrm{lr}\) scaling of CartPole and the all-or-nothing behavior of Breakout. Our reading is that both mechanisms operate in sequence here: an outward field inflates \(|Q|\), the sharpness grows with the weights, and the step-size instability finishes the run. This is also what the CartPole trajectories show at the very end (product 1.5 at the last finite measurement), only on CartPole the first stage takes thousands of steps and on Space Invaders a few hundred.

**Acrobot and MountainCar: no divergence, at the edge of stability.** Nothing diverges in 10,000 steps at either learning rate. Mean \(|Q|\) rises to 82–88 at \(\gamma = 0.99\), which is not a pathology: both environments pay \(-1\) per step, their episodes run 200–500 steps under these policies, and the discounted return is \(-(1 - \gamma^T)/(1 - \gamma) \approx -87\) (\(T = 200\)) to \(-99\) (\(T = 500\)). The no-target runs are learning the correct value scale; the \(\gamma = 0.9\) controls sit at \(|Q| = 10 = 1/(1 - \gamma)\) for the same reason. The target-network runs reach only \(|Q| = 31\)–\(34\) in the same budget (they are slower, not more correct) and have the better Acrobot policy (median −212 against −485 to −500). At lr \(3 \times 10^{-3}\) the product \(\mathrm{lr} \times \mathrm{sharpness}\) has run median 2.5–2.7, above the quadratic threshold, with no divergence: plain SGD on the TD loss is hovering at the edge of stability (Cohen et al., 2021) exactly as it does on supervised losses; the per-run medians are 1.8–2.9 across the 16 runs. Breakout at lr \(3 \times 10^{-2}\) hovers at the same place and does not survive; the difference is the sparse reward and the 400-dimensional input, and we do not have a sharper explanation than that.

### 6.4 The symmetric part is positive, and so is everyone's

The tracked \(\lambda_{\max}(S)\) of the online field is positive at some measurement in every run with at least one finite measurement: 50 of 50 on CartPole and 158 of 158 on the other four environments (two Space Invaders runs at lr \(3 \times 10^{-2}\) diverged before the first measurement completed). It is positive at the first measurement in 50 of 50 CartPole and 154 of 158 other runs; the four exceptions are Breakout runs where the first estimate is \(-10^{-4}\). On CartPole the \(\gamma = 0.9\) runs decay to within \(\pm 0.01\) of zero while the \(\gamma = 0.99\) runs, with or without a target network, stay at 0.02–0.1. On Acrobot and MountainCar at \(\gamma = 0.99\) the median tracked value over the run is 1.1–9.5 at the two learning rates, larger than in the diverging CartPole runs (per-run medians 0.03–1.0), and nothing diverges. This is Section 4.3 again, now in 208 runs: a target network does not make the online field's symmetric part stable, it changes which field is followed; and an unstable symmetric part is where divergence lives when it happens, not a sign that it will. The curl fraction is 0.06–0.35 in every condition and does not separate stable from unstable runs; here too the rotation is a bystander.

### 6.5 What the measurement says to a practitioner

The two remedies everyone uses act on different terms of Proposition 1. A smaller learning rate shrinks \(\alpha^2\|K_\varepsilon\|^2\), the overshoot term, and does nothing to the sign of \(\lambda_{\max}(S)\); it fixes Breakout and delays CartPole. A target network or a shorter horizon changes the field itself, and fixes both. The single measurement that tells the cases apart, \(\mathrm{lr} \times \mathrm{sharpness}\) at the moment \(|Q|\) starts to grow, costs a few Jacobian-vector products every 100 steps and can be logged in any DQN implementation. We are not aware of it having been reported for TD before, and it changes the advice: if the product at onset is far below 2, lowering the learning rate will delay the divergence in proportion and not prevent it.

---

## 7. SD-TD: damping the symmetric spectrum from samples

Sections 4–5 operate on the exact matrix \(K\). The deadly triad happens in an algorithm that sees one transition at a time. This section asks whether the correction survives that, what it costs, whether the cost can be removed, and what it does to the networks of Section 6.

### 7.1 An unbiased rank-two estimate of \(S\), and the algorithm

Each transition \((x, \rho, r, x')\) with \(\delta x = \gamma x' - x\) gives an unbiased estimate \(\hat S = \tfrac12 \rho\,(x\,\delta x^\top + \delta x\, x^\top)\) of \(S\), and its product with a vector,
\[
\hat S v = \tfrac{1}{2}\rho\big(x\,(\delta x^\top v) + \delta x\,(x^\top v)\big),
\]
costs \(O(d)\) and needs no storage beyond \(x\) and \(x'\). SD-TD keeps \(k\) unit vectors \(u_1, \ldots, u_k\) and eigenvalue estimates \(\hat\lambda_1, \ldots, \hat\lambda_k\), initialized at random and 0, and updates them with a deflated online Oja step (Oja, 1982) with rate \(\eta\), then takes a TD(0) step with the positive tracked modes damped.

```
SD-TD(α, η, ε, k)
  initialize w, random unit u_1..u_k, λ̂_1..λ̂_k = 0
  for each transition (x, ρ, r, x'):
      δx = γ x' - x
      δ  = r + γ x'ᵀw - xᵀw                      # TD error
      for i = 1..k:                               # deflated Oja on Ŝ
          s  = ½ ρ ( x (δxᵀu_i) + δx (xᵀu_i) )     # Ŝ u_i
          s  = s - Σ_{j<i} λ̂_j u_j (u_jᵀ u_i)        # deflate tracked modes
          q  = u_iᵀ s                               # Rayleigh quotient
          λ̂_i = (1-η) λ̂_i + η q
          u_i = u_i + η (s - q u_i)
          u_i = normalize( u_i - Σ_{j<i} (u_jᵀu_i) u_j )
      damp = Σ_{i: λ̂_i > 0} (λ̂_i + ε) u_i (u_iᵀ (w - w_ref))    # w_ref = 0 for SD-TD
      w = w + α ( ρ δ x - damp )
```

The update is TD(0) plus \(-\alpha D_t (w - w_{\mathrm{ref}})\) with \(D_t\) the current estimate of the damping matrix \(D\) of Section 4.2 and \(w_{\mathrm{ref}} = 0\). Cost per step is \(O(dk)\): the same order as GTD2 and TDC (which carry one extra \(d\)-vector) and ETD (\(k = 0\) plus a scalar trace). We use \(k = 2\), \(\eta = 0.02\), \(\varepsilon = 0.05\) everywhere; nothing was tuned per family. If every \(\hat\lambda_i \le 0\), the update is exactly TD(0). GTD2 and TDC change the objective so that the Jacobian is negative definite everywhere and converge to the TD fixed point when it exists; ETD reweights the update and converges to a different (emphatic) fixed point. SD-TD changes nothing off the damped subspace. The price is that the brake moves the fixed point (Proposition 3), which the baselines do not pay.

### 7.2 Sampled linear comparison against TD(0), GTD2, TDC, and ETD(0)

Systems are *sampled*, not expected: transitions are drawn from the behavior policy. Five families with rewards: Baird's problem with reward 0 (one system); Baird's problem with Gaussian rewards (five draws); 20 Baird-structure systems with random features, kept only if the expected step at \(\alpha = 0.05\) diverges; 20 random softmax systems (\(\gamma \in \{0.9, 0.99\}\)); 20 on-policy systems (behavior equals target, \(\gamma = 0.95\)). Rewards are scaled by \(1 - \gamma\) so that true values are \(O(1)\) and 20,000 transitions is a meaningful horizon. Each (system, algorithm, \(\alpha\), seed) runs for 20,000 transitions, with six seeds and \(\alpha \in \{0.002, 0.005, 0.01, 0.02, 0.05, 0.1\}\); GTD2 and TDC are run with secondary step \(\beta = \alpha\) (main table) and \(\beta = 10\alpha\); the larger secondary step helps TDC on Baird with reward 0 (1.15 vs 1.86) and is otherwise within 0.1 in RMSVE and more divergence-prone over the step-size grid (30–79% of runs diverged over the whole \(\alpha\) grid, against 2–67% at \(\beta = \alpha\)). A run diverges if \(\|w\|\) exceeds \(10^3(1 + \|v_\pi\|)\). We report the RMSVE averaged over the last fifth of the run and, per (family, algorithm), the step size with the best median RMSVE with divergence counted as \(+\infty\). All 16,632 runs are in `results/linear_runs.json`; the per-step-size divergence table is Appendix C.

| Family | TD(0) | GTD2 | TDC | ETD(0) | SD-TD (\(k = 2\)) | predict 0 |
|---|---|---|---|---|---|---|
| Baird, reward 0 | diverged 100% | 1.91 (0%) | 1.86 (0%) | 330 (67%) | \(10^{-28}\) (0%) | 0 (exact) |
| Baird + rewards | diverged 100% | 2.65 (0%) | 2.66 (0%) | 553 (33%) | **0.87** (0%) | 0.87 |
| Baird structure (diverging) | 38.4 (35%) | 0.70 (0%) | 0.75 (0%) | 0.41 (23%) | 0.71 (5%) | ≈ 0.71 |
| Random softmax | 0.305 (0%) | 0.49 (0%) | 0.41 (0%) | 1.17 (31%) | 0.277 (0%) | — |
| On-policy | 0.206 (0%) | 0.225 (0%) | 0.251 (0%) | 0.206 (0%) | 0.206 (0%) | — |

Entries are median tail RMSVE at the best step size, with the fraction of runs diverged at that step size in parentheses. The last column is the RMSVE of the all-zero predictor (median over systems).

![Figure 1](figures/fig1_baird_curves.png)

*Figure 1. Median RMSVE over seeds and systems at each method's best step size. Left: Baird with reward 0 (true values 0). Right: Baird with Gaussian rewards; the dotted line is the error of predicting 0. SD-TD converges onto that line.*

![Figure 2](figures/fig2_family_summary.png)

*Figure 2. Divergence fraction and median RMSVE at the best step size, per family and method.*

**SD-TD stops the divergence that TD and ETD suffer.** On Baird with rewards, TD(0) diverges at every step size and ETD(0) at every step size (33% of runs even at \(\alpha = 0.002\), 100% above), because the emphasis weights on this problem have enormous variance. SD-TD diverges on 0% of runs at every \(\alpha \le 0.05\) and only at \(\alpha = 0.1\), where the explicit step overshoots exactly as in Section 5.3. On the diverging Baird-structure family, TD diverges in 35% of runs at its best step and 74% over all steps; SD-TD in 5% and 32%. GTD2 and TDC, by contrast, begin to diverge at \(\alpha \ge 0.01\) on that family (Appendix C), as a two-time-scale method does when its secondary step is too large.

**SD-TD is inactive on-policy.** The fraction of steps on which any mode was damped is exactly 0 on 55% of on-policy runs and at most \(2.5 \times 10^{-4}\) on the rest, and SD-TD's numbers coincide with TD(0) to three digits at every step size. It is TD(0) plus a correction that switches itself off, as Proposition 3 and Tsitsiklis–Van Roy together say it should.

**GTD2 and TDC are stable and slow.** Both have 0% divergence at their best steps on every off-policy family, as their theory says. On Baird with rewards both sit at RMSVE 2.65 after 20,000 transitions, three times worse than predicting zero; on Baird with reward 0 they are at 1.9, a 16% reduction of the initial error. This is the known slow convergence of gradient-TD on Baird's features (Sutton et al., 2009, ran the problem for an order of magnitude more sweeps). A 20,000-transition budget is a choice, not a verdict. ETD(0), when it does not diverge, is the most accurate method on Baird-structure (0.41 at its best step, below 0.01 at \(\alpha \ge 0.02\) where it diverges on 43–97% of runs), because its fixed point is exact there.

**SD-TD's accuracy on off-policy problems is the accuracy of the zero predictor.** The bold 0.87 is not a learning result: it equals the RMSVE of predicting zero everywhere (dotted line in Figure 1), and Section 7.3 shows why. On Baird with reward 0 the true values are zero, so \(10^{-28}\) is the trivial case of the same fact. On the random family SD-TD and TD(0) differ by a median of 0.000 across 120 paired runs at \(\alpha = 0.005\), with SD-TD better on 48% of them; the damping fires on tiny positive modes (median 3% of steps) and changes nothing within this horizon.

### 7.3 What damping does to the fixed point

Proposition 3 gives \(w_\varepsilon - w^\ast = K_\varepsilon^{-1} D w^\ast\). We computed both fixed points for every system in Section 7.2 (`results/linear_fixed_points.json`, Figure 3). The Baird-type systems have full-rank features, so the TD fixed point is exact (RMSVE 0) wherever TD would converge to it. The damped fixed point has RMSVE equal to 0.99 times that of the zero predictor on Baird with rewards (five systems) and 0.99 (median of 20) on Baird structure. On the random family, seven of the eight systems with a positive symmetric mode have an exact TD fixed point that damping replaces by one at 0.86–1.02 times the zero predictor; the twelve systems without a positive mode are unchanged. On-policy there is no damping and no change.

So on Baird-type problems the unstable mode of \(S\) is essentially the direction along which the predictor carries value, \(D w^\ast \approx \lambda_{\max} u u^\top w^\ast\) is large, and removing the mode removes the value. SD-TD on these systems is a stabilizer that converges to a predictor near zero. It beats GTD2 and TDC in the table above only because 20,000 transitions is not long enough for them, and it would lose to them at any horizon on which they have converged. The sampled SD-TD runs on the random family did not show the bias within 20,000 steps because the positive eigenvalues there are tiny (0.0003–0.044) and the tracker rarely fired; the fixed-point computation says that with exact modes the asymptote would be biased, so the agreement with TD(0) is a property of the horizon and of a noisy estimate of a small eigenvalue, not a guarantee.

![Figure 3](figures/fig3_fixed_points.png)

*Figure 3. RMSVE of the TD fixed point against RMSVE of the damped fixed point, both relative to the zero predictor, one marker per system. Off-policy systems with a positive mode move from the horizontal axis (exact) to the top (≈ zero predictor); on-policy systems and random systems without a positive mode sit on the diagonal.*

### 7.4 Preserving the fixed point: the anchored variant, and two that fail

Proposition 3 says the shift vanishes iff the damping term vanishes at \(w^\ast\). The direct way to arrange that is to damp \(w - w_{\mathrm{ref}}\) rather than \(w\), where \(w_{\mathrm{ref}}\) is any sequence that converges to \(w^\ast\) and does not depend on \(w\). The pseudocode above already allows it.

**Anchored SD-TD.** Take \(w_{\mathrm{ref}}\) to be a TDC iterate run alongside on the same transitions (secondary step \(\beta = \alpha\)). The joint system is block-triangular: \(w_{\mathrm{ref}}\) follows TDC's dynamics unaffected by \(w\), and \(w\) follows the damped field driven by \(w_{\mathrm{ref}}\). It therefore (i) has \(w^\ast\) as its unique equilibrium whenever TDC converges, (ii) is stable whenever TDC and the damped field \(K_\varepsilon\) are both stable, and (iii) coincides with TD(0) on-policy, where nothing is damped. The cost is one extra \(d\)-vector and a TDC update per step, i.e. SD-TD plus TDC.

We ran TD(0), TDC, SD-TD, and anchored SD-TD on Baird (reward 0), three Baird-with-rewards draws, eight diverging Baird-structure systems, eight random-softmax systems, and six on-policy systems, three seeds each, 20,000 transitions, \(\alpha \in \{0.002, 0.005, 0.01, 0.02\}\) (`n1_anchor.py`, `results/anchor_runs_20000.json`; this is a fresh draw of systems, so the zero-predictor levels differ from Section 7.2).

| Family | \(\alpha\) | TD(0) | TDC | SD-TD | Anchored SD-TD |
|---|---:|---|---|---|---|
| Baird, reward 0 | 0.002 | div. | 2.74 | 0.000 | 2.73 |
| | 0.005 | div. | 1.74 | 0.000 | 1.73 |
| | 0.01 | div. | 2.10 | 0.000 | 2.09 |
| | 0.02 | div. | 1.70 | 0.000 | 1.69 |
| Baird + rewards (zero pred. 0.107) | 0.002 | div. | 3.44 | 0.108 | 3.42 |
| | 0.005 | div. | 1.67 | 0.108 | 1.66 |
| | 0.01 | div. | 2.00 | 0.108 | 1.99 |
| | 0.02 | div. | 1.80 | 0.108 | 1.78 |
| Baird structure (zero pred. 0.298) | 0.002 | 1.24 (46%) | 1.04 | 0.292 | 1.07 |
| | 0.005 | 6.64 (62%) | 0.89 | 0.293 | 0.90 |
| | 0.01 | 96.0 (71%) | 0.74 | 0.293 | 0.75 |
| | 0.02 | div. | 0.54 (50%) | 0.293 | 0.53 (50%) |
| Random softmax (zero pred. 0.255) | 0.002 | 0.264 | 0.525 | 0.261 | 0.298 |
| | 0.005 | 0.219 | 0.372 | 0.219 | 0.232 |
| | 0.01 | 0.173 (38%) | 0.234 (38%) | 0.173 (38%) | 0.173 (38%) |
| | 0.02 | 0.228 (50%) | 0.680 (67%) | 0.228 (50%) | 0.674 (62%) |
| On-policy | 0.002–0.02 | 0.131–0.133 | 0.133–0.269 | 0.131–0.133 | 0.131–0.133 |

Median tail RMSVE over systems and seeds; divergence fraction in parentheses where nonzero.

Three things are true at once. The anchored variant is stable wherever TDC is (0% divergence on every Baird-type cell where TDC has 0%, and the same 50% at \(\alpha = 0.02\) on Baird structure, where TDC's secondary step is too large). It preserves the fixed point: on Baird structure it is at 0.53–1.07 and falling, where plain SD-TD is parked at the zero-predictor level 0.29 forever. And its error tracks TDC's to within 0.02 in every cell: 2.73 against 2.74, 1.66 against 1.67, 0.75 against 0.74. Running both for 200,000 transitions on Baird does not separate them either (Appendix F). The reason is the triangular structure: \(w\) is slaved to \(w_{\mathrm{ref}}\) along the damped modes and follows TD(0) elsewhere, so it can be no faster than its reference along the directions that mattered. On-policy it is TD(0) exactly (the damped fraction is 0.000 on every run); on the random family it is TD(0) wherever the tracker does not fire and slightly worse than TD(0) where it does (0.298 against 0.264 at \(\alpha = 0.002\)), because there it is pulled toward the slower TDC iterate.

This closes the question the earlier draft left open, in the modest direction. A sampled method that stabilizes TD and converges to the TD fixed point exists, is one line away from SD-TD, and inherits the speed of whatever convergent reference it is anchored to. It is a stabilizer for TD(0) that falls back to TDC's answer, not a faster path to that answer. A reader who wants the TD fixed point should run TDC; a reader who wants TD(0) wherever TD(0) is fine and TDC's answer only along the unstable directions has that here.

**Two that fail.** We also report the two fixes a reader would propose first. The *oblique spectral flip* \((I - 2P)K\), with \(P\) the spectral projector onto the unstable eigenspace of \(K\), negates the unstable eigenvalues and preserves the fixed point exactly; computed from the exact matrix it stabilized 100% of the diverging systems of Section 5.2. The version one can track online, the orthogonal flip \((I - (1 + c)UU^\top)K\) with \(U\) the top eigenvectors of \(S\), fixed 0 of 73 Baird-type systems for any \(c\) (median largest real part after the flip 0.001, essentially unchanged), and an online estimate of the oblique projector from single transitions did not converge in our hands. The *slow reference copy*, \(\dot w = Kw + b - D(w - \bar w)\), \(\dot{\bar w} = \tau(w - \bar w)\), keeps \(w^\ast\) as the unique equilibrium but is not block-triangular, and its stability is not implied by that of \(K_\varepsilon\): on exact matrices the joint system is stable on 52 of 52 diverging Baird-structure systems at \(\tau = 10^{-3}\) but only 5 of 15 random and 2 of 21 aliased systems at any \(\tau\) (Appendix F), and in sampled runs the slow time scale makes convergence within 20,000 steps the exception. Both fail for the same reason the anchored variant works: preserving the fixed point requires a reference that converges to it, and \(S\) alone does not know where it is.

### 7.5 A nonlinear value function, 20 seeds

The value function is a SiLU network with widths 8, 32, 1 (321 parameters) on Baird's features, reward 0, true values 0, trained by semi-gradient on the squared TD error with the bootstrap value stopped in the backward pass and refreshed every step. Divergence means \(\|\theta\| > 10^3\) within 600 steps.

The earlier draft projected the update off the top eigenvectors of the symmetric Jacobian of the *loss gradient*, re-estimated every 50 steps by plain power iteration, and reported that it worked on two seeds of four. That was the wrong set of eigenvectors. The loss gradient \(g\) is minus the field, so the unstable field modes are the *most negative* eigenvectors of \(\operatorname{sym}(Dg)\), and plain power iteration returns the eigenvector of largest magnitude, which on this network is usually a stable direction (the sharpest direction of the loss). The fix is a shifted, deflated power iteration: estimate the dominant eigenvalue \(\sigma\) of \(\operatorname{sym}(Dg)\), then iterate on \(\sigma I - \operatorname{sym}(Dg)\), whose top eigenvectors are the most negative of \(\operatorname{sym}(Dg)\) and hence the most positive of the field. The Oja step of SD-TD does this implicitly in the linear case because it maximizes the Rayleigh quotient of \(\hat S\) directly. Tracking the modes every step with two warm-started iterations costs two Jacobian-vector products per mode per step. This is also the estimator behind the \(\lambda_{\max}(S)\) traces of Section 6.

| Learning rate | No correction | Project top 1 (every 50) | Project top 3 (every 50) | Track and damp \(k = 1\) | \(k = 2\) | \(k = 3\) |
|---|---:|---:|---:|---:|---:|---:|
| 0.01 | 5 / 20 | 7 / 20 | 11 / 20 | **20 / 20** | **20 / 20** | **20 / 20** |
| 0.05 | 5 / 20 | 0 / 20 | 4 / 20 | **20 / 20** | **20 / 20** | **20 / 20** |

Seeds stable out of 20. Median final \(\|\theta\|\) for the tracked variants is 8.1 (initial norm 7.8); for the uncorrected runs 4,000 and 13,750. The old projection method is worse than nothing at lr 0.05 with one mode because it projects off a stable direction and leaves the unstable one untouched. One tracked mode is enough here.

![Figure 5](figures/fig5_nonlinear.png)

*Figure 5. Nonlinear Baird network, fraction of 20 seeds stable, by method and learning rate.*

### 7.6 SD-DQN on the five environments

SD-DQN is the DQN update plus \((\hat\lambda_i + \varepsilon)\, u_i u_i^\top\theta\) on the tracked positive modes, with one warm-started shifted power iteration per mode per step (two extra Jacobian-vector products per mode) and an exponential average of the eigenvalue estimate; \(k = 2\) in the table, \(k = 1\) also on CartPole.

| Environment, configuration | DQN, no target | SD-DQN \(k = 2\) | Target network | SD-DQN final \(|Q|\) | SD-DQN median return (DQN target net) |
|---|---:|---:|---:|---:|---|
| CartPole, lr \(10^{-3}\) | 5 / 5 diverged | 0 / 5 | — | 18 | 165 (—) |
| CartPole, lr \(3 \times 10^{-3}\) | 5 / 5 | 0 / 5 (\(k = 1\): 1 / 5) | 0 / 5 | 32 | 217 (390) |
| Acrobot, lr \(3 \times 10^{-3}\) | 0 / 8 | 0 / 8 | 0 / 8 | 4.2 | −494 (−212) |
| MountainCar, lr \(3 \times 10^{-3}\) | 0 / 8 | 0 / 8 | 0 / 8 | 7.1 | −200 (−200) |
| Breakout, lr \(3 \times 10^{-2}\) | 7 / 8 | 1 / 8 | 0 / 8 | 1.5 | 2.3 (2.4) |
| Space Invaders, lr \(3 \times 10^{-2}\) | 8 / 8 | 4 / 8 | 0 / 8 | 2.3 | 3.8 (5.8) |

**Where the mechanism is an outward field, damping the tracked mode removes the divergence.** On CartPole SD-DQN prevents divergence in 19 of 20 runs across the four configurations. Its \(\mathrm{lr} \times \mathrm{sharpness}\) stays below 0.1 at lr \(10^{-3}\) and its \(|Q|\) plateaus at 15–35, about where the target-network baseline plateaus (34–36). It learns worse than a target network: median final return 128–217 against 390, with high variance across seeds (15 to 359). The \(\gamma = 0.9\) controls, which reach return 470–480 with no target network, show that the problem is specific to the long horizon and not to the absence of a target network as such.

**Where the mechanism is overshoot, it helps less, and a target network helps more.** On Breakout SD-DQN reduces divergence from 7/8 to 1/8 and on Space Invaders from 8/8 to 4/8; the target network is 0/8 on both. Damping the outward modes lowers \(\mathrm{lr} \times \mathrm{sharpness}\) (run median 0.75 and 0.14 against 1.1 and 1.2 uncorrected) but does not control the sharpness directly, and on Space Invaders half the runs still cross the threshold.

**Where no divergence was coming, it costs the value scale.** On Acrobot and MountainCar SD-DQN holds mean \(|Q|\) at 3–15 (median 4.2 and 7.1) where the uncorrected runs correctly reach 82–88 and the target network 31–34. The tracked mode is positive throughout (Section 6.4), so the brake is on throughout, and it brakes the direction along which the network is learning that every step costs \(-1\). Acrobot returns are −494 against −212 for the target network. As in the linear case, damping toward zero along the mode that carries value is a brake, and the diagnostic that turns it on cannot tell a value-carrying mode from a diverging one. This is the deep-network version of Proposition 3, and it is the reason we present SD-TD as a stabilizer and the measurement of Section 6 as the contribution.

---

## 8. Discussion

**What the result is.** The answer to "which half of the Jacobian diverges" is the symmetric half, and that is a theorem rather than an observation. The rotation is visible before the weights move, it is a correlate of a positive symmetric mode, and on Baird it reduces the growth rate from 1.02 to 0.24 without changing its sign. The constructive statements are Propositions 1 and 2: damp the positive modes of \(S\), then take a step that tolerates the stiff negative ones, explicitly if \(\alpha < 2\varepsilon'/\|K_\varepsilon\|^2\) and implicitly otherwise. Proposition 3 is the bill.

**What the diagnostic is and is not.** \(\lambda_{\max}(S) > 0\) is necessary for divergence and that is all it is. In 900 linear systems it has a 42% false-positive rate; in 208 deep runs it is positive in every one, including sixteen Acrobot and MountainCar runs where the tracked value is larger than in the diverging CartPole runs and nothing diverges. A positive symmetric mode is also what a network has while it is learning a value scale of \(1/(1 - \gamma)\) from zero, which is why a brake triggered by it costs exactly that. The quantity that does discriminate in the deep case, \(\mathrm{lr} \times \mathrm{sharpness}\) at growth onset, does not say whether a run will diverge; it says *how* a run that is diverging is doing it, and therefore which remedy will work.

**Two mechanisms, two remedies.** On CartPole the growth starts at a product of 0.002–0.009 and the onset scales as \(1/\mathrm{lr}\); lowering the learning rate delays it in proportion and never prevents it, and a target network or shorter horizon does. On MinAtar Breakout the growth starts at 1.9 and a third of the step size prevents it. Space Invaders shows both in sequence. We think the practical content of the paper is this measurement: it costs a few Jacobian-vector products every hundred steps, and it tells a practitioner whether "lower the learning rate" is a fix or a delay.

**What the algorithm is.** SD-TD runs the damping from single transitions at the cost of TD. When it does, it stops the divergence of TD(0) and ETD(0) on every off-policy linear family we tried, stabilizes 20 of 20 seeds of a nonlinear Baird network and 19 of 20 CartPole runs, and reduces MinAtar divergence by a factor of 2–7. Its fixed point is not the TD fixed point, and on the classical counterexample family it is nearly the zero predictor. The anchored variant repairs the fixed point and inherits TDC's speed; it is TD(0) wherever TD(0) is stable and TDC's answer along the directions where it is not. A reader who wants a *corrected learner* should use GTD2/TDC (and wait) or ETD (and accept the variance). A reader who wants TD(0) with a guarantee that it will not blow up has SD-TD, and pays with the value scale on problems where the instability lives in a value-carrying direction, which, in the deep runs, is most of them.

**Why the obvious fixes fail and the anchored one works.** Preserving the fixed point requires knowing where it is, and \(S\) alone does not carry that information: the symmetric spectrum tells you where the field points outward, not how the fixed point sits relative to that direction. The oblique flip needs the spectral projector of \(K\); the slow reference copy needs the reduced dynamics \((K - D)^{-1}K\) to be stable, which they are not in general; the anchored variant borrows the information from a method that already has it, and inherits that method's speed along with it. Any sampled method that both stabilizes and *accelerates* convergence to the TD fixed point will need an estimate of the oblique geometry of \(K\) that is better than TDC's, and we do not have one.

---

## 9. Limitations

Everything is small. The linear systems have at most eight states; the nonlinear results are one 321-parameter network and 4,600–43,000-parameter Q-networks on classic control and MinAtar with 5–8 seeds per configuration and a 10,000-step budget. The budget matters for the negative results: Acrobot and MountainCar did not diverge in 10,000 steps, and we cannot say they would not at 100,000; the MinAtar step sizes are ten times the classic-control ones because plain SGD does not move at \(10^{-3}\) there, and a practitioner would use Adam, whose preconditioning changes the sharpness the step sees. The sharpness and \(\lambda_{\max}(S)\) traces are warm-started power-iteration estimates on a single 256-transition batch every 100 steps; the growth-onset product is read from a trace with 100-step resolution, so on Space Invaders at lr \(3 \times 10^{-2}\), where divergence takes 300–400 steps, we have no onset measurement at all. The mechanism labels (outward field, overshoot) are inferences from the value of one product and one scaling law, not from the eigenstructure of the full Jacobian, which we cannot compute for these networks. SD-TD has three hyperparameters (\(k\), \(\eta\), \(\varepsilon\)) that we fixed once and did not sweep, and the Oja tracker needs the top eigengap of \(S\) to be resolvable from single transitions; the \(\alpha = 0.1\) divergences on Baird are where the explicit step outruns both the tracker and Proposition 1. The gradient-TD comparison uses a 20,000-step budget on which GTD2 and TDC have not converged, which flatters SD-TD's error numbers and is why we print the zero-predictor baseline next to them; the anchored variant's 200,000-step check (Appendix F) is on Baird only. The anchored comparison uses three seeds per cell. The oblique flip and the reference copy are negative results on our constructions, not proofs that no faster fixed-point-preserving spectral correction exists. The families that never diverged (self-loops, cycles at small steps, the random family at small steps) bound the scope: a claim that every off-policy backup has an unstable symmetric mode would be false, and the claim we make is that *when* the explicit semi-gradient step diverged, its symmetric part was unstable, damping the positive modes removed that instability, and the failures that remained were explicit overshoot of a large negative eigenvalue.

---

## 10. Conclusion

Off-policy TD diverges along its symmetric spectrum. That is Bendixson's inequality applied to the TD Jacobian, and it means that the rotation which makes semi-gradient TD "not a gradient" is never the cause of divergence and sometimes a brake on it. The sign of the symmetric spectrum is a necessary condition and a poor predictor, in 900 linear systems and 208 deep runs alike. What distinguishes the ways a deep Q-network diverges is the learning rate times the sharpness of its TD loss at the moment the values start to grow: far below 2 on CartPole, where a smaller step only delays the blow-up, and at 2 on MinAtar Breakout, where a smaller step prevents it. Damping the positive symmetric modes stabilizes every diverging linear system we constructed, can be done from single transitions at the cost of TD, transfers to networks, and is inactive when it is not needed; it also moves the fixed point by an amount we give in closed form, and the variant that repairs the fixed point inherits the speed of the convergent method it is anchored to. The spectral picture tells you where divergence is, how to stop it, and what stopping it costs; the sharpness measurement tells you which kind of divergence you have. Both are a few Jacobian-vector products away in any implementation.

---

## References

- Achiam, J., Knight, E., and Abbeel, P. (2019). Towards characterizing divergence in deep Q-learning. arXiv:1903.08894.
- Asadi, K., Sabach, S., Liu, Y., Gottesman, O., and Fakoor, R. (2023). TD convergence: An optimization perspective. *NeurIPS*.
- Baird, L. (1995). Residual algorithms: Reinforcement learning with function approximation. *ICML*.
- Bendixson, I. (1902). Sur les racines d'une équation fondamentale. *Acta Mathematica* 25.
- Bhandari, J., Russo, D., and Singal, R. (2018). A finite time analysis of temporal difference learning with linear function approximation. *COLT*.
- Cohen, J., Kaur, S., Li, Y., Kolter, J. Z., and Talwalkar, A. (2021). Gradient descent on neural networks typically occurs at the edge of stability. *ICLR*.
- Ghosh, D. and Bellemare, M. G. (2020). Representations for stable off-policy reinforcement learning. *ICML*.
- Kolter, J. Z. (2011). The fixed points of off-policy TD. *NeurIPS*.
- Lyle, C., Rowland, M., and Dabney, W. (2022). Understanding and preventing capacity loss in reinforcement learning. *ICLR*.
- Lyle, C., Zheng, Z., Nikishin, E., Avila Pires, B., Pascanu, R., and Dabney, W. (2023). Understanding plasticity in neural networks. *ICML*.
- Oja, E. (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology* 15.
- Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D., Szepesvári, C., and Wiewiora, E. (2009). Fast gradient-descent methods for temporal-difference learning with linear function approximation. *ICML*.
- Sutton, R. S., Mahmood, A. R., and White, M. (2016). An emphatic approach to the problem of off-policy temporal-difference learning. *JMLR* 17.
- Tsitsiklis, J. N. and Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. *IEEE Transactions on Automatic Control* 42.
- Van Hasselt, H., Doron, Y., Strub, F., Hessel, M., Sonnerat, N., and Modayil, J. (2018). Deep reinforcement learning and the deadly triad. arXiv:1812.02648.
- Young, K. and Tian, T. (2019). MinAtar: An Atari-inspired testbed for thorough and reproducible reinforcement learning experiments. arXiv:1903.03176.
- Zhang, S., Yao, H., and Whiteson, S. (2021). Breaking the deadly triad with a target network. *ICML*.

---

## Appendix A. Proofs and remarks

**A.1 Lemma 1 for the \(d_\mu\)-weighted inner product.** Tsitsiklis and Van Roy's on-policy argument is that \(K\) is negative definite in the inner product \(\langle u, v \rangle_{D} = u^\top \Phi^\top D_\mu \Phi v\). Lemma 1 holds in any inner product with \(S\) replaced by the symmetric part in that inner product. We use the Euclidean inner product throughout because SD-TD estimates \(S\) from samples, where the Euclidean symmetric part is the one with an unbiased rank-two estimator; the choice of inner product changes which matrix is called \(S\) but not the lemma.

**A.2 The numerical range remark after Lemma 1.** The set \(\{v^* K v : \|v\| = 1\}\) (numerical range) has real part exactly \([\lambda_{\min}(S), \lambda_{\max}(S)]\) and contains the spectrum. \(\lambda_{\max}(S) > 0\) says the numerical range reaches into the right half-plane; it does not say an eigenvalue does. The 186 systems of Section 4.3 are those where it does not.

**A.3 Proposition 1, tightness.** Equality in Proposition 1 requires a unit vector that is simultaneously a top eigenvector of \(S_\varepsilon\) (for the \(-2\alpha\varepsilon'\) term) and a top right-singular vector of \(K_\varepsilon\). Generically it is not tight; it is the bound we need, and it was never violated over 1,800 checks.

**A.4 Proposition 3 and ETD.** ETD's fixed point solves an emphasis-reweighted system \(K_M w + b_M = 0\). With full-rank features (Baird structure has \(d = n + 1 > n\)) both the TD and the emphatic fixed point are the exact value function, which is why ETD is exact on Baird structure in Section 7.2 whenever it converges. SD-TD's shift \(K_\varepsilon^{-1} D w^\ast\) is nonzero whenever \(D w^\ast \ne 0\), full-rank features or not.

**A.5 The anchored variant as a triangular system.** With \(w_{\mathrm{ref}}\) following TDC, the expected joint dynamics are \(\dot w_{\mathrm{ref}} = F(w_{\mathrm{ref}})\) (TDC's field, which does not involve \(w\)) and \(\dot w = Kw + b - D(w - w_{\mathrm{ref}}) = K_\varepsilon w + D w_{\mathrm{ref}} + b\). If \(w_{\mathrm{ref}} \to w^\ast\) then the second equation is a stable linear system (Proposition 1 or 2) driven by an input converging to \(D w^\ast\), so \(w \to -K_\varepsilon^{-1}(D w^\ast + b) = -K_\varepsilon^{-1}(D w^\ast - K w^\ast) = -K_\varepsilon^{-1}(-K_\varepsilon w^\ast) = w^\ast\). The rate along the damped modes is governed by the slower of \(\varepsilon'\) and TDC's rate; along the undamped modes it is TD(0)'s. The two-time-scale TDC analysis (Sutton et al., 2009) covers the sampled \(w_{\mathrm{ref}}\); we do not give a sampled proof for the joint system.

## Appendix B. System generators

All random draws use `numpy.random.default_rng` with the seeds in the scripts.

- **Baird (reward 0 / with rewards).** Features as in Section 5.1. Transitions: solid action to state 6, dashed uniform on states 0–5. Target solid w.p. 1, behavior solid w.p. \(1/7\). Rewards 0, or \(\mathcal N(0, 1) \times (1 - 0.99)\) per (state, action). \(\gamma = 0.99\). Initial weights \((1,1,1,1,1,1,10,1)\).
- **Baird structure.** \(n \in \{4,5,6,7\}\) states, features \(\mathcal N(0,1)^{n \times (n+1)}\); solid to the last state, dashed uniform on the rest; target solid, behavior solid w.p. \(1/n\); rewards \(\mathcal N(0,1) \times 0.01\); \(\gamma = 0.99\); initial weights \(\mathcal N(0, I)\). For Section 7 a draw is kept only if \(\rho(I + 0.05 K) > 1 + 10^{-8}\).
- **Random softmax.** \(n \in \{3, \ldots, 7\}\) states, dimension \(\in \{2, \ldots, n+1\}\), features \(\mathcal N(0,1)\), transitions softmax of \(\mathcal N(0,1)\) logits per (state, action), two actions, target \((0.95, 0.05)\), behavior \((0.05, 0.95)\), \(\gamma \in \{0.9, 0.99\}\), rewards \(\mathcal N(0,1)(1 - \gamma)\).
- **On-policy.** \(n \in \{4, \ldots, 8\}\), dimension \(\in \{2, \ldots, n-1\}\) (rank-deficient on purpose), transitions softmax of \(2\mathcal N(0,1)\), target = behavior = softmax of \(\mathcal N(0,1)\) per state, \(\gamma = 0.95\), rewards \(\mathcal N(0,1) \times 0.05\).
- **Cycles, self-loops, aliased** (Sections 4.3, 5.2, 5.3 only): as in `probes/push_further.py`.

Sampling (Section 7) draws \(s_0 \sim d_\mu\) and then 20,000 steps of the behavior chain; \(\rho = \pi(a|s)/\mu(a|s)\).

## Appendix C. Divergence fraction and median RMSVE by step size (Section 7.2)

Each cell is "fraction diverged / median tail RMSVE among non-diverged runs" (nan when all diverged). GTD2 and TDC at \(\beta = \alpha\).

| Family | Algorithm | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|---|
| Baird (reward 0) | TD | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – |
| | SD-TD | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 1.00 / – |
| | GTD2 | 0 / 1.93 | 0 / 1.93 | 0 / 1.92 | 0 / 1.91 | 1.00 / – | 1.00 / – |
| | TDC | 0 / 2.03 | 0 / 2.18 | 0 / 3.10 | 0 / 1.86 | 1.00 / – | 1.00 / – |
| | ETD | 0.67 / 330 | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – |
| Baird + rewards | TD | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – |
| | SD-TD | 0 / 0.87 | 0 / 0.87 | 0 / 0.87 | 0 / 0.87 | 0 / 0.87 | 1.00 / – |
| | GTD2 | 0 / 2.66 | 0 / 2.66 | 0 / 2.65 | 0 / 2.67 | 1.00 / – | 1.00 / – |
| | TDC | 0 / 2.66 | 0 / 2.73 | 0 / 2.68 | 0 / 2.80 | 1.00 / – | 1.00 / – |
| | ETD | 0.33 / 553 | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – | 1.00 / – |
| Baird structure | TD | 0.35 / 38.4 | 0.62 / 13.7 | 0.74 / 6.0 | 0.80 / 12.0 | 0.95 / 163 | 1.00 / – |
| | SD-TD | 0.05 / 0.71 | 0.10 / 0.55 | 0.10 / 0.56 | 0.15 / 0.73 | 0.57 / 0.28 | 0.95 / 0.28 |
| | GTD2 | 0 / 0.88 | 0 / 0.70 | 0.24 / 0.77 | 0.75 / 1.76 | 1.00 / – | 1.00 / – |
| | TDC | 0 / 0.75 | 0 / 0.76 | 0.03 / 0.79 | 0.25 / 0.82 | 0.87 / 1.90 | 1.00 / – |
| | ETD | 0.23 / 0.41 | 0.33 / 0.14 | 0.38 / 0.05 | 0.43 / 0.00 | 0.84 / 0.00 | 0.97 / 0.00 |
| Random softmax | TD | 0 / 0.30 | 0 / 0.31 | 0.20 / 0.36 | 0.70 / 0.53 | 1.00 / – | 1.00 / – |
| | SD-TD | 0 / 0.28 | 0 / 0.28 | 0.19 / 0.36 | 0.70 / 0.54 | 1.00 / – | 1.00 / – |
| | GTD2 | 0 / 0.49 | 0.31 / 0.59 | 0.76 / 0.58 | 0.95 / 8.19 | 1.00 / – | 1.00 / – |
| | TDC | 0 / 0.42 | 0 / 0.41 | 0.33 / 0.57 | 0.75 / 0.58 | 1.00 / – | 1.00 / – |
| | ETD | 0.31 / 1.17 | 0.51 / 2.20 | 0.74 / 2.87 | 0.97 / 5.01 | 1.00 / – | 1.00 / – |
| On-policy | TD | 0 / 0.24 | 0 / 0.25 | 0 / 0.23 | 0 / 0.21 | 0 / 0.21 | 0 / 0.21 |
| | SD-TD | 0 / 0.24 | 0 / 0.25 | 0 / 0.23 | 0 / 0.21 | 0 / 0.21 | 0 / 0.21 |
| | GTD2 | 0 / 0.34 | 0 / 0.28 | 0 / 0.22 | 0 / 0.25 | 0.03 / 0.25 | 0.35 / 0.19 |
| | TDC | 0 / 0.34 | 0 / 0.28 | 0 / 0.25 | 0 / 0.25 | 0 / 0.27 | 0.10 / 0.25 |
| | ETD | 0 / 0.21 | 0 / 0.21 | 0.34 / 0.24 | 0.70 / 0.21 | 0.91 / 0.41 | 1.00 / – |

The random-softmax family diverges for *every* method at \(\alpha \ge 0.05\), including SD-TD and TD(0), although the *expected* explicit step is stable at \(\alpha = 0.05\) on all 20 systems. This is the variance of the sampled update, not an outward expected field: the importance ratio reaches \(0.95/0.05 = 19\), so a single sampled step \(\alpha\rho\, x\,\delta x^\top\) can have norm well above 1. Damping the symmetric spectrum of the expected field does not and should not help there; it is the regime of step-size and variance control.

## Appendix D. Trajectories before divergence

Selected logged rows (every 100 steps), `results/cartpole_runs.json` and `results/deep_envs_runs.json`. "lr·sh" is learning rate × sharpness; \(\hat\lambda\) is the tracked \(\lambda_{\max}(S)\) of the field.

**CartPole.**

| Config | step | mean \(|Q|\) | lr·sh | \(\hat\lambda\) | curl | \(\|\theta\|\) |
|---|---:|---:|---:|---:|---:|---:|
| lr \(10^{-3}\), seed 0 (NaN at 4,800) | 300 | 0.11 | 0.002 | 1.07 | 0.01 | 15.6 |
| | 1,000 | 1.76 | 0.003 | 1.54 | 0.01 | 15.8 |
| | 2,000 | 12.1 | 0.053 | 0.04 | 1.10 | 16.5 |
| | 3,000 | 16.6 | 0.064 | 0.01 | 0.18 | 17.4 |
| | 4,000 | 55.6 | 0.356 | 0.00 | 0.14 | 20.7 |
| | 4,500 | 91.9 | 0.797 | 0.24 | 0.17 | 23.6 |
| | 4,700 | 213 | 2.35 | 0.03 | 0.15 | 27.0 |
| lr \(3 \times 10^{-3}\), seed 0 (NaN at 1,600) | 300 | 0.27 | 0.007 | 1.04 | 0.00 | 15.6 |
| | 700 | 8.2 | 0.067 | 0.09 | 0.42 | 16.2 |
| | 1,000 | 13.1 | 0.142 | 0.01 | 0.33 | 16.9 |
| | 1,300 | 22.5 | 0.360 | 0.03 | 0.35 | 18.8 |
| | 1,500 | 65.6 | 1.09 | 0.00 | 0.33 | 21.8 |
| lr \(10^{-2}\), seed 0 (NaN at 700) | 300 | 0.97 | 0.027 | 1.00 | 0.00 | 15.7 |
| | 400 | 17.1 | 0.701 | 0.19 | 0.02 | 16.6 |
| | 600 | 38.3 | 2.16 | 0.02 | 0.28 | 19.9 |

In every case \(|Q|\) has grown by one to two orders of magnitude before lr·sh reaches 0.5. The first tracked \(\hat\lambda\) is near 1 in all configurations, including the stable ones; it is the tracked estimate on the first batch and decays as the network moves off initialization.

**MinAtar, last six measurements before divergence** (mean \(|Q|\) / lr·sh at each 100-step measurement).

| Config | Trajectory | NaN at |
|---|---|---:|
| Breakout, lr \(3 \times 10^{-2}\), seed 2 | 1.3/1.35, 2.4/1.84, 2.0/1.86, 2.2/1.88, 2.3/1.10, 4.3/2.67 | 1,800 |
| Breakout, lr \(3 \times 10^{-2}\), seed 1 | 2.2/1.48, 2.3/1.80, 3.4/2.25, 4.2/1.57, 3.7/1.94, 4.1/1.71 | 6,200 |
| Breakout, lr \(3 \times 10^{-2}\), seed 4 | 3.4/1.99, 2.0/1.24, 2.2/1.38, 2.8/1.77, 2.6/2.61, 4.1/2.38 | 7,500 |
| Breakout, lr \(3 \times 10^{-2}\), seed 0 | 3.0/3.01, 4.0/2.13, 3.9/1.48, 3.7/1.86, 3.2/1.65, 3.7/1.57 | 9,800 |
| Space Invaders, lr \(10^{-2}\), seed 0 | 0.54/0.09, 4.8/0.55 | 500 |
| Space Invaders, lr \(10^{-2}\), seed 3 | 0.22/0.03, 1.1/0.08, 6.4/0.59, 11/1.35 | 700 |
| Space Invaders, lr \(10^{-2}\), seed 7 | 5.4/0.51, 6.0/0.72, 6.6/0.86, 6.5/0.74, 9.5/1.12, 13/1.52 | 1,200 |
| Space Invaders, lr \(10^{-2}\), seed 2 | 5.9/0.50, 7.4/0.73, 6.7/0.53, 7.6/0.68, 8.2/1.57, 8.5/1.10 | 2,000 |

On Breakout, \(|Q|\) does not trend upward before the blow-up; lr·sh oscillates around 1.5–2.5 for thousands of steps and the run ends within 100–300 steps of a measurement above 2. On Space Invaders at lr \(10^{-2}\), \(|Q|\) grows by a factor of 5–20 over 200–400 steps while lr·sh rises from below 0.1 to above 1.

**Per-seed final returns (CartPole).** Target network 378, 354, 499, 439, 390; no-target \(\gamma = 0.9\) at lr \(3\times10^{-3}\): 471, 478, 500, 497, 433; SD-DQN \(k = 2\), lr \(3\times10^{-3}\): 217, 359, 185, 160, 242; SD-DQN \(k = 2\), lr \(10^{-3}\): 165, 224, 15, 154, 287; SD-DQN \(k = 1\), lr \(10^{-3}\): 70, 191, 229, 128, 74; SD-DQN \(k = 1\), lr \(3\times10^{-3}\): 223, (diverged), 258, 172, 134.

**Per-seed final mean \(|Q|\) (Acrobot and MountainCar, \(\gamma = 0.99\)).** Acrobot no target lr \(10^{-3}\): 52, 74, 74, 84, 93, 94, 95, 96; lr \(3 \times 10^{-3}\): 64, 73, 74, 82, 83, 89, 92, 93; target network: 28–37; SD-DQN \(k = 2\): 3.0, 3.2, 3.3, 3.6, 4.8, 4.9, 5.6, 11.3. MountainCar no target lr \(10^{-3}\): 78–95; lr \(3 \times 10^{-3}\): 77–94; target network: 32–35; SD-DQN \(k = 2\): 3.4, 4.7, 6.2, 7.0, 7.2, 9.5, 9.5, 15.1. Weight norms in the no-target runs end at 18.5–19.7 (initial ≈ 15.9): the values grow because the last layer learns the scale, not because the weights are blowing up.

## Appendix E. Hyperparameters

| Component | Setting |
|---|---|
| SD-TD | \(k = 2\), \(\eta = 0.02\), \(\varepsilon = 0.05\); modes initialized \(\mathcal N(0, I)\) normalized, \(\hat\lambda = 0\) |
| Anchored SD-TD | as SD-TD; \(w_{\mathrm{ref}}\) is a TDC iterate with \(\beta = \alpha\), initialized at \(w_0\), updated on the same transition before the SD-TD step |
| GTD2 / TDC | secondary weights initialized 0; \(\beta \in \{\alpha, 10\alpha\}\), better reported |
| ETD(0) | followon trace \(F_t = \gamma\rho_{t-1}F_{t-1} + 1\), interest 1, emphasis \(M_t = F_t\) |
| Linear runs | 20,000 transitions, 6 seeds (3 for the anchored comparison), \(\alpha \in \{0.002, 0.005, 0.01, 0.02, 0.05, 0.1\}\), divergence at \(\|w\| > 10^3(1 + \|v_\pi\|)\), tail = last 20% |
| Nonlinear Baird | SiLU MLP [8, 32, 1], 600 steps, lr \(\in \{0.01, 0.05\}\), 20 seeds; projection re-estimated every 50 steps with 12 power iterations; tracking: 2 warm-started iterations per step; damping margin 0.05 |
| Deep runs (all five environments) | SiLU MLP [obs, 64, 64, actions], buffer 20,000, batch 64, warm-up 256, \(\varepsilon\): 1.0 → 0.05 over 5,000 steps, 10,000 env steps, plain SGD; observation scales CartPole \((2.4, 3.0, 0.21, 3.0)\), Acrobot \((1,1,1,1,12.57,28.27)\), MountainCar \((1.2, 0.07)\), MinAtar none (binary); MinAtar sticky actions off, 1,000-step episode cap; SD-DQN: 1 warm-started shifted iteration per mode per step, \(\hat\lambda\) EMA 0.8/0.2, margin 0.05; measurements every 100 steps on a 256-batch; 3 curl probes; divergence at mean \(|Q| > 10^4\) or non-finite weights |
| Theory checks | 900 systems; \(\varepsilon = 0.05\); matrices with \(\|K_\varepsilon\| < 10^{-6}\) skipped |

## Appendix F. Fixed-point-preserving variants: additional results

**F.1 Slow reference copy, joint-system stability.** Joint linear system \(\dot w = (K - D)w + D\bar w + b\), \(\dot{\bar w} = \tau(w - \bar w)\), stability by the eigenvalues of the \(2d \times 2d\) block matrix, and of its explicit Euler step at \(\alpha = 0.05\). "Diverging" counts systems whose undamped explicit step at \(\alpha = 0.05\) diverges.

| \(\tau\) | Baird structure (52 diverging) | Random (15) | Aliased (21) |
|---:|---:|---:|---:|
| 0.1 | 41 stable | 4 | 2 |
| 0.01 | 50 | 5 | 2 |
| 0.001 | 52 | 5 (6 for the explicit step) | 2 |

Sampled runs (20,000 steps, three seeds; RMSVE at the end): Baird with rewards, \(\alpha = 0.01\): SD-TD 0.279, 0.278, 0.277 (zero predictor 0.280); reference variant \(\tau = 0.01\): 1.59, 1.13, 1.00; \(\tau = 10^{-3}\): 11.0, 10.5, 9.6; \(\tau = 10^{-4}\): 10.5, 10.4, 10.3 (initial RMSVE 5.5). Baird-structure system 3 (\(\lambda_{\max}(S) = 0.68\), zero predictor 1.54): SD-TD 1.53, 1.52, 1.52; reference \(\tau = 0.01\), \(\alpha = 0.02\): 0.001, 0.001, 0.001 (converged to the true fixed point); system 1 (\(\lambda_{\max}(S) = 1.12\)): both variants diverged at \(\alpha = 0.005\) on all seeds.

**F.2 Anchored variant at 200,000 transitions.** TDC, SD-TD, and anchored SD-TD on Baird (reward 0) and one Baird-with-rewards draw, \(\alpha \in \{0.005, 0.02\}\), two seeds, RMSVE logged every 2,000 steps. Raw numbers in `results/anchor_long.txt` (`n1_anchor_long.py`).

| System | \(\alpha\) | seed | TDC at 20k / 100k / 198k | Anchored SD-TD at 20k / 100k / 198k | SD-TD |
|---|---:|---:|---|---|---|
| Baird, reward 0 | 0.005 | 0 | 2.17 / 2.13 / 2.11 | 2.16 / 2.13 / 2.11 | 0.000 |
| | 0.005 | 1 | 1.25 / 1.23 / 1.14 | 1.24 / 1.23 / 1.13 | 0.000 |
| | 0.02 | 0 | 0.59 / 0.54 / 0.47 | 0.59 / 0.54 / 0.47 | 0.000 |
| | 0.02 | 1 | 12.1 / 13.5 / 10.8 | 12.1 / 13.4 / 10.7 | 0.000 |
| Baird + rewards (zero pred. 0.104) | 0.005 | 0 | 2.06 / 2.03 / 2.01 | 2.05 / 2.03 / 2.01 | 0.104 |
| | 0.005 | 1 | 1.15 / 1.13 / 1.04 | 1.14 / 1.12 / 1.03 | 0.104 |
| | 0.02 | 0 | 0.69 / 0.63 / 0.55 | 0.69 / 0.63 / 0.55 | 0.103 |
| | 0.02 | 1 | 12.0 / 13.4 / 10.7 | 12.0 / 13.3 / 10.6 | 0.103 |

Initial RMSVE 5.3 (reward 0) and 5.2 (with rewards). Over 200,000 transitions TDC and the anchored variant never differ by more than 0.02; both are still far from the fixed point at \(\alpha = 0.005\), and one seed at \(\alpha = 0.02\) sits at RMSVE 11–13 for the whole run (TDC's slow mode on Baird's features, inherited by the anchored variant). Plain SD-TD is at the zero-predictor level from 20,000 steps on.

## Appendix G. Additional expected-update results from the earlier draft

- **Skew scaling on Baird.** Largest real part of \(S + \beta A\): \(\beta = 0\): 1.021; \(\beta = 1\): 0.239; \(\beta = 1.5\): 0.131. Decreasing in \(\beta\) over the range we scanned and never negative. Lemma 1 bounds this quantity above by \(\lambda_{\max}(S) = 1.021\) for every \(\beta\); it gives no lower bound, and we did not find a \(\beta\) at which the real part changed sign.
- **Adaptive freeze.** Freezing the target the first time the curl fraction exceeds 0.2 keeps the weights at norm 12.7 on all four seeds. It works because the frozen-target update is gradient descent on a quadratic.
- **Clip versus implicit on cycles at \(\alpha = 0.5\).** Clipping the eigenvalues of \(S_\varepsilon\) into \((-2/\alpha + \varepsilon, -\varepsilon)\) fixed 3 of 5 inspected cycle failures. Clipping the symmetric spectrum controls only the \(-2\alpha\varepsilon'\) term of Proposition 1, not the \(\alpha^2\|K_\varepsilon\|_2^2\) term, so it can fail when the skew part is large; we did not diagnose the two misses further. The implicit step fixed all 5.
- **Stable families.** Self-loops never diverged at either step size; cycles diverged only at \(\alpha = 0.5\), and all 12 were overshoot (fixed by the implicit step, not by damping).
