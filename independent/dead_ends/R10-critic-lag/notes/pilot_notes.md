# R10 pilot notes

Two runs, both batched DDPG on Pendulum, 3 actor LRs × 4 critic LRs × 8 seeds = 96 runs, 30k steps, CPU. **Pilot**: evaluation every 1k steps, lags k = 0..6 (0–6k steps), 12 min; `logs/pilot.log`, `results/`. **Fine-resolution follow-up** (pre-registered after the pilot, README): evaluation every 100 steps, lags 0..10 (0–1000 steps), 31 min; `logs/fine.log`, `results/fine/`. Ground truth: exact Monte-Carlo Q^{π}(s, a) from 64 probe states, 250-step horizon, γ = 0.98, deterministic dynamics.

## Verdicts

- Pilot: **INCONCLUSIVE** by rule (K1 true, K2 false, K3 true). But K1's estimator is biased: k* = argmin over seven noisy, nearly flat errors lands on k > 0 by selection. The *mean* error profile is minimised at k = 0 in every cell (`results/posthoc.txt`), so K1's pass is an artefact.
- Fine-resolution follow-up: **KILLED.** F1 false (slow-critic cells have the mean profile minimised at k = 0 with support 0.00–0.33; the three cells with kmin > 0 differ from k = 0 by < 2% and are the *fast*-critic cells with flat profiles). F2 false (ρ = +0.24).

**Bottom line: at any resolution from 100 to 6000 steps, the critic is not detectably evaluating a past policy.** Where the policy is still moving (critic LR 1e-4, drift 5–10 Q-units per 100 steps), the critic's error against the current policy is 43–48, three times that of fast critics, but it is equally wrong about every past policy, and slightly more so the older the policy. The error is inaccuracy, not staleness.

## Fine-resolution profiles (mean |Q_critic − Q^{π_{t−k}}(s, π_t(s))|, k = 0, 1, …, 10 × 100 steps; window: steps ≥ 10k)

| actor LR | critic LR | k = 0 | k = 2 | k = 5 | k = 10 | drift / 100 steps | final return (pilot) |
|---|---|---|---|---|---|---|---|
| 1e-4 | 1e-4 | 46.9 | 47.3 | 47.9 | 48.6 | 5.3 | −176 |
| 3e-4 | 1e-4 | 48.1 | 48.4 | 48.8 | 49.6 | 6.4 | −161 |
| 1e-3 | 1e-4 | 42.7 | 42.9 | 43.3 | 43.8 | 10.2 | −168 |
| 1e-4 | 3e-4 | 23.1 | 23.2 | 23.1 | 23.0 | 0.9 | −129 |
| 3e-4 | 1e-3 | 15.5 | 15.6 | 15.6 | 15.8 | 0.6 | −127 |
| 1e-3 | 3e-3 | 14.4 | 14.4 | 14.4 | 14.4 | 0.5 | −127 |

## What survived

- **Critic error against the current policy predicts performance.** Across the 12 LR cells, Spearman(err0, final return) = −0.78 (pilot). Slow critics (1e-4) finish 35–50 return points worse than everything else; above 3e-4 the critic LR barely matters.
- **Critic error tracks policy drift.** Spearman(drift, err0) = +0.90 across cells; pooled over (run, eval), log err0 on log drift has correlation 0.67. The two-timescale intuition is right about the *consequence* (a critic that cannot keep up is wrong, and the actor suffers) and wrong about the *form* (it is not evaluating an old policy; it is evaluating the current one badly).
- **The instrument works.** Exact MC Q against stored past actors is cheap in a deterministic simulator and gives a clean, low-variance error measurement. The flatness of the profiles is not noise: bootstrap support for kmin = 0 in the slow cells is 0.67–1.00.

## Why the hypothesis was wrong (reading)

A critic trained by TD on replayed data is not a delayed copy of the true Q of some past policy. Its targets mix the Polyak target actor (200-update timescale), bootstrapped values that are themselves wrong, and actions from the whole buffer history. The resulting error is a smooth bias field over the state-action space that does not line up with any single policy's Q. "Lag" was the wrong metaphor; "tracking error" is the right one, and it is set by critic LR relative to drift, as the error-vs-drift correlation shows.

## Caveats

- Pendulum only, DDPG only (single critic, no twin, no delay), 64-unit networks. A slower-moving, higher-dimensional task might show a lag that Pendulum does not.
- Probe states are uniform over the state space, not the on-policy distribution.
- The MC horizon of 250 steps at γ = 0.98 truncates < 1% of the return.

## Where this goes

Not a NeurIPS paper on its own. Two salvageable pieces: (1) the tracking-error-vs-drift relation as a diagnostic for setting the critic LR (a short workshop note); (2) the instrument (exact MC against stored actors) is reusable for any critic-quality question, including R11's planned overestimation measurement. The "anticipatory targets" stub (S4-2) should not assume a lag exists; the pilot says to measure tracking error instead.
