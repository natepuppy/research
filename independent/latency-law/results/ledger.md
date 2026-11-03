# Numbers used in `paper/main.md`

Copied from the pilot verdicts and drafts. Nothing here was re-fit.

## Hard targets — `R12-bootstrap-speed-limit`

- Law: t = (⌈L/n⌉ − 1)·K + c·⌈L/n⌉·L/B.
- Pre-registered max() form: killed (SGD R² 0.883).
- Target-limited cells, median t / ((h−1)K): 1.002 (SGD), 1.004 (Adam). L = 64, K = 1000, n = 1: 63 003 and 63 005 against 63 000.
- Confirmation grid, c fixed at 2.63 (SGD) and 4.21 (Adam): identity-fit R² 0.997 and 0.971.
- Source: `../R12-bootstrap-speed-limit/paper/main.md` §§4.2–4.3, `results/verdict.txt`.

## Polyak — `R21-soft-target-horizon`

- h(t) ≈ n·t / (1/τ + 1/α). Slope within 0.98–1.10 of prediction in 18/18 cells, R² = 1.000.
- Switch time: 54/54 within [0.79, 1.03]×, log–log slope 1.011, R² = 0.9997.
- τ = α = 0.01 doubles the latency of α = 1 (19 750 vs 9 855). 12/12.
- Hard K = 100 matches τ = 0.01 (9 901 vs 9 855).
- At τ = 0.01, α = 1, n = 1: horizon 2, 11, 31, 100 at updates 100, 1 000, 3 000, 10 000.
- Source: `../R21-soft-target-horizon/paper/main.md` §4.

## GAE — `R19-gae-speed-limit`

- t_V = 1.25·D(1 − λ) + 1.9 on 30 cells, R² = 0.999. Slopes 0.95 (D) and 0.88 (1 − λ).
- λ = 1: t_V = 1 at every depth through 128.
- Spearman(t_π, t_V) = 0.99 over 70 cells. All 70 cells eventually solve.
- Critic frozen: λ = 0.95 fails at D = 128 (final success 0.16).
- Rollouts: 1/h = (1 − λ) + 1/T_eff, T_eff = 6.1, R² = 0.998. The pre-registered min() form was worse.
- Source: `../R19-gae-speed-limit/paper/main.md` §4.

## Neural critics — `R22-generalisation-hops`

- η = 1.02–1.03 for one-hot, random, thermometer, coordinate. Slopes of T_p on K: 38.9–39.1. R² ≥ 0.99998.
- L = 40, K = 800: T_p = 31 250 for every encoding.
- Gap (states 15–24 empty): coordinate within 1% of V* at the full-coverage time; one-hot values 24% high.
- Policy shortcut: coordinate greedy-correct at update 250, values at 7 850.
- Source: `../R22-generalisation-hops/paper/main.md` §4.

## Choosing n — `R24-nstep-rule`

- 1 344 learners. Spearman of time-to-correct on (⌈21/n⌉ − 1)K + 1: 1.00 over 66 cells. Median T/law = 1.05.
- Learned value vs closed-form bias: r = 0.985, RMSE 0.029, 168 cells.
- Largest safe n within one grid step of n* in 9/12 cells; the three misses have predicted margins 0.026, 0.005, 0.001 against seed noise 0.06. Same n̂ at K = 200 and K = 2000 in 12/12.
- ε = 0.01, K = 2000, gap ≥ 0.2: T = 40 200 at n = 1 and 2 200 at n = 12, 8/8 seeds.
- Source: `../R24-nstep-rule/results/verdict.txt`, `results/posthoc.txt`, `paper/main.md` §3.

## Exploration — `R20-exploration-is-propagation`

- K = 2000, n = 1: discovery 346 / 1 862 / 9 324 and propagation 18 154 / 37 716 / 57 648 at depth 10 / 20 / 30. Shares 0.98 / 0.95 / 0.86.
- Two-term fit: coefficient 0.96 on (⌈D/n⌉ − 1)K, R² = 0.996.
- Count bonus shortens discovery in 5/18 cells, lengthens propagation in 16/18, worsens total time in 16/18. The pre-registered "bonus halves discovery" test failed.
- Source: `../R20-exploration-is-propagation/paper/main.md` §4.
