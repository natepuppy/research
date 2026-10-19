# R19 pilot notes

**Verdict: SUPPORTED** (K1, K2, K4 pass; K3 half-pass: scaling right, constant wrong). `results/verdict.txt`.

## What was run
Binary-choice corridor (reward 1 at the end of corridor A, 0 at the end of B, `forward`/`stall` actions, γ = 1, cap 3D). Tabular softmax policy, one SGD step per iteration on batch-normalised GAE advantages (32 episodes); tabular critic fit exactly to λ-return targets each iteration; segment length T ∈ {∞, 16}. Grid D ∈ {8..128} × λ ∈ {0, 0.5, 0.8, 0.9, 0.95, 0.99, 1} × T × 8 seeds, 600 iterations. ~4 min CPU.

## Results
- **K1 (law in D and λ)**: on the 16 cells with D(1 − λ) ≥ 4, log t_V on log D slope 0.95, on log(1 − λ) slope 0.88, R² = 0.999. Post-hoc linear fit on all 30 λ < 1 cells: t_V = 1.25·D(1 − λ) + 1.9, R² = 0.999.
- **K2 (λ = 1 removes it)**: t_V = 1 for every D (slope 0.00).
- **K3 (rollout length)**: T = 16, λ = 1 scales linearly in D (slope 1.03) but is 1.5–2.3× slower than λ = 0.95, T = ∞, not within 1.5×. Fails as pre-registered. Post-hoc: the two slowdowns add as rates, 1/h_eff = (1 − λ) + 1/T_eff with T_eff ≈ 6 states for a 16-step segment (R² 0.998), rather than the min() I pre-registered (R² 0.988). T_eff < 16 because stalls make 16 time steps cover ~8 states and fixed-time boundaries land on different states across episodes.
- **K4 (policy inherits it)**: Spearman(t_π, t_V) = 0.99; t_π − t_V is −2..+1 iterations. The start-state policy moves in the iteration the critic value arrives.
- **Latency, not decay**: all 70 cells reach success prob ≥ 0.98 by 600 iterations. With the critic frozen at 0 (post-hoc P7), λ = 0.95 fails at D = 128 and λ = 0.8 fails everywhere — the decay view is correct *only* without a critic.

## What I believe now
The 1/(1 − γλ) horizon is an iterations-per-state speed, not a cutoff. In an iterated actor-critic loop a reward D steps away reaches the start after ≈ D(1 − γλ) critic fits (≈ D/T_eff if the rollout segment is shorter), and the policy follows immediately. Setting λ = 1 (or a decoupled λ_critic = 1, as VC-PPO does) makes arrival one iteration, but only if the rollout segment covers the distance — a short `num_steps` reintroduces the same latency at any λ.

## Caveats
- Tabular, exact critic fit, SGD policy: a neural critic fits targets imperfectly (fewer than "one full propagation step" per iteration), and Adam rescales small gradients. Both change the constant, neither the D(1 − λ) scaling, but this must be shown on the 3080 (CleanRL PPO on long-corridor MiniGrid, sweep λ and `num_steps`, log the iteration at which the start-state advantage becomes informative).
- The constant 1.25 and T_eff ≈ 6 are specific to this environment's stall structure.
- One SGD step per batch: PPO's multiple epochs and clipping change how far the policy moves per iteration, not when the signal arrives.
- γ = 1. With γ < 1 the horizon is 1/(1 − γλ) and nothing else changes.

## Next
3080: neural PPO replication; vary learning-rate/epochs to show the latency is in iterations, not samples; RLHF-style check with a frozen reward at the end of a long generation (λ_critic ∈ {0.95, 1}) to connect to VC-PPO.
