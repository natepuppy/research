# R20 pilot notes

**Verdict: SUPPORTED** (K1 and K2 pass; K3, the bonus control, fails at D = 30). `results/verdict.txt`.

## What was run
Corridors D ∈ {10, 20, 30}, reward 1 at the far end, ε = 0.1 with random greedy tie-breaking (untrained agent = random walk). Tabular Q-learning, uniform replay 20k, batch 32, α = 0.1, γ = 0.99, uncorrected n-step, hard target table with period K. K ∈ {1, 200, 2000} × n ∈ {1, 5} × count bonus β ∈ {0, 0.1} × 8 seeds, 300k steps, exact greedy rollout every 500 steps. All 288 runs discovered and solved. ~22 min CPU.

## Results
- **K1**: in the Nature-DQN-like cell (K = 2000, n = 1, no bonus) the time from first reward to a working greedy policy exceeds the time to find the reward by 52× (D = 10), 20× (D = 20), 6× (D = 30). Post-discovery share of time-to-solve: 0.98 / 0.95 / 0.86.
- **K2**: Spearman(median T_p, (⌈D/n⌉ − 1)K + 1) = 0.92 over 18 no-bonus cells. Post-hoc two-term R12 fit: T_p = 0.96·(⌈D/n⌉ − 1)K + 59·D²/(nB) + 584, R² = 0.996. The target-term coefficient is 0.96: the post-discovery time is literally the target-sync count.
- **K3 (bonus halves T_d)**: true at D = 10 and 20, false at D = 30 (9324 vs 10055). The bonus was a weak explorer here (shortens median T_d in only 5/18 cells overall) and it lengthens T_p in 16/18 cells, so total time-to-solve is worse with the bonus in 16/18 cells. Candidate mechanism: stale bonuses in the buffer inject persistent value noise that masks the γ^D-attenuated true signal. Not tested.
- Removing the propagation latency (K = 1, n = 5) cuts total time-to-solve 31× / 18× / 7× relative to (K = 2000, n = 1) with the same exploration.

## What I believe now
In this family of tasks, "hard exploration" under standard value-based settings is mostly post-discovery latency with a closed form, and the share shrinks as D grows only because discovery time grows faster (random-walk ∝ D²-ish with the episode cap) than the linear target-sync term. An exploration bonus attacks the smaller half and, as implemented with stored bonuses, damages the larger half.

## Caveats
- Tabular, deterministic, one-dimensional; the propagation law is R12's and we already knew it holds tabularly. The new content is the accounting (share) and the bonus trade-off.
- The bonus control is weak (β = 0.1, stale). Must redo with β ∈ {0.3, 1} and a recomputed (non-stale) bonus before claiming anything about bonus methods.
- Discovery times here are small (hundreds to ~10k steps) by design; in MiniGrid/Montezuma discovery can be 10⁶+ steps and the share will be smaller. The paper claim must be stated as a function of (T_d, K, n, D), not as a universal percentage.
- Greedy success evaluated exactly; ε-greedy return would lag further.

## Next
3080: DQN on MinAtar and MiniGrid hard-exploration tasks logging first extrinsic reward and first successful greedy episode under K × n × {none, RND, count} bonus. Pre-register: the post-discovery share under Nature-DQN defaults, and whether bonus methods lengthen T_p there as they do here.
