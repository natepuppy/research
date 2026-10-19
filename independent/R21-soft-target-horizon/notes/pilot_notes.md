# R21 pilot notes

**Verdict: SUPPORTED** (K1, K2, K3, K4 all pass). `results/verdict.txt`.

## What was run
Deterministic synchronous tabular updates (every state every step, no replay, no exploration), Q ← Q + α(y − Q) with n-step targets bootstrapped from a Polyak table Q̄ ← Q̄ + τ(Q − Q̄), γ = 0.99. Chain of 200 states with reward 1 per step for the horizon measurement; patience forks (now = 1 vs wait d steps for R with Rγ^d = 2) for the switch time. τ ∈ {0.1, 0.01, 0.001} × α ∈ {1, 0.1, 0.01} × n ∈ {1, 5}, d ∈ {10, 30, 100}, plus hard-sync controls at K = 1/τ. 17 s CPU. No seeds: the system is deterministic.

## Results
- **K1**: the effective horizon h(t) grows linearly at n/(1/τ + 1/α) per update in 18/18 cells (ratio of measured to predicted slope 0.98–1.10, R² = 1.000).
- **K2**: the patience switch happens at t* = (m − 1)/τ + m/α − max(1/τ, 1/α)/3 with m = ⌈d/n⌉: log–log slope 1.011, R² = 0.9997, 54/54 cells within [0.7, 1.4] (actual range 0.79–1.03).
- **K3**: τ and α enter symmetrically; cells with τ = α take the predicted (2m − 1)/(m − 1) multiple of the α = 1 time, 12/12.
- **K4**: hard sync with K = 1/τ matches soft τ within 1.08× in all 15 cells with m ≥ 6; the hard front is a staircase, the soft front a straight line of the same mean slope.
- Smoke-test correction: the first draft counted m target stages; the segment adjacent to the reward needs no bootstrap, so it is m − 1. Fixed by re-derivation before the pilot, documented in the README.

## What I believe now
A Polyak target table is a cascade of first-order filters, one per n-step segment between a decision and its consequence, and the critic's fit rate α adds a second filter per segment. The effective horizon of the critic grows linearly in updates, h(t) ≈ n·t/(1/τ + 1/α); until it reaches the task horizon the agent is a shorter-horizon agent, i.e. soft target updates are an implicit discount schedule. At τ = 0.01 the agent is a γ_eff ≈ 0.9 agent for the first ~1000 updates and reaches γ_eff = 0.99 only after ~10 000.

## Caveats
- This is a linear system and the pilot confirms a derivation; the content is the interpretation and the α-symmetry, both of which the deep version must earn. The exactness here will look trivial to a reviewer unless the deep measurement of h(t) is the headline.
- α_eff for a replay-trained neural critic is unknown and is the quantity that decides whether τ or fitting dominates in SAC/TD3 — that is the first 3080 measurement (regress the learned V on truncated Monte-Carlo returns to read h(t) directly during Hopper training).
- Synchronous updates; replay adds R12's second term.
- γ_eff is defined by horizon matching, not by any formal equivalence of policies.

## Next
3080: TD3 and SAC on Hopper/Walker, τ ∈ {0.05, 0.005, 0.0005} × n ∈ {1, 5}, measuring h(t) from the critic against truncated MC returns and the patience-gridworld switch time; compare the measured per-segment delay to 1/τ to extract α_eff; test whether a τ schedule (large early, small late) beats the fixed default as the law suggests.
