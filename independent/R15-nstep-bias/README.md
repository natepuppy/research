# R15: Why does uncorrected n-step work? A bias accounting

**Question.** Every strong value-based agent (Rainbow, R2D2, Agent57, MuZero's reanalyse) uses n-step returns built from *behaviour* actions taken from the replay buffer with no off-policy correction. This is theoretically wrong (the fixed point is not Q*), and empirically it is what makes large replay buffers useful at all (Fedus et al. 2020), while importance-weighted corrections do not help (Hernandez-Garcia & Sutton 2019). Nobody has explained the paradox. Our hypothesis: the uncorrected n-step target blends the behaviour policy's value into the target, Q ← (T^μ)^{n−1} T* Q, whose fixed point lies *below* Q* for any ε-greedy μ, so n-step carries a pessimistic bias that grows with n and ε; Q-learning under noise carries the opposite, positive max-bias (Thrun & Schwartz 1993; van Hasselt 2010). In noisy environments the two cancel, and that cancellation, not the horizon, is why uncorrected n-step helps. Three testable consequences: overestimation falls with n; the best n rises with noise; removing the max-bias (double Q) removes the n-dependence.

**Status: INCONCLUSIVE by rule; directional predictions replicated on fresh seeds.** Three σ grids (all reported). Pilot grid passed K1/K2 by rule but its noisy agents never learned (discounted). Replication on fresh seeds (σ 0/0.05/0.1): overestimation falls with n in every noisy cell (Spearman −1.00), n ≥ 2 is pessimistic without noise (costs 11–23% of return at n = 10), best n rises 1 → 2–3 → 3 with σ, and the best n is one of the two n at which the two biases net to zero in 10/10 noisy cells. The competence threshold I set (plain Q ≥ 50% of V* at σ = 0.1) was missed at 45%. Double Q halves the max-bias but does not remove the n-dependence (K3 failed in all three grids). Decomposition shows the max-bias shrinking with n is the dominant term, the pessimistic term the smaller opposite-signed one. See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`, `paper/main.md`. Paper version: DQN / Double DQN on MinAtar with injected reward noise on the 3080; prediction that the n-step gain is largest for plain DQN in noisy games and smallest for Double DQN in deterministic ones; a corrected-vs-uncorrected comparison that controls for propagation speed (R12).

Adjacent stub in `research/ideas` (S2-5, "n-step by transition age") proposes a *method* (age-dependent n). This is the *diagnosis* of why uncorrected n-step works at all.

## Pre-registered kill test (written before the pilot ran)

Design: 8×8 grid, slip 0.1, γ = 0.95, reward 1 on entering the goal (terminal), zero-mean Gaussian reward noise σ ∈ {0, 0.5, 1.5} on every step (Q* is unchanged); behaviour ε-greedy with constant ε ∈ {0.1, 0.3}, exploring starts (uniform); n ∈ {1, 2, 3, 5, 10}; method ∈ {Q, double-Q (online argmax, target evaluation)}; FIFO buffer 5000, batch 32, α = 0.1 per sampled copy, target table refreshed every 100 updates; 150k steps, 4 seeds. Overestimation = Σ_s d^{π_g}(s)·(max_a Q(s, a) − V*(s)) with d the greedy policy's discounted visitation from the start state; greedy return = exact V^{π_g}(s₀). Both averaged over the last five checkpoints.

- **K1 (sign).** For plain Q, Spearman(n, overestimation) ≤ −0.8 in every noisy cell (σ > 0, each ε), and at σ = 0 every n ≥ 3 is pessimistic (overestimation ≤ 0).
- **K2 (optimal n rises with noise).** For plain Q, the n with the highest area-under-curve of greedy return is larger at σ = 1.5 than at σ = 0, for both ε.
- **K3 (double-Q removes the dependence).** The slope of overestimation on log n, averaged over noisy cells, is less than half as large in magnitude for double-Q as for plain Q.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither.

### Pre-registered follow-up (written after the pilot, before the follow-up ran)

The pilot passed K1 and K2 by rule, but K2 is degenerate: at σ ∈ {0.5, 1.5} every plain-Q agent's greedy return is ≈ 0.001 against V* = 0.468 (max-bias 2–9, i.e. 5–20× V*), so "best n" there ranks failed agents. The σ grid was set too high for α = 0.1. Follow-up, same code and seeds, `--sigmas 0 0.1 0.2 0.3 --tag band`: a noise band where the agents should still learn. Same K1/K2/K3 with σ_max = 0.3 in place of 1.5. The follow-up verdict replaces the pilot verdict for K2; K1 at σ = 0 (pessimistic n ≥ 3) stands either way. Rule for reporting: SUPPORTED only if K1 and K2 hold **and** the plain-Q agents at σ_max reach ≥ 50% of V* for at least one n (so the "best n" is a comparison between working agents).

*Outcome of `band`:* K1 and K2 pass again (best n = 1, 3, 5–10, 10 at σ = 0, 0.1, 0.2, 0.3), but the working-agent condition fails at σ_max = 0.3 (best plain-Q return 0.7% of V*). It holds at σ = 0.1 (55% of V*), where best n = 3 and the overestimation of n = 3 is the one nearest zero. By the rule above, `band` is **not** SUPPORTED; the σ grid was still too wide.

### Pre-registered replication (written after `band`, before `band2` ran)

Third and final σ grid, chosen from `band` so that agents work at σ_max: `--sigmas 0 0.05 0.1 --seed-base 4 --tag band2`, **fresh seeds** (4–7). Predictions, fixed in advance: (P1) K1 holds (overestimation falls with n at σ = 0.05 and 0.1; n ≥ 3 pessimistic at σ = 0); (P2) best n by AUC is larger at σ = 0.1 than at σ = 0 for both ε, and best n at σ = 0.05 lies between them inclusive; (P3) the plain-Q agent at σ = 0.1 reaches ≥ 50% of V* for at least one n; (P4) at σ = 0.1 the best n is one of the two n whose over_all is nearest zero (the cancellation prediction). Final verdict: SUPPORTED if P1–P3 hold; P4 reported as the mechanism check; INCONCLUSIVE if P2 fails only at σ = 0.05's ordering; KILLED if P2 fails at σ = 0.1 or P1 fails. All three σ grids are reported in the paper.

*Outcome of `band2`:* P1 holds (−1.00 in all four noisy cells); P2 holds (best n = 1, 1 / 3, 2 / 3, 3); P3 **fails** (45% of V*, vs 55% in the first seed set); P4 holds (and 10/10 across `band` + `band2`). P3's failure is not enumerated in the rule above, so it is not KILLED and not SUPPORTED: **INCONCLUSIVE**.

## Layout

| folder | contents |
|---|---|
| `code/` | `r15_nstep.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (config × seed × checkpoint), `summary.csv`, `verdict.txt` (all three grids), `posthoc.txt`; `band/`, `band2/` follow-up grids |
| `logs/` | `pilot.log`, `band.log`, `band2.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r15_nstep.py --smoke                 # ~10 s
.venv/bin/python code/r15_nstep.py --steps 150000 --seeds 4 # pilot grid, ~15 min CPU
.venv/bin/python code/r15_nstep.py --steps 150000 --seeds 4 --sigmas 0 0.1 0.2 0.3 --tag band
.venv/bin/python code/r15_nstep.py --steps 150000 --seeds 4 --seed-base 4 --sigmas 0 0.05 0.1 --tag band2
.venv/bin/python code/r15_nstep.py --summarize [--tag band2]
```
