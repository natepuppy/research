# R18: The sup-norm is the wrong yardstick: which value-error statistic predicts greedy-policy performance?

**Question.** Value-based RL reports value accuracy in two ways: theory bounds the greedy policy's suboptimality by the sup-norm of the value error, ‖Q̂ − Q*‖∞ · 2/(1−γ) (Singh & Yee 1994; Bertsekas & Tsitsiklis 1996), and practice plots "estimated versus true Q" or "overestimation" (van Hasselt et al. 2016; Fujimoto et al. 2018) and argues that reducing it improves performance. Fujimoto et al. (2022) showed that the Bellman error does not track the value error. We ask the next question: does any value-error statistic track *performance*? Our hypothesis: no norm of the error does, because the greedy policy's loss depends only on where the error is large *relative to the action gap* at states the greedy policy itself visits; the sup-norm is dominated by irrelevant states; and the overestimation plots of the deep-RL literature measure a quantity that is, at fixed task, nearly uncorrelated with the policy's return. We test this on a corpus of random MDPs with exact Q*, using Q̂s produced by actual Q-learning runs (so the errors have the structure learning produces) and comparing eleven statistics as predictors of the exact suboptimality of the greedy policy, with training time held fixed so that "everything improves together" cannot carry the correlation.

**Status: KILLED.** Laptop pilot (CPU, 24 Garnet MDPs × 24 Q-learning configurations × 10 checkpoints = 5760 Q̂s, exact truth). At fixed task and training time every norm predicts suboptimality moderately (sup 0.72, RMS 0.75–0.78); the gap-crossing statistic predicts better (0.93) but the pre-registered claim was that norms fail, and they do not; visitation weighting adds 0.02. The corpus's errors are diffuse (sup/RMS = 2.9, the i.i.d. value), which is why; the localised-error regime the hypothesis is about was not produced. Two results stand: the Bellman error is the weakest predictor (0.58), and overestimation has the *wrong sign* at fixed time (ρ = −0.65: learners that are behind under-estimate more and perform worse), so overestimation compared across methods at matched steps measures progress, not bias. See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`, `paper/main.md`. Paper version: the same horse race on learned Q-networks (CartPole, MinAtar) with exact Monte-Carlo ground truth from probe states (R10/R11 instrument), to test whether the published "overestimation → performance" story survives at fixed task; a gap-aware accuracy metric for value learning.

Adjacent: Farahmand (2011) proves tighter bounds when the action gap is large; Bellemare et al. (2016) increase the gap by design. Neither tests which statistic *predicts* performance empirically. No lab stub covers this.

## Pre-registered kill test (written before the pilot ran)

**Corpus.** 24 Garnet MDPs (|S| = 50, |A| = 4, branching 3, γ = 0.9), 12 with dense rewards N(0, 1) per (s, a) and 12 with sparse rewards (5% of (s, a) pairs nonzero, N(0, 1)); evaluation distribution ρ uniform. Learners: online tabular Q-learning with ε-greedy behaviour and a restart to a uniform random state every 50 steps; α ∈ {0.05, 0.2} × ε ∈ {0.1, 0.5} × reward noise σ ∈ {0, 0.5} × 3 seeds = 24 configurations, 100k steps, Q̂ recorded at 10 checkpoints (10k … 100k). For every Q̂: greedy π̂, exact V^{π̂}, Δ = Σ_s ρ(s)(V*(s) − V^{π̂}(s)) ≥ 0, visitations d^{π*}_ρ and d^{π̂}_ρ.

**Statistics.** (1) `sup` ‖Q̂ − Q*‖∞; (2) `rms` uniform RMS; (3) `rms_dstar` RMS weighted by d^{π*}; (4) `rms_dhat` RMS weighted by d^{π̂}; (5) `bell_sup` ‖Q̂ − T*Q̂‖∞; (6) `bell_rms_dhat`; (7) `over_dhat` mean of (max_a Q̂ − V*) under d^{π̂} (the overestimation plotted in the literature); (8) `abs_over_dhat` its absolute value; (9) `cross_unif` fraction of states where the greedy action is suboptimal; (10) `cross_dstar` the same mass under d^{π*}; (11) `cross_dhat` the same mass under d^{π̂}. (9)–(11) are "gap-crossing" statistics: an error matters only if it flips the argmax, and only where the policy goes.

**Correlation protocol.** For each (MDP, checkpoint) group the 24 learner Q̂s are ranked; Spearman(statistic, Δ) is computed within the group and averaged over the 240 groups ("time-controlled"). The pooled within-MDP version over all 240 Q̂s is also reported.

- **K1 (the sup-norm fails).** Time-controlled mean Spearman(`sup`, Δ) ≤ 0.5.
- **K2 (gap crossing where the policy goes works).** Time-controlled mean Spearman(`cross_dhat`, Δ) ≥ 0.8.
- **K3 (no norm comes close).** `cross_dhat` exceeds the best of (1)–(8) by ≥ 0.1, and `cross_dhat` exceeds `cross_unif` by ≥ 0.05 (weighting by the policy's own visitation matters).

SUPPORTED if K1, K2 and K3. INCONCLUSIVE if exactly two. KILLED otherwise. Secondary, reported: time-controlled |Spearman(`over_dhat`, Δ)| ≤ 0.5 (the overestimation plot does not predict performance at fixed task and time).

## Layout

| folder | contents |
|---|---|
| `code/` | `r18_yardstick.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (one row per Q̂ with all statistics and Δ), `summary.csv` (per-statistic correlations), `verdict.txt`, `posthoc.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r18_yardstick.py --smoke
.venv/bin/python code/r18_yardstick.py            # pilot, ~3 min CPU
.venv/bin/python code/r18_yardstick.py --summarize
```
