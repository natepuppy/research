# R1 pilot: evaluating policies that choose when to check

Run 15 Oct 2026 on an M1 Pro CPU. Two loggers: the default clinician (waits randomised 30% of the time, `results/`) and a near-protocol clinician (2%, `results/protocol/`). 5 dataset seeds × 3000 patients each; ground truth from 3 × 5000 simulated patients per target. Simulator: `code/lib/ctsim.py` (continuous-time glucose, rescue + insulin actions, policy-chosen check interval, check cost).

## Verdict

`KILLED` under both loggers by the pre-registered rule. The identification trick (mask the extra checks, reweight for the logger's "keep waiting" decisions) did not help on the `sparse` target in either regime, and the support flag did not separate `dense` from `sparse`. What the pilot did show is below, and two of those results are worth keeping.

## Mean bias (estimate − truth) over 5 seeds, default logger

| target (truth) | bin1h | bin4h | exo | timing | timing+mask | timing+ipw |
|---|---|---|---|---|---|---|
| mode (8.77) | −5.04 | +7.32 | −0.05 | **+0.08** | +0.26 | +2.02 |
| sparse (8.02) | −4.30 | +8.07 | **+0.69** | −0.50 | +0.94 | +3.25 |
| dense (12.72) | −9.00 | +3.36 | −4.01 | **−1.04** | −0.71 | −0.57 (sd 1.65) |
| dosehi (−2.50) | −39.1 | +5.86 | −5.76 | −8.71 | −8.44 | −5.79 |
| doselo (3.95) | −9.00 | −11.4 | −4.29 | −3.06 | −2.97 | −2.47 |

Near-protocol logger (`results/protocol/summary.csv`): same pattern for binning and `exo`; `timing` becomes high-variance on `sparse` (−1.9 ± 3.9) and `dense` (−2.4 ± 2.4) because the rare random waits are all the support there is; masking gets worse (`sparse`: naive −4.1, IPW −6.4, IPW capped at 10 −7.0).

## What it says

1. **Binning is miscalibrated in both directions for the same policy.** 1 h bins say the clinician's own modal rule is worth 3.7; 4 h bins say 16.1; the truth is 8.8. The sign flips with Δ, so no single "binning inflates" story holds; each Δ evaluates a different fictional policy (K3 held: 4/6 binned estimates sat closer to the fixed-grid policy's truth than to the target's). This is the cleanest version of the Frost result I have seen and costs nothing to reproduce.
2. **Treating timing as exogenous is fine until the target changes it.** `exo` (Insulin4RL protocol) is within 0.7 for `mode` and `sparse` but off by −4.0 for `dense`. Putting the wait into the action (`timing`, an SMDP FQE with γ^τ) brings `dense` to −1.0 *when the logger randomises waits*. That is a real, simple improvement to the Insulin4RL recipe, and it needs no identification trick.
3. **The masking/IPW augmentation failed, and the reason is informative.** For a sparser schedule to be identified, the logger must sometimes have looked and done nothing when the reading was bad. Clinicians (and this simulator's logger) act on bad readings almost always, so the "keep waiting" probability in hypo/hyper bands is ~0.04 × 0.01 and the inverse weights hit the cap. The Robins no-direct-effect result is correct and useless here: it identifies the value of looking less only on patients who were stable anyway. Under the randomised logger the augmentation added bias (naive +0.9, IPW +3.3) on top of an estimator that was already fine. Drop it as a method; keep it as a one-paragraph impossibility statement.
4. **The support flag as built is too blunt.** Fraction of target look-ups landing on *unvisited* state-actions stayed ≤ 0.03 everywhere because a few random waits visit every cell. A count-weighted or ESS-style flag is needed before K2 can be tested properly.
5. `dosehi` and `doselo` are badly estimated by everything (sd 5–6 across seeds): the fine glucose band is not enough state for a policy that doses differently, and the coarse 4-bin time feature leaks horizon effects. That is a limitation of the tabular pilot, not of any estimator.

## Caveats

- One synthetic simulator with hand-set dynamics; the value gaps between targets (8.0 vs 8.8 vs 12.7) were tuned so timing matters.
- Tabular FQE on 96 states × 15 actions. A neural SMDP FQE might do better on `dosehi`/`doselo` but would not change points 1–3.
- IPW cap 50 and 10; no self-normalisation across schedules beyond the per-cell weighting.

## What this means for the paper

The "sensing policy OPE" framing (Idea 1 as written) loses its method contribution. The surviving contributions are: (a) the degenerate-policy account of binning with the sign flip, (b) the SMDP (dose, wait) FQE as a drop-in fix to the Insulin4RL protocol, with the result that it only works when the logger's timing varies, and (c) a clean impossibility statement about evaluating sparser monitoring from protocol-driven logs. That is a workshop or ML4H paper from this simulator, and an AISTATS/ML4H paper if (a)–(c) hold on Frost's UVA/Padova simulator and Insulin4RL. Fold the binning theory into R3, which is where the stronger result turned up.

## Re-running

```bash
python code/r1_timing_ope.py                                  # default logger, ~5 min
python code/r1_timing_ope.py --eps-wait 0.02 --tag protocol  # near-protocol logger
python code/r1_timing_ope.py --summarize [--tag protocol]
python code/r1_timing_ope.py --full                           # 20 seeds × 10000 patients
```
