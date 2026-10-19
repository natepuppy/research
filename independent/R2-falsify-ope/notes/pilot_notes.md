# R2 pilot: falsification tests vs. the usual checks

Run 15 Oct 2026 on an M1 Pro CPU, ~5 min. Grid: 5 failure modes × 20 MDPs × n ∈ {500, 2000} × 3 data replicates × 3 targets × 5 estimators = 9000 rows (`results/runs.csv`). Kill test scored on FQE / WIS / DR only.

## Verdict

`KILLED` by the pre-registered rule, but read the table before dropping the idea. K1 failed (battery AUROC 0.63 vs standard checks 0.60 for separating biased from unbiased estimates). K2 failed because two of the four failure modes were barely detected. K3 failed narrowly (agreement power against confounding 0.35, not ≤ 0.2).

## Power at 10% false alarm (thresholds calibrated on the no-failure configurations)

| failure | frac biased | agree | ess | self | nco_is | nco_model | ncp | placebo | battery | standard |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.16 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| confounding | 0.46 | 0.35 | 0.01 | 0.12 | 0.24 | 0.53 | **0.99** | 0.35 | 0.99 | 0.35 |
| miscal | 0.24 | 0.48 | 0.00 | **0.95** | 0.47 | 0.10 | 0.80 | 0.39 | 0.98 | 0.46 |
| abstraction | 0.18 | 0.07 | 0.06 | 0.02 | 0.11 | 0.07 | 0.38 | 0.10 | 0.27 | 0.09 |
| support | 0.17 | 0.27 | 0.10 | 0.10 | 0.14 | 0.11 | 0.07 | 0.30 | 0.18 | 0.28 |

Mean |error| of FQE by failure: none 0.07, confounding 0.33, miscal 0.07, abstraction 0.19, support 0.13 (returns are on a 0–10 scale). DR under confounding: 1.08. WIS: ~1.1 everywhere, including under no failure.

## What it says

- **The two tests that have a named failure mode work.** The negative-control policy (`ncp`: perturb the target only in inert states, nothing should move) catches confounding by indication with power 0.99 where estimator agreement gets 0.35. Behaviour self-evaluation (`self`) catches a miscalibrated behaviour model with power 0.95. Both are cheap and need nothing but the logs. This is the part of the idea worth keeping.
- **Agreement is weak but not useless here**, because WIS and FQE have *different* biases under confounding, so they disagree. The claim "agreement has zero power against shared bias" needs a setting where every estimator shares the bias (e.g. all model-based, or a preprocessing failure). The pilot did not build that setting.
- **Abstraction and support were under-injected.** Merging states in threes moved FQE by 0.19 and the support gap by 0.13, so there was little to detect; the tests' low power there says nothing yet. Both need harsher versions (merge across reward-relevant boundaries; targets that put most of their mass off-support).
- **K1 was the wrong metric.** The "biased" label was dominated by WIS, which is off by ~1 regardless of failure because its effective sample size is 1–5% at horizon 10. The tests are not designed to detect variance, and scoring them on it penalised them. Next version: score detection of *failure presence* (as in the power table) and, for bias, label on FQE/DR only.
- `nco_model` (does E[baseline covariate | s0, a0] depend on a0 under the target?) had power 0.53 against confounding: a real but weaker signal than `ncp`. `nco_is` is noisy because of the weights.

## Caveats

- Tabular, horizon 10, 12 states. Everything model-based is nearly exact under no failure, which flatters FQE.
- One confounding mechanism (a binary frailty variable that tilts actions and lowers rewards). Power against confounding should be checked against weaker κ.
- Bootstrap standard errors with B = 12–25 are crude; some of the false-alarm calibration noise comes from that.

## What to do next

1. Re-run with harsher abstraction and support injections and a shared-bias setting (model-based estimators only, plus a timing-shift failure from R1's simulator).
2. Replace K1 with: AUROC for failure presence per test, and a formal statement of what `ncp` tests (it is an exclusion-restriction test: in inert states the action can only matter through confounding).
3. Then the sepsis simulator (Oberst & Sontag) and the MIMIC-IV demo pipeline.

## Re-running

```bash
python code/r2_falsify.py                 # ~5 min
python code/r2_falsify.py --summarize
python code/r2_falsify.py --full          # 40 MDPs, n up to 8000, 5 replicates
```
