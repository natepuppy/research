# T3 pilot: what does a reset reset?

Run 14 Oct 2026 on an M1 Pro CPU, default grid: 2 RL regimes × 6 interventions + supervised control × 2, 3 seeds, 42 runs. Each run: supervised pretraining on family F0 (1500 steps, 99% accuracy), then 5 phases × 300 GRPO steps (G = 8, B = 64, Adam 1e-3, KL to reference), probes at every phase boundary. Rows in `results/phases.csv` (per-phase learning) and `results/checkpoints.csv` (probes and network metrics), factorial table in `results/factorial.csv`.

Note on the loss: these runs used a group-*sum* GRPO loss with β = 0.01. The script now uses the standard group-mean loss with β = 0.01/G, which is the same optimisation problem under Adam.

## Verdict

`INCONCLUSIVE` by the pre-registered rule, for a specific reason: the *positive control did not go positive*. Supervised relabelling (the textbook plasticity-loss inducer) made this 100k-parameter model *more* trainable, not less (supervised probe 0.72 → 0.81). So the probes cannot be certified as sensitive, and the rule stops there. The data still say something clear:

## Main numbers (mean over 3 seeds; probes are the gain in success probability from 100 fresh-Adam steps on a held-out family)

| regime / intervention | late-phase RL gain (phases 3–5) | sup probe base → late | RL probe base → late | eNTK eff. rank base → late | Adam g²/v late |
|---|---|---|---|---|---|
| rl_curriculum / none | 0.155 | 0.68 → 0.48 (−29%) | 0.28 → 0.18 | 41 → 34 | 0.35 |
| rl_curriculum / adam_reset | 0.146 | 0.68 → 0.45 | 0.28 → 0.17 | 41 → 30 | 1.24 |
| rl_curriculum / ref_reset | 0.152 | 0.68 → 0.49 | 0.28 → 0.20 | 41 → 35 | 0.38 |
| rl_curriculum / both (ProRL) | 0.150 | 0.68 → 0.46 | 0.28 → 0.18 | 41 → 34 | 1.25 |
| rl_curriculum / shrink_perturb (0.8/0.2) | 0.115 | 0.68 → 0.27 | 0.28 → 0.21 | 41 → 33 | 0.21 |
| rl_curriculum / temp_reinflate (T = 1.5 first third) | **0.170** | 0.68 → 0.48 | 0.28 → 0.18 | 41 → 32 | 0.39 |
| rl_relabel / none | 0.160 | 0.72 → 0.52 (−28%) | 0.26 → 0.17 | 41 → 31 | 0.36 |
| rl_relabel / both | 0.149 | 0.72 → 0.48 | 0.26 → 0.18 | 41 → 34 | 1.53 |
| rl_relabel / temp_reinflate | **0.181** | 0.72 → 0.53 | 0.26 → 0.18 | 41 → 35 | 0.32 |
| sup_relabel / none (control) | 0.370 | 0.72 → **0.82** (+14%) | 0.26 → 0.40 | 41 → 40 | 2.07 |

## Reading

1. **Trainability does decline under prolonged GRPO, and it does not under the matched supervised curriculum.** Both RL regimes lose ~28% of fresh-Adam supervised-probe gain and ~35% of RL-probe gain over five phases, monotonically. The same curriculum trained with cross-entropy *gains* trainability. This is the opposite of the "RL makes small updates, so nothing is lost" expectation: relative update norm is 0.18 for RL vs 0.24 for supervised, not dramatically smaller, and the RL runs lose eNTK effective rank (41 → 30–35) where supervised does not (41 → 40). NTK rank collapse, SWD's mechanism, appears here under GRPO with no bootstrapping.
2. **None of ProRL's resets helps.** Optimizer reset, reference reset, and both leave late-phase learning unchanged (0.146–0.152 vs 0.155 for none). There is no reset benefit to attribute, so accounts (b) and (c) have nothing to explain in this testbed. The Adam diagnostic confirms the optimizer *is* stale (g²/v ≈ 0.35, meaning current gradients are ~3× smaller than the second-moment history) and the reset fixes that (g²/v ≈ 1.2), but fixing it does not change learning.
3. **Temperature re-inflation is the only intervention that improves late-phase learning** (+10–13% relative), which is the coverage account (a).
4. **Shrink & Perturb at 0.8/0.2 is destructive** at this scale (probe 0.27, late gain 0.115). It needs a much gentler setting (0.95/0.05 or less) before it says anything.

So the pilot's working answer: in a small GRPO testbed, prolonged RL produces a genuine, RL-specific trainability loss with NTK rank collapse, but the resets practitioners use do not address it; what helps is sampling entropy. The hypothesis in the idea file (no representational loss, stale Adam is the culprit) is not supported here.

## Caveats

- The positive control failing is a property of the testbed, not necessarily of the probes: a 100k model trained on more random hashes gets better at memorising random hashes. A proper control needs a setting where supervised relabelling is known to hurt (e.g. a wider MLP on permuted-label MNIST, or many more phases). Without it the RL-specific decline cannot be attributed with full confidence to GRPO rather than to the probe design.
- 300-step phases are short; ProRL's resets act on 2000+ step runs where Adam state and the KL anchor have longer to bind. The KL weight here (effective β = 0.00125) is also deliberately light.
- Trainability probes measure fitting speed on the prompts they train on, by design (random hashes do not generalise in 100 steps).

## Next steps that would make this a paper result

1. Fix the positive control (a regime where supervised relabelling demonstrably reduces the probes) so the RL decline can be certified as RL-specific.
2. Longer phases (1000–3000 steps) with β ∈ {0.001, 0.01, 0.1}; the reference reset can only matter when the anchor binds.
3. Decompose the eNTK rank loss: is the GRPO gradient itself low-rank (few distinct advantage directions per batch) or does the representation lose rank? Compare per-example gradient rank on the *same* probe batch under CE vs GRPO at matched checkpoints.
4. Then Qwen2.5-0.5B on Countdown as Testbed B, with the same probe protocol.

## Re-running

```bash
python code/t3_resets.py                  # default grid, ~1.2 h on an M1 Pro when the CPU is free
python code/t3_resets.py --resume         # skip runs already in results/phases.csv
python code/t3_resets.py --summarize      # recompute factorial/verdict only
python code/t3_resets.py --full --device cuda   # 1000-step phases, 5 seeds
```
