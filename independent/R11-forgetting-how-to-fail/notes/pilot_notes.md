# R11 pilot notes

Run: batched DQN on CartPole, 40k steps, greedy eval every 1k (4 episodes), 3 conditions × 24 seeds (same seeds across conditions), CPU, 7 min. Reservoir: 256 transitions, 4 of every 32 minibatch samples. Probe set: 256 (s, a) pairs from a random policy where a terminates the episode; Q*(s, a) = 1 exactly. Log: `logs/pilot.log`. Raw: `results/runs.csv`, `results/per_run.csv`. Post hoc: `results/posthoc.txt`.

## Verdict

**INCONCLUSIVE** by the pre-registered rule: K2 (fix) and K3 (specificity) pass, K1 (mechanism) half-passes.

## Headline table

| condition | collapses / run | runs with ≥ 1 collapse | final (sd) | mean failure-Q (true = 1) | min buffer terminal % |
|---|---|---|---|---|---|
| control | 2.46 | 24/24 | 119 (40) | 21.2 | 0.60 |
| reservoir_term | **0.08** | **2/24** | **154 (23)** | **1.0** | 0.28 |
| reservoir_rand | 2.17 | 23/24 | 115 (31) | 9.3 | 0.57 |

Paired on seed, reservoir_term beats control on final score in 19/24 runs (mean +34.5; two-sided sign test p = 0.007).

## Time course (means over 24 seeds)

| step | control score / failQ / term% | reservoir_term score / failQ / term% | reservoir_rand score / failQ |
|---|---|---|---|
| 4k | 156 / 2.5 / 3.6 | 157 / 1.2 / 3.5 | 157 / 2.6 |
| 8k | 151 / 3.7 / 2.3 | 173 / 1.2 / 2.2 | 169 / 3.8 |
| 12k | 121 / 5.1 / 1.3 | 154 / 1.2 / 1.0 | 112 / 5.3 |
| 20k | 122 / 15.2 / 0.9 | 168 / 1.0 / 0.4 | 100 / 8.5 |
| 28k | 109 / 25.9 / 0.9 | 152 / 1.0 / 0.5 | 112 / 9.7 |
| 40k | 134 / 34.8 / 0.9 | 148 / 1.0 / 0.5 | 108 / 11.3 |

## What this says

- **The fix works and is specific (K2, K3).** Keeping 256 terminal transitions in every minibatch stream removes collapses almost entirely (2.46 → 0.08 per run) and raises the final score by 35 points with half the variance. The same amount of *random* old data does nothing (2.17 collapses per run, final 115). So it is not "old data" or "a slower effective buffer" that helps; it is the terminals.
- **The network really does forget how to fail.** In control runs the Q-value for actions that end the episode drifts from ~2 at 4k steps to ~35 at 40k, against a true value of exactly 1, while terminals fall below 1% of the buffer. With the reservoir, failure-Q stays at 1.0 for the whole run. The inflation is not a small miscalibration: by the end, the agent believes an action that kills it in one step is worth 35 steps of reward.
- **But inflation does not time the collapses (K1 half-fails).** The rank correlation between failure-Q and terminal fraction is −0.53 (passes). The AUROC of failure-Q at a good evaluation for a collapse within the next two evaluations is 0.61 (needed 0.7), and across control runs the level of inflation does not predict how many collapses a run has (Spearman −0.20). Inflation is chronic and monotone; collapses are episodic and happen throughout. So failure-Q is a *marker* of lost anchoring, not a *trigger* of each collapse.

## Interpretation and what to test next

Two readings fit the data. (a) The reservoir works because it keeps the terminal anchor in the loss, which pins the whole value scale (all Q-values are bootstrapped chains that end either at a terminal or nowhere); without it, overestimation grows everywhere and the greedy policy becomes brittle in a way that is not specific to failure states. (b) The reservoir works because the network keeps a correct ranking near the failure boundary. The pilot cannot separate these. The next run should measure overestimation on *non-failure* states against Monte-Carlo returns, and add a condition that keeps terminal transitions but *removes* their terminal flag (same states, no anchor) — if (a), that condition should fail; if (b), it should still work.

## Caveats

- CartPole, one architecture, one buffer size (10k). The effect size may be specific to a buffer that turns over every 50 good episodes.
- Collapse is defined on 4-episode greedy evaluations; the threshold pair (175, 100) was chosen before the run but is still a choice.
- Time-limit truncation is handled correctly (bootstrapped), so the effect is not the Pardo et al. truncation-as-terminal bug.
- The reservoir changes the minibatch distribution (12.5% terminals vs < 1% in control); some of the effect could be a reweighting effect rather than a memory effect. A condition that prioritises terminals from the FIFO buffer (no protected memory) would separate these.

## 3080 plan

1. Mechanism split: terminal-with-flag vs terminal-without-flag reservoir; overestimation on non-failure states vs MC truth; fork a good run at t, add the reservoir to one child and not the other (R7 machinery) for a per-run causal claim.
2. Scale: MinAtar (Breakout, Asterix have terminals; Freeway does not, a natural control) and MuJoCo Hopper/Walker (fall = terminal) with SAC. Buffer sizes 10⁴–10⁶: the prediction is that the effect shrinks as the buffer grows, because terminals survive longer.
3. Compare to the plasticity toolbox: resets, LayerNorm, Pop-Art. The reservoir is cheaper than all of them; the question is whether it is a special case of "keep the anchor" that those methods achieve indirectly.
