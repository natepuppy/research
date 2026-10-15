"""S4-46. Anytime-valid comparisons of RL algorithms.

Two ways to spend a seed budget:

  * fixed: always run N seeds, then a Welch t-test
  * optional stopping: add seeds until the t-test is significant (invalid;
    this is what inflates false positives)
  * anytime: add seeds until a betting confidence sequence for the paired
    difference excludes zero, which controls the error at every stopping time

The drop-in tool is lib/anytime.py (BettingCS). This script does the
experiment from the idea: build seed pools, then simulate hundreds of papers
that draw from those pools under each protocol.

Pools:
  * synthetic Beta returns with a known null and a known small gap, so the
    false-positive rate is meaningful
  * a real tabular gridworld (lib/gridworld.py), Q-learning against a copy
    with learning rate 0, so the same protocols are scored on actual RL seed noise

No gym, gymnax, or Craftax install. The statistical question does not depend
on which environment produced the bounded returns.

Kill test: no compute saving at a matched false-positive rate.

    python code/s4_anytime.py
    python code/s4_anytime.py --smoke
    python code/s4_anytime.py --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.anytime import BettingCS, null_mean_on_unit_interval, rescale_to_unit  # noqa: E402
from lib.gridworld import return_bounds, run_seed  # noqa: E402
from lib.io_utils import kill_test, save_csv  # noqa: E402


def _one_sample_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 1.0
    if float(np.std(values)) < 1e-8:
        return 0.0 if abs(float(np.mean(values))) > 1e-8 else 1.0
    pvalue = float(stats.ttest_1samp(values, 0.0).pvalue)
    if pvalue != pvalue:
        return 1.0
    return pvalue


def welch_p(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return 1.0
    result = stats.ttest_ind(left, right, equal_var=False)
    pvalue = float(result.pvalue)
    if pvalue != pvalue:
        return 1.0
    return pvalue


def protocol_once(diff: np.ndarray, low: float, high: float, alpha: float, fixed_n: int) -> dict:
    """One simulated paper. `diff` is a pool of paired differences, shuffled by the caller."""
    unit = rescale_to_unit(diff, low, high)
    null = null_mean_on_unit_interval(low, high)
    sequence = BettingCS(alpha=alpha)
    anytime_n = len(unit)
    anytime_reject = False
    for index, value in enumerate(unit, start=1):
        sequence.update(float(value))
        if sequence.excludes(null):
            anytime_n = index
            anytime_reject = True
            break
    optional_n = fixed_n
    optional_reject = False
    # One-sample t-test of mean(diff) == 0, matching the paired confidence sequence.
    for n in range(2, fixed_n + 1):
        pvalue = _one_sample_p(diff[:n])
        if pvalue < alpha:
            optional_n = n
            optional_reject = True
            break
    fixed_p = _one_sample_p(diff[:fixed_n])
    return {
        "anytime_n": anytime_n,
        "anytime_reject": int(anytime_reject),
        "optional_n": optional_n,
        "optional_reject": int(optional_reject),
        "fixed_n": fixed_n,
        "fixed_reject": int(fixed_p == fixed_p and fixed_p < alpha),
    }


def simulate(pool: np.ndarray, low: float, high: float, alpha: float, fixed_n: int, papers: int, rng: np.random.Generator, setting: str) -> list[dict]:
    rows = []
    for paper in range(papers):
        take = min(fixed_n, len(pool))
        choice = rng.choice(pool, size=take, replace=True)
        outcome = protocol_once(choice, low, high, alpha, take)
        outcome["paper"] = paper
        outcome["setting"] = setting
        rows.append(outcome)
    return rows


def paired_pool(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rng.shuffle(left)
    rng.shuffle(right)
    n = min(len(left), len(right))
    return left[:n] - right[:n]


def rates(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    fixed_n = rows[0]["fixed_n"] if rows else 0
    return {
        "fixed_fpr_or_power": mean("fixed_reject"),
        "optional_fpr_or_power": mean("optional_reject"),
        "anytime_fpr_or_power": mean("anytime_reject"),
        "optional_mean_n": mean("optional_n"),
        "anytime_mean_n": mean("anytime_n"),
        "fixed_n": fixed_n,
        "anytime_saving": 1.0 - mean("anytime_n") / fixed_n if fixed_n else float("nan"),
    }


def beta_pool(a: float, b: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.beta(a, b, size=n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", default="cpu", help="unused; accepted so run_all.py can forward it")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    if args.smoke:
        pool_n, papers, fixed_n, grid_seeds = 80, 40, 40, 20
    elif args.full:
        pool_n, papers, fixed_n, grid_seeds = 400, 300, 80, 30
    else:
        pool_n, papers, fixed_n, grid_seeds = 200, 150, 60, 24

    # Null: both arms Beta(12, 12), mean 0.5. Alternative: means 0.4 and 0.6.
    # A 0.07 gap is smaller than a Hoeffding-style boundary at these sample
    # sizes, so the alternative is the gap a variance-adaptive betting
    # sequence can actually close inside the fixed budget. Paired differences
    # of [0, 1] variables lie in [-1, 1].
    null_left = beta_pool(12, 12, pool_n, rng)
    null_right = beta_pool(12, 12, pool_n, rng)
    alt_left = beta_pool(10, 15, pool_n, rng)
    alt_right = beta_pool(15, 10, pool_n, rng)
    synthetic_rows = []
    synthetic_rows += simulate(paired_pool(null_left.copy(), null_right.copy(), rng), -1.0, 1.0, args.alpha, fixed_n, papers, rng, "beta_null")
    synthetic_rows += simulate(paired_pool(alt_left.copy(), alt_right.copy(), rng), -1.0, 1.0, args.alpha, fixed_n, papers, rng, "beta_gap")

    low, high = return_bounds(horizon=30, step_cost=0.01, goal_reward=1.0)
    # Paired difference of two returns in [low, high] lies in [low - high, high - low].
    diff_low, diff_high = low - high, high - low
    episodes = 20 if args.smoke else 40
    # Learning rate 0 never updates Q, so greedy evaluation stays a random walk.
    # The competent agent is ordinary Q-learning on a 4x4 grid. Same bounds:
    # a horizon of 30 with step cost 0.01 cannot leave [-0.3, 1].
    strong = np.array(
        [run_seed(eps=0.1, seed=1000 + i, episodes=episodes, size=4, lr=0.5) for i in range(grid_seeds)]
    )
    weak = np.array(
        [run_seed(eps=0.1, seed=2000 + i, episodes=episodes, size=4, lr=0.0) for i in range(grid_seeds)]
    )
    print(f"gridworld Q-learning mean {strong.mean():.3f} sd {strong.std():.3f}")
    print(f"gridworld lr=0       mean {weak.mean():.3f} sd {weak.std():.3f}")
    grid_null = simulate(paired_pool(strong.copy(), strong.copy(), rng), diff_low, diff_high, args.alpha, min(fixed_n, grid_seeds), papers, rng, "grid_null")
    grid_gap = simulate(paired_pool(strong.copy(), weak.copy(), rng), diff_low, diff_high, args.alpha, min(fixed_n, grid_seeds), papers, rng, "grid_gap")

    rows = synthetic_rows + grid_null + grid_gap
    summary = []
    for setting in ("beta_null", "beta_gap", "grid_null", "grid_gap"):
        subset = [row for row in rows if row["setting"] == setting]
        stats_row = rates(subset)
        stats_row["setting"] = setting
        summary.append(stats_row)
        print(
            f"{setting}: fixed {stats_row['fixed_fpr_or_power']:.2f}  "
            f"optional-stopping {stats_row['optional_fpr_or_power']:.2f}  "
            f"anytime {stats_row['anytime_fpr_or_power']:.2f}  "
            f"anytime mean n {stats_row['anytime_mean_n']:.1f} / {stats_row['fixed_n']}  "
            f"saving {stats_row['anytime_saving']:.2f}"
        )

    null = next(row for row in summary if row["setting"] == "beta_null")
    alt = next(row for row in summary if row["setting"] == "beta_gap")
    # Matched error: anytime must not exceed alpha (with Monte Carlo slack),
    # and it has to reject the alternative while using fewer seeds than the
    # fixed budget.
    fpr_ok = null["anytime_fpr_or_power"] <= args.alpha + 0.03
    saves = alt["anytime_saving"] >= 0.15 and alt["anytime_fpr_or_power"] >= 0.5
    supported = bool(fpr_ok and saves)
    if not fpr_ok and not saves:
        supported_out: bool | None = False
    elif fpr_ok and saves:
        supported_out = True
    else:
        supported_out = False
    text = (
        f"under the beta null, anytime false-positive rate {null['anytime_fpr_or_power']:.3f} "
        f"(optional stopping {null['optional_fpr_or_power']:.3f}, nominal alpha {args.alpha}). "
        f"Under the beta gap, anytime power {alt['anytime_fpr_or_power']:.2f} at mean "
        f"{alt['anytime_mean_n']:.1f} seeds versus fixed {alt['fixed_n']} "
        f"(saving {alt['anytime_saving']:.0%})."
    )
    verdict = kill_test(supported_out, text)
    save_csv(args.out / "papers.csv", rows)
    save_csv(args.out / "summary.csv", summary)
    save_csv(args.out / "gridworld_seeds.csv", [{"agent": "eps_0.05", "return": float(v)} for v in strong] + [{"agent": "eps_0.60", "return": float(v)} for v in weak])
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
