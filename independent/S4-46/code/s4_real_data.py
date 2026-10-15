"""Second and third real datasets, and fresh (non-bootstrap) seed streams, for the anytime-testing paper.

Reuses the tests and the `run_setting` driver of s4_extensions.py on:

  1. TD-MPC2 benchmark results (Hansen, Su, Wang 2024; github.com/nicklashansen/tdmpc2/results):
     DMControl, 39 tasks, reward in [0, 1000] -> score = reward / 1000 in [0, 1];
     Meta-World, 50 tasks, success rate in [0, 1] (the 15 MyoSuite tasks are excluded);
     algorithms dreamerv3, sac, tdmpc, tdmpc2; 3 seeds per task. One aggregate "seed" is a bundle
     of one run per task for each algorithm, scored by the mean over tasks (as for BBF).
  2. Fresh runs (data/fresh/fresh_runs.csv, produced by s4_fresh_runs.py): three DQN variants on
     CartPole, Acrobot, MinAtar breakout and space_invaders, 30-60 seeds each, scores in [0, 1] by
     construction. Used in two ways:
       * genuine paired streams: a random disjoint pairing of agent A seeds with agent B seeds gives
         one stream of i.i.d. paired differences with no resampling; different pairings give
         different streams (an exact permutation ensemble, not a bootstrap);
       * genuine null streams: random disjoint pairing of an agent's seeds with itself.
     Bootstrap pools of the same data are also run to cap 200 for comparability with BBF.

    python code/s4_real_data.py [--quick] [--skip-fresh]
Writes results/real_data_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from s4_extensions import OUT, run_setting  # noqa: E402

TDMPC2 = PROJECT / "data" / "tdmpc2"
FRESH = PROJECT / "data" / "fresh" / "fresh_runs.csv"


# ----------------------------------------------------------------------------
# TD-MPC2
# ----------------------------------------------------------------------------


def load_tdmpc2(algo: str, suite: str) -> pd.DataFrame:
    rows = []
    for path in sorted((TDMPC2 / algo).glob("*.csv")):
        frame = pd.read_csv(path)
        task = path.stem
        is_mw = task.startswith("mw-")
        if suite == "metaworld" and not is_mw:
            continue
        if suite == "dmcontrol" and ("reward" not in frame.columns or is_mw or task.startswith("myo-") or task in ("lift-cube", "pick-cube", "stack-cube", "turn-faucet")):
            continue
        col = "success" if "success" in frame.columns else "reward"
        last = frame.sort_values("step").groupby("seed").tail(1)
        for _, r in last.iterrows():
            score = float(r[col]) if col == "success" else float(r[col]) / 1000.0
            rows.append({"task": task, "seed": int(r["seed"]), "score": float(np.clip(score, 0.0, 1.0))})
    return pd.DataFrame(rows)


def aggregate_path(left: pd.DataFrame, right: pd.DataFrame, key: str = "task", n: int = 400, subset: int | None = None):
    """Aggregate seed stream. With `subset`, each path first draws a random subset of that many tasks
    (a small benchmark slice, as in many papers) and aggregates over it."""
    tasks = sorted(set(left[key]) & set(right[key]))
    pools_l = [left.loc[left[key] == t, "score"].to_numpy() for t in tasks]
    pools_r = [right.loc[right[key] == t, "score"].to_numpy() for t in tasks]

    def draw(rng: np.random.Generator) -> np.ndarray:
        idx = np.arange(len(tasks)) if subset is None else rng.choice(len(tasks), size=subset, replace=False)
        agg_l = np.zeros(n)
        agg_r = np.zeros(n)
        for i in idx:
            agg_l += pools_l[i][rng.integers(0, len(pools_l[i]), size=n)]
            agg_r += pools_r[i][rng.integers(0, len(pools_r[i]), size=n)]
        return (agg_l - agg_r) / len(idx)

    return draw


# ----------------------------------------------------------------------------
# Fresh runs
# ----------------------------------------------------------------------------


def load_fresh() -> pd.DataFrame:
    frame = pd.read_csv(FRESH)
    frame = frame[frame["score"].notna()]
    return frame


def fresh_pool(frame: pd.DataFrame, env: str, agent: str) -> np.ndarray:
    return frame.loc[(frame["env"] == env) & (frame["agent"] == agent), "score"].to_numpy(dtype=float)


def bootstrap_pair_path(a: np.ndarray, b: np.ndarray, n: int = 400):
    def draw(rng: np.random.Generator) -> np.ndarray:
        return a[rng.integers(0, len(a), size=n)] - b[rng.integers(0, len(b), size=n)]

    return draw


def genuine_pair_path(a: np.ndarray, b: np.ndarray):
    """Disjoint random pairing of A seeds with B seeds: every stream is an i.i.d. sample of paired
    differences with no run reused. Length = min(|A|, |B|)."""
    m = min(len(a), len(b))

    def draw(rng: np.random.Generator) -> np.ndarray:
        return a[rng.permutation(len(a))[:m]] - b[rng.permutation(len(b))[:m]]

    return draw


def genuine_null_path(a: np.ndarray):
    """Disjoint random pairing of an agent's seeds with itself: an exact null on real seed noise."""
    m = len(a) // 2

    def draw(rng: np.random.Generator) -> np.ndarray:
        perm = rng.permutation(len(a))
        return a[perm[:m]] - a[perm[m : 2 * m]]

    return draw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-fresh", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--prior-sd", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    alpha, cap = args.alpha, 200
    paths = 120 if args.quick else 400
    null_paths = 300 if args.quick else 1000
    rng = np.random.default_rng(1)
    settings: list[dict] = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        # --- TD-MPC2 ---
        data = {suite: {algo: load_tdmpc2(algo, suite) for algo in ("dreamerv3", "sac", "tdmpc", "tdmpc2")} for suite in ("dmcontrol", "metaworld")}
        for suite, frames in data.items():
            ntask = frames["tdmpc2"]["task"].nunique()
            print(f"{suite}: {ntask} tasks, " + ", ".join(f"{a}: mean {f['score'].mean():.3f}" for a, f in frames.items()))
            for algo in ("tdmpc2", "sac"):
                settings.append(run_setting(f"null_{suite}_{algo}_vs_{algo}_aggregate", aggregate_path(frames[algo], frames[algo]), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
            for left, right in (("tdmpc2", "dreamerv3"), ("tdmpc2", "sac"), ("tdmpc2", "tdmpc"), ("dreamerv3", "sac"), ("tdmpc", "sac")):
                settings.append(run_setting(f"{suite}_{left}_vs_{right}_aggregate", aggregate_path(frames[left], frames[right]), -1.0, 1.0, alpha, cap, paths, args.prior_sd, pool, rng))
            # Declared-bound sweep on the smallest-gap comparison of the suite.
            small = ("tdmpc2", "tdmpc")
            for half in (0.5, 0.25, 0.1):
                settings.append(run_setting(f"{suite}_{small[0]}_vs_{small[1]}_aggregate_bound{half:g}", aggregate_path(frames[small[0]], frames[small[1]]), -half, half, alpha, cap, paths, args.prior_sd, pool, rng))
            # Five-task slices: the common "a handful of tasks" benchmark, with much larger seed noise.
            settings.append(run_setting(f"null_{suite}_tdmpc2_vs_tdmpc2_subset5", aggregate_path(frames["tdmpc2"], frames["tdmpc2"], subset=5), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
            for left, right in (("tdmpc2", "tdmpc"), ("tdmpc2", "dreamerv3"), ("tdmpc", "sac")):
                settings.append(run_setting(f"{suite}_{left}_vs_{right}_subset5", aggregate_path(frames[left], frames[right], subset=5), -1.0, 1.0, alpha, cap, paths, args.prior_sd, pool, rng))

        # --- Fresh runs ---
        if not args.skip_fresh and FRESH.exists():
            fresh = load_fresh()
            summary_rows = fresh.groupby(["env", "agent"])["score"].agg(["count", "mean", "std"]).reset_index()
            print(summary_rows.to_string(index=False))
            for env in sorted(fresh["env"].unique()):
                pools = {agent: fresh_pool(fresh, env, agent) for agent in ("dqn", "ddqn", "dqn_lr3e-4")}
                if any(len(p) < 10 for p in pools.values()):
                    continue
                # Exact nulls on genuine disjoint pairings.
                for agent in ("dqn", "ddqn"):
                    m = len(pools[agent]) // 2
                    settings.append(run_setting(f"null_fresh_{env}_{agent}_genuine", genuine_null_path(pools[agent]), -1.0, 1.0, alpha, m, null_paths, args.prior_sd, pool, rng, is_null=True))
                settings.append(run_setting(f"null_fresh_{env}_dqn_bootstrap", bootstrap_pair_path(pools["dqn"], pools["dqn"]), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
                # Power: genuine disjoint pairings (length min(|A|,|B|)) and bootstrap pools (cap 200).
                for left, right in (("dqn", "ddqn"), ("dqn", "dqn_lr3e-4"), ("ddqn", "dqn_lr3e-4")):
                    m = min(len(pools[left]), len(pools[right]))
                    settings.append(run_setting(f"fresh_{env}_{left}_vs_{right}_genuine", genuine_pair_path(pools[left], pools[right]), -1.0, 1.0, alpha, m, paths, args.prior_sd, pool, rng))
                    settings.append(run_setting(f"fresh_{env}_{left}_vs_{right}_bootstrap", bootstrap_pair_path(pools[left], pools[right]), -1.0, 1.0, alpha, cap, paths, args.prior_sd, pool, rng))
    out = OUT / "real_data_summary.json"
    out.write_text(json.dumps({"alpha": alpha, "cap": cap, "prior_sd": args.prior_sd, "settings": settings}, indent=1))
    print(f"wrote {out}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
