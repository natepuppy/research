"""R12: The speed limit of bootstrapping under replay and target networks.

Tabular TD prediction on a deterministic chain of length L with a single terminal reward. Transitions
are sampled uniformly from a (stationary) replay buffer in minibatches of B; targets are n-step
bootstrapped from a target table refreshed every K updates. We measure the number of updates until
the value of the start state reaches half its true value (propagation time), across a grid of
(L, n, K, B, optimiser), and test the two-regime law

    t_prop  ~  ceil(L / n) * max(K, c * L / B)

(target-limited when K dominates, replay-limited when the time to sample the frontier dominates).
All configurations are stepped in lockstep as one vectorised program. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, save_csv, load_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GAMMA = 0.99
LS = [8, 16, 32, 64]
NS = [1, 2, 4, 8]
KS = [1, 10, 100, 1000]
BS = [4, 32]
OPTS = ["sgd", "adam"]
ALPHA_SGD = 0.3      # per-hit step size (sum reduction: every sampled copy applies one step)
LR_ADAM = 0.05       # per-state Adam, mean reduction
LMAX = max(LS)


def run(args):
    rng = np.random.default_rng(args.seed)
    cfgs = [(L, n, K, B, o) for L in LS for n in NS for K in KS for B in BS for o in OPTS]
    C = len(cfgs)
    LMAX = max(LS)
    L = np.array([c[0] for c in cfgs]); n = np.array([c[1] for c in cfgs]); K = np.array([c[2] for c in cfgs])
    B = np.array([c[3] for c in cfgs]); is_adam = np.array([c[4] == "adam" for c in cfgs])
    Bmax = max(BS)
    V = np.zeros((C, LMAX + 1)); Vt = np.zeros((C, LMAX + 1))           # index LMAX is a dummy
    m = np.zeros_like(V); v = np.zeros_like(V); tcount = np.zeros_like(V)
    true0 = GAMMA ** (L - 1)
    t_prop = np.full(C, -1)
    rows_c = np.arange(C)[:, None]
    bmask = (np.arange(Bmax)[None] < B[:, None])                           # (C, Bmax) valid batch slots
    t0 = time.time()
    for t in range(1, args.max_updates + 1):
        active = t_prop < 0
        if not active.any():
            break
        # sample states uniformly over the L transitions of each chain (stationary buffer = uniform over the chain)
        s = (rng.random((C, Bmax)) * L[:, None]).astype(np.int64)
        # n-step target along the deterministic chain: reward 1 only on the last transition (L-1 -> terminal)
        s_n = s + n[:, None]
        reach = s_n >= L[:, None]
        y = np.where(reach, GAMMA ** (L[:, None] - 1 - s), GAMMA ** n[:, None] * Vt[rows_c, np.minimum(s_n, LMAX)])
        delta = y - V[rows_c, s]
        delta = np.where(bmask, delta, 0.0)
        # scatter: per-state sum of deltas and counts
        sum_d = np.zeros((C, LMAX + 1)); cnt = np.zeros((C, LMAX + 1))
        np.add.at(sum_d, (rows_c, s), delta); np.add.at(cnt, (rows_c, s), bmask.astype(float))
        # sgd (sum reduction, step per hit) -- note: multiple hits of one state in one batch act as one summed step
        upd_sgd = ALPHA_SGD * sum_d
        # adam (mean reduction over the batch, per-state moments, bias-corrected)
        gmean = sum_d / B[:, None]
        hit = cnt > 0
        m = np.where(hit, 0.9 * m + 0.1 * gmean, m); v = np.where(hit, 0.999 * v + 0.001 * gmean ** 2, v)
        tcount = tcount + hit
        mhat = m / (1 - 0.9 ** np.maximum(tcount, 1)); vhat = v / (1 - 0.999 ** np.maximum(tcount, 1))
        upd_adam = np.where(hit, LR_ADAM * mhat / (np.sqrt(vhat) + 1e-8), 0.0)
        upd = np.where(is_adam[:, None], upd_adam, upd_sgd)
        upd[~active] = 0.0
        V += upd
        V[:, LMAX] = 0.0
        refresh = (t % K == 0) & active
        Vt[refresh] = V[refresh]
        done = active & (V[:, 0] >= 0.5 * true0)
        t_prop[done] = t
        if t % 20000 == 0:
            print(f"t={t} [{time.time() - t0:.0f}s] remaining {int(active.sum())}/{C}", flush=True)
    rows = []
    for i, (Lc, nc, Kc, Bc, oc) in enumerate(cfgs):
        rows.append(dict(seed=args.seed, L=Lc, n=nc, K=Kc, B=Bc, opt=oc, t_prop=int(t_prop[i]) if t_prop[i] > 0 else args.max_updates,
                         converged=bool(t_prop[i] > 0), hops=int(np.ceil(Lc / nc)), replay_time=Lc / Bc))
    return rows


def fit_loglog(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(slope), float(icpt), float(r2)


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    keys = sorted({(r["L"], r["n"], r["K"], r["B"], r["opt"]) for r in rows})
    summary = []
    for k in keys:
        sel = [r for r in rows if (r["L"], r["n"], r["K"], r["B"], r["opt"]) == k]
        t = np.array([r["t_prop"] for r in sel], float)
        summary.append(dict(L=k[0], n=k[1], K=k[2], B=k[3], opt=k[4], t_prop=float(np.exp(np.mean(np.log(t)))),
                            all_converged=all(r["converged"] for r in sel), hops=sel[0]["hops"], replay_time=sel[0]["replay_time"]))
    text_parts = []
    verdicts = {}
    for opt in OPTS:
        S = [s for s in summary if s["opt"] == opt and s["all_converged"]]
        # fit the constant c in max(K, c L/B) by grid search on R^2 of the single-predictor law
        best = None
        for c in np.exp(np.linspace(np.log(0.05), np.log(50), 60)):
            pred = [s["hops"] * max(s["K"], c * s["replay_time"]) for s in S]
            slope, icpt, r2 = fit_loglog(pred, [s["t_prop"] for s in S])
            if best is None or r2 > best[0]:
                best = (r2, c, slope)
        r2_law, c_star, slope_law = best
        # baseline: horizon only (hops) and K only
        _, _, r2_hops = fit_loglog([s["hops"] for s in S], [s["t_prop"] for s in S])
        _, _, r2_K = fit_loglog([s["hops"] * s["K"] for s in S], [s["t_prop"] for s in S])
        # K2: regime slopes. Target-limited cells: K >= 10 * c L/B ; replay-limited: K <= 0.1 * c L/B
        tl = [s for s in S if s["K"] >= 10 * c_star * s["replay_time"]]
        rl = [s for s in S if s["K"] <= 0.1 * c_star * s["replay_time"]]

        def slope_on_K(cells):
            # within groups of fixed (L, n, B), slope of log t on log K; average over groups with >= 2 K values
            groups = {}
            for s in cells:
                groups.setdefault((s["L"], s["n"], s["B"]), []).append(s)
            sl = [fit_loglog([s["K"] for s in g], [s["t_prop"] for s in g])[0] for g in groups.values() if len({s["K"] for s in g}) >= 2]
            return float(np.mean(sl)) if sl else float("nan"), len(sl)

        s_tl, n_tl = slope_on_K(tl); s_rl, n_rl = slope_on_K(rl)
        k2 = (s_tl >= 0.8) and (s_rl <= 0.2)
        # K3: slope on n in target-limited regime, within (L, K, B) groups
        groups = {}
        for s in tl:
            groups.setdefault((s["L"], s["K"], s["B"]), []).append(s)
        sn = [fit_loglog([s["n"] for s in g], [s["t_prop"] for s in g])[0] for g in groups.values() if len({s["n"] for s in g}) >= 3]
        slope_n = float(np.mean(sn)) if sn else float("nan")
        k3 = -1.3 <= slope_n <= -0.7
        k1 = r2_law >= 0.9
        verdicts[opt] = (k1, k2, k3)
        text_parts.append(f"[{opt}] K1 (R2 of log t_prop on log hops*max(K, c L/B) = {r2_law:.3f} >= 0.9; c* = {c_star:.2f}, slope {slope_law:.2f}; "
                          f"baselines hops-only R2 = {r2_hops:.2f}, hops*K R2 = {r2_K:.2f}) = {k1}. "
                          f"K2 (slope on K: target-limited {s_tl:.2f} >= 0.8 over {n_tl} groups; replay-limited {s_rl:.2f} <= 0.2 over {n_rl} groups) = {k2}. "
                          f"K3 (slope on n in target-limited regime = {slope_n:.2f} in [-1.3, -0.7]) = {k3}.")
        for s in summary:
            if s["opt"] == opt:
                s["pred"] = s["hops"] * max(s["K"], c_star * s["replay_time"])
    save_csv(RESULTS / "summary.csv", summary)
    k1_all = all(v[0] for v in verdicts.values()); k2_all = all(v[1] for v in verdicts.values())
    verdict = True if (k1_all and k2_all) else (None if (k1_all or k2_all) else False)
    line = kill_test(verdict, " ".join(text_parts))
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nopt   L   n    K    B   hops  L/B    t_prop     pred")
    for s in summary:
        if s["B"] == 32 and s["n"] in (1, 8):
            print(f"{s['opt']:4s} {s['L']:3d} {s['n']:3d} {s['K']:5d} {s['B']:3d}  {s['hops']:4d} {s['replay_time']:5.1f}  {s['t_prop']:9.0f} {s['pred']:9.0f}")


# ----------------------------------------------------------------------------- confirmation run
# Pre-registered AFTER the pilot and BEFORE this run (see README, "Confirmation"): the pilot's data suggested an
# additive law  t_prop = (hops - 1) * K + c * hops * L / B  with c fitted on the pilot (c_sgd = 2.63, c_adam = 4.21).
# The confirmation grid shares no (L, n, K, B) values with the pilot grid; c is held fixed, so the test is out of sample.
CONFIRM = dict(LS=[12, 24, 48, 96], NS=[1, 3, 6], KS=[3, 30, 300, 3000], BS=[8, 16])
C_FIXED = dict(sgd=2.63, adam=4.21)


def r2_identity_log(pred, y):
    x, y = np.log(np.asarray(pred, float)), np.log(np.asarray(y, float))
    return float(1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def summarize_confirm(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "confirm" / "runs.csv")
    keys = sorted({(r["L"], r["n"], r["K"], r["B"], r["opt"]) for r in rows})
    summary = []
    for k in keys:
        sel = [r for r in rows if (r["L"], r["n"], r["K"], r["B"], r["opt"]) == k]
        t = float(np.exp(np.mean(np.log([r["t_prop"] for r in sel]))))
        hops = sel[0]["hops"]
        pred = (hops - 1) * k[2] + C_FIXED[k[4]] * hops * k[0] / k[3]
        summary.append(dict(L=k[0], n=k[1], K=k[2], B=k[3], opt=k[4], t_prop=t, pred=pred, hops=hops,
                            all_converged=all(r["converged"] for r in sel)))
    save_csv(RESULTS / "confirm" / "summary.csv", summary)
    parts, ok = [], {}
    for opt in OPTS:
        S = [s for s in summary if s["opt"] == opt and s["all_converged"]]
        r2 = r2_identity_log([s["pred"] for s in S], [s["t_prop"] for s in S])
        ratios = np.array([s["t_prop"] / s["pred"] for s in S])
        thr = 0.95 if opt == "sgd" else 0.90
        ok[opt] = r2 >= thr
        parts.append(f"[{opt}] C1 (out-of-sample identity-fit R2 in log space of (hops-1)K + c*hops*L/B with c = {C_FIXED[opt]} fixed from the pilot = {r2:.3f} >= {thr}) = {ok[opt]}; "
                     f"t/pred median {np.median(ratios):.2f}, 10-90% [{np.quantile(ratios, 0.1):.2f}, {np.quantile(ratios, 0.9):.2f}], n = {len(S)}, unconverged excluded = {len([s for s in summary if s['opt'] == opt]) - len(S)}.")
    line = kill_test(True if all(ok.values()) else (None if any(ok.values()) else False), "CONFIRMATION. " + " ".join(parts))
    (RESULTS / "confirm" / "verdict.txt").write_text(line + "\n")


def main():
    global LS, NS, KS, BS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-updates", type=int, default=400_000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--confirm", action="store_true", help="run the out-of-sample confirmation grid for the refined law")
    args = ap.parse_args()
    if args.summarize:
        (summarize_confirm if args.confirm else summarize)(); return
    if args.smoke:
        args.seeds, args.max_updates = 1, 20_000
    out = RESULTS / "confirm" if args.confirm else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    if args.confirm:
        LS, NS, KS, BS = CONFIRM["LS"], CONFIRM["NS"], CONFIRM["KS"], CONFIRM["BS"]
    rows = []
    for sd in range(args.seeds):
        args.seed = 100 + sd if args.confirm else sd
        rows += run(args)
        print(f"seed {args.seed} done: {sum(r['converged'] for r in rows)}/{len(rows)} converged", flush=True)
    save_csv(out / "runs.csv", rows)
    (summarize_confirm if args.confirm else summarize)(rows)


if __name__ == "__main__":
    main()
