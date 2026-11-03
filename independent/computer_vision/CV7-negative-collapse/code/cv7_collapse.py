"""CV7: negative exemplars attenuate a concept segmenter globally.

Synthetic episodes. A negative exemplar lowers the presence score and shifts
every proposal logit by about the same amount, whether or not the proposal
resembles the negative. The fix is a second pass: keep the positive-only
presence, and reject proposals whose embedding matches the negative."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

N_PROPS = 8
PRES_THR = 0.5
# How far one negative pulls the presence score. CASE exclusion is the steeper of the two.
BENCH = {"case": 0.26, "coco": 0.23}

CFG = dict(seeds=8, n_test=2000, n_val=400)
SMOKE = dict(seeds=2, n_test=800, n_val=300)


def make_batch(rng: np.random.Generator, n: int, n_neg: int, drop: float) -> dict:
    is_target = rng.random((n, N_PROPS)) < 0.40
    is_target[~is_target.any(1), 0] = True
    sim = np.where(
        is_target, rng.normal(0.28, 0.09, (n, N_PROPS)), rng.normal(0.74, 0.09, (n, N_PROPS)),
    )
    sim = np.clip(sim, 0.0, 1.0)
    # Targets sit near the decision boundary, so a uniform logit shift is visible,
    # and distractors account for the false positives a negative is supposed to remove.
    logit = np.where(
        is_target, rng.normal(0.55, 0.30, (n, N_PROPS)), rng.normal(-0.55, 0.35, (n, N_PROPS)),
    )
    presence = rng.uniform(0.62, 0.92, n)
    if n_neg == 0:
        logit_drop = np.zeros_like(logit)
        return dict(is_target=is_target, sim=sim, logit=logit, presence=presence,
                    logit_neg=logit, presence_neg=presence, logit_drop=logit_drop, n_neg=0)
    # The leak term is the only place similarity touches the drop, and it is small.
    logit_drop = 0.50 + 0.03 * n_neg + 0.02 * (sim - 0.5) + rng.normal(0.0, 0.06, (n, N_PROPS))
    presence_neg = presence - drop * n_neg + rng.normal(0.0, 0.025, n)
    return dict(is_target=is_target, sim=sim, logit=logit, presence=presence,
                logit_neg=logit - logit_drop, presence_neg=presence_neg,
                logit_drop=logit_drop, n_neg=n_neg)


def mean_miou(presence, logit, is_target, sim, reject: bool, thr: float, oracle: bool) -> float:
    alive = presence >= PRES_THR
    emitted = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))) >= 0.5
    if oracle:
        emitted = emitted & is_target
    elif reject:
        emitted = emitted & (sim < thr)
    emitted = emitted & alive[:, None]
    tp = (emitted & is_target).sum(1).astype(float)
    fp = (emitted & ~is_target).sum(1).astype(float)
    fn = (~emitted & is_target).sum(1).astype(float)
    den = tp + fp + fn
    iou = np.divide(tp, den, out=np.zeros(len(den)), where=den > 0)
    return float(iou.mean() * 100.0)


def modes(batch: dict, thr: float) -> dict[str, float]:
    tgt, sim = batch["is_target"], batch["sim"]
    # Nothing to reject when the episode has no negative exemplar.
    do_reject = batch["n_neg"] > 0
    pos = mean_miou(batch["presence"], batch["logit"], tgt, sim, False, thr, False)
    presence_only = mean_miou(batch["presence_neg"], batch["logit"], tgt, sim, False, thr, False)
    both = mean_miou(batch["presence_neg"], batch["logit_neg"], tgt, sim, False, thr, False)
    fix = mean_miou(batch["presence"], batch["logit"], tgt, sim, do_reject, thr, False)
    # Oracle ceiling standing in for a trained adapter: label rejection, same logits.
    cpa = mean_miou(batch["presence"], batch["logit"], tgt, sim, False, thr, True)
    return dict(pos=pos, presence_only=presence_only, both=both, fix=fix, cpa=cpa)


def fit_threshold(rng: np.random.Generator, n: int, drop: float) -> float:
    val = make_batch(rng, n, 1, drop)
    best_thr, best = 0.5, -1.0
    for thr in np.linspace(0.20, 0.80, 25):
        score = modes(val, float(thr))["fix"]
        if score > best:
            best_thr, best = float(thr), score
    return best_thr


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def presence_share(m: dict) -> float:
    """Share of the collapse that the presence drop produces on its own."""
    collapsed = m["pos"] - m["both"]
    if collapsed < 1.0:
        return float("nan")
    return (m["pos"] - m["presence_only"]) / collapsed


def bench_metrics(rng: np.random.Generator, n: int, drop: float, thr: float) -> dict:
    batch = make_batch(rng, n, 1, drop)
    m = modes(batch, thr)
    gap = m["pos"] - m["both"]
    recovery = (m["fix"] - m["both"]) / gap if gap >= 1.0 else float("nan")
    bare = make_batch(rng, n, 0, drop)
    bare_m = modes(bare, thr)
    curve = {}
    for k in (0, 1, 3, 5):
        curve[k] = modes(make_batch(rng, n, k, drop), thr)["both"]
    return dict(
        pos=m["pos"], neg=m["both"], fix=m["fix"], cpa=m["cpa"], recovery=recovery,
        rho=pearson(batch["logit_drop"], batch["sim"]), presence_share=presence_share(m),
        zero_gap=abs(bare_m["fix"] - bare_m["pos"]),
        n0=curve[0], n1=curve[1], n3=curve[3], n5=curve[5],
    )


def run_cell(seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(2000 + seed)
    out = dict(seed=seed)
    for name, drop in BENCH.items():
        thr = fit_threshold(rng, cfg["n_val"], drop)
        m = bench_metrics(rng, cfg["n_test"], drop, thr)
        out[f"{name}_thr"] = thr
        for key, value in m.items():
            out[f"{name}_{key}"] = value
    return out


def summarize(rows: list[dict]) -> tuple[list[dict], str]:
    med = lambda k: float(np.nanmedian([r[k] for r in rows]))  # noqa: E731
    n = len(rows)
    need = 6 if n >= 8 else n
    k1_seeds = sum(
        abs(r["case_rho"]) < 0.20 and abs(r["coco_rho"]) < 0.20
        and r["case_presence_share"] >= 0.50 and r["coco_presence_share"] >= 0.50
        for r in rows
    )
    k2_seeds = sum(r["case_recovery"] >= 0.70 and r["coco_recovery"] >= 0.60 for r in rows)
    k3_seeds = sum(
        (r["case_cpa"] - r["case_fix"]) <= 3.0
        and r["case_zero_gap"] <= 1.0 and r["coco_zero_gap"] <= 1.0
        for r in rows
    )
    k1, k2, k3 = k1_seeds >= need, k2_seeds >= need, k3_seeds >= need
    if k1 and k2:
        verdict = True
    elif k1 or k2:
        verdict = None
    else:
        verdict = False
    summ = [dict(
        n_seeds=n,
        case_rho=med("case_rho"), coco_rho=med("coco_rho"),
        case_presence_share=med("case_presence_share"), coco_presence_share=med("coco_presence_share"),
        case_pos=med("case_pos"), case_neg=med("case_neg"), case_fix=med("case_fix"),
        case_cpa=med("case_cpa"), case_recovery=med("case_recovery"),
        coco_pos=med("coco_pos"), coco_neg=med("coco_neg"), coco_fix=med("coco_fix"),
        coco_cpa=med("coco_cpa"), coco_recovery=med("coco_recovery"),
        case_n0=med("case_n0"), case_n1=med("case_n1"), case_n3=med("case_n3"), case_n5=med("case_n5"),
        coco_n0=med("coco_n0"), coco_n1=med("coco_n1"), coco_n3=med("coco_n3"), coco_n5=med("coco_n5"),
        k1_seeds=k1_seeds, k2_seeds=k2_seeds, k3_seeds=k3_seeds,
    )]
    text = (
        f"K1 (|rho| of logit drop vs similarity < 0.20, and presence accounts for >= 50% of the "
        f"collapse, on both benchmarks: {k1_seeds}/{n}; "
        f"case rho {med('case_rho'):+.3f}, share {med('case_presence_share'):.2f}; "
        f"coco rho {med('coco_rho'):+.3f}, share {med('coco_presence_share'):.2f}) = {k1}. "
        f"K2 (fix recovers >= 70% of the CASE gap and >= 60% of the COCO gap: {k2_seeds}/{n}; "
        f"case {med('case_recovery'):.2f}, coco {med('coco_recovery'):.2f}) = {k2}. "
        f"K3 (CPA ceiling within 3 mIoU of the fix on CASE, and zero negatives leave the "
        f"positive-only score within 1: {k3_seeds}/{n}; "
        f"case fix {med('case_fix'):.1f} vs cpa {med('case_cpa'):.1f}) = {k3}."
    )
    return summ, kill_test(verdict, text)


def print_table(summ: list[dict]) -> None:
    s = summ[0]
    print("\n bench   pos    neg    fix    cpa    rec    rho    presence   collapse n=0/1/3/5")
    print(f" case   {s['case_pos']:5.1f}  {s['case_neg']:5.1f}  {s['case_fix']:5.1f}  {s['case_cpa']:5.1f}  "
          f"{s['case_recovery']:.2f}  {s['case_rho']:+.3f}  {s['case_presence_share']:.2f}     "
          f"{s['case_n0']:.0f}/{s['case_n1']:.0f}/{s['case_n3']:.0f}/{s['case_n5']:.0f}")
    print(f" coco   {s['coco_pos']:5.1f}  {s['coco_neg']:5.1f}  {s['coco_fix']:5.1f}  {s['coco_cpa']:5.1f}  "
          f"{s['coco_recovery']:.2f}  {s['coco_rho']:+.3f}  {s['coco_presence_share']:.2f}     "
          f"{s['coco_n0']:.0f}/{s['coco_n1']:.0f}/{s['coco_n3']:.0f}/{s['coco_n5']:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.summarize:
        rows = load_csv(RESULTS / "runs.csv")
    else:
        cfg = SMOKE if args.smoke else CFG
        t0 = time.time()
        rows = [run_cell(seed, cfg) for seed in range(cfg["seeds"])]
        print(f"{len(rows)} seeds [{time.time() - t0:.1f}s]", flush=True)
        if not args.smoke:
            save_csv(RESULTS / "runs.csv", rows)
    summ, line = summarize(rows)
    print_table(summ)
    if not args.smoke:
        save_csv(RESULTS / "summary.csv", summ)
        (RESULTS / "verdict.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
