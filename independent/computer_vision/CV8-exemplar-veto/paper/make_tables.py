"""Tables and every number quoted in the paper, computed from results/*/episodes.csv.

Writes paper/tab/*.tex and paper/numbers.json. Intervals are 95% percentile intervals over
2,000 image-level bootstrap resamples; they are descriptive and are not the pre-registered test.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
TAB = HERE / "tab"
N_BOOT = 2000
HARD = 0.10
TAUS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name / "episodes.csv")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def fast_auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.sort(np.asarray(neg, float))
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    below = np.searchsorted(neg, pos, side="left")
    upto = np.searchsorted(neg, pos, side="right")
    return float(((below + upto) / 2).mean() / len(neg))


class Boot:
    """Image-level bootstrap for one experiment."""

    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self.imgs = pd.Index(sorted(df["image"].unique()))
        self.idx = np.random.default_rng(seed).integers(0, len(self.imgs), size=(N_BOOT, len(self.imgs)))

    def _per_image(self, sub: pd.DataFrame, values) -> tuple[np.ndarray, np.ndarray]:
        v = pd.Series(np.asarray(values, float), index=sub["image"].to_numpy())
        total = v.groupby(level=0).sum().reindex(self.imgs, fill_value=0.0).to_numpy()
        count = v.groupby(level=0).size().reindex(self.imgs, fill_value=0).to_numpy().astype(float)
        return total, count

    def _samples(self, sub, values) -> tuple[float, np.ndarray]:
        total, count = self._per_image(sub, values)
        return total.sum() / count.sum(), total[self.idx].sum(1) / count[self.idx].sum(1)

    @staticmethod
    def _ci(point, samples) -> tuple[float, float, float]:
        lo, hi = np.nanpercentile(samples, [2.5, 97.5])
        return float(point), float(lo), float(hi)

    def mean(self, sub, values):
        return self._ci(*self._samples(sub, values))

    def diff(self, sub_a, a, sub_b, b):
        pa, sa = self._samples(sub_a, a)
        pb, sb = self._samples(sub_b, b)
        return self._ci(pa - pb, sa - sb)

    def rel_drop(self, sub, before, after):
        pb, sb = self._samples(sub, before)
        pa, sa = self._samples(sub, after)
        return self._ci(1 - pa / pb, 1 - sa / sb)

    def auroc(self, df, score: str, pos_mask, neg_mask, n: int = 1000):
        s = df[score].to_numpy(float)
        pos_mask, neg_mask = np.asarray(pos_mask, bool), np.asarray(neg_mask, bool)
        point = fast_auroc(s[pos_mask], s[neg_mask])
        code = pd.Categorical(df["image"], categories=self.imgs).codes
        order = np.argsort(code, kind="stable")
        starts = np.searchsorted(code[order], np.arange(len(self.imgs) + 1))
        rows = [order[starts[i]:starts[i + 1]] for i in range(len(self.imgs))]
        out = []
        for take in self.idx[:n]:
            r = np.concatenate([rows[i] for i in take])
            out.append(fast_auroc(s[r][pos_mask[r]], s[r][neg_mask[r]]))
        return self._ci(point, np.asarray(out))


def fmt(x: float, d: int) -> str:
    """Round half up, so 4.25 prints as 4.3 rather than the float-dependent 4.2."""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP))


def pct(t, d: int = 1) -> str:
    return fmt(100 * t[0], d)


def pct_ci(t, d: int = 1) -> str:
    return f"{fmt(100 * t[0], d)} {{\\scriptsize[{fmt(100 * t[1], d)}, {fmt(100 * t[2], d)}]}}"


def num_ci(t, d: int = 3) -> str:
    return f"{fmt(t[0], d)} {{\\scriptsize[{fmt(t[1], d)}, {fmt(t[2], d)}]}}"


def passes(name: str) -> dict[str, int]:
    seeds = pd.read_csv(RES / name / "seeds.csv")
    return {c: int(seeds[c].astype(str).eq("True").sum()) for c in seeds.columns if len(c) == 2 and c[1].isdigit()}


N: dict = {}


def veto(name: str, key: str) -> tuple[pd.DataFrame, Boot]:
    """Absent-phrase statistics shared by the COCO, LVIS pool and LVIS held-out runs."""
    df = load(name)
    b = Boot(df)
    tn, btn, tp, btp = (df[df.cond == c] for c in ("T-", "B+T-", "T+", "B+T+"))
    N[key] = dict(
        n_images=int(df.image.nunique()), n_absent=int(len(tn)), n_present_text=int(len(tp)), n_consistent=int(len(btp)),
        text_ge0=b.mean(tn, tn.presence >= 0), text_any=b.mean(tn, tn.n_kept > 0), text_mean=b.mean(tn, tn.presence),
        box_ge0=b.mean(btn, btn.presence >= 0), box_clamp=b.mean(btn, btn.presence >= 9.99), box_any=b.mean(btn, btn.n_kept > 0),
        box_covers=b.mean(btn, btn.covers_anchor), box_mean=b.mean(btn, btn.presence), box_median=float(btn.presence.median()),
        present_text_ge0=b.mean(tp, tp.presence >= 0), present_text_mean=b.mean(tp, tp.presence),
        present_text_any=b.mean(tp, tp.n_kept > 0), present_text_iou=b.mean(tp, tp.iou_cat),
        present_box_ge0=b.mean(btp, btp.presence >= 0), present_box_mean=b.mean(btp, btp.presence),
        present_box_clamp=b.mean(btp, btp.presence >= 9.99), present_box_iou=b.mean(btp, btp.iou_cat),
        auroc_text=fast_auroc(tp.presence, tn.presence), auroc_box=fast_auroc(btp.presence, btn.presence),
    )
    if "freq" in df:
        for f in "rcf":
            sub = tp[tp.freq == f]
            N[key][f"present_text_ge0_{f}"] = (float((sub.presence >= 0).mean()) if len(sub) else float("nan"), int(len(sub)))
    return df, b


def hard_curve(df: pd.DataFrame, b: Boot, taus=TAUS) -> list[dict]:
    btn, btp = df[df.cond == "B+T-"], df[df.cond == "B+T+"]
    out = []
    for tau in taus:
        ok_n = sigmoid(btn.gate_presence) >= tau
        ok_p = sigmoid(btp.gate_presence) >= tau
        out.append(dict(
            tau=tau,
            fp=b.mean(btn, (btn.n_kept > 0) & ok_n),
            drop=b.rel_drop(btn, btn.n_kept > 0, (btn.n_kept > 0) & ok_n),
            iou=b.mean(btp, btp.iou_cat * ok_p),
            cost=b.diff(btp, btp.iou_cat, btp, btp.iou_cat * ok_p),
            consistent_rejected=b.mean(btp, ~ok_p),
        ))
    return out


def detector(name: str, key: str) -> None:
    df = load(name)
    b = Boot(df)
    correct = df.correct == 1
    absent = (df.correct == 0) & (df.class_absent == 1)
    wrong = df.correct == 0
    present_wrong = wrong & (df.class_absent == 0)
    hard = (df.accepted == 1) & (sigmoid(df.text_presence) >= HARD)
    N[key] = dict(
        n_images=int(df.image.nunique()), n_det=int(len(df)), n_correct=int(correct.sum()), n_absent=int(absent.sum()),
        n_present_wrong=int(present_wrong.sum()),
        accept_absent=b.mean(df[absent], df[absent].accepted), accept_correct=b.mean(df[correct], df[correct].accepted),
        accept_present_wrong=b.mean(df[present_wrong], df[present_wrong].accepted),
        mult_absent=b.mean(df[absent], df[absent].accepted_gated), mult_correct=b.mean(df[correct], df[correct].accepted_gated),
        hard_absent=b.mean(df[absent], hard[absent]), hard_correct=b.mean(df[correct], hard[correct]),
        hard_present_wrong=b.mean(df[present_wrong], hard[present_wrong]),
        box_presence_ge0_absent=b.mean(df[absent], df[absent].presence >= 0),
        text_ge0_absent=b.mean(df[absent], df[absent].text_presence >= 0),
        text_ge0_correct=b.mean(df[correct], df[correct].text_presence >= 0),
        det_score_correct=float(df[correct].det_score.mean()), det_score_absent=float(df[absent].det_score.mean()),
        det_score_absent_hard_accepted=float(df[absent & hard].det_score.mean()),
        det_score_absent_hard_rejected=float(df[absent & ~hard].det_score.mean()),
        auroc_det=b.auroc(df, "det_score", correct, absent), auroc_ungated=b.auroc(df, "max_score", correct, absent),
        auroc_gated=b.auroc(df, "max_score_gated", correct, absent),
        auroc_det_all=b.auroc(df, "det_score", correct, wrong), auroc_ungated_all=b.auroc(df, "max_score", correct, wrong),
        auroc_gated_all=b.auroc(df, "max_score_gated", correct, wrong),
    )
    # Paired difference: gated SAM 3 score minus the detector's own score, same resamples.
    s_g, s_d = df.max_score_gated.to_numpy(float), df.det_score.to_numpy(float)
    code = pd.Categorical(df["image"], categories=b.imgs).codes
    order = np.argsort(code, kind="stable")
    starts = np.searchsorted(code[order], np.arange(len(b.imgs) + 1))
    rows = [order[starts[i]:starts[i + 1]] for i in range(len(b.imgs))]
    c, a = correct.to_numpy(), absent.to_numpy()
    gaps = []
    for take in b.idx[:1000]:
        r = np.concatenate([rows[i] for i in take])
        gaps.append(fast_auroc(s_g[r][c[r]], s_g[r][a[r]]) - fast_auroc(s_d[r][c[r]], s_d[r][a[r]]))
    point = N[key]["auroc_gated"][0] - N[key]["auroc_det"][0]
    N[key]["auroc_gap"] = (float(point), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))
    top = df[absent & hard].cls.value_counts().head(6)
    N[key]["residual_classes"] = {k: int(v) for k, v in top.items()}


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)

    # ---------------- phase 1, COCO
    coco, bc = veto("real", "coco")
    bn, bx, bo = (coco[coco.cond == c] for c in ("B-T-", "B+Tx", "B"))
    btn, btp = coco[coco.cond == "B+T-"], coco[coco.cond == "B+T+"]
    N["coco"].update(
        n_neg=int(len(bn)), neg_ge0=bc.mean(bn, bn.presence >= 0), neg_floor=bc.mean(bn, bn.presence <= -9.99),
        neg_any=bc.mean(bn, bn.n_kept > 0), neg_mean=bc.mean(bn, bn.presence),
        n_conflict=int(len(bx)), conflict_covers=bc.mean(bx, bx.covers_anchor), conflict_text=bc.mean(bx, bx.iou_cat > 0.5),
        conflict_ge0=bc.mean(bx, bx.presence >= 0),
        n_boxonly=int(len(bo)), boxonly_mean=bc.mean(bo, bo.presence), boxonly_ge0=bc.mean(bo, bo.presence >= 0),
        boxonly_any=bc.mean(bo, bo.n_kept > 0), boxonly_covers=bc.mean(bo, bo.covers_anchor),
        mult_fp=bc.mean(btn, btn.n_kept_gated > 0), mult_drop=bc.rel_drop(btn, btn.n_kept > 0, btn.n_kept_gated > 0),
        mult_iou=bc.mean(btp, btp.iou_cat_gated), mult_cost=bc.diff(btp, btp.iou_cat, btp, btp.iou_cat_gated),
    )

    # ---------------- phase 2, LVIS pool
    pool, bp = veto("real_lvis", "pool")
    btn, btp, bg = (pool[pool.cond == c] for c in ("B+T-", "B+T+", "Bg+T-"))
    N["pool"].update(
        mult_fp=bp.mean(btn, btn.n_kept_gated > 0), mult_drop=bp.rel_drop(btn, btn.n_kept > 0, btn.n_kept_gated > 0),
        mult_iou=bp.mean(btp, btp.iou_cat_gated), mult_cost=bp.diff(btp, btp.iou_cat, btp, btp.iou_cat_gated),
        bg_ge0=bp.mean(bg, bg.presence >= 0), bg_mean=bp.mean(bg, bg.presence), bg_any=bp.mean(bg, bg.n_kept > 0),
        bg_clamp=bp.mean(bg, bg.presence >= 9.99), bg_covers_box=bp.mean(bg, bg.covers_anchor),
        consistent_text_negative=bp.mean(btp, btp.gate_presence < 0),
        consistent_text_negative_iou=float(btp[btp.gate_presence < 0].iou_cat.mean()),
        curve=hard_curve(pool, bp),
    )
    for f in "rcf":
        sub = btp[btp.anchor_freq == f]
        N["pool"][f"mult_by_freq_{f}"] = (int(len(sub)), float(sub.iou_cat.mean()), float(sub.iou_cat_gated.mean()))

    # ---------------- phase 3, LVIS held-out
    held, bh = veto("real_heldout", "held")
    assert not set(held.image) & set(pool.image), "held-out images overlap the pool"
    btn, btp, bg = (held[held.cond == c] for c in ("B+T-", "B+T+", "Bg+T-"))
    N["held"].update(
        hard_fp=bh.mean(btn, btn.n_kept_hard > 0), hard_drop=bh.rel_drop(btn, btn.n_kept > 0, btn.n_kept_hard > 0),
        hard_iou=bh.mean(btp, btp.iou_hard), hard_cost=bh.diff(btp, btp.iou_cat, btp, btp.iou_hard),
        geom_fp=bh.mean(btn, btn.n_kept_geom > 0), geom_drop=bh.rel_drop(btn, btn.n_kept > 0, btn.n_kept_geom > 0),
        geom_iou=bh.mean(btp, btp.iou_geom), geom_cost=bh.diff(btp, btp.iou_cat, btp, btp.iou_geom),
        bg_ge0=bh.mean(bg, bg.presence >= 0), bg_mean=bh.mean(bg, bg.presence), bg_any=bh.mean(bg, bg.n_kept > 0),
        bg_clamp=bh.mean(bg, bg.presence >= 9.99),
        mae=bh.mean(btp, (btp.n_kept - btp.n_gt).abs()), mae_hard=bh.mean(btp, (btp.n_kept_hard - btp.n_gt).abs()),
        consistent_rejected=bh.mean(btp, sigmoid(btp.gate_presence) < HARD),
        consistent_text_negative=bh.mean(btp, btp.gate_presence < 0),
        curve=hard_curve(held, bh),
    )
    at = [c for c in N["held"]["curve"] if c["tau"] == HARD][0]
    assert abs(at["fp"][0] - N["held"]["hard_fp"][0]) < 1e-9 and abs(at["iou"][0] - N["held"]["hard_iou"][0]) < 1e-9

    # ---------------- ablations
    abl = load("real_ablate")
    ba = Boot(abl)
    N["ablate"] = dict(n_images=int(abl.image.nunique()))
    for cond in ("text", "b1", "b2", "b3", "half", "double", "shift"):
        sub = abl[abl.cond == cond]
        N["ablate"][cond] = dict(n=int(len(sub)), ge0=ba.mean(sub, sub.presence >= 0), mean=ba.mean(sub, sub.presence),
                                 any=ba.mean(sub, sub.n_kept > 0), covers=ba.mean(sub, sub.covers_anchor))

    # ---------------- detectors
    detector("real_detector", "det_pool")
    detector("real_detector_heldout", "det_held")
    detector("real_retinanet", "det_retina")
    check = pd.read_csv(RES / "real_detector" / "lvis_check_posthoc.csv").iloc[0].to_dict()
    N["det_pool"]["lvis_check"] = {k: float(v) for k, v in check.items()}

    # ---------------- counting
    cnt = load("real_count")
    bk = Boot(cnt)
    text, box, absent = (cnt[cnt.cond == c] for c in ("text", "box", "absent_box"))
    N["count"] = dict(
        n_images=int(cnt.image.nunique()), gt_mean=float(box["gt"].mean()), gt_median=float(box["gt"].median()),
        text_mae=bk.mean(text, (text.n_kept - text["gt"]).abs()), box_mae=bk.mean(box, (box.n_kept - box["gt"]).abs()),
        hard_mae=bk.mean(box, (box.n_kept_hard - box["gt"]).abs()),
        mae_cost=bk.diff(box, (box.n_kept_hard - box["gt"]).abs(), box, (box.n_kept - box["gt"]).abs()),
        false_count=bk.mean(absent, absent.n_kept >= 1), false_count_hard=bk.mean(absent, absent.n_kept_hard >= 1),
        false_drop=bk.rel_drop(absent, absent.n_kept >= 1, absent.n_kept_hard >= 1),
        absent_ge0=bk.mean(absent, absent.presence >= 0), absent_text_ge0=bk.mean(absent, absent.text_presence >= 0),
        false_count_mean=float(absent.n_kept.mean()), false_count_median=float(absent.n_kept.median()),
        true_rejected=bk.mean(box, sigmoid(box.text_presence) < HARD),
    )

    # ---------------- precision
    pre = load("real_precision")
    tn, btn = pre[pre.cond == "T-"], pre[pre.cond == "B+T-"]
    N["precision"] = dict(
        n_images=int(pre.image.nunique()), n_episodes=int(len(pre)), n_absent=int(len(tn)),
        text_ge0=float((tn.presence >= 0).mean()), box_ge0=float((btn.presence >= 0).mean()),
        text_ge0_16=float((tn.presence16 >= 0).mean()), box_ge0_16=float((btn.presence16 >= 0).mean()),
        gate_agree=float(((sigmoid(tn.presence) >= HARD) == (sigmoid(tn.presence16) >= HARD)).mean()),
        mean_abs_diff=float((pre.presence - pre.presence16).abs().mean()), max_abs_diff=float((pre.presence - pre.presence16).abs().max()),
        sign_agree=float(((pre.presence >= 0) == (pre.presence16 >= 0)).mean()),
    )

    N["passes"] = {name: passes(name) for name in ("real", "real_lvis", "real_detector", "real_heldout", "real_ablate",
                                                   "real_retinanet", "real_detector_heldout", "real_count", "real_precision")}
    (HERE / "numbers.json").write_text(json.dumps(N, indent=1))
    write_tables()
    print(f"wrote {sorted(p.name for p in TAB.glob('*.tex'))} and numbers.json; "
          f"{N['n_criteria']} criteria, {N['n_criteria_failed']} failed")


def write_tables() -> None:
    c, p, h, a = N["coco"], N["pool"], N["held"], N["ablate"]

    # Table 1: the veto is lost.
    rows = []
    for label, src, d in (("COCO", "80 classes", c), ("LVIS, selection", "verified", p), ("LVIS, held-out", "verified", h)):
        rows.append(f"{label} & {src} & {d['n_absent']:,} & {pct(d['text_ge0'])} & {pct(d['text_any'])} & "
                    f"{pct_ci(d['box_ge0'])} & {pct(d['box_clamp'])} & {pct(d['box_any'])} & {pct_ci(d['box_covers'])} \\\\")
    (TAB / "veto.tex").write_text(
        "\\begin{tabular}{@{}llr cc cccc@{}}\n\\toprule\n"
        " & & & \\multicolumn{2}{c}{Text only} & \\multicolumn{4}{c}{Text + positive box on the largest object} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(l){6-9}\n"
        "Images & Absence & Episodes & $s_p\\!\\geq\\!0$ & any mask & $s_p\\!\\geq\\!0$ & $s_p\\!=\\!10$ & any mask & boxed object returned \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # Table 2: what the presence head responds to.
    def line(name, n, ge0, mean, anym, cov):
        mean = mean if mean == "--" else f"${mean}$"  # math mode, so that negative means get a minus sign
        return f"{name} & {n:,} & {ge0} & {mean} & {anym} & {cov} \\\\"

    bgp = N["pool"]
    rows = [
        "\\multicolumn{6}{@{}l}{\\emph{Absent phrase, LVIS held-out}} \\\\",
        line("text only", h["n_absent"], pct(h["text_ge0"]), f"{h['text_mean'][0]:.2f}", pct(h["text_any"]), "--"),
        line("+ box on largest object", h["n_absent"], pct(h["box_ge0"]), f"{h['box_mean'][0]:.2f}", pct(h["box_any"]), pct(h["box_covers"])),
        line("+ box on background", h["n_images"], pct(h["bg_ge0"]), f"{h['bg_mean'][0]:.2f}", pct(h["bg_any"]), "--"),
        "\\multicolumn{6}{@{}l}{\\emph{Absent phrase, LVIS selection images}} \\\\",
        line("text only", a["text"]["n"], pct(a["text"]["ge0"]), f"{a['text']['mean'][0]:.2f}", "--", "--"),
        line("+ 1 box", a["b1"]["n"], pct(a["b1"]["ge0"]), f"{a['b1']['mean'][0]:.2f}", pct(a["b1"]["any"]), pct(a["b1"]["covers"])),
        line("+ 2 boxes", a["b2"]["n"], pct(a["b2"]["ge0"]), f"{a['b2']['mean'][0]:.2f}", pct(a["b2"]["any"]), pct(a["b2"]["covers"])),
        line("+ 3 boxes", a["b3"]["n"], pct(a["b3"]["ge0"]), f"{a['b3']['mean'][0]:.2f}", pct(a["b3"]["any"]), pct(a["b3"]["covers"])),
        line("+ box at half scale", a["half"]["n"], pct(a["half"]["ge0"]), f"{a['half']['mean'][0]:.2f}", pct(a["half"]["any"]), pct(a["half"]["covers"])),
        line("+ box at double scale", a["double"]["n"], pct(a["double"]["ge0"]), f"{a['double']['mean'][0]:.2f}", pct(a["double"]["any"]), pct(a["double"]["covers"])),
        line("+ box shifted by half its width", a["shift"]["n"], pct(a["shift"]["ge0"]), f"{a['shift']['mean'][0]:.2f}", pct(a["shift"]["any"]), pct(a["shift"]["covers"])),
        "\\multicolumn{6}{@{}l}{\\emph{COCO (absence by the 80-class label set)}} \\\\",
        line("absent phrase + \\emph{negative} box", c["n_neg"], pct(c["neg_ge0"]), f"{c['neg_mean'][0]:.2f}", pct(c["neg_any"]), "--"),
        line("positive box, no text", c["n_boxonly"], pct(c["boxonly_ge0"]), f"{c['boxonly_mean'][0]:.2f}", pct(c["boxonly_any"]), pct(c["boxonly_covers"])),
        line("other present class + box", c["n_conflict"], pct(c["conflict_ge0"]), "--", "--", pct(c["conflict_covers"])),
    ]
    (TAB / "controls.tex").write_text(
        "\\begin{tabular}{@{}lrcccc@{}}\n\\toprule\n"
        "Prompt & $n$ & $s_p\\!\\geq\\!0$ & mean $s_p$ & any mask & object \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    del bgp

    # Table 3: gates.
    p10 = [x for x in p["curve"] if x["tau"] == HARD][0]
    rows = [
        "\\multicolumn{6}{@{}l}{\\emph{LVIS selection images (500)}} \\\\",
        f"none & -- & {pct(p['box_any'])} & -- & {p['present_box_iou'][0]:.3f} & -- \\\\",
        f"multiplicative & pre-reg., \\textbf{{fails}} & {pct(p['mult_fp'])} & {pct(p['mult_drop'])} & {p['mult_iou'][0]:.3f} & {num_ci(p['mult_cost'])} \\\\",
        f"hard, $\\tau\\!=\\!0.10$ & post hoc & {pct(p10['fp'])} & {pct(p10['drop'])} & {p10['iou'][0]:.3f} & {num_ci(p10['cost'])} \\\\",
        "\\multicolumn{6}{@{}l}{\\emph{LVIS held-out images (500)}} \\\\",
        f"none & -- & {pct(h['box_any'])} & -- & {h['present_box_iou'][0]:.3f} & -- \\\\",
        f"hard, $\\tau\\!=\\!0.10$ & pre-reg., passes & {pct_ci(h['hard_fp'])} & {pct(h['hard_drop'])} & {h['hard_iou'][0]:.3f} & {num_ci(h['hard_cost'])} \\\\",
        f"geometry & pre-reg., \\textbf{{fails}} & {pct_ci(h['geom_fp'])} & {pct(h['geom_drop'])} & {h['geom_iou'][0]:.3f} & {num_ci(h['geom_cost'])} \\\\",
    ]
    (TAB / "gates.tex").write_text(
        "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
        "Gate & Status & False-pos.\\ episodes (\\%) & Drop (\\%) & mIoU & mIoU cost \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # Table 4: detector verification.
    rows = []
    for label, key in (("Faster R-CNN, COCO set", "det_pool"), ("Faster R-CNN, held-out", "det_held"), ("RetinaNet, COCO set", "det_retina")):
        d = N[key]
        rows.append(f"{label} & {d['n_det']:,} & {d['n_absent']} & {pct(d['accept_absent'])} & {pct(d['hard_absent'])} & {pct(d['hard_correct'])} & "
                    f"{d['auroc_det'][0]:.3f} & {d['auroc_ungated'][0]:.3f} & {num_ci(d['auroc_gated'])} \\\\")
    (TAB / "detector.tex").write_text(
        "\\begin{tabular}{@{}lrr ccc ccc@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Detections} & \\multicolumn{3}{c}{Accepted by SAM 3 (\\%)} & \\multicolumn{3}{c}{AUROC, correct vs.\\ class-absent} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(l){7-9}\n"
        "Detector, images & all & class-absent & absent & absent, gated & correct, gated & detector & SAM 3 & SAM 3, gated \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # Table 5: counting.
    k = N["count"]
    (TAB / "count.tex").write_text(
        "\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
        "Prompt & MAE, true class & False count (\\%) \\\\\n\\midrule\n"
        f"text only & {k['text_mae'][0]:.2f} & -- \\\\\n"
        f"text + box & {k['box_mae'][0]:.2f} & {pct_ci(k['false_count'])} \\\\\n"
        f"text + box, hard gate & {k['hard_mae'][0]:.2f} & {pct_ci(k['false_count_hard'])} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")

    # Table 6: every pre-registered criterion.
    ps = N["passes"]
    dp, dh, dr = N["det_pool"], N["det_held"], N["det_retina"]
    crit = [
        ("1", "K1", "presence AUROC $\\geq 0.95$ text only, $\\leq 0.60$ with box", f"{c['auroc_text']:.3f}, {c['auroc_box']:.3f}", ps["real"]["k1"]),
        ("1", "K2", "boxed object returned $\\geq 70\\%$; box beats text $2{:}1$ in conflicts", f"{pct(c['box_covers'])}\\%; {pct(c['conflict_covers'])}\\% vs.\\ {pct(c['conflict_text'])}\\%", ps["real"]["k2"]),
        ("1", "K3", "multiplicative gate: FP drop $\\geq 80\\%$, mIoU cost $\\leq 0.02$", f"{pct(c['mult_drop'])}\\%, {c['mult_cost'][0]:.3f}", ps["real"]["k3"]),
        ("2", "L1", "absent $s_p \\geq 0$: $\\leq 10\\%$ text only, $\\geq 90\\%$ with box", f"{pct(p['text_ge0'])}\\%, {pct(p['box_ge0'])}\\%", ps["real_lvis"]["l1"]),
        ("2", "L2", "boxed object returned $\\geq 70\\%$", f"{pct(p['box_covers'])}\\%", ps["real_lvis"]["l2"]),
        ("2", "L3", "multiplicative gate: FP drop $\\geq 80\\%$, mIoU cost $\\leq 0.03$", f"{pct(p['mult_drop'])}\\%, {p['mult_cost'][0]:.3f}", ps["real_lvis"]["l3"]),
        ("2", "D1", "class-absent detections accepted $\\geq 90\\%$", f"{pct(dp['accept_absent'])}\\%", ps["real_detector"]["d1"]),
        ("2", "D2", "multiplicative gate: drop $\\geq 70\\%$, correct lost $\\leq 5$ pts", f"{100 * (1 - dp['mult_absent'][0] / dp['accept_absent'][0]):.1f}\\%, {100 * (dp['accept_correct'][0] - dp['mult_correct'][0]):.1f} pts", ps["real_detector"]["d2"]),
        ("3", "H1", "as L1, held-out images", f"{pct(h['text_ge0'])}\\%, {pct(h['box_ge0'])}\\%", ps["real_heldout"]["h1"]),
        ("3", "H2", "as L2, held-out images", f"{pct(h['box_covers'])}\\%", ps["real_heldout"]["h2"]),
        ("3", "H3", "hard gate: FP drop $\\geq 80\\%$, mIoU cost $\\leq 0.02$", f"{pct(h['hard_drop'])}\\%, {h['hard_cost'][0]:.3f}", ps["real_heldout"]["h3"]),
        ("3", "H4", "geometry gate: same bars as H3", f"{pct(h['geom_drop'])}\\%, {h['geom_cost'][0]:.3f}", ps["real_heldout"]["h4"]),
        ("3", "H5", "background box: $s_p \\geq 0$ in $\\geq 90\\%$", f"{pct(h['bg_ge0'])}\\%", ps["real_heldout"]["h5"]),
        ("3", "H6", "false counts $\\geq 90\\%$, gate drop $\\geq 80\\%$, count MAE rises by $\\leq 0.30$", f"{pct(h['box_any'])}\\%, {pct(h['hard_drop'])}\\%, $+${h['mae_hard'][0] - h['mae'][0]:.3f}", ps["real_heldout"]["h6"]),
        ("3", "A1", "$s_p \\geq 0$ in $\\geq 90\\%$ with 1, 2, 3 boxes", f"{pct(a['b1']['ge0'])}, {pct(a['b2']['ge0'])}, {pct(a['b3']['ge0'])}\\%", ps["real_ablate"]["a1"]),
        ("3", "A2", "$s_p \\geq 0$ in $\\geq 90\\%$ for half, double, shifted box", f"{pct(a['half']['ge0'])}, {pct(a['double']['ge0'])}, {pct(a['shift']['ge0'])}\\%", ps["real_ablate"]["a2"]),
        ("3", "A3", "half-scale box covers the object $\\leq 50\\%$", f"{pct(a['half']['covers'])}\\%", ps["real_ablate"]["a3"]),
        ("3", "R1", "RetinaNet: class-absent accepted $\\geq 90\\%$", f"{pct(dr['accept_absent'])}\\%", ps["real_retinanet"]["d1"]),
        ("3", "R2", "RetinaNet: gated AUROC $\\geq 0.90$ and above detector", f"{dr['auroc_gated'][0]:.3f} vs.\\ {dr['auroc_det'][0]:.3f}", ps["real_retinanet"]["d2"]),
        ("3", "R1$'$", "Faster R-CNN, held-out: as R1", f"{pct(dh['accept_absent'])}\\%", ps["real_detector_heldout"]["d1"]),
        ("3", "R2$'$", "Faster R-CNN, held-out: as R2", f"{dh['auroc_gated'][0]:.3f} vs.\\ {dh['auroc_det'][0]:.3f}", ps["real_detector_heldout"]["d2"]),
        ("3", "F1", "FSC-147: absent class + exemplar counts $\\geq 1$ in $\\geq 80\\%$", f"{pct(N['count']['false_count'])}\\%", ps["real_count"]["f1"]),
        ("3", "F2", "FSC-147: hard gate drops that by $\\geq 70\\%$", f"{pct(N['count']['false_drop'])}\\%", ps["real_count"]["f2"]),
        ("3", "F3", "FSC-147: true-class MAE rises by $\\leq 2.0$", f"$+${N['count']['mae_cost'][0]:.2f}", ps["real_count"]["f3"]),
        ("4", "P1", "fp32: as L1 on 100 held-out images", f"{100 * N['precision']['text_ge0']:.1f}\\%, {100 * N['precision']['box_ge0']:.1f}\\%", ps["real_precision"]["p1"]),
        ("4", "P2", "fp32 and fp16 gate decisions agree $\\geq 95\\%$", f"{100 * N['precision']['gate_agree']:.1f}\\%", ps["real_precision"]["p2"]),
        ("4", "P3", "mean $|s_p^{32} - s_p^{16}| \\leq 0.5$", f"{N['precision']['mean_abs_diff']:.3f}", ps["real_precision"]["p3"]),
    ]
    body, last = [], None
    for ph, cid, text, obs, k in crit:
        if last is not None and ph != last:
            body.append("\\addlinespace[2pt]")
        last = ph
        mark = f"{k}/8" if k >= 6 else f"\\textbf{{{k}/8}}"
        body.append(f"{ph} & {cid} & {text} & {obs} & {mark} \\\\")
    (TAB / "criteria.tex").write_text(
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cllll@{}}\n\\toprule\n"
        "Phase & ID & Criterion (fixed before the run) & Observed & Pass \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular*}\n")
    n_fail = sum(x[4] < 6 for x in crit)
    N["n_criteria"], N["n_criteria_failed"] = len(crit), n_fail
    (HERE / "numbers.json").write_text(json.dumps(N, indent=1))


if __name__ == "__main__":
    main()
