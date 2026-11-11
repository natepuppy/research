"""Intruders and duplicate views in VGGT-1B, on TUM RGB-D.

Train the probes on freiburg1_desk. Test them on freiburg1_xyz. An intruder is one
view swapped for a frame from the other sequence. A duplicate is one view replaced
by a copy of another view in the same window. ScanNet, DL3DV and pi3 were not run.
Thresholds are the README's, written before the run. Eight views are requested.
If one eight-view forward does not fit, the whole run uses four and says so.

K1. Held-out intruder AUROC of the token probe >= 0.85 and >= 0.10 above anchor
    agreement (minus the minimum pairwise token cosine), on >= 6 of 8 bootstraps.
K2. Held-out duplicate AUROC of a probe fit on duplicates >= 0.80, on >= 6 of 8.
    Anchor agreement and mean depth confidence are logged beside it.
K3. On held-out intruder sets, dropping the least-agreeing view and running a
    second forward reduces mean rotation error of the genuine views by >= 40%.
    Reported. Pose error is after a similarity alignment to TUM ground truth.
SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one, or if the test slice has
fewer than 12 clean and 12 contaminated sets. KILLED if neither.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real"
DATA = Path(r"C:\Users\natha\dev\cv-data\tum")
CODE = Path(r"C:\Users\natha\dev\vggt")
SEQUENCES = ("rgbd_dataset_freiburg1_desk", "rgbd_dataset_freiburg1_xyz")
URLS = {
    "rgbd_dataset_freiburg1_desk": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "rgbd_dataset_freiburg1_xyz": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
}
TRAIN, TEST = SEQUENCES
N_PER_KIND = 16


def ensure_sequences() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if (DATA / name / "rgb.txt").exists():
            continue
        archive = DATA / f"{name}.tgz"
        if not archive.exists() or archive.stat().st_size < 1_000_000:
            print(f"downloading {name}", flush=True)
            subprocess.check_call(["curl.exe", "-L", "--retry", "10", "--retry-all-errors", "-C", "-", "-o", str(archive), url])
        print(f"extracting {name}", flush=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(DATA)


def ensure_code() -> None:
    if (CODE / "vggt" / "models" / "vggt.py").exists():
        return
    print("cloning VGGT", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/vggt.git", str(CODE)])
    sys.path.insert(0, str(CODE))


def read_list(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        stamp, name = line.split()[:2]
        rows.append((float(stamp), name))
    return rows


def frames_of(seq_name: str) -> list[dict]:
    seq = DATA / seq_name
    rgb, depth = read_list(seq / "rgb.txt"), read_list(seq / "depth.txt")
    gt = []
    for line in (seq / "groundtruth.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        gt.append((float(parts[0]), np.array(parts[1:4], float), np.array(parts[4:8], float)))
    j = k = 0
    frames = []
    for stamp, rgb_name in rgb:
        while j + 1 < len(depth) and abs(depth[j + 1][0] - stamp) < abs(depth[j][0] - stamp):
            j += 1
        while k + 1 < len(gt) and abs(gt[k + 1][0] - stamp) < abs(gt[k][0] - stamp):
            k += 1
        pose = None
        if abs(gt[k][0] - stamp) <= 0.02:
            pose = (gt[k][1], gt[k][2])
        if abs(depth[j][0] - stamp) <= 0.02:
            frames.append(dict(stamp=stamp, rgb=str(seq / rgb_name), pose=pose))
    return frames[::2]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    u, _, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    sign[-1, -1] = np.sign(np.linalg.det(u @ vt))
    return u @ sign @ vt


def rotation_error(extrinsic: np.ndarray, poses: list, genuine: list[int]) -> float:
    centers_p, centers_g, rots_p, rots_g = [], [], [], []
    for i in genuine:
        if poses[i] is None:
            continue
        rotation = extrinsic[i, :3, :3]
        translation = extrinsic[i, :3, 3]
        c2w = rotation.T
        centers_p.append(-c2w @ translation)
        centers_g.append(poses[i][0])
        rots_p.append(c2w)
        rots_g.append(quat_to_rot(poses[i][1]))
    if len(centers_p) < 3:
        return float("nan")
    align = umeyama(np.stack(centers_p), np.stack(centers_g))
    angles = []
    for pred, gt in zip(rots_p, rots_g):
        cosine = (np.trace(gt.T @ (align @ pred)) - 1) / 2
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return float(np.mean(angles))


def make_sets(n_views: int) -> list[dict]:
    pool = {name: frames_of(name) for name in SEQUENCES}
    rng = np.random.default_rng(0)
    sets = []
    for seq_name in SEQUENCES:
        frames = pool[seq_name]
        other = pool[TRAIN if seq_name == TEST else TEST]
        starts = list(range(0, len(frames) - n_views + 1, n_views))
        rng.shuffle(starts)
        starts = starts[:N_PER_KIND]
        for kind in ("clean", "intruder", "duplicate"):
            for start in starts:
                window = frames[start:start + n_views]
                paths = [item["rgb"] for item in window]
                poses = [item["pose"] for item in window]
                planted = n_views // 2
                if kind == "intruder":
                    donor = other[int(rng.integers(0, len(other)))]
                    paths[planted] = donor["rgb"]
                    poses[planted] = None
                elif kind == "duplicate":
                    paths[planted] = paths[0]
                    poses[planted] = None
                genuine = [i for i in range(n_views) if not (kind != "clean" and i == planted)]
                sets.append(dict(
                    index=len(sets), seq=seq_name, split="train" if seq_name == TRAIN else "test",
                    kind=kind, paths=paths, poses=poses, genuine=genuine, planted=planted, n_views=n_views,
                ))
    return sets


def view_matrix(tokens: torch.Tensor, ps_idx: int, n_views: int) -> np.ndarray:
    value = tokens.detach().float()
    while value.ndim > 3:
        value = value[0]
    if value.ndim == 2:
        value = value[None]
    if value.shape[0] != n_views and value.shape[1] == n_views:
        value = value.transpose(0, 1)
    patch = value[:, ps_idx:, :].mean(dim=1).cpu().numpy()
    return patch


def forward(model, paths: list[str], device: str):
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images(paths).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        batched = images[None]
        tokens, ps_idx = model.aggregator(batched)
        pose = model.camera_head(tokens)[-1]
        extrinsic, _ = pose_encoding_to_extri_intri(pose, images.shape[-2:])
        _, confidence = model.depth_head(tokens, batched, ps_idx)
    last = tokens[-1] if isinstance(tokens, (list, tuple)) else tokens
    matrix = view_matrix(last, int(ps_idx), len(paths))
    conf = float(confidence.detach().float().mean())
    extrinsic_np = extrinsic.detach().float().cpu().numpy()
    if extrinsic_np.ndim == 4:
        extrinsic_np = extrinsic_np[0]
    return matrix, extrinsic_np, conf


def loo_features(tokens: np.ndarray) -> tuple[np.ndarray, float, int]:
    norm = tokens / np.clip(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-6, None)
    sims = norm @ norm.T
    np.fill_diagonal(sims, np.nan)
    agreement = np.nanmean(sims, axis=1)
    return np.sort(agreement), float(np.nanmin(sims)), int(np.nanargmin(agreement))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    n_pos, n_neg = int(label.sum()), int((1 - label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_probe(rows: list[dict], positive: str) -> np.ndarray:
    chosen = [r for r in rows if r["split"] == "train" and r["kind"] in ("clean", positive)]
    x = np.array([np.array([float(v) for v in r["loo"].split(";")]) for r in chosen])
    y = np.array([1.0 if r["kind"] == positive else 0.0 for r in chosen])
    bias = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    weight = np.zeros(bias.shape[1])
    for _ in range(400):
        prob = 1 / (1 + np.exp(-np.clip(bias @ weight, -20, 20)))
        grad = bias.T @ (prob - y) / len(y) + 0.1 * np.r_[weight[:-1], 0.0]
        weight -= 0.5 * grad
    return weight


def probe_score(row: dict, weight: np.ndarray) -> float:
    feat = np.array([float(v) for v in row["loo"].split(";")])
    logit = float(np.r_[feat, 1.0] @ weight)
    return 1 / (1 + np.exp(-np.clip(logit, -20, 20)))


def bootstrap(rows: list[dict], weight: np.ndarray, positive: str, rng: np.random.Generator) -> tuple[float, float]:
    chosen = [r for r in rows if r["split"] == "test" and r["kind"] in ("clean", positive)]
    take = rng.choice(len(chosen), size=len(chosen), replace=True)
    sample = [chosen[i] for i in take]
    score = np.array([probe_score(row, weight) for row in sample])
    anchor = np.array([-float(row["anchor"]) for row in sample])
    label = np.array([1.0 if row["kind"] == positive else 0.0 for row in sample])
    return auroc(score, label), auroc(anchor, label)


def anchor_auroc(rows: list[dict], positive: str) -> float:
    chosen = [r for r in rows if r["split"] == "test" and r["kind"] in ("clean", positive)]
    score = np.array([-float(r["anchor"]) for r in chosen])
    label = np.array([1.0 if r["kind"] == positive else 0.0 for r in chosen])
    return auroc(score, label)


def verdict(rows: list[dict], n_views: int) -> None:
    test_clean = [r for r in rows if r["split"] == "test" and r["kind"] == "clean"]
    test_in = [r for r in rows if r["split"] == "test" and r["kind"] == "intruder"]
    test_dup = [r for r in rows if r["split"] == "test" and r["kind"] == "duplicate"]
    measured = len(test_clean) >= 12 and len(test_in) >= 12 and len(test_dup) >= 12
    if measured:
        w_in, w_dup = fit_probe(rows, "intruder"), fit_probe(rows, "duplicate")
        k1_hits, k2_hits = [], []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            intruder, anchor = bootstrap(rows, w_in, "intruder", rng)
            duplicate, _ = bootstrap(rows, w_dup, "duplicate", rng)
            k1_hits.append(intruder >= 0.85 and intruder - anchor >= 0.10)
            k2_hits.append(duplicate >= 0.80)
        point_in, anchor_in = bootstrap(rows, w_in, "intruder", np.random.default_rng(0))
        point_dup, anchor_dup = bootstrap(rows, w_dup, "duplicate", np.random.default_rng(0))
    else:
        k1_hits, k2_hits = [], []
        point_in = point_dup = anchor_in = anchor_dup = float("nan")
    repaired = [r for r in test_in if r.get("err_before", "") not in ("", None) and r.get("err_after", "") not in ("", None)]
    before = np.array([float(r["err_before"]) for r in repaired])
    after = np.array([float(r["err_after"]) for r in repaired])
    usable = before > 1.0
    reduction = float(np.mean((before[usable] - after[usable]) / before[usable])) if usable.any() else float("nan")
    k1 = measured and sum(k1_hits) >= 6
    k2 = measured and sum(k2_hits) >= 6
    k3 = measured and reduction == reduction and reduction >= 0.40
    if not measured:
        outcome = None
    elif k1 and k2:
        outcome = True
    elif k1 or k2:
        outcome = None
    else:
        outcome = False
    conf = float(np.mean([float(r["conf"]) for r in test_dup])) if test_dup else float("nan")
    text = (
        f"K1 (intruder probe >= 0.85 and >= 0.10 above anchor, {sum(k1_hits)}/8; "
        f"point {point_in:.3f} vs anchor {anchor_in:.3f}) = {bool(k1)}. "
        f"K2 (duplicate probe >= 0.80, {sum(k2_hits)}/8; point {point_dup:.3f}, "
        f"anchor {anchor_dup:.3f}, mean depth confidence {conf:.3f}) = {bool(k2)}. "
        f"K3 (pose-error reduction {reduction:.3f} on {int(usable.sum()) if measured else 0} sets, 2 forwards) = {bool(k3)}. "
        f"Views {n_views}. Test clean/intruder/duplicate {len(test_clean)}/{len(test_in)}/{len(test_dup)}. "
        f"VGGT-1B on TUM freiburg1; ScanNet, DL3DV and pi3 were not run."
    )
    line = kill_test(outcome, text)
    (OUT / "verdict.txt").write_text(line + "\n")
    save_csv(OUT / "summary.csv", [dict(
        n_views=n_views, point_intruder=point_in, anchor_intruder=anchor_in,
        point_duplicate=point_dup, anchor_duplicate=anchor_dup, confidence=conf,
        reduction=reduction, k1_seeds=sum(k1_hits), k2_seeds=sum(k2_hits),
    )])


def evaluate(n_views: int) -> None:
    ensure_code()
    sys.path.insert(0, str(CODE))
    from vggt.models.vggt import VGGT

    device = "cuda"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    sets = make_sets(n_views)
    try:
        forward(model, sets[0]["paths"], device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if n_views == 4:
            raise
        print("8 views did not fit; rerunning every set at 4 views. Thresholds unchanged.", flush=True)
        del model
        torch.cuda.empty_cache()
        evaluate(4)
        return
    done = {int(r["index"]) for r in load_csv(OUT / "runs.csv") if int(r.get("n_views", 0)) == n_views}
    written = [r for r in load_csv(OUT / "runs.csv") if int(r.get("n_views", 0)) == n_views]
    for item in sets:
        if item["index"] in done:
            continue
        tokens, extrinsic, conf = forward(model, item["paths"], device)
        features, anchor, suspect = loo_features(tokens)
        err_before = rotation_error(extrinsic, item["poses"], item["genuine"])
        err_after = ""
        if item["split"] == "test" and item["kind"] == "intruder":
            kept = [i for i in range(item["n_views"]) if i != suspect]
            paths = [item["paths"][i] for i in kept]
            _, extrinsic_2, _ = forward(model, paths, device)
            genuine = [kept.index(i) for i in item["genuine"] if i in kept]
            err_after = rotation_error(extrinsic_2, [item["poses"][i] for i in kept], genuine)
        record = dict(
            index=item["index"], seq=item["seq"], split=item["split"], kind=item["kind"],
            n_views=n_views, loo=";".join(f"{v:.6f}" for v in features), anchor=anchor,
            conf=conf, err_before=err_before, err_after=err_after, suspect=suspect,
            planted=item["planted"],
        )
        written.append(record)
        save_csv(OUT / "runs.csv", written)
        print(f"{item['index']:03d} {item['split']} {item['kind']} err={err_before}", flush=True)
        torch.cuda.empty_cache()
    verdict(written, n_views)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        rows = load_csv(OUT / "runs.csv")
        verdict(rows, int(rows[-1]["n_views"]) if rows else args.views)
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    ensure_sequences()
    try:
        evaluate(args.views)
    except Exception as exc:
        line = f"NOT RUN. {type(exc).__name__}: {exc}"
        print(line, flush=True)
        (OUT / "error.txt").write_text(line + "\n")
        if not (OUT / "verdict.txt").exists():
            (OUT / "verdict.txt").write_text(line + "\n")
        raise


if __name__ == "__main__":
    main()
