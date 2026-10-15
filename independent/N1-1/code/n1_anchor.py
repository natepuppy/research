"""SD-TD anchored to a TDC reference (fixed-point preserving) vs SD-TD vs TDC vs TD on the sampled linear families.

    python code/n1_anchor.py [steps]
"""
import sys, json
import numpy as np
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # code/
PROJECT = HERE.parent
import n1_extensions as n1

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
SEEDS = 3
ALPHAS = [0.002, 0.005, 0.01, 0.02]
ALGOS = ["td", "tdc", "sdtd", "sdtd_anchor"]

def build():
    rng = np.random.default_rng(7)
    systems = [n1.baird_system(None, 0.0)]
    systems += [n1.baird_system(rng, 1.0) for _ in range(3)]
    found = []
    while len(found) < 8:
        s = n1.baird_structure_system(rng)
        if np.max(np.abs(np.linalg.eigvals(np.eye(len(s.key)) + 0.05 * s.key))) > 1 + 1e-8:  # expected step diverges
            found.append(s)
    systems += found
    systems += [n1.random_system(rng) for _ in range(8)]
    systems += [n1.on_policy_system(rng) for _ in range(6)]
    return systems

def job(args):
    idx, system, algo, alpha, seed = args
    out = n1.run_algorithm(system, algo, alpha, 1.0, STEPS, seed)
    out["system_index"] = idx
    out.pop("curve", None)
    out["rmsve_zero"] = system.rmsve(np.zeros(system.phi.shape[1]))
    return out

if __name__ == "__main__":
    systems = build()
    jobs = [(i, s, a, al, sd) for i, s in enumerate(systems) for a in ALGOS for al in ALPHAS for sd in range(SEEDS)]
    with Pool(10) as pool:
        res = pool.map(job, jobs, chunksize=4)
    fams = sorted(set(r["system"] for r in res), key=lambda f: ["baird", "baird_reward", "baird_structure", "random", "on_policy"].index(f))
    print(f"steps={STEPS}  cells: diverged frac / median tail RMSVE (non-diverged) ; zero-predictor RMSVE median per family")
    for f in fams:
        zero = np.median([r["rmsve_zero"] for r in res if r["system"] == f])
        print(f"\n== {f}  (predict-0 RMSVE {zero:.3f})")
        for a in ALGOS:
            cells = []
            for al in ALPHAS:
                sub = [r for r in res if r["system"] == f and r["algo"] == a and r["alpha"] == al]
                div = np.mean([r["diverged"] for r in sub])
                ok = [r["tail_rmsve"] for r in sub if not r["diverged"]]
                cells.append(f"{al:g}: {div:.2f}/{np.median(ok):.3f}" if ok else f"{al:g}: {div:.2f}/  nan")
            print(f"  {a:12s}", " | ".join(cells))
    json.dump(res, open(fstr(PROJECT / "results" / f"anchor_runs_{STEPS}.json"), "w"))
