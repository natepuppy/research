"""200,000-transition check of TDC vs SD-TD vs anchored SD-TD on Baird (Appendix F.2). Prints one tuple per run."""
import sys, numpy as np
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # code/
PROJECT = HERE.parent
import n1_extensions as n1
STEPS = 200000
def job(a):
    sysname, system, algo, alpha, seed = a
    out = n1.run_algorithm(system, algo, alpha, 1.0, STEPS, seed, eval_every=2000)
    c = [v for _, v in out["curve"]]
    return sysname, algo, alpha, seed, out["diverged"], [round(c[i],3) for i in (0, 10, 25, 50, 75, 99) if i < len(c)]
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    systems = [("baird", n1.baird_system(None, 0.0)), ("baird_reward", n1.baird_system(rng, 1.0))]
    jobs = [(n, s, a, al, sd) for n, s in systems for a in ("tdc", "sdtd", "sdtd_anchor") for al in (0.005, 0.02) for sd in range(2)]
    with Pool(10) as p:
        for r in p.map(job, jobs):
            print(r)
