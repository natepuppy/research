"""3080 measurements for the latency-law paper.

Same corridor as R22 (length L, right reaches a terminal reward of 1, V*(s) = gamma**(L-1-s)),
but the data is collected online by the agent and the network trains on CUDA.

Three mechanisms, one instrument (the value front: contiguous states from the goal whose
prediction has reached half the truth):

  dqn     hard target, refreshed every K updates. Law: arrival ~ (L-1)*K.
  polyak  target <- (1-tau)*target + tau*online every update. Law: arrival ~ (L-1)/tau + L/alpha.
  ppo     neural actor-critic, GAE(lambda), one critic fit per iteration.
          Law: lambda=1 arrives immediately once a return is seen; smaller lambda is slower,
          t ~ L*((1-gamma*lambda) + 1/T) critic fits when the rollout is the cap.

Numbers written to results/gpu/. Nothing is filled in by hand.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gpu"
GAMMA = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class QNet(nn.Module):
    def __init__(self, fin: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, fin: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(fin, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, 2)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def features(states: torch.Tensor, enc: str, L: int) -> torch.Tensor:
    if enc == "onehot":
        return F.one_hot(states, L).float()
    if enc == "coord":
        return (states.float() / L).unsqueeze(-1)
    raise ValueError(enc)


def fin_of(enc: str, L: int) -> int:
    return L if enc == "onehot" else 1


def vstar(L: int, device) -> torch.Tensor:
    s = torch.arange(L, device=device)
    return GAMMA ** (L - 1 - s).float()


@torch.no_grad()
def value_front(q_max: torch.Tensor, vs: torch.Tensor) -> int:
    """Contiguous run from the goal whose value has reached half the truth. q_max: (L,)."""
    arrived = q_max >= 0.5 * vs
    # index 0 of the flipped vector is the goal
    flipped = arrived.flip(0)
    if not bool(flipped[0]):
        return 0
    # first False stops the run
    bad = (~flipped).nonzero(as_tuple=False)
    if bad.numel() == 0:
        return int(arrived.numel())
    return int(bad.view(-1)[0])


def step_env(s: int, a: int, L: int):
    if a == 1:
        sp = s + 1
        if sp >= L:
            return 0, 1.0, True
        return sp, 0.0, False
    return max(s - 1, 0), 0.0, False


class Replay:
    def __init__(self, cap: int):
        self.s = np.zeros(cap, np.int64)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.sp = np.zeros(cap, np.int64)
        self.d = np.zeros(cap, np.float32)
        self.cap = cap
        self.n = 0
        self.i = 0

    def add(self, s, a, r, sp, d):
        self.s[self.i] = s
        self.a[self.i] = a
        self.r[self.i] = r
        self.sp[self.i] = sp
        self.d[self.i] = d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, B: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, B)
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.d[idx]


def run_q(kind: str, L: int, enc: str, K_or_tau: float, seed: int, max_updates: int, reset: str, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fin = fin_of(enc, L)
    online = QNet(fin).to(DEVICE)
    target = QNet(fin).to(DEVICE)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    vs = vstar(L, DEVICE)
    buf = Replay(50_000)
    s = int(rng.integers(0, L)) if reset == "random" else 0
    front_at = -1
    eps = 0.2
    B = 64
    t0 = time.time()
    last_front = 0
    for t in range(1, max_updates + 1):
        with torch.no_grad():
            x = features(torch.tensor([s], device=DEVICE), enc, L)
            a = int(online(x).argmax(1)) if rng.random() > eps else int(rng.integers(0, 2))
        sp, r, done = step_env(s, a, L)
        buf.add(s, a, r, 0 if done else sp, float(done))
        if done or (reset == "random" and rng.random() < 0.02):
            s = int(rng.integers(0, L)) if reset == "random" else 0
        else:
            s = sp
        if buf.n >= 1000:
            sb, ab, rb, spb, db = buf.sample(B, rng)
            sb_t = torch.tensor(sb, device=DEVICE)
            ab_t = torch.tensor(ab, device=DEVICE)
            rb_t = torch.tensor(rb, device=DEVICE)
            sp_t = torch.tensor(spb, device=DEVICE)
            db_t = torch.tensor(db, device=DEVICE)
            with torch.no_grad():
                qn = target(features(sp_t, enc, L)).max(1).values
                y = rb_t + GAMMA * (1.0 - db_t) * qn
            q = online(features(sb_t, enc, L)).gather(1, ab_t.view(-1, 1)).squeeze(1)
            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            opt.step()
            if kind == "dqn":
                if t % int(K_or_tau) == 0:
                    target.load_state_dict(online.state_dict())
            else:
                tau = float(K_or_tau)
                with torch.no_grad():
                    for pt, po in zip(target.parameters(), online.parameters()):
                        pt.mul_(1.0 - tau).add_(po, alpha=tau)
        if t % 100 == 0 or t == max_updates:
            with torch.no_grad():
                xs = features(torch.arange(L, device=DEVICE), enc, L)
                qm = online(xs).max(1).values
                fr = value_front(qm, vs)
            last_front = fr
            if fr >= L and front_at < 0:
                front_at = t
                break
    elapsed = time.time() - t0
    solved = front_at > 0
    t_arr = front_at if solved else max_updates
    log(f"  {kind} enc={enc} knob={K_or_tau} reset={reset} seed={seed} front={last_front} T={t_arr} solved={solved} {elapsed:.1f}s")
    return dict(kind=kind, enc=enc, knob=K_or_tau, reset=reset, seed=seed, L=L,
                T=t_arr, solved=int(solved), front=last_front, seconds=round(elapsed, 2))


def run_ppo(L: int, lam: float, Tseg: int, seed: int, max_iter: int, log):
    """Neural PPO on the same corridor. Onehot features. Start every episode at 0.
    Tseg is the bootstrap horizon (rollout is cut into segments of this length)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = "onehot"
    net = ActorCritic(L).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    vs = vstar(L, DEVICE)
    n_ep = 24
    cap = max(Tseg, min(L * L, 320))
    front_at = -1
    t0 = time.time()
    last_front = 0
    for it in range(1, max_iter + 1):
        # rollouts
        S = np.zeros((n_ep, cap), np.int64)
        A = np.zeros((n_ep, cap), np.int64)
        R = np.zeros((n_ep, cap), np.float32)
        D = np.zeros((n_ep, cap), np.float32)
        M = np.zeros((n_ep, cap), np.float32)
        logp_old = np.zeros((n_ep, cap), np.float32)
        s = np.zeros(n_ep, np.int64)
        alive = np.ones(n_ep, bool)
        for k in range(cap):
            S[:, k] = s
            M[:, k] = alive
            with torch.no_grad():
                logits, _ = net(features(torch.tensor(s, device=DEVICE), enc, L))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample().cpu().numpy()
                logp_old[:, k] = dist.log_prob(torch.tensor(a, device=DEVICE)).cpu().numpy()
            A[:, k] = a
            for i in range(n_ep):
                if not alive[i]:
                    continue
                sp, r, done = step_env(int(s[i]), int(a[i]), L)
                R[i, k] = r
                D[i, k] = float(done)
                s[i] = 0 if done else sp
                alive[i] = not done
        # GAE. A segment boundary bootstraps from V and does not carry the advantage across.
        with torch.no_grad():
            flat_s = torch.tensor(S.reshape(-1), device=DEVICE)
            _, v_all = net(features(flat_s, enc, L))
            V = v_all.view(n_ep, cap).cpu().numpy()
            s_last = torch.tensor(s, device=DEVICE)
            _, v_last = net(features(s_last, enc, L))
            v_boot = v_last.cpu().numpy() * alive.astype(np.float32)
        adv = np.zeros_like(R)
        last_gae = np.zeros(n_ep, np.float32)
        for k in range(cap - 1, -1, -1):
            if k == cap - 1 or (k + 1) % Tseg == 0:
                last_gae = np.zeros(n_ep, np.float32)
            v_next = V[:, k + 1] if k + 1 < cap else v_boot
            nonterm = (1.0 - D[:, k]) * M[:, k]
            delta = (R[:, k] + GAMMA * v_next * (1.0 - D[:, k]) - V[:, k]) * M[:, k]
            last_gae = delta + GAMMA * lam * nonterm * last_gae
            adv[:, k] = last_gae
        ret = adv + V
        mask = M.reshape(-1) > 0
        sb = torch.tensor(S.reshape(-1)[mask], device=DEVICE)
        ab = torch.tensor(A.reshape(-1)[mask], device=DEVICE)
        rb = torch.tensor(ret.reshape(-1)[mask], device=DEVICE)
        advb = torch.tensor(adv.reshape(-1)[mask], device=DEVICE)
        old = torch.tensor(logp_old.reshape(-1)[mask], device=DEVICE)
        advb = (advb - advb.mean()) / (advb.std() + 1e-8)
        xb = features(sb, enc, L)
        for _ in range(4):
            logits, v = net(xb)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(ab)
            ratio = torch.exp(logp - old)
            clip = torch.clamp(ratio, 0.8, 1.2)
            pol = -torch.min(ratio * advb, clip * advb).mean()
            val = F.mse_loss(v, rb)
            ent = dist.entropy().mean()
            loss = pol + 0.5 * val - 0.01 * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
        if it % 2 == 0 or it == max_iter:
            with torch.no_grad():
                _, vgrid = net(features(torch.arange(L, device=DEVICE), enc, L))
                fr = value_front(vgrid, vs)
            last_front = fr
            if it % 5 == 0:
                log(f"    ppo it={it} reward={float(R.sum()):.1f} V0={float(vgrid[0]):.3f} front={fr}")
            if fr >= L and front_at < 0:
                front_at = it
                break
    elapsed = time.time() - t0
    solved = front_at > 0
    t_arr = front_at if solved else max_iter
    log(f"  ppo lam={lam} T={Tseg} seed={seed} L={L} front={last_front} iters={t_arr} solved={solved} {elapsed:.1f}s")
    return dict(kind="ppo", enc="onehot", knob=lam, reset=f"T{Tseg}", seed=seed, L=L,
                T=t_arr, solved=int(solved), front=last_front, seconds=round(elapsed, 2))


def fit_eta(rows, kind, enc, reset, L):
    sel = [r for r in rows if r["kind"] == kind and r["enc"] == enc and r["reset"] == reset and r["solved"]]
    knobs = sorted(set(r["knob"] for r in sel))
    if len(knobs) < 2:
        return None
    xs, ys = [], []
    for k in knobs:
        cell = [r["T"] for r in sel if r["knob"] == k]
        if not cell:
            return None
        xs.append(float(k))
        ys.append(float(np.median(cell)))
    x = np.array(xs)
    y = np.array(ys)
    if kind == "dqn":
        # T ≈ slope * K + intercept; hops per sync = L / slope
        X = np.column_stack([x, np.ones_like(x)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ b
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        eta = L / b[0] if b[0] > 1e-9 else float("inf")
        return dict(kind=kind, enc=enc, reset=reset, slope=b[0], intercept=b[1], r2=r2, eta=eta, n_knobs=len(knobs))
    if kind == "polyak":
        # T ≈ slope * (1/tau) + intercept; slope should be near L-1
        inv = 1.0 / x
        X = np.column_stack([inv, np.ones_like(inv)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ b
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        return dict(kind=kind, enc=enc, reset=reset, slope=b[0], intercept=b[1], r2=r2, eta=(L - 1) / b[0] if b[0] > 1e-9 else float("inf"), n_knobs=len(knobs))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--which", default="all", choices=["all", "dqn", "polyak", "ppo"])
    args = ap.parse_args()
    print(f"device={DEVICE} torch={torch.__version__}", flush=True)
    if DEVICE.type != "cuda":
        raise SystemExit("CUDA is not available; this script is the 3080 measurement.")
    log = lambda m: print(m, flush=True)
    rows = []
    if args.smoke:
        L, seeds, max_u, max_it = 20, 1, 4000, 40
        ks, taus, lams, tsegs = [40], [0.05], [1.0, 0.9], [16]
        resets = ["random"]
        encs = ["onehot"]
        ppo_L = 12
    else:
        L, seeds, max_u, max_it = 40, 4, 80000, 60
        ks, taus, lams, tsegs = [50, 200, 800], [0.05, 0.01, 0.002], [1.0, 0.95, 0.8], [16, 64]
        resets = ["random", "start"]
        encs = ["onehot", "coord"]
        ppo_L = 16
    do_q = args.which in ("all", "dqn")
    do_p = args.which in ("all", "polyak")
    do_ppo = args.which in ("all", "ppo")
    if do_q:
        for enc in encs:
            for reset in resets:
                for K in ks:
                    for sd in range(seeds):
                        rows.append(run_q("dqn", L, enc, K, 1000 + sd, max_u, reset, log))
                        save_csv(OUT / "runs.csv", rows)
    if do_p:
        for enc in encs:
            for tau in taus:
                for sd in range(seeds):
                    rows.append(run_q("polyak", L, enc, tau, 2000 + sd, max_u, "random", log))
                    save_csv(OUT / "runs.csv", rows)
    if do_ppo:
        for lam in lams:
            for Tseg in tsegs:
                for sd in range(seeds):
                    rows.append(run_ppo(ppo_L, lam, Tseg, 3000 + sd, max_it, log))
                    save_csv(OUT / "runs.csv", rows)
    fits = []
    if do_q:
        for enc in encs:
            for reset in resets:
                f = fit_eta(rows, "dqn", enc, reset, L)
                if f:
                    fits.append(f)
                    log(f"FIT dqn {enc} {reset}: slope={f['slope']:.2f} eta={f['eta']:.3f} R2={f['r2']:.4f} intercept={f['intercept']:.0f}")
    if do_p:
        for enc in encs:
            f = fit_eta(rows, "polyak", enc, "random", L)
            if f:
                fits.append(f)
                log(f"FIT polyak {enc}: slope={f['slope']:.2f} (want ~{L-1}) eta={f['eta']:.3f} R2={f['r2']:.4f} intercept={f['intercept']:.0f}")
    if fits:
        save_csv(OUT / "fits.csv", [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()} for f in fits])
    # PPO ordering, written as its own summary
    ppo_rows = [r for r in rows if r["kind"] == "ppo"]
    if ppo_rows:
        summ = []
        for lam in sorted(set(r["knob"] for r in ppo_rows)):
            for reset in sorted(set(r["reset"] for r in ppo_rows)):
                cell = [r for r in ppo_rows if r["knob"] == lam and r["reset"] == reset]
                solved = [r["T"] for r in cell if r["solved"]]
                summ.append(dict(lam=lam, reset=reset, n=len(cell), solved=len(solved),
                                  T_med=float(np.median(solved)) if solved else -1))
                log(f"PPO lam={lam} {reset}: solved {len(solved)}/{len(cell)} median iters {summ[-1]['T_med']}")
        save_csv(OUT / "ppo_summary.csv", summ)
    print("done", flush=True)


if __name__ == "__main__":
    main()
