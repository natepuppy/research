"""Deep diagnostic on more environments: is DQN divergence without a target network an outward field
(positive symmetric mode, lr x sharpness << 2) or step-size overshoot (lr x sharpness ~ 2)?

Same measurements as the CartPole experiment in n1_extensions.py, on gymnasium Acrobot-v1 and
MountainCar-v0 and on MinAtar breakout / space_invaders (episode cap 1000 steps). Plain SGD.

    python code/n1_deep_envs.py [--quick] [--seeds 8] [--workers 10] [--envs acrobot,mountaincar,breakout,space_invaders]

Writes results/deep_envs_runs.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # code/
PROJECT = HERE.parent

import n1_extensions as n1  # noqa: E402


class GymEnv:
    """gymnasium classic-control wrapper. Observations scaled by a fixed vector."""

    def __init__(self, name: str, seed: int):
        import gymnasium as gym

        self.env = gym.make(name)
        self.seed = seed
        high = np.abs(self.env.observation_space.high).astype(np.float64)
        high = np.where(np.isfinite(high) & (high < 1e6), high, 1.0)
        if name.startswith("Acrobot"):
            high = np.array([1.0, 1.0, 1.0, 1.0, 12.57, 28.27])
        if name.startswith("MountainCar"):
            high = np.array([1.2, 0.07])
        self.scale = high
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.n_actions = int(self.env.action_space.n)
        self.count = 0

    def reset(self, rng):
        obs, _ = self.env.reset(seed=int(self.seed * 100003 + self.count))
        self.count += 1
        return np.asarray(obs, dtype=np.float64) / self.scale

    def step(self, a: int):
        obs, r, term, trunc, _ = self.env.step(int(a))
        return np.asarray(obs, dtype=np.float64) / self.scale, float(r), bool(term), bool(trunc)


class MinAtarEnv:
    """MinAtar wrapper with a 1000-step episode cap (so returns are bounded by construction)."""

    def __init__(self, game: str, seed: int, cap: int = 1000):
        from minatar import Environment

        self.env = Environment(game, sticky_action_prob=0.0)
        self.env.seed(int(seed))
        self.actions = list(self.env.minimal_action_set())
        self.n_actions = len(self.actions)
        self.obs_dim = int(np.prod(self.env.state_shape()))
        self.scale = 1.0
        self.cap = cap
        self.t = 0

    def reset(self, rng):
        self.env.reset()
        self.t = 0
        return self.env.state().astype(np.float64).reshape(-1)

    def step(self, a: int):
        r, term = self.env.act(self.actions[int(a)])
        self.t += 1
        trunc = self.t >= self.cap and not term
        return self.env.state().astype(np.float64).reshape(-1), float(r), bool(term), bool(trunc)


def make_env(name: str, seed: int):
    if name == "acrobot":
        return GymEnv("Acrobot-v1", seed)
    if name == "mountaincar":
        return GymEnv("MountainCar-v0", seed)
    if name in ("breakout", "space_invaders", "asterix", "freeway", "seaquest"):
        return MinAtarEnv(name, seed)
    raise ValueError(name)


def diagnostic_run(env_name: str, seed: int, lr: float, tau: float, damp_k: int, steps: int, gamma: float = 0.99, margin: float = 0.05, measure_every: int = 100, hidden: int = 64) -> dict:
    import torch

    from lib.nets import FuncMLP

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(env_name, seed)
    net = FuncMLP([env.obs_dim, hidden, hidden, env.n_actions], torch.device("cpu"), seed=seed)
    theta = net.theta.detach().clone()
    target = theta.clone()
    cap = 20000
    buf_s = np.zeros((cap, env.obs_dim), dtype=np.float32)
    buf_a = np.zeros(cap, dtype=np.int64)
    buf_r = np.zeros(cap, dtype=np.float32)
    buf_sp = np.zeros((cap, env.obs_dim), dtype=np.float32)
    buf_d = np.zeros(cap, dtype=np.float32)
    n, ptr = 0, 0
    s = env.reset(rng)
    ep_ret, returns = 0.0, []
    modes = [torch.randn(theta.numel()) for _ in range(max(damp_k, 1))]
    lams = [0.0] * max(damp_k, 1)
    sigma_vec = torch.randn(theta.numel())
    log = []
    diverged_at = None

    def make_batch(idx):
        return (
            torch.tensor(buf_s[idx]),
            torch.tensor(buf_a[idx]),
            torch.tensor(buf_r[idx]),
            torch.tensor(buf_sp[idx]),
            torch.tensor(buf_d[idx]),
        )

    for t in range(steps):
        eps = max(0.05, 1.0 - t / (0.5 * steps))
        if rng.random() < eps:
            a = int(rng.integers(0, env.n_actions))
        else:
            with torch.no_grad():
                q = net.forward(theta, torch.tensor(s, dtype=torch.float32)[None])
            a = int(q.argmax().item())
        sp, r, done, trunc = env.step(a)
        ep_ret += r
        buf_s[ptr], buf_a[ptr], buf_r[ptr], buf_sp[ptr], buf_d[ptr] = s, a, r, sp, float(done)
        ptr = (ptr + 1) % cap
        n = min(n + 1, cap)
        if done or trunc:
            returns.append(ep_ret)
            ep_ret = 0.0
            s = env.reset(rng)
        else:
            s = sp
        if n < 256:
            continue
        batch = make_batch(rng.integers(0, n, size=64))
        boot_theta = theta if tau >= 1.0 else target
        leaf = theta.detach().requires_grad_(True)
        pred = net.forward(leaf, batch[0]).gather(1, batch[1][:, None]).squeeze(1)
        with torch.no_grad():
            boot = net.forward(boot_theta, batch[3]).max(dim=1).values
            backup = batch[2] + gamma * (1 - batch[4]) * boot
        loss = (pred - backup).pow(2).mean()
        (grad,) = torch.autograd.grad(loss, leaf)
        grad = grad.detach()
        if damp_k > 0:
            new_modes, new_lams, sigma_vec = n1.unstable_modes_from_loss_jacobian(
                n1._cartpole_apply_factory(net, theta, batch, gamma), theta.numel(), damp_k, iters=1, warm=modes, sigma_warm=sigma_vec
            )
            damp = torch.zeros_like(theta)
            for i in range(damp_k):
                modes[i] = new_modes[i]
                lams[i] = 0.8 * lams[i] + 0.2 * new_lams[i]
                if lams[i] > 0:
                    damp = damp + (lams[i] + margin) * modes[i] * (modes[i] @ theta)
            grad = grad + damp
        theta = (theta - lr * grad).detach()
        target = ((1 - tau) * target + tau * theta).detach()
        if t % measure_every == 0:
            with torch.no_grad():
                q_abs = float(net.forward(theta, batch[0]).abs().mean())
            big = make_batch(rng.integers(0, n, size=256))
            _, lam_list, _ = n1.unstable_modes_from_loss_jacobian(n1._cartpole_apply_factory(net, theta, big, gamma), theta.numel(), 1, iters=10)
            sharpness = n1.unstable_modes_from_loss_jacobian.last_sharpness
            cf = n1._cartpole_curl_fraction(net, theta, big, gamma)
            bad = (not np.isfinite(q_abs)) or q_abs > 1e4 or not torch.isfinite(theta).all()
            log.append(
                {
                    "step": t,
                    "q_abs": q_abs,
                    "lam_max_S": lam_list[0],
                    "sharpness": sharpness,
                    "lr_times_sharpness": lr * sharpness,
                    "curl_fraction": cf,
                    "theta_norm": float(theta.norm()),
                    "recent_return": float(np.mean(returns[-10:])) if returns else 0.0,
                    "tracked_lam": float(lams[0]) if damp_k else None,
                }
            )
            if bad:
                diverged_at = t
                break
    return {
        "env": env_name,
        "seed": seed,
        "lr": lr,
        "tau": tau,
        "damp_k": damp_k,
        "gamma": gamma,
        "diverged_at": diverged_at,
        "log": log,
        "final_return": float(np.mean(returns[-10:])) if returns else 0.0,
        "n_params": int(theta.numel()),
    }


CONFIGS = [
    {"lr": 1e-3, "tau": 1.0, "damp_k": 0, "gamma": 0.99},
    {"lr": 3e-3, "tau": 1.0, "damp_k": 0, "gamma": 0.99},
    {"lr": 3e-3, "tau": 0.005, "damp_k": 0, "gamma": 0.99},
    {"lr": 3e-3, "tau": 1.0, "damp_k": 0, "gamma": 0.9},
    {"lr": 3e-3, "tau": 1.0, "damp_k": 2, "gamma": 0.99},
]


def _job(args):
    env_name, seed, cfg, steps = args
    try:
        return diagnostic_run(env_name, seed, steps=steps, **cfg)
    except Exception as exc:  # keep the sweep alive; record the failure
        return {"env": env_name, "seed": seed, **cfg, "error": repr(exc), "log": [], "diverged_at": None, "final_return": float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--envs", default="acrobot,mountaincar,breakout,space_invaders")
    args = ap.parse_args()
    steps = 3000 if args.quick else 10000
    envs = args.envs.split(",")
    # MinAtar rewards are sparse and Q is small; plain SGD needs a 10x larger step to move at all in 10k steps.
    def configs_for(env_name: str):
        mult = 10.0 if env_name in ("breakout", "space_invaders", "asterix", "freeway", "seaquest") else 1.0
        return [{**cfg, "lr": cfg["lr"] * mult} for cfg in CONFIGS]

    jobs = [(e, s, cfg, steps) for e in envs for cfg in configs_for(e) for s in range(args.seeds)]
    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for out in pool.imap_unordered(_job, jobs, chunksize=1):
            results.append(out)
            last = out["log"][-1] if out.get("log") else {}
            print(
                f"{out['env']:14s} lr {out['lr']:g} tau {out['tau']:g} gamma {out['gamma']:g} k {out['damp_k']} seed {out['seed']}: "
                f"div {out['diverged_at']} q_abs {last.get('q_abs', float('nan')):.2f} lr*sh {last.get('lr_times_sharpness', float('nan')):.3f} "
                f"lam {last.get('lam_max_S', float('nan')):.3f} ret {out['final_return']:.1f} {out.get('error', '')}",
                flush=True,
            )
    (PROJECT / "results").mkdir(exist_ok=True)
    out_path = PROJECT / "results" / ("deep_envs_runs_quick.json" if args.quick else "deep_envs_runs.json")
    out_path.write_text(json.dumps({"steps": steps, "runs": results}))
    print(f"wrote {out_path} ({len(results)} runs, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
