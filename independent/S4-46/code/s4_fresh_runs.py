"""Fresh (non-bootstrap) seed runs for the anytime-testing paper.

Three cheap agents on four cheap environments, many seeds each, so that paired seed differences
can be formed from genuinely independent runs rather than bootstrap resamples:

  agents : dqn (Adam 1e-3, Polyak target 0.005), ddqn (double DQN, same), dqn_lr3e-4 (lr 3e-4)
  envs   : CartPole-v1 (30k steps), Acrobot-v1 (50k), MinAtar breakout (100k), MinAtar space_invaders (100k)
  seeds  : dqn 60, ddqn 40, dqn_lr3e-4 40 per classic-control environment; 40/30/30 on MinAtar

Score = mean return of 10 evaluation episodes (epsilon 0.05) at the end of training. Normalized to [0, 1]
by construction: CartPole R/500, Acrobot (R+500)/500, MinAtar min(R/100, 1) with a 1000-step episode cap.

    python code/s4_fresh_runs.py [--quick] [--workers 8] [--envs ...]

Writes data/fresh/fresh_runs.csv (one row per run).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

ENV_STEPS = {"cartpole": 30000, "acrobot": 50000, "breakout": 100000, "space_invaders": 100000}
AGENT_SEEDS = {"dqn": 60, "ddqn": 40, "dqn_lr3e-4": 40}
MINATAR_SEEDS = {"dqn": 40, "ddqn": 30, "dqn_lr3e-4": 30}
MINATAR = ("breakout", "space_invaders", "asterix", "freeway", "seaquest")


class GymEnv:
    def __init__(self, name: str, seed: int):
        import gymnasium as gym

        self.env = gym.make(name)
        self.seed = seed
        self.count = 0
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.n_actions = int(self.env.action_space.n)
        self.scale = {"CartPole-v1": np.array([2.4, 3.0, 0.21, 3.0]), "Acrobot-v1": np.array([1, 1, 1, 1, 12.57, 28.27])}[name].astype(np.float32)

    def reset(self):
        obs, _ = self.env.reset(seed=int(self.seed * 100003 + self.count))
        self.count += 1
        return (np.asarray(obs, dtype=np.float32) / self.scale).astype(np.float32)

    def step(self, a: int):
        obs, r, term, trunc, _ = self.env.step(int(a))
        return (np.asarray(obs, dtype=np.float32) / self.scale).astype(np.float32), float(r), bool(term), bool(trunc)


class MinAtarEnv:
    def __init__(self, game: str, seed: int, cap: int = 1000):
        from minatar import Environment

        self.env = Environment(game, sticky_action_prob=0.0)
        self.env.seed(int(seed))
        self.actions = list(self.env.minimal_action_set())
        self.n_actions = len(self.actions)
        self.obs_dim = int(np.prod(self.env.state_shape()))
        self.cap = cap
        self.t = 0

    def reset(self):
        self.env.reset()
        self.t = 0
        return self.env.state().astype(np.float32).reshape(-1)

    def step(self, a: int):
        r, term = self.env.act(self.actions[int(a)])
        self.t += 1
        trunc = self.t >= self.cap and not term
        return self.env.state().astype(np.float32).reshape(-1), float(min(r, 1.0)), bool(term), bool(trunc)


def make_env(name: str, seed: int):
    if name == "cartpole":
        return GymEnv("CartPole-v1", seed)
    if name == "acrobot":
        return GymEnv("Acrobot-v1", seed)
    if name in MINATAR:
        return MinAtarEnv(name, seed)
    raise ValueError(name)


def normalize(env_name: str, ret: float) -> float:
    if env_name == "cartpole":
        return ret / 500.0
    if env_name == "acrobot":
        return (ret + 500.0) / 500.0
    return min(ret / 100.0, 1.0)


def train(env_name: str, agent: str, seed: int, steps: int) -> dict:
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(env_name, seed)
    lr = 3e-4 if agent == "dqn_lr3e-4" else 1e-3
    double = agent == "ddqn"
    gamma, tau, batch_size, buf_cap, warm = 0.99, 0.005, 64, 50000, 1000
    hidden = 128 if env_name in MINATAR else 64

    def mlp():
        return nn.Sequential(nn.Linear(env.obs_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, env.n_actions))

    q, q_targ = mlp(), mlp()
    q_targ.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf_s = np.zeros((buf_cap, env.obs_dim), dtype=np.float32)
    buf_a = np.zeros(buf_cap, dtype=np.int64)
    buf_r = np.zeros(buf_cap, dtype=np.float32)
    buf_sp = np.zeros((buf_cap, env.obs_dim), dtype=np.float32)
    buf_d = np.zeros(buf_cap, dtype=np.float32)
    n, ptr = 0, 0
    s = env.reset()
    ep_ret, train_returns = 0.0, []
    for t in range(steps):
        eps = max(0.05, 1.0 - t / (0.4 * steps))
        if rng.random() < eps:
            a = int(rng.integers(0, env.n_actions))
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(s)[None]).argmax().item())
        sp, r, term, trunc = env.step(a)
        ep_ret += r
        buf_s[ptr], buf_a[ptr], buf_r[ptr], buf_sp[ptr], buf_d[ptr] = s, a, r, sp, float(term)
        ptr = (ptr + 1) % buf_cap
        n = min(n + 1, buf_cap)
        if term or trunc:
            train_returns.append(ep_ret)
            ep_ret = 0.0
            s = env.reset()
        else:
            s = sp
        if n < warm:
            continue
        idx = rng.integers(0, n, size=batch_size)
        bs, ba, br, bsp, bd = (torch.from_numpy(buf_s[idx]), torch.from_numpy(buf_a[idx]), torch.from_numpy(buf_r[idx]), torch.from_numpy(buf_sp[idx]), torch.from_numpy(buf_d[idx]))
        with torch.no_grad():
            if double:
                a_star = q(bsp).argmax(dim=1, keepdim=True)
                boot = q_targ(bsp).gather(1, a_star).squeeze(1)
            else:
                boot = q_targ(bsp).max(dim=1).values
            target = br + gamma * (1 - bd) * boot
        pred = q(bs).gather(1, ba[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), q_targ.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
    # Evaluation: 10 episodes, epsilon 0.05, fresh env instance seeded differently from training.
    eval_env = make_env(env_name, seed + 10_000)
    eval_returns = []
    for _ in range(10):
        s = eval_env.reset()
        done, ret = False, 0.0
        while not done:
            if rng.random() < 0.05:
                a = int(rng.integers(0, eval_env.n_actions))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s)[None]).argmax().item())
            s, r, term, trunc = eval_env.step(a)
            ret += r
            done = term or trunc
        eval_returns.append(ret)
    ret = float(np.mean(eval_returns))
    return {
        "env": env_name,
        "agent": agent,
        "seed": seed,
        "steps": steps,
        "eval_return": ret,
        "score": normalize(env_name, ret),
        "last10_train_return": float(np.mean(train_returns[-10:])) if train_returns else float("nan"),
    }


def _job(args):
    env_name, agent, seed, steps = args
    t0 = time.time()
    try:
        out = train(env_name, agent, seed, steps)
    except Exception as exc:
        out = {"env": env_name, "agent": agent, "seed": seed, "steps": steps, "eval_return": float("nan"), "score": float("nan"), "last10_train_return": float("nan"), "error": repr(exc)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs", default="cartpole,acrobot,breakout,space_invaders")
    args = ap.parse_args()
    envs = args.envs.split(",")
    jobs = []
    for env_name in envs:
        steps = ENV_STEPS[env_name] // (10 if args.quick else 1)
        seeds_table = MINATAR_SEEDS if env_name in MINATAR else AGENT_SEEDS
        for agent, n_seeds in seeds_table.items():
            for seed in range(2 if args.quick else n_seeds):
                jobs.append((env_name, agent, seed, steps))
    out_dir = PROJECT / "data" / "fresh"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("fresh_runs_quick.csv" if args.quick else "fresh_runs.csv")
    fields = ["env", "agent", "seed", "steps", "eval_return", "score", "last10_train_return", "seconds", "error"]
    t0 = time.time()
    with out_path.open("w", newline="") as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, out in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            writer.writerow({**{k: "" for k in fields}, **out})
            f.flush()
            print(f"[{i + 1}/{len(jobs)}] {out['env']:14s} {out['agent']:11s} seed {out['seed']:3d}: return {out['eval_return']:8.1f} score {out['score']:.3f} ({out['seconds']}s) {out.get('error', '')}", flush=True)
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
