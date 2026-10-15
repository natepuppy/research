"""T3. What does a reset reset? Trainability under prolonged GRPO, with a positive control.

Pilot version (Testbed A-lite). A tiny transformer (lib/tiny_lm.py) is
pretrained on task family F0, then trained with exact GRPO (G = 8, KL to a
reference policy) through a curriculum of new families F1..F5, one phase
each. Three competing accounts of why prolonged RLVR stalls, and what each
predicts a reset should do:

  (a) coverage / KL anchor   : only the reference reset or temperature
                               re-inflation helps; trainability probes are flat
  (b) optimizer-state staleness : an optimizer-only reset recovers most of the
                               benefit of ProRL's full reset; probes with a
                               *carried* Adam state degrade, fresh-Adam probes do not
  (c) representational plasticity loss : fresh-Adam probes degrade across
                               phases; Shrink & Perturb on the weights is needed

Regimes
  rl_curriculum : each family has its own prefix token (new tasks accumulate)
  rl_relabel    : no family token, so each phase overwrites the previous
                  answers (the classic plasticity-loss inducer, but under GRPO)
  sup_relabel   : the same overwriting curriculum trained with cross-entropy
                  (positive control: this is where plasticity loss is known
                  to occur, so the probes must detect it here)

Interventions at every phase boundary (factorial over regimes)
  none, adam_reset, ref_reset, both (ProRL), shrink_perturb, temp_reinflate

Probes at every phase boundary, on a copy of the model:
  rl probe   : 100 GRPO steps on a held-out family, fresh Adam
  rl probe (carried Adam) : same, but starting from the run's current Adam state
  sup probe  : 100 cross-entropy steps on a second held-out family, fresh Adam
Network metrics at every boundary: entropy, relative update norm and
sparsity vs base, effective rank of per-example logit gradients (eNTK
proxy), optimization readiness (gradient norm x cross-half cosine), Adam
second-moment staleness (mean g^2 / mean v).

Kill test (hypothesis: no representational loss under RLVR, but stale Adam
matters). SUPPORTED if fresh-Adam probes stay within 10% of base in both RL
regimes and adam_reset recovers >= 50% of the late-phase gain that `both`
recovers over `none`. KILLED if fresh-Adam probes fall > 20% in an RL regime,
or adam_reset recovers < 25% while ref_reset recovers most. The positive
control must show a probe decline, otherwise the probes are not sensitive
and the verdict is INCONCLUSIVE.

    python code/t3_resets.py --smoke
    python code/t3_resets.py
    python code/t3_resets.py --full
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.tiny_lm import TinyLM, entropy, grpo_loss, make_family, make_prompts, success_probs  # noqa: E402

CONTENT_VOCAB, PROMPT_LEN, N_ANSWERS = 32, 8, 16
N_FAMILIES = 8  # F0 base, F1..F5 curriculum, H_rl, H_sup
VOCAB = CONTENT_VOCAB + N_FAMILIES
INTERVENTIONS = ["none", "adam_reset", "ref_reset", "both", "shrink_perturb", "temp_reinflate"]
G, BATCH, LR = 8, 64, 1e-3
# KL weight relative to the group-mean GRPO loss. The pilot results were produced with the
# earlier group-sum normalisation and beta = 0.01, which is the same optimisation problem
# under Adam (scale-invariant) as group-mean with beta = 0.01 / G.
BETA = 0.01 / G


def with_token(prompts: torch.Tensor, family_id: int, use_token: bool) -> torch.Tensor:
    tok = CONTENT_VOCAB + (family_id if use_token else 0)
    return torch.cat([torch.full((prompts.shape[0], 1), tok, dtype=prompts.dtype), prompts], 1)


class World:
    def __init__(self, seed: int, n_prompts: int, device):
        self.device = device
        self.families = [make_family(PROMPT_LEN, N_ANSWERS, CONTENT_VOCAB, seed=seed * 100 + f) for f in range(N_FAMILIES)]
        self.prompts = make_prompts(n_prompts, PROMPT_LEN, CONTENT_VOCAB, seed=seed * 100 + 50)
        self.probe_prompts = make_prompts(512, PROMPT_LEN, CONTENT_VOCAB, seed=seed * 100 + 51)

    def data(self, family_id: int, use_token: bool, probe: bool = False):
        base = self.probe_prompts if probe else self.prompts
        return with_token(base, family_id, use_token).to(self.device), self.families[family_id].answers(base).to(self.device)


def fresh_model(seed: int, device) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(VOCAB, PROMPT_LEN + 1, N_ANSWERS).to(device)


def supervised_steps(model, opt, prompts, answers, steps, gen, batch=BATCH) -> None:
    n = prompts.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        loss = F.cross_entropy(model(prompts[idx]), answers[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def grpo_steps(model, opt, prompts, answers, steps, gen, ref=None, beta=0.0, temperature=1.0, batch=BATCH) -> list[float]:
    n = prompts.shape[0]
    accs = []
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        ref_logits = None
        if ref is not None and beta > 0:
            with torch.no_grad():
                ref_logits = ref(prompts[idx])
        loss, info = grpo_loss(model, prompts[idx], answers[idx], torch.full((batch,), G, dtype=torch.long), ref_logits, beta, temperature, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        accs.append(info["batch_acc"])
    return accs


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


@torch.no_grad()
def adam_staleness(model, opt, prompts, answers) -> tuple[float, float]:
    """mean g^2 / mean v over parameters with Adam state, and the fraction of v below 1e-12."""
    with torch.enable_grad():
        loss = F.cross_entropy(model(prompts[:256]), answers[:256])
        grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    g2, v, dead, total = 0.0, 0.0, 0, 0
    for p, g in zip(model.parameters(), grads):
        state = opt.state.get(p, {})
        if "exp_avg_sq" not in state or g is None:
            continue
        g2 += float(g.square().sum())
        v += float(state["exp_avg_sq"].sum())
        dead += int((state["exp_avg_sq"] < 1e-12).sum())
        total += state["exp_avg_sq"].numel()
    return (g2 / max(v, 1e-30), dead / max(total, 1))


def network_metrics(model, base_theta: torch.Tensor, prompts, answers, opt) -> dict:
    theta = flat_params(model)
    delta = theta - base_theta
    scale = base_theta.square().mean().sqrt()
    params = {k: v.detach() for k, v in model.named_parameters()}

    def correct_logit(p, x, y_onehot):
        out = functional_call(model, p, (x[None],))
        return (out[0] * y_onehot).sum()

    xs, ys = prompts[:48], F.one_hot(answers[:48], N_ANSWERS).float()
    per_example = vmap(grad(correct_logit), in_dims=(None, 0, 0))(params, xs, ys)
    Gmat = torch.cat([g.reshape(48, -1) for g in per_example.values()], 1)
    s = torch.linalg.svdvals(Gmat).clamp_min(1e-12)
    p = s.square() / s.square().sum()
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    # readiness: gradient strength x reliability across two halves of a CE batch
    def ce_grad(sl):
        loss = F.cross_entropy(model(prompts[sl]), answers[sl])
        gs = torch.autograd.grad(loss, list(model.parameters()))
        return torch.cat([g.reshape(-1) for g in gs])

    g1, g2 = ce_grad(slice(0, 128)), ce_grad(slice(128, 256))
    g_full = 0.5 * (g1 + g2)
    cosine = float(F.cosine_similarity(g1, g2, dim=0))
    stale, dead_v = adam_staleness(model, opt, prompts, answers)
    return {
        "entropy": entropy(model, prompts),
        "rel_update_norm": float(delta.norm() / base_theta.norm()),
        "update_sparsity": float((delta.abs() < 1e-3 * scale).float().mean()),
        "entk_eff_rank": eff_rank,
        "readiness": float(g_full.norm()) * cosine,
        "grad_cosine": cosine,
        "adam_g2_over_v": stale,
        "adam_dead_v_frac": dead_v,
    }


def probe(model, world, use_token, steps, seed, kind: str, carried_opt=None) -> float:
    """Fresh-copy trainability probe. Returns the gain in mean success probability on the held-out family.

    Trainability means how fast the network fits new targets, so the probe is
    scored on the prompts it trains on (the families are random hashes; there
    is nothing to generalise to within 100 steps).
    """
    fam = 6 if kind == "rl" else 7
    prompts, answers = world.data(fam, use_token, probe=True)
    eval_prompts, eval_answers = prompts, answers
    m = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    before = float(success_probs(m, eval_prompts, eval_answers).mean())
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    if carried_opt is not None:
        state = copy.deepcopy(carried_opt.state_dict())
        opt.load_state_dict(state)
    if kind == "rl":
        grpo_steps(m, opt, prompts, answers, steps, gen)
    else:
        supervised_steps(m, opt, prompts, answers, steps, gen)
    after = float(success_probs(m, eval_prompts, eval_answers).mean())
    return after - before


def run(regime: str, intervention: str, seed: int, n_prompts: int, pre_steps: int, phase_steps: int, probe_steps: int, device) -> tuple[list[dict], list[dict]]:
    world = World(seed, n_prompts, device)
    use_token = regime == "rl_curriculum"
    model = fresh_model(seed, device)
    gen = torch.Generator().manual_seed(seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    p0, a0 = world.data(0, use_token)
    supervised_steps(model, opt, p0, a0, pre_steps, gen)
    base_theta = flat_params(model)
    ref = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)  # RL starts with a fresh optimizer, as in practice
    checkpoints, phases = [], []

    def checkpoint(phase: int):
        row = {"regime": regime, "intervention": intervention, "seed": seed, "phase": phase}
        row.update(network_metrics(model, base_theta, p0, a0, opt))
        row["probe_rl_fresh"] = probe(model, world, use_token, probe_steps, seed, "rl")
        row["probe_rl_carried"] = probe(model, world, use_token, probe_steps, seed, "rl", carried_opt=opt)
        row["probe_sup_fresh"] = probe(model, world, use_token, probe_steps, seed, "sup")
        checkpoints.append(row)

    checkpoint(0)
    for phase in range(1, 6):
        prompts, answers = world.data(phase, use_token)
        if phase > 1:
            if intervention in ("adam_reset", "both"):
                opt = torch.optim.Adam(model.parameters(), lr=LR)
            if intervention in ("ref_reset", "both"):
                ref = copy.deepcopy(model).eval()
            if intervention == "shrink_perturb":
                fresh = fresh_model(seed + 1000 + phase, device)
                with torch.no_grad():
                    for p, q in zip(model.parameters(), fresh.parameters()):
                        p.mul_(0.8).add_(0.2 * q)
        before = float(success_probs(model, prompts, answers).mean())
        if regime == "sup_relabel":
            supervised_steps(model, opt, prompts, answers, phase_steps, gen)
        else:
            temp = 1.5 if intervention == "temp_reinflate" else 1.0
            warm = phase_steps // 3
            grpo_steps(model, opt, prompts, answers, warm, gen, ref, BETA, temp)
            grpo_steps(model, opt, prompts, answers, phase_steps - warm, gen, ref, BETA, 1.0)
        after = float(success_probs(model, prompts, answers).mean())
        phases.append({"regime": regime, "intervention": intervention, "seed": seed, "phase": phase, "p_before": before, "p_after": after, "gain": after - before})
        checkpoint(phase)
    return phases, checkpoints


def summarise(phases: list[dict], checkpoints: list[dict]) -> tuple[list[dict], str, bool | None]:
    def mean(rows, key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    table = []
    for regime in ("rl_curriculum", "rl_relabel", "sup_relabel"):
        for iv in INTERVENTIONS:
            ph = [r for r in phases if r["regime"] == regime and r["intervention"] == iv]
            if not ph:
                continue
            ck = [r for r in checkpoints if r["regime"] == regime and r["intervention"] == iv]
            base = [r for r in ck if r["phase"] == 0]
            late = [r for r in ck if r["phase"] >= 3]
            table.append({
                "regime": regime, "intervention": iv,
                "gain_phase1": mean([r for r in ph if r["phase"] == 1], "gain"),
                "gain_late": mean([r for r in ph if r["phase"] >= 3], "gain"),
                "probe_rl_fresh_base": mean(base, "probe_rl_fresh"), "probe_rl_fresh_late": mean(late, "probe_rl_fresh"),
                "probe_rl_carried_late": mean(late, "probe_rl_carried"),
                "probe_sup_fresh_base": mean(base, "probe_sup_fresh"), "probe_sup_fresh_late": mean(late, "probe_sup_fresh"),
                "entk_rank_base": mean(base, "entk_eff_rank"), "entk_rank_late": mean(late, "entk_eff_rank"),
                "readiness_base": mean(base, "readiness"), "readiness_late": mean(late, "readiness"),
                "adam_g2_over_v_late": mean(late, "adam_g2_over_v"),
                "rel_update_norm_late": mean(late, "rel_update_norm"),
                "entropy_late": mean(late, "entropy"),
            })

    def row(regime, iv):
        for r in table:
            if r["regime"] == regime and r["intervention"] == iv:
                return r
        return None

    notes, probe_drops, recoveries = [], [], []
    for regime in ("rl_curriculum", "rl_relabel"):
        none = row(regime, "none")
        if none is None:
            continue
        drop = 1.0 - none["probe_sup_fresh_late"] / none["probe_sup_fresh_base"] if none["probe_sup_fresh_base"] > 1e-6 else float("nan")
        probe_drops.append(drop)
        both, adam, refr = row(regime, "both"), row(regime, "adam_reset"), row(regime, "ref_reset")
        if both and adam and refr:
            benefit = both["gain_late"] - none["gain_late"]
            adam_rec = (adam["gain_late"] - none["gain_late"]) / benefit if abs(benefit) > 1e-3 else float("nan")
            ref_rec = (refr["gain_late"] - none["gain_late"]) / benefit if abs(benefit) > 1e-3 else float("nan")
            recoveries.append((regime, benefit, adam_rec, ref_rec))
            notes.append(f"{regime}: sup-probe change {-drop:+.0%}, late gain none {none['gain_late']:.3f} both {both['gain_late']:.3f}, adam-only recovers {adam_rec:.0%}, ref-only {ref_rec:.0%}")
    control = row("sup_relabel", "none")
    control_drop = float("nan")
    if control and control["probe_sup_fresh_base"] > 1e-6:
        control_drop = 1.0 - control["probe_sup_fresh_late"] / control["probe_sup_fresh_base"]
        notes.append(f"positive control (supervised relabel) sup-probe change {-control_drop:+.0%}")
    text = "; ".join(notes) + "."
    if not recoveries or not np.isfinite(control_drop):
        return table, text, None
    if control_drop < 0.1:
        return table, text + " Probes did not detect loss in the positive control, so they are not sensitive enough here.", None
    worst_drop = float(np.nanmax(probe_drops))
    adam_recs = [r[2] for r in recoveries if np.isfinite(r[2]) and r[1] > 0.02]
    ref_recs = [r[3] for r in recoveries if np.isfinite(r[3]) and r[1] > 0.02]
    if not adam_recs:
        return table, text + " Resets gave no late-phase benefit over none, so there is nothing to attribute (coverage/none account).", None
    if worst_drop <= 0.10 and float(np.mean(adam_recs)) >= 0.5:
        return table, text, True
    if worst_drop > 0.20 or (float(np.mean(adam_recs)) < 0.25 and ref_recs and float(np.mean(ref_recs)) > 0.5):
        return table, text, False
    return table, text, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--resume", action="store_true", help="skip (seed, regime, intervention) runs already in results/phases.csv")
    parser.add_argument("--summarize", action="store_true", help="only re-summarise existing results")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.smoke:
        n_prompts, pre_steps, phase_steps, probe_steps, seeds = 256, 50, 30, 10, [0]
        plan = [("rl_curriculum", "none"), ("rl_relabel", "both"), ("sup_relabel", "none")]
    elif args.full:
        n_prompts, pre_steps, phase_steps, probe_steps, seeds = 4096, 3000, 1000, 200, list(range(5))
        plan = [(r, iv) for r in ("rl_curriculum", "rl_relabel") for iv in INTERVENTIONS] + [("sup_relabel", "none"), ("sup_relabel", "shrink_perturb")]
    else:
        n_prompts, pre_steps, phase_steps, probe_steps, seeds = 2048, 1500, 300, 100, [0, 1, 2]
        plan = [(r, iv) for r in ("rl_curriculum", "rl_relabel") for iv in INTERVENTIONS] + [("sup_relabel", "none"), ("sup_relabel", "shrink_perturb")]
    phases, checkpoints = [], []
    if args.resume or args.summarize:
        phases, checkpoints = load_csv(args.out / "phases.csv"), load_csv(args.out / "checkpoints.csv")
    done = {(r["seed"], r["regime"], r["intervention"]) for r in phases}
    for seed in seeds:
        for regime, iv in plan:
            if args.summarize or (seed, regime, iv) in done:
                continue
            t0 = time.time()
            ph, ck = run(regime, iv, seed, n_prompts, pre_steps, phase_steps, probe_steps, device)
            phases.extend(ph)
            checkpoints.extend(ck)
            gains = " ".join(f"{r['gain']:+.2f}" for r in ph)
            probes = " ".join(f"{r['probe_sup_fresh']:.2f}" for r in ck)
            print(f"seed {seed} {regime:13s} {iv:14s} gains [{gains}] sup-probe [{probes}] rank {ck[0]['entk_eff_rank']:.1f}->{ck[-1]['entk_eff_rank']:.1f} ({time.time()-t0:.0f}s)", flush=True)
            save_csv(args.out / "phases.csv", phases)
            save_csv(args.out / "checkpoints.csv", checkpoints)
    table, text, supported = summarise(phases, checkpoints)
    verdict = kill_test(supported, text)
    for r in table:
        print(f"{r['regime']:13s} {r['intervention']:14s} gain p1 {r['gain_phase1']:+.3f} late {r['gain_late']:+.3f} | sup-probe {r['probe_sup_fresh_base']:.3f}->{r['probe_sup_fresh_late']:.3f} rl-probe fresh {r['probe_rl_fresh_base']:.3f}->{r['probe_rl_fresh_late']:.3f} carried {r['probe_rl_carried_late']:.3f} | rank {r['entk_rank_base']:.1f}->{r['entk_rank_late']:.1f} g2/v {r['adam_g2_over_v_late']:.2f}")
    save_csv(args.out / "factorial.csv", table)
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
