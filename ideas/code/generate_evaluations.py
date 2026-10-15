import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import json, re

with open("parsed_ideas.json", "r") as f:
    ideas = json.load(f)

print(f"Total ideas to evaluate: {len(ideas)}")

def classify_category(section, title, body):
    s = (section + " " + title + " " + body).lower()
    if any(k in s for k in ["llm", "reasoning", "rlvr", "grpo", "prompt", "token-level", "language model", "verifiable"]):
        return "LLM RL & Reasoning"
    elif any(k in s for k in ["multi-agent", "coordination", "self-play", "marl", "partner", "team", "stigmergy"]):
        return "Multi-Agent RL"
    elif any(k in s for k in ["world model", "dreamer", "rssm", "imagination", "model-based", "td-mpc"]):
        return "World Models & MBRL"
    elif any(k in s for k in ["offline", "d4rl", "ogbench", "replay", "dataset", "buffer", "fine-tuning", "wsrl", "cal-ql"]):
        return "Offline RL & Data"
    elif any(k in s for k in ["exploration", "novelty", "bonus", "rnd", "curiosity", "intrinsic"]):
        return "Exploration"
    elif any(k in s for k in ["robotics", "control", "locomotion", "manipulation", "myosuite", "contact", "shield", "safe"]):
        return "Control, Robotics & Safety"
    elif any(k in s for k in ["evaluation", "benchmark", "seed", "metric", "methodology", "anytime-valid", "variance"]):
        return "Evaluation & Methodology"
    elif any(k in s for k in ["representation", "interpretability", "cka", "dormant", "sae", "probing", "bisimulation", "linear"]):
        return "Representation & Interpretability"
    elif any(k in s for k in ["architecture", "depth", "1000 layer", "simba", "ssm", "transformer", "normalization"]):
        return "Architecture & Scaling"
    else:
        return "Learning Dynamics & Optimization"

def score_idea(idea):
    title = idea["title"]
    title_l = title.lower()
    body_l = idea["raw_body"].lower()
    comp_l = idea["compute_note"].lower()
    exp_l = idea["experiment"].lower()
    core_l = idea["core_idea"].lower()
    sec_l = idea["section"].lower()
    category = classify_category(idea["section"], title, idea["raw_body"])

    # -------------------------------------------------------------
    # 1. RTX 3080 FEASIBILITY (1 - 100)
    # Hardware: 10 GB VRAM, 30 TFLOPS FP32, PCIe 4.0
    # Factors: VRAM overhead, training duration, simulator speed, data availability.
    # -------------------------------------------------------------
    feas = 84.0
    
    # Positive factors
    if any(k in comp_l for k in ["days", "tiny", "hours", "cheap"]):
        feas += 8.0
    if any(k in body_l for k in ["gymnax", "craftax", "minatar", "jaxmarl", "brax", "playground", "mjx"]):
        feas += 5.0  # Vectorized GPU sims are 100x-1000x faster than CPU MuJoCo
    if any(k in comp_l for k in ["piggybacks", "small nets", "jvps are cheap"]):
        feas += 4.0
    if "0.5b" in body_l or "0.8b" in body_l:
        feas += 2.0  # 0.5B LoRA fits comfortably in 10GB (<4GB VRAM with FP16/BF16)
        
    # Negative factors
    if any(k in comp_l for k in ["three weeks", "2 to 3 weeks", "two to three weeks"]):
        feas -= 14.0
    elif any(k in comp_l for k in ["two weeks", "1 to 2 weeks", "one to two weeks"]):
        feas -= 5.0
        
    if "atari-100k" in body_l or "atari" in body_l:
        feas -= 6.0
        if "20 seeds" in body_l or "leaderboard" in body_l or "subset" in body_l:
            feas -= 6.0
    if "dreamerv3" in body_l or "world model" in body_l:
        feas -= 5.0
        if "pixel" in body_l:
            feas -= 7.0
    if "pretrain" in body_l and ("from scratch" in body_l or "20–100m" in body_l or "synthetic corpora" in body_l):
        feas -= 12.0
    if "1.5b" in body_l:
        feas -= 4.0  # 1.5B with multi-sample GRPO gets close to 10GB VRAM ceiling
    if "isaac lab" in body_l or "humanoidbench" in body_l:
        feas -= 6.0
    if "mcts" in body_l or "search at test time" in body_l:
        feas -= 7.0
    if "50+ seeds" in body_l or "100+ seeds" in body_l:
        if not any(k in body_l for k in ["gymnax", "craftax", "playground", "mjx"]):
            feas -= 8.0
            
    feas = max(38.0, min(98.0, feas))

    # -------------------------------------------------------------
    # 2. NEURIPS ACCEPTANCE LIKELIHOOD (1 - 100)
    # Loves: Foundational insights, deep empirical discoveries, scaling laws,
    # unexpected phenomena, clean kill tests, causal understanding.
    # -------------------------------------------------------------
    neurips = 70.0
    if any(k in body_l for k in ["scaling law", "chinchilla", "edge of stability", "grokking", "curl", "hamiltonian", "symplectic", "lyapunov", "information curse", "plasticity collapse"]):
        neurips += 16.0
    if any(k in title_l for k in ["why does", "what fraction", "revealed time preference", "does rl really", "depth helps"]):
        neurips += 9.0
    if any(k in body_l for k in ["entk", "leverage score", "rank collapse", "singular-value", "spectral", "unifying"]):
        neurips += 8.0
    if any(k in body_l for k in ["data attribution", "influence", "causal", "emergence", "emergent"]):
        neurips += 6.0
    if category in ["Learning Dynamics & Optimization", "Architecture & Scaling"]:
        neurips += 4.0
    if any(k in title_l for k in ["hyperparameter", "wrapper sensitivity", "default", "checkpoint selection", "notes across"]):
        neurips -= 8.0  # NeurIPS often rejects pure tuning or incremental agent tricks
    if "flagged as at-risk" in body_l:
        neurips -= 4.0
    neurips = max(42.0, min(95.0, neurips))

    # -------------------------------------------------------------
    # 3. ICML ACCEPTANCE LIKELIHOOD (1 - 100)
    # Loves: Mathematical optimization, Bellman contraction, exact estimators,
    # residual gradient proofs/checks, statistical testing rigor, formal bounds.
    # -------------------------------------------------------------
    icml = 71.0
    if any(k in body_l for k in ["contraction modulus", "bellman operator", "sobolev", "estimator", "variance reduction", "crn", "pegasus", "residual gradient", "jacobian", "durbin–watson", "autocorrelation", "anytime-valid", "martingale"]):
        icml += 16.0
    if any(k in body_l for k in ["power iteration", "hessian", "extragradient", "semigradient", "fitted q-iteration"]):
        icml += 9.0
    if any(k in body_l for k in ["pathwise", "score-function", "reppo", "vgf", "double sampling"]):
        icml += 7.0
    if category in ["Learning Dynamics & Optimization", "Evaluation & Methodology"]:
        icml += 4.0
    if any(k in body_l for k in ["prompt", "restating the problem", "hint", "paraphrase", "notes across episodes"]):
        icml -= 14.0  # ICML is notoriously skeptical of LLM prompting / soft prompt ideas
    if "flagged as at-risk" in body_l:
        icml -= 4.0
    icml = max(38.0, min(96.0, icml))

    # -------------------------------------------------------------
    # 4. ICLR ACCEPTANCE LIKELIHOOD (1 - 100)
    # Loves: Neural representation learning, deep architectures (Simba, 1000-layer, SSMs),
    # world model latent dynamics, LLM post-training mechanics (RLVR, GRPO, LoRA, logit lens).
    # -------------------------------------------------------------
    iclr = 73.0
    if any(k in body_l for k in ["representation", "architecture", "depth", "1000 layer", "simba", "world model", "dreamer", "rssm", "transformer", "ssm", "latch", "quasimetric", "bisimulation", "cka", "dormant units"]):
        iclr += 15.0
    if any(k in body_l for k in ["rlvr", "grpo", "lora", "logit lens", "reasoning", "llm", "soft prompt", "prefix-only"]):
        iclr += 12.0
    if any(k in body_l for k in ["flow policy", "diffusion policy", "hyperspherical", "normalization"]):
        iclr += 7.0
    if category in ["Architecture & Scaling", "Representation & Interpretability", "LLM RL & Reasoning", "World Models & MBRL"]:
        iclr += 4.0
    if "gridworld" in body_l and not any(k in body_l for k in ["representation", "deep", "neural", "latent"]):
        iclr -= 10.0  # ICLR strongly disfavors tabular-only or low-dim gridworlds
    if "flagged as at-risk" in body_l:
        iclr -= 4.0
    iclr = max(44.0, min(96.0, iclr))

    # -------------------------------------------------------------
    # 5. AAAI ACCEPTANCE LIKELIHOOD (1 - 100)
    # Loves: Multi-agent coordination, zero-shot coordination, stigmergy,
    # heuristic search + RL, planning, safe RL / shields, cognitive biases,
    # decision-theoretic algorithms, benchmark and evaluation clarity.
    # -------------------------------------------------------------
    aaai = 71.0
    if any(k in body_l for k in ["multi-agent", "coordination", "zero-shot coordination", "stigmergy", "self-play", "partner", "overcooked", "team", "convention"]):
        aaai += 17.0
    if any(k in body_l for k in ["safe", "shield", "constraint", "cognitive-bias", "curriculum", "hierarchy", "heuristic", "mcts", "planning", "search"]):
        aaai += 12.0
    if category in ["Multi-Agent RL", "Control, Robotics & Safety"]:
        aaai += 5.0
    if any(k in body_l for k in ["benchmark", "offline rl", "fine-tuning", "task subset"]):
        aaai += 4.0
    if any(k in body_l for k in ["adam's ε", "numerical precision", "fp16", "compensated targets", "jvp", "curl"]):
        aaai -= 10.0  # AAAI reviews often penalize ultra-fine numerical optimizer diagnostics
    if "flagged as at-risk" in body_l:
        aaai -= 3.0
    aaai = max(45.0, min(94.0, aaai))

    # Composite scores
    avg_conf = (neurips + icml + iclr + aaai) / 4.0
    sweet_spot = (avg_conf * 0.55) + (feas * 0.45)

    # Determine best conference fit
    conf_scores = {"NeurIPS": neurips, "ICML": icml, "ICLR": iclr, "AAAI": aaai}
    best_conf = max(conf_scores, key=conf_scores.get)
    
    # Generate brief rationales
    conf_reason = f"Primary fit: {best_conf} ({conf_scores[best_conf]:.0f}). "
    if best_conf == "NeurIPS":
        conf_reason += "High conceptual depth & fundamental scaling/learning phenomenon."
    elif best_conf == "ICML":
        conf_reason += "Strong algorithmic/operator formulation & statistical rigor."
    elif best_conf == "ICLR":
        conf_reason += "Deep representation learning, architecture, or modern reasoning mechanics."
    else:
        conf_reason += "Strong multi-agent, planning, or decision-theoretic formulation."
        
    comp_reason = f"Feas {feas:.0f}/100. "
    if feas >= 90:
        comp_reason += "Fast GPU simulator (Gymnax/MJX/Craftax/MinAtar) or lightweight model. Days to 1 week."
    elif feas >= 80:
        comp_reason += "Standard DMC/MuJoCo or 0.5B LoRA. Fits easily in 10GB VRAM. 1-2 weeks."
    elif feas >= 65:
        comp_reason += "Dreamer/1.5B LLM or multiple seeds across DMC. High single-GPU load. 2-3 weeks."
    else:
        comp_reason += "Heavy pixel Atari-100k or pretraining from scratch. Significant single 3080 bottleneck."

    return {
        "id": idea["id"],
        "file": idea["file"],
        "set_name": idea["set_name"],
        "num": idea["num"],
        "section": idea["section"],
        "category": category,
        "title": idea["title"],
        "core_idea": idea["core_idea"],
        "experiment": idea["experiment"],
        "kill_test": idea["kill_test"],
        "compute_note": idea["compute_note"],
        "neurips": round(neurips, 1),
        "icml": round(icml, 1),
        "iclr": round(iclr, 1),
        "aaai": round(aaai, 1),
        "avg_conf": round(avg_conf, 1),
        "feasibility_3080": round(feas, 1),
        "sweet_spot": round(sweet_spot, 1),
        "best_conf": best_conf,
        "conf_reason": conf_reason,
        "comp_reason": comp_reason
    }

evaluated = [score_idea(x) for x in ideas]

with open("all_250_evaluated.json", "w") as f:
    json.dump(evaluated, f, indent=2)

print(f"Evaluated all {len(evaluated)} ideas successfully.")
