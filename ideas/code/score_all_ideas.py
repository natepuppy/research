import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import json, re

with open("parsed_ideas.json", "r") as f:
    ideas = json.load(f)

scored_ideas = []

# Helper keywords and criteria for scoring
def evaluate_idea(idea):
    title = idea["title"].lower()
    body = idea["raw_body"].lower()
    compute_note = idea["compute_note"].lower()
    section = idea["section"].lower()
    core = idea["core_idea"].lower()
    exp = idea["experiment"].lower()
    
    # -------------------------------------------------------------
    # 1. FEASIBILITY ON RTX 3080 (10 GB) (1-100)
    # -------------------------------------------------------------
    # Base feasibility starts around 85 for standard RL on single GPU
    feas = 85.0
    
    # Positive compute indicators
    if "days" in compute_note or "tiny" in compute_note:
        feas += 8.0
    if "gymnax" in body or "craftax" in body or "minatar" in body or "jaxmarl" in body or "brax" in body or "playground" in body or "mjx" in body:
        feas += 5.0
    if "small nets" in compute_note or "jvps are cheap" in compute_note or "power iteration" in compute_note:
        feas += 4.0
    if "piggybacks" in compute_note:
        feas += 7.0
    if "0.5b" in body or "0.5–1b" in body or "0.8b" in body:
        feas += 2.0  # Fits easily in 10GB with LoRA and short seqs
        
    # Negative compute indicators (strain on 10GB 3080)
    if "three weeks" in compute_note or "2 to 3 weeks" in compute_note or "two to three weeks" in compute_note:
        feas -= 14.0
    elif "two weeks" in compute_note:
        feas -= 6.0
    if "atari-100k" in body or "atari" in body:
        feas -= 7.0
        if "20 seeds" in body or "leaderboard" in body:
            feas -= 10.0
    if "dreamerv3" in body or "rssm" in body or "world model" in body:
        feas -= 5.0
        if "pixel" in body:
            feas -= 8.0
    if "pretrain" in body and ("20–100m" in body or "from scratch" in body):
        feas -= 12.0
    if "1.5b" in body:
        feas -= 4.0 # 1.5B with GRPO multiple rollouts can be tight on 10GB
    if "50+ seeds" in body or "100+ seeds" in body or "500+ short runs" in body:
        if not ("gymnax" in body or "craftax" in body or "playground" in body):
            feas -= 8.0
    if "search" in body and "mcts" in body:
        feas -= 8.0
    if "isaac lab" in body or "humanoidbench" in body:
        feas -= 5.0
        
    feas = max(35.0, min(98.0, feas))
    
    # -------------------------------------------------------------
    # 2. NEURIPS (1-100)
    # Loves: foundational theories, surprising phenomena, scaling laws, 
    # causal interventions, grokking, rank/plasticity theory, clean kill tests.
    # -------------------------------------------------------------
    neurips = 72.0
    if any(k in body for k in ["scaling law", "chinchilla", "edge of stability", "grokking", "curl", "hamiltonian", "symplectic", "lyapunov", "information curse", "phenomenon", "fundamental", "causal", "plasticity collapse"]):
        neurips += 14.0
    if any(k in body for k in ["why does", "what fraction", "revealed time preference", "does rl really", "unifying", "paradox", "emergence", "emergent"]):
        neurips += 8.0
    if any(k in body for k in ["entk", "leverage score", "rank collapse", "singular-value", "spectral"]):
        neurips += 7.0
    if "benchmark" in body and "procedural" in body:
        neurips += 2.0
    if any(k in body for k in ["hyperparameter", "wrapper sensitivity", "default", "checkpoint selection"]):
        neurips -= 6.0 # NeurIPS dislikes pure tuning/wrapper papers unless framed as deep theory
    if any(k in body for k in ["engineering", "drop-in tool"]):
        neurips -= 5.0
        
    neurips = max(40.0, min(95.0, neurips))

    # -------------------------------------------------------------
    # 3. ICML (1-100)
    # Loves: mathematical optimization, Bellman operators, contraction modulus, 
    # exact estimators (PEGASUS, CRN, Vine, Sobolev), statistical rigor, sequential testing.
    # -------------------------------------------------------------
    icml = 72.0
    if any(k in body for k in ["bellman", "contraction modulus", "operator", "sobolev", "estimator", "variance reduction", "crn", "pegasus", "residual gradient", "jacobian", "durbin–watson", "autocorrelation", "anytime-valid", "martingale", "statistical"]):
        icml += 15.0
    if any(k in body for k in ["optimization", "gradient", "hessian", "power iteration", "semigradient", "convergence", "extragradient"]):
        icml += 8.0
    if any(k in body for k in ["discrete vgf", "pathwise", "score-function"]):
        icml += 6.0
    if any(k in body for k in ["prompt", "restating the problem", "hint", "paraphrase"]):
        icml -= 12.0 # ICML reviewers are notorious for bouncing soft LLM prompt tricks
    if any(k in body for k in ["wrapper sensitivity", "seeds versus evaluation episodes"]):
        icml += 3.0 # ICML likes statistical methodology papers (like rliable)
        
    icml = max(38.0, min(96.0, icml))

    # -------------------------------------------------------------
    # 4. ICLR (1-100)
    # Loves: representation learning, deep architectures, 1000-layer, 
    # simba, world models, latent dynamics, bistable SSM, grokking, 
    # LLM reasoning post-training (GRPO, RLVR, logit lens, prefix-tuning).
    # -------------------------------------------------------------
    iclr = 74.0
    if any(k in body for k in ["representation", "architecture", "depth", "1000 layer", "simba", "world model", "dreamer", "rssm", "transformer", "ssm", "mamba", "latch", "quasimetric", "bisimulation", "feature", "cka", "dormant"]):
        iclr += 14.0
    if any(k in body for k in ["rlvr", "grpo", "lora", "logit lens", "reasoning", "llm", "soft prompt", "prefix", "token-level"]):
        iclr += 11.0
    if any(k in body for k in ["flow policy", "diffusion policy", "normalization", "hyperspherical"]):
        iclr += 8.0
    if any(k in body for k in ["tabular", "gridworld"]) and not any(k in body for k in ["representation", "latent", "deep"]):
        iclr -= 12.0 # ICLR hates pure tabular/gridworld unless demonstrating deep representation
        
    iclr = max(42.0, min(96.0, iclr))

    # -------------------------------------------------------------
    # 5. AAAI (1-100)
    # Loves: multi-agent coordination, zero-shot coordination, stigmergy, 
    # planning, heuristic search + RL, safe RL / shields, cognitive biases,
    # benchmarks, practical algorithms with strong general AI appeal.
    # -------------------------------------------------------------
    aaai = 71.0
    if any(k in body for k in ["multi-agent", "coordination", "zero-shot coordination", "stigmergy", "self-play", "partner", "overcooked", "team", "convention"]):
        aaai += 16.0
    if any(k in body for k in ["safe", "shield", "constraint", "cognitive-bias", "curriculum", "hierarchy", "heuristic", "mcts", "planning", "search"]):
        aaai += 11.0
    if any(k in body for k in ["benchmark", "task subset", "procedural", "evaluation", "offline rl", "fine-tuning"]):
        aaai += 6.0
    if any(k in body for k in ["adam's ε", "numerical precision", "fp16", "compensated targets", "jvp"]):
        aaai -= 10.0 # AAAI reviewers often care less about micro-numerical optimizer details
        
    aaai = max(45.0, min(94.0, aaai))

    # Fine-tuning adjustments based on specific ideas from author notes
    # (e.g. author explicitly noted "Why it lands" or "Kill test" or "flagged at-risk")
    if "flagged as at-risk" in body:
        neurips -= 4.0
        icml -= 4.0
        iclr -= 4.0
        aaai -= 3.0

    # Composite / Average Conference score
    avg_conf = (neurips + icml + iclr + aaai) / 4.0
    # Sweet spot score: balance between conference acceptance and feasibility on 3080
    sweet_spot = (avg_conf * 0.55) + (feas * 0.45)
    
    return {
        "id": idea["id"],
        "file": idea["file"],
        "set_name": idea["set_name"],
        "num": idea["num"],
        "section": idea["section"],
        "title": idea["title"],
        "core_idea": idea["core_idea"][:180] + ("..." if len(idea["core_idea"]) > 180 else ""),
        "experiment": idea["experiment"][:180] + ("..." if len(idea["experiment"]) > 180 else ""),
        "compute_note": idea["compute_note"],
        "neurips": round(neurips, 1),
        "icml": round(icml, 1),
        "iclr": round(iclr, 1),
        "aaai": round(aaai, 1),
        "feasibility_3080": round(feas, 1),
        "avg_conf": round(avg_conf, 1),
        "sweet_spot": round(sweet_spot, 1)
    }

for idea in ideas:
    scored_ideas.append(evaluate_idea(idea))

# Sort by sweet spot
scored_ideas_sorted = sorted(scored_ideas, key=lambda x: x["sweet_spot"], reverse=True)

with open("scored_ideas.json", "w") as f:
    json.dump(scored_ideas_sorted, f, indent=2)

print("Top 10 Sweet Spot Ideas:")
for i, d in enumerate(scored_ideas_sorted[:10], 1):
    print(f"{i}. [{d['set_name']}] #{d['num']}: {d['title']}")
    print(f"   NeurIPS: {d['neurips']} | ICML: {d['icml']} | ICLR: {d['iclr']} | AAAI: {d['aaai']} | 3080 Feas: {d['feasibility_3080']} | SweetSpot: {d['sweet_spot']}")
