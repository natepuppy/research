import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import re, json, csv, os

with open("parsed_ideas.json", "r") as f:
    ideas = json.load(f)

print(f"Loaded {len(ideas)} ideas.")

# Categorize research domain
def categorize(title, body, section):
    s = (title + " " + body + " " + section).lower()
    if any(k in s for k in ["llm", "rlvr", "grpo", "language model", "reasoning", "token-level", "soft prompt", "prefix-only", "text environment"]):
        return "LLM RL & Reasoning"
    elif any(k in s for k in ["multi-agent", "coordination", "marl", "self-play", "partner", "team", "stigmergy"]):
        return "Multi-Agent RL"
    elif any(k in s for k in ["world model", "dreamer", "rssm", "imagination", "model-based", "td-mpc"]):
        return "World Models & MBRL"
    elif any(k in s for k in ["offline", "d4rl", "ogbench", "replay", "buffer", "dataset", "fine-tuning", "wsrl", "cal-ql"]):
        return "Offline RL & Data"
    elif any(k in s for k in ["exploration", "novelty", "bonus", "rnd", "curiosity", "intrinsic motivation"]):
        return "Exploration"
    elif any(k in s for k in ["robotics", "control", "locomotion", "manipulation", "myosuite", "contact", "shield", "safety"]):
        return "Control, Robotics & Safety"
    elif any(k in s for k in ["evaluation", "benchmark", "seed", "methodology", "anytime-valid", "mutation testing", "checkpoint selection"]):
        return "Evaluation & Methodology"
    elif any(k in s for k in ["representation", "interpretability", "cka", "dormant", "sae", "probing", "bisimulation", "quasimetric"]):
        return "Representation & Interpretability"
    elif any(k in s for k in ["architecture", "depth", "1000 layer", "simba", "ssm", "transformer", "normalization", "plasticity"]):
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
    cat = categorize(title, idea["raw_body"], idea["section"])

    # -------------------------------------------------------------------------
    # 1. RTX 3080 FEASIBILITY (1 - 100)
    # Hardware constraints: 10 GB VRAM, ~30 TFLOPS FP32, PCIe 4.0.
    # -------------------------------------------------------------------------
    feas = 82.0

    # Environment & SIM advantages
    if any(k in body_l for k in ["gymnax", "craftax", "minatar", "jaxmarl", "brax", "playground", "mjx"]):
        feas += 8.0  # Vectorized JAX / GPU-parallel sims run at millions of steps/sec
    elif any(k in body_l for k in ["gridworld", "tabular", "point-maze", "four rooms"]):
        feas += 9.0  # Tabular / toy environments finish in minutes to hours
    elif any(k in comp_l for k in ["tiny", "days", "hours"]):
        feas += 6.0
    
    if any(k in comp_l for k in ["small nets", "jvps are cheap", "power iteration", "piggybacks"]):
        feas += 4.0
    if any(k in body_l for k in ["0.5b", "0.8b", "smollm", "pythia"]):
        feas += 2.0  # Small models with LoRA easily fit in 4-6 GB VRAM

    # Compute bottlenecks
    if any(k in comp_l for k in ["three weeks", "2 to 3 weeks", "two to three weeks"]):
        feas -= 14.0
    elif any(k in comp_l for k in ["two weeks", "1 to 2 weeks", "one to two weeks"]):
        feas -= 5.0

    if "atari-100k" in body_l or "atari" in body_l:
        feas -= 7.0
        if any(k in body_l for k in ["20 seeds", "leaderboard", "8 games", "bbf", "efficientzero"]):
            feas -= 10.0  # Running multiple seeds across Atari is notorious on single GPU
    if "dreamerv3" in body_l or "world model" in body_l:
        feas -= 5.0
        if "pixel" in body_l:
            feas -= 7.0  # Pixel-level RSSM training takes 5-8 days per seed
    if "pretrain" in body_l and ("from scratch" in body_l or "20–100m" in body_l or "synthetic corpora" in body_l):
        feas -= 14.0  # Pretraining multiple LMs from scratch is compute prohibitive
    if "1.5b" in body_l:
        feas -= 4.0  # 1.5B with multi-sample GRPO approaches 10 GB limit
    if "isaac lab" in body_l or "humanoidbench" in body_l:
        feas -= 5.0
    if "50+ seeds" in body_l or "100+ seeds" in body_l:
        if not any(k in body_l for k in ["gymnax", "craftax", "playground", "mjx"]):
            feas -= 8.0

    feas = max(38.0, min(98.0, round(feas, 1)))

    # -------------------------------------------------------------------------
    # 2. NEURIPS ACCEPTANCE LIKELIHOOD (1 - 100)
    # Culture: Foundational phenomena, scaling laws, deep theory, counter-intuitive
    # empirical discoveries, causal mechanisms, broad impact.
    # -------------------------------------------------------------------------
    nips = 68.0
    # Core NeurIPS themes
    if any(k in body_l for k in ["scaling law", "chinchilla", "edge of stability", "grokking", "curl", "hamiltonian", "symplectic", "lyapunov", "revealed time preference"]):
        nips += 18.0
    if any(k in title_l for k in ["why does", "what fraction", "does rl really", "depth helps"]):
        nips += 10.0
    if any(k in body_l for k in ["entk", "leverage score", "rank collapse", "singular-value", "spectral", "unifying", "plasticity collapse"]):
        nips += 8.0
    if any(k in body_l for k in ["data attribution", "causal", "emergence", "emergent bisimulation", "emergent equivariance"]):
        nips += 7.0
    if cat in ["Learning Dynamics & Optimization", "Architecture & Scaling"]:
        nips += 3.0
    # Disfavored by NeurIPS
    if any(k in title_l for k in ["wrapper sensitivity", "default", "checkpoint selection", "notes across episodes", "drop-in tool"]):
        nips -= 10.0
    if "flagged as at-risk" in body_l:
        nips -= 4.0

    nips = max(42.0, min(95.0, round(nips, 1)))

    # -------------------------------------------------------------------------
    # 3. ICML ACCEPTANCE LIKELIHOOD (1 - 100)
    # Culture: Mathematical optimization, Bellman contraction, operator properties,
    # exact estimators (CRN, Sobolev, Vine), variance proofs, statistical testing.
    # -------------------------------------------------------------------------
    icml = 69.0
    # Core ICML themes
    if any(k in body_l for k in ["contraction modulus", "bellman operator", "sobolev", "estimator", "variance reduction", "crn", "pegasus", "residual gradient", "jacobian", "durbin–watson", "autocorrelation", "anytime-valid"]):
        nips_bonus = 18.0
        icml += nips_bonus
    if any(k in body_l for k in ["power iteration", "hessian", "extragradient", "semigradient", "fitted q-iteration", "double sampling"]):
        icml += 9.0
    if any(k in body_l for k in ["pathwise", "score-function", "reppo", "vgf", "subspace", "offline-to-online"]):
        icml += 6.0
    if cat in ["Learning Dynamics & Optimization", "Evaluation & Methodology"]:
        icml += 4.0
    # Disfavored by ICML
    if any(k in body_l for k in ["prompt", "restating the problem", "hint", "paraphrase", "notes across"]):
        icml -= 15.0  # ICML is notoriously harsh on soft prompting / LLM heuristics
    if "flagged as at-risk" in body_l:
        icml -= 4.0

    icml = max(40.0, min(96.0, round(icml, 1)))

    # -------------------------------------------------------------------------
    # 4. ICLR ACCEPTANCE LIKELIHOOD (1 - 100)
    # Culture: Neural representations, deep architectures (Simba, depth, SSMs),
    # world model latent dynamics, LLM reasoning post-training (RLVR, GRPO, logit lens).
    # -------------------------------------------------------------------------
    iclr = 72.0
    # Core ICLR themes
    if any(k in body_l for k in ["representation", "architecture", "depth", "1000 layer", "simba", "world model", "dreamer", "rssm", "transformer", "ssm", "latch", "quasimetric", "bisimulation", "cka", "dormant units"]):
        iclr += 16.0
    if any(k in body_l for k in ["rlvr", "grpo", "lora", "logit lens", "reasoning", "llm", "soft prompt", "prefix-only"]):
        iclr += 12.0
    if any(k in body_l for k in ["flow policy", "diffusion policy", "hyperspherical", "normalization", "plasticity"]):
        iclr += 7.0
    if cat in ["Architecture & Scaling", "Representation & Interpretability", "LLM RL & Reasoning", "World Models & MBRL"]:
        iclr += 4.0
    # Disfavored by ICLR
    if "gridworld" in body_l and not any(k in body_l for k in ["representation", "deep", "neural", "latent", "probe"]):
        iclr -= 12.0  # ICLR strongly dislikes low-dim tabular MDPs unless representations are analyzed
    if "flagged as at-risk" in body_l:
        iclr -= 4.0

    iclr = max(44.0, min(96.0, round(iclr, 1)))

    # -------------------------------------------------------------------------
    # 5. AAAI ACCEPTANCE LIKELIHOOD (1 - 100)
    # Culture: Multi-agent coordination, zero-shot coordination, stigmergy,
    # heuristic search + RL, planning, safe RL / shields, cognitive decision biases,
    # comprehensive empirical benchmarks.
    # -------------------------------------------------------------------------
    aaai = 70.0
    # Core AAAI themes
    if any(k in body_l for k in ["multi-agent", "coordination", "zero-shot coordination", "stigmergy", "self-play", "partner", "overcooked", "team", "convention", "leader-follower"]):
        aaai += 18.0
    if any(k in body_l for k in ["safe", "shield", "constraint", "cognitive-bias", "curriculum", "hierarchy", "heuristic", "mcts", "planning", "search at test time"]):
        aaai += 13.0
    if cat in ["Multi-Agent RL", "Control, Robotics & Safety"]:
        aaai += 5.0
    if any(k in body_l for k in ["benchmark", "offline rl", "fine-tuning", "task subset", "evaluation"]):
        aaai += 4.0
    # Disfavored by AAAI
    if any(k in body_l for k in ["adam's ε", "numerical precision", "fp16", "compensated targets", "jvp", "curl"]):
        aaai -= 11.0  # AAAI reviews rarely care about micro-numerical optimizer details
    if "flagged as at-risk" in body_l:
        aaai -= 3.0

    aaai = max(45.0, min(94.0, round(aaai, 1)))

    # -------------------------------------------------------------------------
    # Composite Scores & Notes
    # -------------------------------------------------------------------------
    avg_conf = round((nips + icml + iclr + aaai) / 4.0, 1)
    sweet_spot = round((avg_conf * 0.55) + (feas * 0.45), 1)

    conf_scores = {"NeurIPS": nips, "ICML": icml, "ICLR": iclr, "AAAI": aaai}
    best_conf = max(conf_scores, key=conf_scores.get)

    # Rationale
    if best_conf == "NeurIPS":
        rationale = f"Primary fit: NeurIPS ({nips:.0f}). High conceptual depth addressing fundamental scaling or learning phenomena with clean falsifiable mechanisms."
    elif best_conf == "ICML":
        rationale = f"Primary fit: ICML ({icml:.0f}). Rigorous algorithmic/operator formulation, exact estimator properties, or statistical optimization guarantees."
    elif best_conf == "ICLR":
        rationale = f"Primary fit: ICLR ({iclr:.0f}). Strong representation-learning novelty, deep architecture/normalization scaling, or modern LLM reasoning mechanics."
    else:
        rationale = f"Primary fit: AAAI ({aaai:.0f}). High relevance to multi-agent coordination, decision-theoretic agent behavior, planning, or safe RL."

    # Compute note
    if feas >= 92:
        comp_notes = f"RTX 3080 Feasibility: {feas:.0f}/100. Fast GPU-native environment (Gymnax/MJX/Craftax/MinAtar) or lightweight diagnostic runs. Turnaround in 2 to 5 days."
    elif feas >= 80:
        comp_notes = f"RTX 3080 Feasibility: {feas:.0f}/100. Standard MuJoCo/DMC or 0.5B LoRA GRPO. Fits comfortably within 10 GB VRAM. Turnaround in 1 to 2 weeks."
    elif feas >= 65:
        comp_notes = f"RTX 3080 Feasibility: {feas:.0f}/100. Moderate single-GPU load (DMC Dreamer or multi-seed continuous runs). Approaches VRAM ceiling. Turnaround in 2 to 3 weeks."
    else:
        comp_notes = f"RTX 3080 Feasibility: {feas:.0f}/100. High single-GPU bottleneck (pixel Atari-100k multi-seed sweep or pretraining from scratch). High risk of thermal/runtime strain."

    return {
        "id": idea["id"],
        "file": idea["file"],
        "set_name": idea["set_name"],
        "num": idea["num"],
        "section": idea["section"],
        "category": cat,
        "title": title,
        "neurips": nips,
        "icml": icml,
        "iclr": iclr,
        "aaai": aaai,
        "avg_conf": avg_conf,
        "feasibility_3080": feas,
        "sweet_spot": sweet_spot,
        "best_conf": best_conf,
        "rationale": rationale,
        "compute_data_notes": comp_notes,
        "core_idea": idea["core_idea"],
        "experiment": idea["experiment"],
        "kill_test": idea["kill_test"],
        "compute_note": idea["compute_note"]
    }

scored = [score_idea(item) for item in ideas]

# Assign ranks for each individual conference, feasibility, and sweet spot
def rank_by_key(items, key, reverse=True):
    sorted_items = sorted(items, key=lambda x: x[key], reverse=reverse)
    for r, it in enumerate(sorted_items, 1):
        it[f"{key}_rank"] = r

rank_by_key(scored, "neurips")
rank_by_key(scored, "icml")
rank_by_key(scored, "iclr")
rank_by_key(scored, "aaai")
rank_by_key(scored, "feasibility_3080")
rank_by_key(scored, "sweet_spot")

# Sort final output by sweet spot rank
scored_final = sorted(scored, key=lambda x: x["sweet_spot_rank"])

print("Writing clean CSV...")
csv_path = "research/brainstorming/the_independent_rankings/rl_ideas_ranked_250.csv"
fieldnames = [
    "overall_rank",
    "source_file",
    "set_name",
    "idea_number",
    "title",
    "category",
    "neurips_score",
    "neurips_rank",
    "icml_score",
    "icml_rank",
    "iclr_score",
    "iclr_rank",
    "aaai_score",
    "aaai_rank",
    "avg_conference_score",
    "rtx_3080_feasibility",
    "feasibility_rank",
    "sweet_spot_score",
    "best_venue",
    "rationale",
    "compute_data_notes",
    "core_hypothesis_or_question",
    "experiment_protocol",
    "kill_test",
    "original_compute_estimate"
]

def clean_val(v):
    if v is None:
        return ""
    s = str(v).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for row in scored_final:
        writer.writerow({
            "overall_rank": row["sweet_spot_rank"],
            "source_file": clean_val(row["file"]),
            "set_name": clean_val(row["set_name"]),
            "idea_number": row["num"],
            "title": clean_val(row["title"]),
            "category": clean_val(row["category"]),
            "neurips_score": row["neurips"],
            "neurips_rank": row["neurips_rank"],
            "icml_score": row["icml"],
            "icml_rank": row["icml_rank"],
            "iclr_score": row["iclr"],
            "iclr_rank": row["iclr_rank"],
            "aaai_score": row["aaai"],
            "aaai_rank": row["aaai_rank"],
            "avg_conference_score": row["avg_conf"],
            "rtx_3080_feasibility": row["feasibility_3080"],
            "feasibility_rank": row["feasibility_3080_rank"],
            "sweet_spot_score": row["sweet_spot"],
            "best_venue": row["best_conf"],
            "rationale": clean_val(row["rationale"]),
            "compute_data_notes": clean_val(row["compute_data_notes"]),
            "core_hypothesis_or_question": clean_val(row["core_idea"]),
            "experiment_protocol": clean_val(row["experiment"]),
            "kill_test": clean_val(row["kill_test"]),
            "original_compute_estimate": clean_val(row["compute_note"])
        })

print(f"Successfully wrote {len(scored_final)} ranked ideas to {csv_path}")
