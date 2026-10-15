import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import json

with open("all_250_evaluated.json") as f:
    data = json.load(f)

sweet_sorted = sorted(data, key=lambda x: x["sweet_spot"], reverse=True)
nips_sorted = sorted(data, key=lambda x: x["neurips"], reverse=True)
icml_sorted = sorted(data, key=lambda x: x["icml"], reverse=True)
iclr_sorted = sorted(data, key=lambda x: x["iclr"], reverse=True)
aaai_sorted = sorted(data, key=lambda x: x["aaai"], reverse=True)
feas_sorted = sorted(data, key=lambda x: x["feasibility_3080"], reverse=True)

md = []
md.append("# Complete Evaluation and Rankings of 250 RL Research Ideas\n")
md.append("Evaluated across the Top Four Conferences (**NeurIPS, ICML, ICLR, AAAI**) and single **RTX 3080 (10GB)** feasibility.\n")
md.append("\n## Top 20 Sweet Spot Ideas (High Conference Acceptance + Fast 3080 Turnaround)\n")
md.append("| # | Idea & Source | Category | NeurIPS | ICML | ICLR | AAAI | 3080 Feas | Sweet Spot | Best Venue |\n")
md.append("|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

for i, d in enumerate(sweet_sorted[:20], 1):
    md.append(f"| {i} | **{d['title']}**<br>`{d['file']}` #{d['num']} | {d['category']} | {d['neurips']:.0f} | {d['icml']:.0f} | {d['iclr']:.0f} | {d['aaai']:.0f} | **{d['feasibility_3080']:.0f}** | **{d['sweet_spot']:.1f}** | `{d['best_conf']}` |\n")

with open("sales_conversation_ai/rl_ideas_rankings_summary.md", "w") as f:
    f.write("".join(md))

print("Saved sales_conversation_ai/rl_ideas_rankings_summary.md")
