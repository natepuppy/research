import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import json

with open("all_250_evaluated.json") as f:
    data = json.load(f)

print(f"Total: {len(data)}")
for conf in ["neurips", "icml", "iclr", "aaai", "feasibility_3080", "sweet_spot"]:
    scores = [d[conf] for d in data]
    print(f"{conf:18s} | Min: {min(scores):4.1f} | Max: {max(scores):4.1f} | Mean: {sum(scores)/len(scores):4.1f}")

print("\n--- TOP 5 NEURIPS ---")
for d in sorted(data, key=lambda x: x["neurips"], reverse=True)[:5]:
    print(f"[{d['neurips']}] {d['file']} #{d['num']}: {d['title']}")

print("\n--- TOP 5 ICML ---")
for d in sorted(data, key=lambda x: x["icml"], reverse=True)[:5]:
    print(f"[{d['icml']}] {d['file']} #{d['num']}: {d['title']}")

print("\n--- TOP 5 ICLR ---")
for d in sorted(data, key=lambda x: x["iclr"], reverse=True)[:5]:
    print(f"[{d['iclr']}] {d['file']} #{d['num']}: {d['title']}")

print("\n--- TOP 5 AAAI ---")
for d in sorted(data, key=lambda x: x["aaai"], reverse=True)[:5]:
    print(f"[{d['aaai']}] {d['file']} #{d['num']}: {d['title']}")

print("\n--- TOP 5 FEASIBILITY ON 3080 ---")
for d in sorted(data, key=lambda x: x["feasibility_3080"], reverse=True)[:5]:
    print(f"[{d['feasibility_3080']}] {d['file']} #{d['num']}: {d['title']}")

print("\n--- TOP 10 SWEET SPOT (High Conf + High Feas) ---")
for d in sorted(data, key=lambda x: x["sweet_spot"], reverse=True)[:10]:
    print(f"[{d['sweet_spot']}] {d['file']} #{d['num']}: {d['title']} (N:{d['neurips']}, M:{d['icml']}, R:{d['iclr']}, A:{d['aaai']}, F:{d['feasibility_3080']})")
