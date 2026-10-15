import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import json

with open("all_250_evaluated.json") as f:
    data = json.load(f)

# Sort by sweet spot
data_sorted = sorted(data, key=lambda x: x["sweet_spot"], reverse=True)

with open("sales_conversation_ai/rl_ideas_ranked_250.json", "w") as f:
    json.dump(data_sorted, f, indent=2)

print("Saved sales_conversation_ai/rl_ideas_ranked_250.json")
