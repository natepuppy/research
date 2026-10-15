import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import re, json, csv, os

files = [
    ("rl_ideas_50.md", "sales_conversation_ai/rl_ideas_50.md", "Set 1: Paper Follow-ups (2025-2026)"),
    ("rl_ideas_novel_50.md", "sales_conversation_ai/rl_ideas_novel_50.md", "Novel Set 1: Mechanisms & Measurements"),
    ("rl_ideas_novel_50_set2.md", "sales_conversation_ai/rl_ideas_novel_50_set2.md", "Novel Set 2: Hyperparams, Data, Control & LLM"),
    ("rl_ideas_novel_50_set3.md", "sales_conversation_ai/rl_ideas_novel_50_set3.md", "Novel Set 3: Learning Dynamics & Eval"),
    ("rl_ideas_novel_50_set4.md", "sales_conversation_ai/rl_ideas_novel_50_set4.md", "Novel Set 4: Value Learning, Control & Clones")
]

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()

parsed_ideas = []

for shortname, path, set_title in files:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Find where the ideas stop (before "--- \n## How to pick" or similar)
    stop_match = re.search(r"\n---\s*\n\s*##\s*How to pick", text)
    ideas_text = text[:stop_match.start()] if stop_match else text

    matches = list(re.finditer(r"\*\*(\d+)\.\s*(.*?)\*\*", ideas_text))
    
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(ideas_text)
        chunk = ideas_text[start:end]

        preceding = ideas_text[:m.start()]
        headers = re.findall(r"##\s+([A-K]\.\s+.*)", preceding)
        section = headers[-1].strip() if headers else "General"

        header_in_chunk = re.search(r"\n\s*##\s+[A-K]\.\s+.*", chunk)
        if header_in_chunk:
            chunk = chunk[:header_in_chunk.start()]

        fields = {}
        current_key = None
        for line in chunk.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("- ") or line.startswith("* "):
                key_match = re.match(r"^[-*]\s*([^:]+):\s*(.*)$", line)
                if key_match:
                    current_key = key_match.group(1).strip().lower()
                    fields[current_key] = key_match.group(2).strip()
            elif current_key and not line.startswith("#"):
                fields[current_key] += " " + line

        core = fields.get("hypothesis", "") or fields.get("question", "")
        gap = fields.get("gap", "") or fields.get("builds on", "")
        exp = fields.get("experiment", "")
        why = fields.get("why it lands", "")
        kill = fields.get("kill test", "")
        comp = fields.get("compute", "")

        parsed_ideas.append({
            "id": f"{shortname.replace('.md','')}_{num}",
            "file": shortname,
            "set_name": set_title,
            "num": num,
            "section": section,
            "title": title,
            "core": clean_text(core),
            "gap": clean_text(gap),
            "experiment": clean_text(exp),
            "why_lands": clean_text(why),
            "kill_test": clean_text(kill),
            "compute": clean_text(comp),
            "full_text": clean_text(f"{title}. {core}. {gap}. {exp}. {why}. {kill}. {comp}")
        })

print(f"Total parsed ideas: {len(parsed_ideas)}")
assert len(parsed_ideas) == 250
