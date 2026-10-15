import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import re, json

files = [
    ("rl_ideas_50.md", "sales_conversation_ai/rl_ideas_50.md", "Set 1: Paper Follow-ups (2025-2026)"),
    ("rl_ideas_novel_50.md", "sales_conversation_ai/rl_ideas_novel_50.md", "Novel Set 1: Mechanisms & Measurements"),
    ("rl_ideas_novel_50_set2.md", "sales_conversation_ai/rl_ideas_novel_50_set2.md", "Novel Set 2: Hyperparams, Data, Control & LLM"),
    ("rl_ideas_novel_50_set3.md", "sales_conversation_ai/rl_ideas_novel_50_set3.md", "Novel Set 3: Learning Dynamics & Evaluation"),
    ("rl_ideas_novel_50_set4.md", "sales_conversation_ai/rl_ideas_novel_50_set4.md", "Novel Set 4: Value Learning, Control & Clones")
]

all_ideas = []

for shortname, filepath, tag in files:
    with open(filepath, "r") as f:
        text = f.read()
    
    matches = list(re.finditer(r"\*\*(\d+)\.\s*(.*?)\*\*", text))
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[start:end].strip()
        
        # Extract section if any
        # find closest preceding ## header
        preceding = text[:m.start()]
        headers = re.findall(r"##\s+([A-K]\.\s+.*)", preceding)
        section = headers[-1] if headers else "General"
        
        # extract specific fields if present
        builds_on = ""
        question = ""
        experiment = ""
        why_lands = ""
        compute = ""
        hypothesis = ""
        gap = ""
        kill_test = ""
        
        m_b = re.search(r"-\s*Builds on:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_b: builds_on = m_b.group(1).strip()
        
        m_q = re.search(r"-\s*Question:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_q: question = m_q.group(1).strip()
        
        m_h = re.search(r"-\s*Hypothesis:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_h: hypothesis = m_h.group(1).strip()
        
        m_g = re.search(r"-\s*Gap:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_g: gap = m_g.group(1).strip()
        
        m_e = re.search(r"-\s*Experiment:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_e: experiment = m_e.group(1).strip()
        
        m_w = re.search(r"-\s*Why it lands:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_w: why_lands = m_w.group(1).strip()
        
        m_k = re.search(r"-\s*Kill test:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_k: kill_test = m_k.group(1).strip()
        
        m_c = re.search(r"-\s*Compute:\s*(.*?)(?=\n-\s*|\n\*\*|\Z)", body, re.DOTALL)
        if m_c: compute = m_c.group(1).strip()
        
        all_ideas.append({
            "id": f"{shortname.replace('.md','')}_{num}",
            "file": shortname,
            "set_name": tag,
            "num": num,
            "section": section,
            "title": title,
            "core_idea": hypothesis if hypothesis else question,
            "gap_context": gap if gap else builds_on,
            "experiment": experiment,
            "why_lands": why_lands,
            "kill_test": kill_test,
            "compute_note": compute,
            "raw_body": body
        })

print(f"Loaded {len(all_ideas)} ideas successfully.")
with open("parsed_ideas.json", "w") as out:
    json.dump(all_ideas, out, indent=2)
