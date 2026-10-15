# S4-46: Anytime validity is nearly free; distribution-free validity is what costs seeds

**Question.** When comparing RL algorithms across seeds, do anytime-valid confidence sequences cut the seed budget without inflating false positives, and where does the remaining cost come from?

**Status.** Full manuscript drafted (`paper/paper.md`), conference-length version in `paper/paper_rlc.md`. Verdict: `KILL TEST: SUPPORTED` (see `results/verdict.txt`).

## Layout

| folder | contents |
|---|---|
| `code/` | `s4_anytime.py` (beta and gridworld runs), `s4_extensions.py` (synthetic + BBF), `s4_real_data.py` (TD-MPC2 and fresh runs), `s4_fresh_runs.py` (generates `data/fresh/`), `make_figures.py`, `make_figures_real.py`; `code/lib/` is the vendored shared library (`lib/anytime.py` is the `BettingCS` tool) |
| `data/` | `bbf/`, `tdmpc2/` (published per-seed scores), `fresh/` (runs produced by `code/s4_fresh_runs.py`) |
| `results/` | summaries and per-seed CSVs, `verdict.txt` |
| `paper/` | `paper.md` (full), `paper_rlc.md`, `paper_short.md`, `figures/` (written by the two `make_figures*.py` scripts) |

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/s4_anytime.py            # --smoke / --full
.venv/bin/python code/s4_extensions.py
.venv/bin/python code/s4_real_data.py
.venv/bin/python code/make_figures.py && .venv/bin/python code/make_figures_real.py
```
