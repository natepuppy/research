# Follow-up probes

Scripts that cut across N1-1, S4-1, and S4-46 after the first grids. They are not experiments of their own, and none of them is cited by a paper. `push_again.py` and `push_further.py` are also copied into `research/independent/N1-1/code/probes/`, which is the copy the N1-1 paper refers to.

| file | what it checks |
|---|---|
| `code/probes/explore_reformulations.py` | Whether extra rotation stabilizes Baird, how many seeds a betting sequence needs, and whether a nonlinear value function on Baird sharpens |
| `code/probes/tweak_followups.py` | Damping the unstable symmetric mode on random linear MDPs and on nonlinear Baird, and betting versus a t-test when the effect size is unknown |
| `code/probes/push_further.py`, `push_again.py` | The same probes the N1-1 paper cites |
| `code/self_check.py` | Curl estimator, Lanczos eigenvalue, Baird divergence, and betting-sequence coverage. No training |

`data/`, `results/`, and `paper/` are empty: these scripts print to the terminal.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/self_check.py
.venv/bin/python code/probes/explore_reformulations.py
```
