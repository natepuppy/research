# Football (NFL tracking data)

Idea-stage project. No hypothesis committed yet.

## Layout

| folder | contents |
|---|---|
| `code/tracking_viewer/` | `server.py` + `static/`: local viewer for Big Data Bowl tracking frames (reads `data/big_data_bowl_2026/`) |
| `data/` | `README.md` documents the local Postgres `nfl` database; `big_data_bowl_2026/` holds the Kaggle evaluation harness |
| `results/` | empty |
| `paper/existing_papers/` | related work (PDFs) |

## Run

```bash
python3 code/tracking_viewer/server.py
```
