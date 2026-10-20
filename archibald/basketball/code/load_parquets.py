"""One-time load of the parquet drop into the `nba` database. Skips per-team
roster files and the duplicate unnumbered frame when a `_0` frame exists.
Only games with a `_done` marker are loaded."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from db import ensure, mark_loaded, write_frame

ROOT = Path(__file__).resolve().parents[1] / "data" / "2025-26"
LEAGUE = ROOT / "league"
GAMES = ROOT / "games"


def load_league() -> None:
    files = sorted(LEAGUE.glob("*.parquet"))
    numbered = {re.sub(r"_\d+$", "", path.stem) for path in files if re.search(r"_\d+$", path.stem)}
    for path in files:
        stem = path.stem
        if stem in numbered:
            continue
        if re.fullmatch(r"roster_[A-Z]{2,3}", stem) or re.fullmatch(r"team_[A-Z]{2,3}", stem):
            continue
        df = pd.read_parquet(path)
        if df.empty and len(df.columns) == 0:
            mark_loaded("league", stem, 0)
            print(f"league.{stem} empty")
            continue
        n_rows = write_frame("league", stem, df, replace=True)
        mark_loaded("league", stem, n_rows)
        parent = re.sub(r"_\d+$", "", stem)
        if parent != stem:
            mark_loaded("league", parent, n_rows)
        print(f"league.{stem} {n_rows}")


def load_games() -> None:
    done = {path.stem for path in (GAMES / "_done").glob("*.json")}
    folders = [path for path in GAMES.iterdir() if path.is_dir() and path.name != "_done"]
    counts = {folder.name: 0 for folder in folders}
    for game_id in sorted(done):
        for folder in folders:
            path = folder / f"{game_id}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            counts[folder.name] += write_frame("games", folder.name, df)
        detail = (GAMES / "_done" / f"{game_id}.json").read_text()
        json.loads(detail)
        mark_loaded("games", game_id, 0, detail)
        print(f"game {game_id}")
    index = ROOT / "games_index.parquet"
    if index.exists():
        write_frame("games", "games_index", pd.read_parquet(index), replace=True)
    print("rows", counts)


def main() -> None:
    ensure()
    load_league()
    load_games()


if __name__ == "__main__":
    main()
