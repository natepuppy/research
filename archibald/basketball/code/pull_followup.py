"""Pull what the follow-up needs, without touching the 2025-26 league tables.

1. 2024-25 team advanced, misc, rebounding, and rosters, plus a spread of
   regular-season games (play-by-play and the four box scores the rebounding
   and push checks use), into schema s2425.
2. The 2025-26 regular-season games that are not loaded yet, same five
   endpoints, into the games schema. Those rows are marked partial so the
   full 11-endpoint puller will still refetch them later.

    python code/pull_followup.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import psycopg
from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscoremiscv3,
    boxscoreplayertrackv3,
    boxscoretraditionalv3,
    commonteamroster,
    leaguedashptstats,
    leaguedashteamstats,
    leaguegamelog,
    playbyplayv3,
)
from nba_api.stats.library.parameters import PtMeasureType
from nba_api.stats.static import teams as static_teams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_2025_26 as base  # noqa: E402
from db import clear_game, connect, mark_loaded, write_frame  # noqa: E402

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
SCHEMA = "s2425"
ENDPOINTS = (
    ("pbp", playbyplayv3.PlayByPlayV3),
    ("box_traditional", boxscoretraditionalv3.BoxScoreTraditionalV3),
    ("box_advanced", boxscoreadvancedv3.BoxScoreAdvancedV3),
    ("box_misc", boxscoremiscv3.BoxScoreMiscV3),
    ("box_tracking", boxscoreplayertrackv3.BoxScorePlayerTrackV3),
)


def ensure_schema() -> None:
    with connect() as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.loaded (
                name text PRIMARY KEY,
                n_rows bigint,
                loaded_at timestamptz DEFAULT now()
            )
            """
        )
        conn.commit()


def seen(name: str) -> bool:
    with connect() as conn:
        row = conn.execute(f"SELECT 1 FROM {SCHEMA}.loaded WHERE name = %s", (name,)).fetchone()
    return row is not None


def mark(name: str, n_rows: int) -> None:
    with connect() as conn:
        conn.execute(
            f"""
            INSERT INTO {SCHEMA}.loaded (name, n_rows)
            VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET n_rows = EXCLUDED.n_rows, loaded_at = now()
            """,
            (name, n_rows),
        )
        conn.commit()


def clear_schema_game(game_id: str) -> None:
    with connect() as conn:
        tables = conn.execute(
            """
            SELECT table_name FROM information_schema.columns
            WHERE table_schema = %s AND column_name = 'game_id'
              AND table_name <> 'loaded'
            """,
            (SCHEMA,),
        ).fetchall()
        for (table,) in tables:
            conn.execute(f"DELETE FROM {SCHEMA}.{table} WHERE game_id = %s", (game_id,))
        conn.commit()


def save_obj(schema: str, game_id: str, kind: str, obj) -> None:
    frames = base.frames_of(obj)
    if kind == "pbp":
        df = frames[0] if frames else pd.DataFrame({"game_id": [game_id]})
        if "game_id" not in df.columns and "gameId" not in df.columns and "GAME_ID" not in df.columns:
            df = df.copy()
            df["game_id"] = game_id
        write_frame(schema, "pbp", df)
        return
    parts = base.split_frames(frames)
    if not parts:
        write_frame(schema, f"{kind}_player", pd.DataFrame({"game_id": [game_id]}))
        return
    for role, df in parts.items():
        if "gameId" not in df.columns and "GAME_ID" not in df.columns and "game_id" not in df.columns:
            df = df.copy()
            df["game_id"] = game_id
        write_frame(schema, f"{kind}_{role}", df)


def pull_game(schema: str, game_id: str) -> dict:
    saved = {}
    for kind, factory in ENDPOINTS:
        base.log(f"{schema} {game_id} {kind}")
        try:
            obj = base.fetch(factory, game_id=game_id)
        except Exception as exc:  # noqa: BLE001
            saved[kind] = f"error: {exc.__class__.__name__}"
            base.log(f"{game_id} {kind} failed: {exc.__class__.__name__}")
            continue
        save_obj(schema, game_id, kind, obj)
        saved[kind] = "ok"
    return saved


def pull_league_2425() -> None:
    if not seen("team_advanced"):
        base.log("2024-25 team advanced")
        obj = base.fetch(
            leaguedashteamstats.LeagueDashTeamStats,
            season="2024-25",
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        frames = base.frames_of(obj)
        mark("team_advanced", write_frame(SCHEMA, "team_advanced", frames[0], replace=True))
    if not seen("team_misc"):
        base.log("2024-25 team misc")
        obj = base.fetch(
            leaguedashteamstats.LeagueDashTeamStats,
            season="2024-25",
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Misc",
            per_mode_detailed="PerGame",
        )
        frames = base.frames_of(obj)
        mark("team_misc", write_frame(SCHEMA, "team_misc", frames[0], replace=True))
    if not seen("tracking_rebounding"):
        base.log("2024-25 team rebounding")
        obj = base.fetch(
            leaguedashptstats.LeagueDashPtStats,
            season="2024-25",
            season_type_all_star="Regular Season",
            pt_measure_type=PtMeasureType.rebounding,
            player_or_team="Team",
            per_mode_simple="PerGame",
        )
        frames = base.frames_of(obj)
        mark("tracking_rebounding", write_frame(SCHEMA, "tracking_rebounding", frames[0], replace=True))
    if not seen("rosters"):
        frames = []
        for team in static_teams.get_teams():
            base.log(f"2024-25 roster {team['abbreviation']}")
            try:
                obj = base.fetch(commonteamroster.CommonTeamRoster, team_id=team["id"], season="2024-25")
            except Exception as exc:  # noqa: BLE001
                base.log(f"roster {team['abbreviation']} failed: {exc.__class__.__name__}")
                continue
            got = base.frames_of(obj)
            if got:
                df = got[0].copy()
                df["team_id"] = team["id"]
                df["team_abbreviation"] = team["abbreviation"]
                frames.append(df)
        if frames:
            mark("rosters", write_frame(SCHEMA, "rosters", pd.concat(frames, ignore_index=True), replace=True))
    if not seen("game_log_team"):
        base.log("2024-25 team game log")
        obj = base.fetch(
            leaguegamelog.LeagueGameLog,
            season="2024-25",
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
        )
        frames = base.frames_of(obj)
        mark("game_log_team", write_frame(SCHEMA, "game_log_team", frames[0], replace=True))
    base.log("LEAGUE READY")


def sample_game_ids() -> list[str]:
    with psycopg.connect(DSN) as conn:
        df = pd.read_sql(
            "SELECT DISTINCT game_id, game_date FROM s2425.game_log_team",
            conn,
        )
    df["game_id"] = df["game_id"].map(lambda v: str(v).split(".")[0].zfill(10))
    df = df.sort_values(["game_date", "game_id"]).drop_duplicates("game_id")
    # every 8th game: about 150 games, about 10 per team, spread across the season
    return df.iloc[::8]["game_id"].tolist()


def pull_sample() -> None:
    ids = [g for g in sample_game_ids() if not seen(g)]
    base.log(f"2024-25 sample todo {len(ids)}")
    for i, game_id in enumerate(ids, 1):
        clear_schema_game(game_id)
        saved = pull_game(SCHEMA, game_id)
        if all(v == "ok" for v in saved.values()):
            mark(game_id, 0)
        else:
            base.log(f"{game_id} incomplete {saved}")
        if i % 10 == 0:
            base.log(f"sample progress {i}/{len(ids)}")
    base.log("SAMPLE DONE")


def missing_2025_26() -> list[str]:
    with psycopg.connect(DSN) as conn:
        df = pd.read_sql(
            """
            SELECT DISTINCT g.game_id
            FROM league.game_log_team_regular_season g
            WHERE g.game_id LIKE '00225%'
              AND NOT EXISTS (
                SELECT 1 FROM games.box_tracking_player t WHERE t.game_id = g.game_id
              )
            ORDER BY 1
            """,
            conn,
        )
    return [str(v).zfill(10) for v in df["game_id"]]


def pull_rest_of_season() -> None:
    ids = missing_2025_26()
    base.log(f"2025-26 backfill todo {len(ids)}")
    for i, game_id in enumerate(ids, 1):
        clear_game(game_id)
        saved = pull_game("games", game_id)
        saved["partial"] = True
        mark_loaded("games", game_id, 0, json.dumps(saved))
        if i % 10 == 0:
            base.log(f"backfill progress {i}/{len(ids)}")
    base.log("BACKFILL DONE")


def main() -> None:
    base.SLEEP = 2.0
    ensure_schema()
    pull_league_2425()
    pull_sample()
    pull_rest_of_season()
    base.log("followup pull done")


if __name__ == "__main__":
    main()
