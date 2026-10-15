#!/usr/bin/env python3
"""Local viewer for game plays and Combine drills."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "big_data_bowl_2026"  # football/data
STATIC = ROOT / "static"
PORT = 8765

_lock = threading.Lock()
_index: dict = {}
_week_cache: dict = {"week": None, "plays": {}}
_combine: dict = {"reps": {}}
_EVENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")

_DRILL_GROUP = {
    "FORTY_YARD_DASH": "Timed",
    "SHORT_SHUTTLE": "Timed",
    "THREE_CONE_DRILL": "Timed",
    "SKILL_DRILLS_DB": "DB",
    "SKILL_DRILLS_DL": "DL",
    "SKILL_DRILLS_LB": "LB",
    "SKILL_DRILLS_OL": "OL",
    "SKILL_DRILLS_TE": "TE",
    "SKILL_DRILLS_WR": "WR",
}


def _null(value: str):
    if value is None or value in ("", "NA", "nan", "NaN"):
        return None
    return value


def _num(value: str, places: int = 2):
    value = _null(value)
    if value is None:
        return None
    return round(float(value), places)


def psql_rows(sql: str) -> list[dict]:
    result = subprocess.run(
        ["psql", "nfl", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER"],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def drill_label(drill_type: str, drill_name: str) -> str:
    group = _DRILL_GROUP.get(drill_type, drill_type.replace("_", " ").title())
    name = drill_name.replace("_", " ").title()
    return f"{group} · {name}"


def height_text(inches: str | None) -> str | None:
    if not inches:
        return None
    total = float(inches)
    feet = int(total // 12)
    rest = round(total - feet * 12, 1)
    if rest == int(rest):
        return f"{feet}-{int(rest)}"
    return f"{feet}-{rest}"


def load_index() -> None:
    games: dict[int, dict[str, dict]] = {}
    plays: dict[tuple[str, str], dict] = {}
    with (DATA / "supplementary_data.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["season"] != "2023":
                continue
            week = int(row["week"])
            game_id = row["game_id"]
            game = games.setdefault(week, {}).setdefault(
                game_id,
                {
                    "id": game_id,
                    "week": week,
                    "home": row["home_team_abbr"],
                    "away": row["visitor_team_abbr"],
                    "date": row["game_date"],
                },
            )
            plays[(game_id, row["play_id"])] = {
                "gameId": game_id,
                "playId": row["play_id"],
                "week": week,
                "home": game["home"],
                "away": game["away"],
                "offense": row["possession_team"],
                "defense": row["defensive_team"],
                "quarter": int(row["quarter"]) if _null(row["quarter"]) else None,
                "clock": row["game_clock"],
                "down": int(row["down"]) if _null(row["down"]) else None,
                "distance": int(float(row["yards_to_go"])) if _null(row["yards_to_go"]) else None,
                "description": row["play_description"],
                "result": row["pass_result"],
                "yards": _num(row["yards_gained"], 0),
                "epa": _num(row["expected_points_added"]),
                "formation": _null(row["offense_formation"]),
                "coverage": _null(row["team_coverage_type"]),
                "route": _null(row["route_of_targeted_receiver"]),
                "playAction": row["play_action"] in ("TRUE", "True", "true", "1"),
                "homeScore": _num(row["pre_snap_home_score"], 0),
                "awayScore": _num(row["pre_snap_visitor_score"], 0),
            }
    _index["games"] = games
    _index["plays"] = plays


def load_week(week: int) -> dict:
    with _lock:
        if _week_cache["week"] == week:
            return _week_cache["plays"]
        plays: dict[tuple[str, str], dict] = {}

        def bucket(game_id: str, play_id: str) -> dict:
            return plays.setdefault(
                (game_id, play_id),
                {"players": {}, "direction": None, "yardline": None, "ball": None},
            )

        input_path = DATA / "train" / f"input_2023_w{week:02d}.csv"
        with input_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                play = bucket(row["game_id"], row["play_id"])
                if play["direction"] is None:
                    play["direction"] = row["play_direction"]
                    play["yardline"] = _num(row["absolute_yardline_number"], 1)
                    play["ball"] = [_num(row["ball_land_x"]), _num(row["ball_land_y"])]
                player = play["players"].setdefault(
                    row["nfl_id"],
                    {
                        "id": row["nfl_id"],
                        "name": row["player_name"],
                        "position": row["player_position"],
                        "side": row["player_side"],
                        "role": row["player_role"],
                        "height": row["player_height"],
                        "weight": _num(row["player_weight"], 0),
                        "predict": row["player_to_predict"] in ("True", "true", "1"),
                        "input": [],
                        "output": [],
                    },
                )
                player["input"].append(
                    {
                        "f": int(row["frame_id"]),
                        "x": _num(row["x"]),
                        "y": _num(row["y"]),
                        "s": _num(row["s"]),
                        "a": _num(row["a"]),
                        "o": _num(row["o"], 1),
                        "dir": _num(row["dir"], 1),
                    }
                )

        output_path = DATA / "train" / f"output_2023_w{week:02d}.csv"
        with output_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                play = plays.get((row["game_id"], row["play_id"]))
                if play is None or row["nfl_id"] not in play["players"]:
                    continue
                play["players"][row["nfl_id"]]["output"].append(
                    {"f": int(row["frame_id"]), "x": _num(row["x"]), "y": _num(row["y"])}
                )

        for play in plays.values():
            for player in play["players"].values():
                player["input"].sort(key=lambda frame: frame["f"])
                player["output"].sort(key=lambda frame: frame["f"])
        _week_cache["week"] = week
        _week_cache["plays"] = plays
        return plays


def load_index_db() -> None:
    games: dict[int, dict[str, dict]] = {}
    plays: dict[tuple[str, str], dict] = {}
    rows = psql_rows(
        """
        SELECT g.week::text, g.game_id::text, g.home_team_abbr, g.visitor_team_abbr,
               g.game_date::text, p.play_id::text, p.quarter::text, p.game_clock,
               p.down::text, p.yards_to_go::text, p.possession_team, p.defensive_team,
               p.play_description, p.pass_result, p.yards_gained::text,
               p.expected_points_added::text, p.offense_formation, p.team_coverage_type,
               p.route_of_targeted_receiver, p.play_action::text,
               p.pre_snap_home_score::text, p.pre_snap_visitor_score::text
        FROM bdb2026.plays p
        JOIN bdb2026.games g USING (game_id)
        WHERE g.season = 2023
          AND EXISTS (
            SELECT 1 FROM bdb2026.tracking_before tb
            WHERE tb.game_id = p.game_id AND tb.play_id = p.play_id
          )
        """
    )
    for row in rows:
        week = int(row["week"])
        game_id = row["game_id"]
        games.setdefault(week, {}).setdefault(
            game_id,
            {
                "id": game_id,
                "week": week,
                "home": row["home_team_abbr"],
                "away": row["visitor_team_abbr"],
                "date": row["game_date"],
            },
        )
        plays[(game_id, row["play_id"])] = {
            "gameId": game_id,
            "playId": row["play_id"],
            "week": week,
            "home": row["home_team_abbr"],
            "away": row["visitor_team_abbr"],
            "offense": row["possession_team"],
            "defense": row["defensive_team"],
            "quarter": int(row["quarter"]) if row["quarter"] else None,
            "clock": row["game_clock"],
            "down": int(row["down"]) if row["down"] else None,
            "distance": int(row["yards_to_go"]) if row["yards_to_go"] else None,
            "description": row["play_description"],
            "result": row["pass_result"],
            "yards": _num(row["yards_gained"], 0),
            "epa": _num(row["expected_points_added"]),
            "formation": _null(row["offense_formation"]),
            "coverage": _null(row["team_coverage_type"]),
            "route": _null(row["route_of_targeted_receiver"]),
            "playAction": row["play_action"] == "true",
            "homeScore": _num(row["pre_snap_home_score"], 0),
            "awayScore": _num(row["pre_snap_visitor_score"], 0),
        }
    _index["games"] = games
    _index["plays"] = plays
    _index["source"] = "postgres"


def play_payload_db(week: int, game_id: str, play_id: str) -> dict | None:
    meta = _index["plays"].get((game_id, play_id))
    if meta is None or meta["week"] != week:
        return None
    game_sql = int(game_id)
    play_sql = int(play_id)
    roster = psql_rows(
        f"""
        SELECT pp.nfl_id::text, pl.player_name, pl.player_position, pl.player_height,
               pl.player_weight::text, pp.player_side, pp.player_role,
               pp.player_to_predict::text, pp.play_direction,
               pp.absolute_yardline_number::text, pp.ball_land_x::text, pp.ball_land_y::text
        FROM bdb2026.player_plays pp
        JOIN bdb2026.players pl USING (nfl_id)
        WHERE pp.game_id = {game_sql} AND pp.play_id = {play_sql}
        """
    )
    if not roster:
        return None
    players = {
        row["nfl_id"]: {
            "id": row["nfl_id"],
            "name": row["player_name"],
            "position": row["player_position"],
            "side": row["player_side"],
            "role": row["player_role"],
            "height": row["player_height"],
            "weight": _num(row["player_weight"], 0),
            "predict": row["player_to_predict"] == "true",
            "input": [],
            "output": [],
        }
        for row in roster
    }
    for phase, table, extra in (
        ("input", "tracking_before", ", s::text, a::text, o::text, dir::text"),
        ("output", "tracking_after", ""),
    ):
        frames = psql_rows(
            f"""
            SELECT nfl_id::text, frame_id::text, x::text, y::text{extra}
            FROM bdb2026.{table}
            WHERE game_id = {game_sql} AND play_id = {play_sql}
            ORDER BY nfl_id, frame_id
            """
        )
        for row in frames:
            player = players.get(row["nfl_id"])
            if player is None:
                continue
            frame = {"f": int(row["frame_id"]), "x": _num(row["x"]), "y": _num(row["y"])}
            if phase == "input":
                frame.update(
                    {
                        "s": _num(row["s"]),
                        "a": _num(row["a"]),
                        "o": _num(row["o"], 1),
                        "dir": _num(row["dir"], 1),
                    }
                )
            player[phase].append(frame)
    sample = roster[0]
    listed = list(players.values())
    return {
        **meta,
        "kind": "play",
        "direction": sample["play_direction"],
        "yardline": _num(sample["absolute_yardline_number"], 1),
        "ballLand": [_num(sample["ball_land_x"]), _num(sample["ball_land_y"])],
        "inputFrames": max((frame["f"] for player in listed for frame in player["input"]), default=0),
        "outputFrames": max((frame["f"] for player in listed for frame in player["output"]), default=0),
        "players": listed,
    }


def load_combine() -> None:
    rows = psql_rows(
        """
        SELECT DISTINCT t.event_id, t.draft_year::text, t.drill_type, t.drill_name,
               t.attempt::text, t.nfl_id::text, p.display_name, p.nfl_position
        FROM bdb2027.combine_tracking t
        JOIN bdb2027.players p USING (nfl_id)
        WHERE t.entity_type = 'PLAYER'
        """
    )
    reps: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['draft_year']}|{row['drill_type']}|{row['drill_name']}"
        reps.setdefault(key, []).append(
            {
                "event": row["event_id"],
                "name": row["display_name"],
                "position": row["nfl_position"],
                "attempt": int(row["attempt"]) if row["attempt"] else None,
            }
        )
    for bucket in reps.values():
        bucket.sort(key=lambda rep: (rep["name"] or "", rep["attempt"] or 0))
    _combine["reps"] = reps


def combine_index() -> dict:
    drills: dict[str, list[dict]] = {}
    years = set()
    for key in _combine["reps"]:
        year, drill_type, drill_name = key.split("|", 2)
        years.add(int(year))
        drills.setdefault(year, []).append(
            {
                "key": key,
                "label": drill_label(drill_type, drill_name),
            }
        )
    for year_drills in drills.values():
        year_drills.sort(key=lambda drill: drill["label"])
    return {
        "years": sorted(years),
        "drills": drills,
        "reps": _combine["reps"],
    }


def combine_payload(event_id: str) -> dict | None:
    if not _EVENT_ID.fullmatch(event_id):
        return None
    rep = next(
        (item for bucket in _combine["reps"].values() for item in bucket if item["event"] == event_id),
        None,
    )
    if rep is None:
        return None
    key = next(name for name, bucket in _combine["reps"].items() if any(item["event"] == event_id for item in bucket))
    year, drill_type, drill_name = key.split("|", 2)
    info = psql_rows(
        f"""
        SELECT p.college_name, p.draft_round::text, p.draft_overall_pick::text,
               r.combine_height::text, r.combine_weight::text, r.forty::text,
               r.ten_yd_split::text, r.short_shuttle::text, r.three_cone::text,
               r.vertical::text, r.broad_jump::text
        FROM bdb2027.players p
        LEFT JOIN bdb2027.combine_results r USING (nfl_id)
        WHERE p.nfl_id = (
          SELECT nfl_id FROM bdb2027.combine_tracking
          WHERE event_id = '{event_id}' AND entity_type = 'PLAYER' LIMIT 1
        )
        """
    )
    detail = info[0] if info else {}
    frames = psql_rows(
        f"""
        SELECT entity_type, time::text, x::text, y::text, s::text, a::text, dir::text
        FROM bdb2027.combine_tracking
        WHERE event_id = '{event_id}'
        ORDER BY time, entity_type
        """
    )
    if not frames:
        return None
    start = frames[0]["time"]

    def stamp(value: str) -> datetime:
        return datetime.fromisoformat(value)

    origin = stamp(start)
    player_frames = []
    ball_frames = []
    xs = []
    ys = []
    for row in frames:
        frame_no = int(round((stamp(row["time"]) - origin).total_seconds() * 10)) + 1
        point = {"f": frame_no, "x": _num(row["x"]), "y": _num(row["y"])}
        if point["x"] is None or point["y"] is None:
            continue
        xs.append(point["x"])
        ys.append(point["y"])
        if row["entity_type"] == "BALL":
            ball_frames.append(point)
        else:
            player_frames.append(
                {
                    **point,
                    "s": _num(row["s"]),
                    "a": _num(row["a"]),
                    "o": _num(row["dir"], 1),
                    "dir": _num(row["dir"], 1),
                }
            )
    marks = [
        ("Forty", _num(detail.get("forty"))),
        ("10-yard split", _num(detail.get("ten_yd_split"))),
        ("Short shuttle", _num(detail.get("short_shuttle"))),
        ("Three cone", _num(detail.get("three_cone"))),
        ("Vertical", _num(detail.get("vertical"), 1)),
        ("Broad jump", _num(detail.get("broad_jump"), 0)),
    ]
    draft = None
    if detail.get("draft_overall_pick"):
        draft = f"Round {detail.get('draft_round')}, pick {detail.get('draft_overall_pick')}"
    label = drill_label(drill_type, drill_name)
    attempt = f"rep {rep['attempt']}" if rep["attempt"] else "rep"
    return {
        "kind": "combine",
        "year": int(year),
        "drill": label,
        "attempt": rep["attempt"],
        "description": f"{rep['name']}, {rep['position']} — {label}, {attempt}",
        "college": detail.get("college_name"),
        "draft": draft,
        "marks": [[name, value] for name, value in marks if value is not None],
        "direction": None,
        "yardline": None,
        "ballLand": None,
        "bounds": [min(xs), min(ys), max(xs), max(ys)] if xs else None,
        "inputFrames": max((frame["f"] for frame in player_frames), default=1),
        "outputFrames": 0,
        "ballFrames": ball_frames,
        "players": [
            {
                "id": event_id,
                "name": rep["name"],
                "position": rep["position"],
                "side": "Athlete",
                "role": label,
                "height": height_text(detail.get("combine_height")),
                "weight": _num(detail.get("combine_weight"), 0),
                "predict": True,
                "input": player_frames,
                "output": [],
            }
        ],
    }


def play_payload(week: int, game_id: str, play_id: str) -> dict | None:
    if _index.get("source") == "postgres":
        return play_payload_db(week, game_id, play_id)
    meta = _index["plays"].get((game_id, play_id))
    if meta is None or meta["week"] != week:
        return None
    tracking = load_week(week).get((game_id, play_id))
    if tracking is None:
        return None
    players = list(tracking["players"].values())
    input_frames = max((frame["f"] for player in players for frame in player["input"]), default=0)
    output_frames = max((frame["f"] for player in players for frame in player["output"]), default=0)
    return {
        **meta,
        "direction": tracking["direction"],
        "yardline": tracking["yardline"],
        "ballLand": tracking["ball"],
        "inputFrames": input_frames,
        "outputFrames": output_frames,
        "players": players,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/"):
            self._api(path)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _api(self, path: str) -> None:
        query = {}
        if "?" in self.path:
            for part in self.path.split("?", 1)[1].split("&"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    query[key] = unquote(value)
        try:
            if path == "/api/index":
                games = {
                    str(week): list(week_games.values())
                    for week, week_games in sorted(_index["games"].items())
                }
                body = {"weeks": list(range(1, 19)), "games": games}
            elif path == "/api/plays":
                week = int(query["week"])
                game_id = query["game"]
                body = [
                    play
                    for play in _index["plays"].values()
                    if play["week"] == week and play["gameId"] == game_id
                ]
                body.sort(key=lambda play: int(play["playId"]))
            elif path == "/api/play":
                body = play_payload(int(query["week"]), query["game"], query["play"])
                if body is None:
                    self.send_error(404, "Play not found")
                    return
            elif path == "/api/combine":
                body = combine_index()
            elif path == "/api/combine/rep":
                body = combine_payload(query["event"])
                if body is None:
                    self.send_error(404, "Rep not found")
                    return
            else:
                self.send_error(404, "Unknown API")
                return
        except (KeyError, ValueError) as exc:
            self.send_error(400, str(exc))
            return
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    print("Reading play index…", flush=True)
    if (DATA / "supplementary_data.csv").exists():
        load_index()
        print(f"Warming week 1 ({len(_index['plays'])} plays indexed)…", flush=True)
        load_week(1)
    else:
        load_index_db()
        print(f"{len(_index['plays'])} plays indexed from Postgres", flush=True)
    print("Reading Combine reps…", flush=True)
    load_combine()
    print(f"{sum(len(bucket) for bucket in _combine['reps'].values())} Combine reps", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Play viewer at http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
