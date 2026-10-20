#!/usr/bin/env python3
"""Local viewer for 2025-26 shot locations in the nba database."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PORT = 8766
GAME_ID = re.compile(r"^\d{10}$")
CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def psql_rows(sql: str) -> list[dict]:
    result = subprocess.run(
        ["psql", "nba", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER"],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def clock_label(value: str) -> str:
    match = CLOCK.fullmatch(value or "")
    if not match:
        return value or ""
    minutes = int(match.group(1))
    seconds = int(float(match.group(2)))
    return f"{minutes}:{seconds:02d}"


def sides(matchup: str) -> tuple[str, str]:
    if " @ " in matchup:
        away, home = matchup.split(" @ ", 1)
        return away.strip(), home.strip()
    if " vs. " in matchup:
        home, away = matchup.split(" vs. ", 1)
        return away.strip(), home.strip()
    if " vs " in matchup:
        home, away = matchup.split(" vs ", 1)
        return away.strip(), home.strip()
    return "", matchup


def num(value: str, places: int = 1):
    if value is None or value == "":
        return None
    return round(float(value), places)


def games() -> list[dict]:
    rows = psql_rows(
        """
        SELECT DISTINCT ON (g.game_id)
               g.game_id, g.game_date, g.matchup, g.season_type,
               (SELECT count(*) FROM games.pbp p
                WHERE p.game_id = g.game_id AND p.is_field_goal = 1) AS shots
        FROM games.games_index g
        WHERE EXISTS (SELECT 1 FROM games.pbp p WHERE p.game_id = g.game_id)
        ORDER BY g.game_id, g.game_date
        """
    )
    listed = []
    for row in rows:
        away, home = sides(row["matchup"])
        listed.append(
            {
                "id": row["game_id"],
                "date": row["game_date"],
                "seasonType": row["season_type"],
                "matchup": row["matchup"],
                "home": home,
                "away": away,
                "shots": int(row["shots"] or 0),
            }
        )
    listed.sort(key=lambda game: (game["date"], game["id"]))
    return listed


def game_payload(game_id: str) -> dict | None:
    if not GAME_ID.fullmatch(game_id):
        return None
    meta = next((game for game in games() if game["id"] == game_id), None)
    if meta is None:
        return None
    actions = []
    for row in psql_rows(
        f"""
        SELECT action_number::text, period::text, clock, team_tricode, person_id::text,
               player_name, action_type, sub_type, description,
               x_legacy::text, y_legacy::text, shot_distance::text, shot_result,
               is_field_goal::text, shot_value::text, score_home, score_away
        FROM games.pbp
        WHERE game_id = '{game_id}'
        ORDER BY action_number
        """
    ):
        x = None if row["x_legacy"] in ("", None) else round(float(row["x_legacy"]) / 10, 2)
        y = None if row["y_legacy"] in ("", None) else round(float(row["y_legacy"]) / 10, 2)
        shot = row["is_field_goal"] == "1" or row["action_type"] == "Heave"
        placed = shot and not (x in (None, 0) and y in (None, 0))
        actions.append(
            {
                "n": int(row["action_number"]),
                "period": int(row["period"]) if row["period"] else None,
                "clock": clock_label(row["clock"]),
                "team": row["team_tricode"] or "",
                "playerId": row["person_id"] or "",
                "player": row["player_name"] or "",
                "action": row["action_type"] or "",
                "sub": row["sub_type"] or "",
                "description": row["description"] or row["action_type"] or "Event",
                "x": x if placed else None,
                "y": y if placed else None,
                "distance": num(row["shot_distance"], 0),
                "result": row["shot_result"] or "",
                "value": num(row["shot_value"], 0),
                "scoreHome": row["score_home"] or "",
                "scoreAway": row["score_away"] or "",
                "shot": placed,
            }
        )
    players = []
    for row in psql_rows(
        f"""
        SELECT person_id::text, first_name, family_name, name_i, team_tricode, position,
               jersey_num, minutes, points::text, field_goals_made::text,
               field_goals_attempted::text, three_pointers_made::text,
               three_pointers_attempted::text, rebounds_total::text, assists::text, comment
        FROM games.box_traditional_player
        WHERE game_id = '{game_id}'
        ORDER BY team_tricode, points DESC NULLS LAST
        """
    ):
        name = row["name_i"] or f"{row['first_name']} {row['family_name']}".strip()
        players.append(
            {
                "id": row["person_id"],
                "name": name,
                "team": row["team_tricode"],
                "position": row["position"] or "",
                "jersey": row["jersey_num"] or "",
                "minutes": row["minutes"] or "",
                "points": num(row["points"], 0),
                "fgm": num(row["field_goals_made"], 0),
                "fga": num(row["field_goals_attempted"], 0),
                "tpm": num(row["three_pointers_made"], 0),
                "tpa": num(row["three_pointers_attempted"], 0),
                "reb": num(row["rebounds_total"], 0),
                "ast": num(row["assists"], 0),
                "comment": row["comment"] or "",
            }
        )
    return {**meta, "actions": actions, "players": players}


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
            if path == "/api/games":
                body = games()
            elif path == "/api/game":
                body = game_payload(query["id"])
                if body is None:
                    self.send_error(404, "Game not found")
                    return
            else:
                self.send_error(404, "Unknown API")
                return
        except (KeyError, ValueError, subprocess.CalledProcessError) as exc:
            self.send_error(400, str(exc))
            return
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Shot viewer at http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
