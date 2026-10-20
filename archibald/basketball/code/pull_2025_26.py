"""Pull the 2025-26 NBA season from stats.nba.com via nba_api.

League tables are one request each. Per-game tables (play-by-play and the box
score variants the game log does not contain) are one request per game. Rows
already in the local Postgres database `nba` are skipped, so a stopped run
can be continued.

    python code/pull_2025_26.py
    python code/pull_2025_26.py --limit 2
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from db import clear_game, ensure, loaded, mark_loaded, read_table, write_frame
from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscoredefensivev2,
    boxscorefourfactorsv3,
    boxscorehustlev2,
    boxscorematchupsv3,
    boxscoremiscv3,
    boxscoreplayertrackv3,
    boxscorescoringv3,
    boxscoretraditionalv3,
    boxscoreusagev3,
    commonallplayers,
    commonplayoffseries,
    commonteamroster,
    iststandings,
    leaguedashlineups,
    leaguedashplayerbiostats,
    leaguedashplayerclutch,
    leaguedashplayerptshot,
    leaguedashplayershotlocations,
    leaguedashplayerstats,
    leaguedashptdefend,
    leaguedashptstats,
    leaguedashptteamdefend,
    leaguedashteamclutch,
    leaguedashteamptshot,
    leaguedashteamshotlocations,
    leaguedashteamstats,
    leaguegamelog,
    leaguehustlestatsplayer,
    leaguehustlestatsteam,
    leagueseasonmatchups,
    leaguestandings,
    playbyplayv3,
    playerestimatedmetrics,
    scheduleleaguev2,
    shotchartleaguewide,
    synergyplaytypes,
    teamdetails,
    teamestimatedmetrics,
)
from nba_api.stats.library.parameters import DefenseCategory, PlayType, PtMeasureType, TypeGrouping
from nba_api.stats.static import teams as static_teams

SEASON = "2025-26"
SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs", "Pre Season", "All Star")
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "pull.log"

MEASURES = ("Base", "Advanced", "Misc", "Scoring", "Usage", "Defense")
PT_MEASURES = (
    PtMeasureType.speed_distance,
    PtMeasureType.rebounding,
    PtMeasureType.possessions,
    PtMeasureType.catch_shoot,
    PtMeasureType.pull_up_shot,
    PtMeasureType.defense,
    PtMeasureType.drives,
    PtMeasureType.passing,
    PtMeasureType.elbowTouch,
    PtMeasureType.postTouch,
    PtMeasureType.paintTouch,
    PtMeasureType.efficiency,
)
PLAY_TYPES = (
    PlayType.transition,
    PlayType.isolation,
    PlayType.pr_ball_handler,
    PlayType.pr_roll_man,
    PlayType.post_up,
    PlayType.spot_up,
    PlayType.handoff,
    PlayType.cut,
    PlayType.off_screen,
    PlayType.putbacks,
    PlayType.misc,
)

# kind -> (endpoint class, which frames to keep)
# Frames are split into player / team / other by column names.
GAME_ENDPOINTS = (
    ("pbp", playbyplayv3.PlayByPlayV3),
    ("box_traditional", boxscoretraditionalv3.BoxScoreTraditionalV3),
    ("box_advanced", boxscoreadvancedv3.BoxScoreAdvancedV3),
    ("box_misc", boxscoremiscv3.BoxScoreMiscV3),
    ("box_scoring", boxscorescoringv3.BoxScoreScoringV3),
    ("box_usage", boxscoreusagev3.BoxScoreUsageV3),
    ("box_four_factors", boxscorefourfactorsv3.BoxScoreFourFactorsV3),
    ("box_tracking", boxscoreplayertrackv3.BoxScorePlayerTrackV3),
    ("box_hustle", boxscorehustlev2.BoxScoreHustleV2),
    ("box_matchups", boxscorematchupsv3.BoxScoreMatchupsV3),
    ("box_defensive", boxscoredefensivev2.BoxScoreDefensiveV2),
)


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(part) for part in col if str(part) not in ("", "None"))
            for col in df.columns
        ]
    return df


def fetch(factory, **kwargs):
    global REQUESTS
    last = None
    for attempt in range(3):
        try:
            REQUESTS += 1
            if REQUESTS % 40 == 0:
                log("pause 45s to stay under the API rate limit")
                time.sleep(45)
            time.sleep(SLEEP)
            return factory(timeout=90, **kwargs)
        except TypeError:
            raise
        except Exception as exc:  # noqa: BLE001 — an empty body is how stats.nba.com throttles
            last = exc
            if attempt == 2:
                break
            wait = 90 * (attempt + 1)
            log(f"retry {attempt + 1} {factory.__name__} in {wait}s: {exc.__class__.__name__}")
            time.sleep(wait)
    raise last


SLEEP = 2.5
REQUESTS = 0


def frames_of(obj) -> list[pd.DataFrame]:
    out = []
    for ds in obj.data_sets:
        df = flatten(ds.get_data_frame())
        if len(df):
            out.append(df)
    return out


def split_frames(frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    buckets: dict[str, list[pd.DataFrame]] = {"player": [], "team": [], "other": []}
    for df in frames:
        cols = {c.lower().replace("_", "") for c in df.columns}
        if "personid" in cols or "playerid" in cols:
            buckets["player"].append(df)
        elif "teamid" in cols:
            buckets["team"].append(df)
        else:
            buckets["other"].append(df)
    saved = {}
    for name, parts in buckets.items():
        if not parts:
            continue
        if len(parts) == 1:
            saved[name] = parts[0]
        else:
            for i, part in enumerate(parts):
                saved[f"{name}_{i}"] = part
    return saved


def pull_one(name: str, factory, **kwargs) -> None:
    if loaded("league", name):
        return
    log(f"league {name}")
    try:
        obj = fetch(factory, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log(f"league {name} failed: {exc.__class__.__name__}: {exc}")
        err = ROOT / "logs" / "errors.jsonl"
        err.parent.mkdir(parents=True, exist_ok=True)
        with err.open("a") as f:
            f.write(json.dumps({"table": name, "error": str(exc)}) + "\n")
        return
    frames = frames_of(obj)
    n_rows = 0
    if len(frames) <= 1:
        n_rows = write_frame("league", name, frames[0] if frames else pd.DataFrame(), replace=True)
    else:
        for i, df in enumerate(frames):
            n_rows += write_frame("league", f"{name}_{i}", df, replace=True)
    mark_loaded("league", name, n_rows)


def slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def pull_league() -> None:
    ensure()
    pull_one(
        "players",
        commonallplayers.CommonAllPlayers,
        season=SEASON,
        is_only_current_season=1,
    )
    pull_one("schedule", scheduleleaguev2.ScheduleLeagueV2, season=SEASON)
    pull_one("standings", leaguestandings.LeagueStandings, season=SEASON)
    pull_one("ist_standings_group", iststandings.ISTStandings, season=SEASON, section="group")
    pull_one("ist_standings_wildcard", iststandings.ISTStandings, season=SEASON, section="wildcard")
    try:
        pull_one("playoff_series", commonplayoffseries.CommonPlayoffSeries, season=SEASON)
    except Exception as exc:  # noqa: BLE001
        log(f"playoff series skipped: {exc}")

    if not loaded("league", "rosters"):
        roster_frames = []
        for team in static_teams.get_teams():
            log(f"roster {team['abbreviation']}")
            try:
                obj = fetch(commonteamroster.CommonTeamRoster, team_id=team["id"], season=SEASON)
            except Exception as exc:  # noqa: BLE001
                log(f"roster {team['abbreviation']} failed: {exc}")
                continue
            frames = frames_of(obj)
            if frames:
                df = frames[0].copy()
                df["team_id"] = team["id"]
                df["team_abbreviation"] = team["abbreviation"]
                roster_frames.append(df)
        if roster_frames:
            n_rows = write_frame("league", "rosters", pd.concat(roster_frames, ignore_index=True), replace=True)
            mark_loaded("league", "rosters", n_rows)

    if not loaded("league", "teams"):
        detail_frames = []
        for team in static_teams.get_teams():
            log(f"team {team['abbreviation']}")
            try:
                obj = fetch(teamdetails.TeamDetails, team_id=team["id"])
            except Exception as exc:  # noqa: BLE001
                log(f"team {team['abbreviation']} failed: {exc}")
                continue
            frames = frames_of(obj)
            if frames:
                detail_frames.append(frames[0])
        if detail_frames:
            n_rows = write_frame("league", "teams", pd.concat(detail_frames, ignore_index=True), replace=True)
            mark_loaded("league", "teams", n_rows)

    for season_type in SEASON_TYPES:
        tag = slug(season_type)
        for who, arg in (("team", "T"), ("player", "P")):
            pull_one(
                f"game_log_{who}_{tag}",
                leaguegamelog.LeagueGameLog,
                season=SEASON,
                season_type_all_star=season_type,
                player_or_team_abbreviation=arg,
            )

    for season_type in ("Regular Season", "Playoffs"):
        tag = slug(season_type)
        for measure in MEASURES:
            pull_one(
                f"player_{slug(measure)}_{tag}",
                leaguedashplayerstats.LeagueDashPlayerStats,
                season=SEASON,
                season_type_all_star=season_type,
                measure_type_detailed_defense=measure,
                per_mode_detailed="PerGame",
            )
            if measure != "Usage":
                pull_one(
                    f"team_{slug(measure)}_{tag}",
                    leaguedashteamstats.LeagueDashTeamStats,
                    season=SEASON,
                    season_type_all_star=season_type,
                    measure_type_detailed_defense=measure,
                    per_mode_detailed="PerGame",
                )
        pull_one(
            f"player_bio_{tag}",
            leaguedashplayerbiostats.LeagueDashPlayerBioStats,
            season=SEASON,
            season_type_all_star=season_type,
        )
        pull_one(
            f"player_clutch_{tag}",
            leaguedashplayerclutch.LeagueDashPlayerClutch,
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
        )
        pull_one(
            f"team_clutch_{tag}",
            leaguedashteamclutch.LeagueDashTeamClutch,
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
        )
        pull_one(
            f"hustle_player_{tag}",
            leaguehustlestatsplayer.LeagueHustleStatsPlayer,
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_time="PerGame",
        )
        pull_one(
            f"hustle_team_{tag}",
            leaguehustlestatsteam.LeagueHustleStatsTeam,
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_time="PerGame",
        )
        pull_one(
            f"estimated_player_{tag}",
            playerestimatedmetrics.PlayerEstimatedMetrics,
            season=SEASON,
            season_type=season_type,
        )
        pull_one(
            f"estimated_team_{tag}",
            teamestimatedmetrics.TeamEstimatedMetrics,
            season=SEASON,
            season_type=season_type,
        )
        if season_type == "Regular Season":
            pull_one("shot_zones", shotchartleaguewide.ShotChartLeagueWide, season=SEASON)
        pull_one(
            f"player_shot_locations_{tag}",
            leaguedashplayershotlocations.LeagueDashPlayerShotLocations,
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
        )
        pull_one(
            f"team_shot_locations_{tag}",
            leaguedashteamshotlocations.LeagueDashTeamShotLocations,
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
        )
        pull_one(
            f"player_pt_shot_{tag}",
            leaguedashplayerptshot.LeagueDashPlayerPtShot,
            season=SEASON,
            season_type_all_star=season_type,
        )
        pull_one(
            f"team_pt_shot_{tag}",
            leaguedashteamptshot.LeagueDashTeamPtShot,
            season=SEASON,
            season_type_all_star=season_type,
        )
        for category in (
            DefenseCategory.overall,
            DefenseCategory.threes,
            DefenseCategory.twos,
            DefenseCategory.less_than_6ft,
            DefenseCategory.less_than_10ft,
            DefenseCategory.greater_than_15ft,
        ):
            pull_one(
                f"pt_defend_{slug(category)}_{tag}",
                leaguedashptdefend.LeagueDashPtDefend,
                season=SEASON,
                season_type_all_star=season_type,
                defense_category=category,
                per_mode_simple="PerGame",
            )
            pull_one(
                f"pt_team_defend_{slug(category)}_{tag}",
                leaguedashptteamdefend.LeagueDashPtTeamDefend,
                season=SEASON,
                season_type_all_star=season_type,
                defense_category=category,
                per_mode_simple="PerGame",
            )
        pull_one(
            f"matchups_{tag}",
            leagueseasonmatchups.LeagueSeasonMatchups,
            season=SEASON,
            season_type_playoffs=season_type,
            per_mode_simple="Totals",
        )
        for measure in ("Base", "Advanced"):
            pull_one(
                f"lineups_{slug(measure)}_{tag}",
                leaguedashlineups.LeagueDashLineups,
                season=SEASON,
                season_type_all_star=season_type,
                measure_type_detailed_defense=measure,
                per_mode_detailed="PerGame",
                group_quantity=5,
            )
        for pt in PT_MEASURES:
            for who in ("Player", "Team"):
                pull_one(
                    f"tracking_{slug(pt)}_{who.lower()}_{tag}",
                    leaguedashptstats.LeagueDashPtStats,
                    season=SEASON,
                    season_type_all_star=season_type,
                    pt_measure_type=pt,
                    player_or_team=who,
                    per_mode_simple="PerGame",
                )
        for play_type in PLAY_TYPES:
            for grouping in (TypeGrouping.offensive, TypeGrouping.defensive):
                for who in ("P", "T"):
                    pull_one(
                        f"synergy_{slug(play_type)}_{grouping}_{'player' if who == 'P' else 'team'}_{tag}",
                        synergyplaytypes.SynergyPlayTypes,
                        season=SEASON,
                        season_type_all_star=season_type,
                        play_type_nullable=play_type,
                        type_grouping_nullable=grouping,
                        player_or_team_abbreviation=who,
                        per_mode_simple="PerGame",
                    )


def game_ids() -> pd.DataFrame:
    frames = []
    for season_type in SEASON_TYPES:
        df = read_table("league", f"game_log_team_{slug(season_type)}")
        if df.empty or "game_id" not in df.columns:
            continue
        df = df.copy()
        df["season_type"] = season_type
        frames.append(df)
    if not frames:
        raise SystemExit("no team game logs; run the league pull first")
    games = pd.concat(frames, ignore_index=True)
    games = games.drop_duplicates("game_id")
    keep = [c for c in ("game_id", "game_date", "matchup", "season_type", "team_abbreviation") if c in games.columns]
    games = games[keep]
    order = {"Regular Season": 0, "PlayIn": 1, "Playoffs": 2, "Pre Season": 3, "All Star": 4}
    games["_ord"] = games["season_type"].map(order).fillna(9)
    return games.sort_values(["_ord", "game_date", "game_id"]).drop(columns="_ord")


def game_done(game_id: str) -> bool:
    return loaded("games", game_id)


def pull_game(game_id: str) -> None:
    clear_game(game_id)
    saved = {}
    for kind, factory in GAME_ENDPOINTS:
        single = kind == "pbp"
        log(f"{game_id} {kind}")
        try:
            obj = fetch(factory, game_id=game_id)
        except Exception as exc:  # noqa: BLE001
            saved[kind] = f"error: {exc.__class__.__name__}"
            err = ROOT / "logs" / "errors.jsonl"
            err.parent.mkdir(parents=True, exist_ok=True)
            with err.open("a") as f:
                f.write(json.dumps({"game_id": game_id, "kind": kind, "error": str(exc)}) + "\n")
            log(traceback.format_exc().splitlines()[-1])
            continue
        frames = frames_of(obj)
        if single:
            df = frames[0] if frames else pd.DataFrame({"game_id": [game_id]})
            if "gameId" not in df.columns and "GAME_ID" not in df.columns:
                df = df.copy()
                df["game_id"] = game_id
            write_frame("games", kind, df)
        else:
            parts = split_frames(frames)
            if not parts:
                write_frame("games", f"{kind}_player", pd.DataFrame({"game_id": [game_id]}))
            for role, df in parts.items():
                if "gameId" not in df.columns and "GAME_ID" not in df.columns:
                    df = df.copy()
                    df["game_id"] = game_id
                write_frame("games", f"{kind}_{role}", df)
        saved[kind] = "ok"
    mark_loaded("games", game_id, 0, json.dumps(saved))


def pull_games(limit: int | None) -> None:
    games = game_ids()
    write_frame("games", "games_index", games, replace=True)
    todo = [g for g in games["game_id"].astype(str) if not game_done(g)]
    if limit is not None:
        todo = todo[:limit]
    log(f"games todo {len(todo)} of {len(games)}")
    for i, game_id in enumerate(todo, 1):
        pull_game(game_id)
        if i % 25 == 0:
            log(f"progress {i}/{len(todo)}")


def main() -> None:
    global SLEEP
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-only", action="store_true")
    parser.add_argument("--games-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.5)
    args = parser.parse_args()
    SLEEP = args.sleep
    ensure()
    if not args.games_only:
        pull_league()
    if not args.league_only:
        pull_games(args.limit)
    log("done")


if __name__ == "__main__":
    main()
