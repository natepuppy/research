# NFL data

Local Postgres database `nfl` on this machine (`psql nfl`). Four schemas. Tracking coordinates are `x`/`y` in yards, `s` speed, `a` acceleration, `dir` motion direction, `o` body orientation.

`nflverse.schedules.old_game_id_key` is the numeric game id used by both Big Data Bowls. `nflverse.players.nfl_id_key` is the Next Gen Stats player id.

## `bdb2027`

Combine movement and later NFL play for 510 prospects (2023–2025).

- `players` — name, position, college, draft slot. Primary key `nfl_id`
- `combine_results` — height, weight, 40, jumps, agility times, Next Gen Stats grades. References `players`
- `combine_tracking` — 10 Hz Combine drill frames (`PLAYER` or `BALL`). References `players`
- `player_career_successes` — career snaps, games, All-Pro, Pro Bowl. References `players`
- `games` — season, week, home and away. References `nflverse.schedules`
- `player_play` — one row per player per snap. References `players` and `games`
- `game_tracking` — 10 Hz in-game frames for those players. References `players` and `games`



## `bdb2026`

Movement while the ball is in the air, 2023 season, plus the prediction sample.

- `games` — season, week, teams. References `nflverse.schedules`
- `plays` — down, formation, coverage, result, expected points. References `games`
- `players` — name, height, weight, position. References `nflverse.players`
- `player_plays` — role on the play (passer, route runner, coverage). References `plays` and `players`
- `tracking_before` — frames before the throw. References `player_plays`
- `tracking_after` — frames after the throw. These `x`/`y` values are the prediction targets. References `player_plays`
- `test_player_plays`, `test_tracking`, `test_targets` — the prediction API sample. Same relationships



## `bdb2023`

Pass rushes from the first eight weeks of the 2021 season. Every player on the field, plus the ball.

- `games` — season, week, home and away. References `nflverse.schedules` and `nflverse.teams`
- `plays` — down, formation, coverage, result. References `games` and `nflverse.teams`. Flagged players reference `players`
- `players` — name, height, weight, college, position. References `nflverse.players`
- `player_plays` — PFF role, pressure, and who was blocked. References `plays` and `players`
- `tracking` — 10 Hz frames from before the snap through the throw. The ball has no `nfl_id`. Player frames reference `player_plays`



## `nflverse`

Season data from 1999 through the current season. `pbp` is every play and references `schedules`. `players` is the id crosswalk (`gsis_id`, `nfl_id_key`, `pfr_id`, `espn_id`).

- `schedules` — every game nflverse published, plus the 147 preseason games from the 2027 bowl that were not in that file
- `pbp` — play-by-play, expected points, win probability. References `schedules`
- `participation` — who was on the field, personnel, coverage, routes
- `stats_player_week`, `stats_player_season`, `stats_team_week`, `stats_team_season` — counting stats. Week tables reference `schedules`. Team tables reference `teams`
- `rosters`, `weekly_rosters` — roster status, college, draft slot
- `depth_charts`, `depth_charts_current` — weekly depth chart. Both reference `players`. The older one also references `teams`
- `injuries` — injury report and practice status. References `players` and `teams`
- `snap_counts` — offensive, defensive, and special-teams snaps. `snap_count_games` is the list of games those counts cover
- `teams` — names, conferences, colors, logos
- `officials` — referee crew. `game_id_key` is the numeric game id
- `trades` — picks and players exchanged
- `draft_picks` — draft slot and later career totals. References `players`
- `contracts` — value, length, guarantees
- `combine` — historical Combine measurements and drill times, not sensor frames
- `ngs_passing`, `ngs_receiving`, `ngs_rushing` — weekly Next Gen Stats summaries. Reference `players`
- `pfr_week_pass`, `pfr_week_rush`, `pfr_week_rec`, `pfr_week_def` and the matching `pfr_season_*` tables — Pro Football Reference advanced stats. Week tables reference `schedules`. Passing tables reference `players`
- `ftn_charting` — motion, play-action, screens, pressure, drops. References `pbp`
- `espn_qbr_week`, `espn_qbr_season` — ESPN QBR

A few links are not enforced because some child rows have no parent: historical roster team codes, about 9,000 participation rows, 25 of the 510 combine prospects, and a handful of Pro Football Reference players.

## `Research Questions`

Need to rank these based on 

1. How interesting is this question to me?
2. How impactful would a clean result be here?
3. How novel is this?
4. Do we have the data to accomplish this?

- How much of the target and the yards come from the formation, and how much from who is actually available that week: depth chart, snap share, and injury report?
- How likely is a trick play to happen on a specific play?
- When should you use a trick play? Teams often do trick plays when people expect it. If you do it when people least expect it is it more effective?
- How do motion, play-action, screens, and RPOs change the best formation and the likely target?
- While the ball is in the air, which receiver and defender paths create separation, and which ones turn into yards after the catch?
- Should I sub a player based on who is defending them? Or on the flip side, who they are defending?
- What different situations effect players in games vs combine? Full pads? Does that effect how fast they actually run? Or how agile they actually are?
- Based on team B's formation, what is the optimal formation of team A?
- Reaction time vs speed correlation? How can you measure reaction time? Is that more important than speed?
- Does a rookie's movement after the stopwatch is off predict anything?
- When the official Combine time and the sensor tracking disagree, which one matches early playing time and production?
- After a trade, do players production go up or down and how can you predict that?
- Does the officiating crew change penalty rate or passing volume enough to change the game plan?
- How much of one receiver's separation was created by a teammate's route that never got the ball?
- Which defenders win by taking away throws, so the value shows up as targets that never come?
- When a player returns from an injury report, is his movement different from his own baseline, and does that difference show up in the next week's snaps?
- Does the direction they are facing indicate their football IQ?
- Does Football IQ player a bigger role or raw power? where is the point of diminishing returns?
- What player is best at executing what types of routes?
- Does speed or size matter more for certain tyypes of routes?


