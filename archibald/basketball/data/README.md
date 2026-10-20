# 2025–26 NBA data

Local Postgres database `nba`. Password blank.

```
postgresql://nathanclark@127.0.0.1:5432/nba
```

Postico: host `127.0.0.1`, port `5432`, user `nathanclark`, database `nba`.

Two schemas, `league` and `games`. Tables were created by `code/db.py` from the stats.nba.com frames, so the column list is whatever the API returned. Whole numbers are `bigint`, decimals are `double precision`, and everything else is `text`. That includes dates, minutes on the box score, the running score, and 0/1 flags such as `is_field_goal`.

There are no foreign keys. The only primary keys are `league.loaded.name` and `games.loaded.game_id`. Per-game tables other than `games_index` have an index on `game_id`.

To see the columns of any table:

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'league' AND table_name = 'player_base_regular_season'
ORDER BY ordinal_position;
```

## Identifiers

`game_id` is 10-character text. The first three digits are the kind of game:

| prefix | what it is | games on the schedule |
|---|---|---|
| `001` | preseason | 71 |
| `002` | regular season, including NBA Cup group and knockout games | 1,230 |
| `003` | All-Star weekend (Rising Stars and the All-Star games) | 7 |
| `004` | playoffs | 85 |
| `005` | play-in | 6 |
| `006` | NBA Cup championship | 1 |

The rest is the season year and the game number. `0022500001` is the first regular-season game of 2025–26.

`season_id` on the game logs is the same idea in five digits: a leading type digit plus `2025`. Preseason `12025`, regular season `22025`, playoffs `42025`, play-in `52025`.

One player id is used everywhere. `league.players.person_id`, `player_id` on the league stat tables, and `person_id` on the per-game box scores and play-by-play are the same number. All 582 players in the regular-season game log match a row in `league.players`. `team_id` is the same number on every table. `league.teams` has the 30 franchises.

## What is loaded

League tables cover the finished season: regular season 21 Oct 2025 through 12 Apr 2026, the play-in, the playoffs (Finals ended 13 Jun 2026), and the preseason.

`games.games_index` lists all 1,392 of those games (everything except the All-Star weekend and the Cup championship). Each game is stored three identical times, so count `DISTINCT game_id`. `team_abbreviation` is one of the two teams, whichever row the team game log kept. The matchup string is the reliable one: `DEN @ GSW` means Denver is away, `DEN vs. PHX` means Denver is home. Home and away, score, arena, and series label live on `league.schedule_0`.

Play-by-play and the box-score tables hold only the games whose id is in `games.loaded`. The pull walks regular-season ids first. When this was written it had the opening stretch of the regular season, on the order of 80 games, and was still running. A game can sit in `games.pbp` for a few minutes before its id lands in `games.loaded`.

```sql
SELECT count(*) FROM games.loaded;
SELECT min(game_id), max(game_id) FROM games.loaded;
```

These requests came back empty. They are recorded in `league.loaded` with `n_rows = 0` and have no table:

- All-Star player and team game logs
- Playoff player-versus-defender matchups
- Synergy defensive-player rows for cuts, putbacks (`offrebound`), transition, and misc

The win-probability endpoint returns an empty body, so there is no win-probability table. `games.loaded.detail` is a jsonb map of endpoint name to `"ok"` or an error string. Older rows still have a `winprob` key from a pull that tried that endpoint.

## How a season stat table is shaped

Most `league` tables are one row per player or one row per team, for one season type. The name ends in `_regular_season` or `_playoffs`. Game logs also have `_playin` and `_pre_season`.

Dash tables were requested **per game**. `min` of `34.8` and `pts` of `27.7` are season averages, not totals. Game-log tables are the exception: one row per appearance, and the counting stats are that game's totals.

Percentages are fractions. `0.569` means 56.9%.

Dash tables from the player and team stat endpoints also carry a `*_rank` column for every stat, plus `team_count` (how many teams that player played for). The rank columns repeat the stat columns and can be ignored for analysis.

A player who played for two teams has one row, with the combined per-game line. `team_abbreviation` is the last team, which is why `team_count` exists.

## Directory

| table | rows | grain | what it is |
|---|---|---|---|
| `players` | 582 | player | name, slug, team, `rosterstatus` (`1` current roster, 525 players; `0` the other 57) |
| `teams` | 30 | franchise | city, arena, capacity, owner, GM, coach |
| `rosters` | 530 | player on a team | jersey, position, height, weight, birth date, age, experience, school, how acquired |
| `schedule_0` | 1,400 | game | the full schedule, including All-Star and the Cup final |
| `schedule_1` | 38 | week | week number, name, start, end |
| `standings` | 30 | team | final regular-season standings |
| `ist_standings_group`, `ist_standings_wildcard` | 30 each | team | NBA Cup group and wildcard standings, with the four group-game results pivoted into `game_id1` … `game_id4` |
| `playoff_series` | 85 | playoff game | `series_id`, `game_num`, home and visitor team ids |
| `shot_zones` | 20 | league-wide zone | field-goal makes and attempts for the regular season. No player id and no coordinates |
| `loaded` | | request | which league calls have been saved, and how many rows |

`schedule_0.game_date` is text like `10/02/2025 00:00:00`. The same row also has `game_date_est` and `game_date_utc`. Home and away are prefixed columns: `home_team_team_tricode`, `home_team_score`, `away_team_team_tricode`, `away_team_score`. `game_label` and `game_sub_label` carry the series name (`West First Round`, `Game 3`, `Emirates NBA Cup`). The broadcaster columns (`national_broadcasters_0_…` through `_5_`, plus radio and home/away) are the same block repeated. `game_subtype` is blank for a normal game, `in-season` or `in-season-knockout` for the Cup, and `Global Games` for the international games.

`standings.wins`, `losses`, and `win_pct` are numbers. Split records (`home`, `road`, `l10`, `vs_east`, `jan`, …) are text like `31-9`. `playoff_rank` is the conference seed. `clinch_indicator` is the API's short flag. The explicit flags are `clinched_conference_title`, `clinched_division_title`, `clinched_playoff_birth`, `eliminated_conference`, and `eliminated_division`.

## Game logs

One row per player or team per game. Counting stats are totals for that game. `min` is whole minutes (`bigint`), so seconds are dropped. `game_date` is `YYYY-MM-DD`. `wl` is `W` or `L`. `matchup` uses `@` for away and `vs.` for home.

| table | rows |
|---|---|
| `game_log_player_regular_season` | 26,651 |
| `game_log_player_playoffs` | 1,921 |
| `game_log_player_playin` | 126 |
| `game_log_player_pre_season` | 1,967 |
| `game_log_team_regular_season` | 2,460 (two rows per game, one per team) |
| `game_log_team_playoffs` | 170 |
| `game_log_team_playin` | 12 |
| `game_log_team_pre_season` | 142 |

Columns: `fgm`, `fga`, `fg_pct`, `fg3_m`, `fg3_a`, `fg3_pct`, `ftm`, `fta`, `ft_pct`, `oreb`, `dreb`, `reb`, `ast`, `stl`, `blk`, `tov`, `pf`, `pts`, `plus_minus`. The player log also has `fantasy_pts` and `video_available`.

This is the full season of box scores. The `games` schema repeats the box score in a wider shape, but only for the games pulled so far.

## Season averages

`player_{measure}_{season}` has about 582 regular-season rows and about 230 playoff rows. `team_{measure}_{season}` has 30 and 16. There is no `team_usage` table. The API has no team usage endpoint.

Shared identity columns: `player_id`, `player_name`, `nickname`, `team_id`, `team_abbreviation`, `age`, `gp`, `w`, `l`, `w_pct`, `min`.

| measure | the columns that matter |
|---|---|
| `player_base`, `team_base` | counting line: `fgm`/`fga`/`fg_pct`, `fg3_*`, `ftm`/`fta`/`ft_pct`, `oreb`, `dreb`, `reb`, `ast`, `tov`, `stl`, `blk`, `blka`, `pf`, `pfd`, `pts`, `plus_minus`, plus double-doubles (`dd2`) and triple-doubles (`td3`) |
| `player_advanced`, `team_advanced` | `off_rating`, `def_rating`, `net_rating`, and the `e_` estimated versions, `ast_pct`, `ast_to`, `ast_ratio`, `oreb_pct`, `dreb_pct`, `reb_pct`, `tm_tov_pct`, `e_tov_pct`, `efg_pct`, `ts_pct`, `usg_pct`, `pace`, `pace_per40`, `pie`, `poss` |
| `player_misc`, `team_misc` | `pts_off_tov`, `pts_2_nd_chance`, `pts_fb`, `pts_paint`, the same four for the opponent, `blk`, `blka`, `pf`, `pfd` |
| `player_scoring`, `team_scoring` | share of attempts and points: `pct_fga_2_pt`, `pct_fga_3_pt`, `pct_pts_paint`, `pct_pts_2_pt_mr`, `pct_pts_3_pt`, `pct_pts_fb`, `pct_pts_ft`, `pct_pts_off_tov`, and assisted vs unassisted (`pct_ast_2_pm`, `pct_uast_2_pm`, `pct_ast_3_pm`, `pct_uast_fgm`) |
| `player_usage` | `usg_pct` and the player's share of team production: `pct_fga`, `pct_fg3_a`, `pct_fta`, `pct_reb`, `pct_ast`, `pct_tov`, `pct_stl`, `pct_blk`, `pct_pts` |
| `player_defense`, `team_defense` | `def_rating`, `dreb`, `dreb_pct`, `stl`, `blk`, opponent points off turnovers, second chance, fast break, and paint, `def_ws` |
| `estimated_player`, `estimated_team` | the `e_` ratings only: offensive, defensive, net, assist ratio, rebound rates, turnover rate, usage, pace |

`player_bio_{season}` adds `player_height`, `player_height_inches`, `player_weight`, `college`, `country`, `draft_year`, `draft_round`, `draft_number`, and a short per-game line (`pts`, `reb`, `ast`, `net_rating`, `usg_pct`, `ts_pct`).

`player_clutch_{season}` and `team_clutch_{season}` use the endpoint default: last five minutes, score within five. Same counting columns as `player_base`. Regular season has 492 players and all 30 teams. Playoffs have 133 and 15.

`hustle_player_{season}` and `hustle_team_{season}`: contested shots (all, 2pt, 3pt), deflections, charges drawn, screen assists and screen-assist points, loose balls recovered, box outs, and the share of those that produced a rebound.

## Shot profile, tracking, matchups, lineups

`player_shot_locations_{season}` and `team_shot_locations_{season}` are makes, attempts, and percentage in eight zones: `restricted_area`, `in_the_paint_non_ra`, `mid_range`, `left_corner_3`, `right_corner_3`, `corner_3`, `above_the_break_3`, `backcourt`. Still per game, still no x/y.

`player_pt_shot_{season}` and `team_pt_shot_{season}` are a single overall shot profile: `fga_frequency`, `fgm`/`fga`/`fg_pct`/`efg_pct`, and the same split into twos and threes. They are not broken out by zone.

`pt_defend_{category}_{season}` is how opponents shot with that player as the closest defender. The defender id is `close_def_person_id`. Columns are `freq`, `d_fgm`, `d_fga`, `d_fg_pct`, `normal_fg_pct` (that shooter's usual percentage), and `pct_plusminus`. Categories: `overall`, `2_pointers`, `3_pointers`, `less_than_6ft`, `less_than_10ft`, `greater_than_15ft`. `pt_team_defend_*` is the same for a team. Not every player defends every distance, so row counts sit a little under 582.

`tracking_{measure}_{player|team}_{season}` is player-tracking, per game. Every table has `gp`, `w`, `l`, `min`. The measure adds:

| measure | what the extra columns are |
|---|---|
| `speeddistance` | `dist_feet`, `dist_miles`, `dist_miles_off`, `dist_miles_def`, `avg_speed`, `avg_speed_off`, `avg_speed_def` |
| `rebounding` | contested and uncontested rebounds, chances, chance percentage, average distance to the rebound, split into offensive and defensive |
| `possessions` | `touches`, `time_of_poss`, `avg_sec_per_touch`, `avg_drib_per_touch`, `pts_per_touch`, elbow / post / paint touches |
| `passing` | `passes_made`, `passes_received`, `potential_ast`, `secondary_ast`, `ast_points_created`, `ast_to_pass_pct` |
| `drives` | `drives`, and for those drives the makes, free throws, points, passes, assists, turnovers, fouls, each with a share (`drive_pts_pct`, `drive_tov_pct`, …) |
| `catchshoot` | catch-and-shoot makes, attempts, points, threes, effective field-goal percentage |
| `pullupshot` | the same shape for pull-up shots |
| `defense` | `stl`, `blk`, `dreb`, and rim protection: `def_rim_fgm`, `def_rim_fga`, `def_rim_fg_pct` |
| `painttouch`, `posttouch`, `elbowtouch` | touches in that area, then makes, free throws, points, passes, assists, turnovers, fouls |
| `efficiency` | points and field-goal percentage rolled up from drives, catch-and-shoot, pull-ups, and the three touch areas |

`synergy_{play_type}_{offensive|defensive}_{player|team}_{season}` is Synergy play-type tracking, per game. Play types: `transition`, `isolation`, `prballhandler`, `prrollman`, `postup`, `spotup`, `handoff`, `cut`, `offscreen`, `offrebound` (putbacks), `misc`. Columns: `percentile`, `poss_pct`, `ppp`, `fg_pct`, `efg_pct`, `poss`, `pts`, `fgm`, `fga`, and the share of possessions that drew a free throw, a turnover, a shooting foul, a plus-one, or a score. A player appears only if they had that play type, so an isolation table has a few hundred rows, not 582. Team tables are 30 and 16. Defensive-player tables are absent for cuts, putbacks, transition, and misc.

`lineups_{base|advanced}_{season}` is five-man units. `group_name` is five abbreviated names. `group_id` is the five player ids wrapped in dashes (`-1629611-1630533-…-`). The API returns its top 2,000 lineups. Regular season is exactly 2,000. Playoffs are 1,442. Rates are per game, so a lineup that played one game can show a high `min`.

`matchups_regular_season` is 147,805 rows of offensive player versus defender, requested as **totals** rather than per game. `off_player_id`, `def_player_id`, `gp`, `matchup_min` (text `38:41`), `partial_poss`, `player_pts`, `team_pts`, assists, turnovers, blocks, field goals, threes, free throws, `sfl` (shooting fouls), and the help-defense line (`help_blk`, `help_fgm`, `help_fga`). No team id on this table. The playoff request came back empty.

## `games` — one game at a time

Every row has `game_id`. Join player box scores on `game_id` plus `person_id`. Join to `league.players` with `person_id = players.person_id`.

A finished game has one row in `games.loaded`. `detail` records which of the 11 endpoints succeeded.

`games.pbp` is every action. `action_number` orders the game. `period` is 1–4, then 5 and 6 for overtime. `clock` is an ISO duration of time remaining, `PT11M36.00S` for 11:36. `team_tricode` and `person_id` are the team and player the action belongs to. `action_type` is `Made Shot`, `Missed Shot`, `Free Throw`, `Rebound`, `Turnover`, `Foul`, `Substitution`, `Timeout`, `Jump Ball`, `Instant Replay`, `period`, or blank. `sub_type` is the finer label (`Pullup Jump shot`, `Cutting Layup Shot`). `description` is the text line. `shot_result` is `Made` or `Missed`. `is_field_goal` is 1 for a field-goal attempt. `shot_value` is 2 or 3. `score_home` and `score_away` are the running score as text. `points_total` is the points on that action.

Shot location is `x_legacy` and `y_legacy`, in tenths of a foot, origin at the basket. Divide by 10 for feet. `x` is left and right of the basket (about −250 to 250). `y` runs from just behind the baseline (small negatives) toward half court, and past 500 for a heave. `shot_distance` is already in feet. Only field-goal attempts have a location. A few hundred attempts are stored at `(0, 0)`, which the shot viewer treats as no location. The viewer is `code/shot_viewer/` at `http://127.0.0.1:8766`.

The box-score tables share a team header (`team_id`, `team_city`, `team_name`, `team_tricode`, `team_slug`) and, on player tables, a player header (`person_id`, `first_name`, `family_name`, `name_i`, `player_slug`, `position`, `jersey_num`, `comment`, `minutes`). `minutes` is text `34:12`. `comment` is blank when the player played. Otherwise it is a did-not-play reason: `DNP - Coach's Decision`, `DND - Injury/Illness`, `NWT - Injury/Illness`.

| table | grain | what it adds |
|---|---|---|
| `box_traditional_player` | player-game | counting box score and `plus_minus_points` |
| `box_traditional_team_0` | team-game, two rows | starters and bench (`starters_bench`) |
| `box_traditional_team_1` | team-game | team totals and `plus_minus_points` |
| `box_advanced_player`, `box_advanced_team` | player-game, team-game | offensive, defensive, and net rating, rebound rates, `efg`, true shooting, usage, pace, possessions, `pie` |
| `box_misc_*` | player-game, team-game | points off turnovers, second chance, fast break, paint, and the opponent's versions |
| `box_scoring_*` | player-game, team-game | share of attempts and points by shot type, assisted vs unassisted |
| `box_usage_*` | player-game, team-game | usage, and share of the team's makes, attempts, rebounds, assists, turnovers |
| `box_four_factors_*` | player-game, team-game | effective field-goal percentage, free-throw rate, turnover percentage, offensive-rebound percentage, and the opponent's four |
| `box_tracking_*` | player-game, team-game | speed, distance, rebound chances, touches, passes, contested and uncontested field goals, shots defended at the rim |
| `box_hustle_*` | player-game, team-game | contested shots, deflections, charges drawn, screen assists, loose balls, box outs |
| `box_defensive_*` | player-game, team-game | matchup minutes, partial possessions, switches, points allowed, steals, blocks, opponent field goals |
| `box_matchups_team` | one offensive player vs one defender, in one game | the same matchup line as `league.matchups_regular_season`, plus both player ids (`person_id_off`, `person_id_def`) and time shares. `matchup_minutes` is `1:32`. `matchup_minutes_sort` is that duration in seconds |

`box_traditional_team_0` is the split the API returns alongside the team total, so a game contributes four rows there (starters and bench for each side) and two rows to `box_traditional_team_1`.
