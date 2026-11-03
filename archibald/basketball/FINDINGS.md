# What we found in the 2025–26 NBA data

Written 29 Oct 2026. Rerun on the full 2025–26 season on 30 Oct. Numbers are from the local `nba` database.

Samples:

- This season, play-by-play: 1,228 of 1,230 regular-season games (246,709 possessions). Two games have no play-by-play. League tables cover the full season. Hustle, defensive, and matchup boxes exist for 766 team-games only; the backfilled games are marked `partial` and do not have them.
- Last season check: 153 games spread across 2024–25, plus the full-season team tables. Schema `s2425`.
- "Both halves" means even-numbered games and odd-numbered games, estimated separately.
- Garbage time is dropped unless noted: last 3 minutes of the 4th or later, or a lead bigger than 20.

Code and outputs live under `research/archibald/basketball/`: `code/scoreboard.py`, `code/unset.py`, `code/early_shots.py`, `code/act.py`, `code/levers.py`, `code/more.py`, `code/more2.py`, `code/more3.py`, `code/matchup_patterns.py`, `code/lineups.py`. Results are in `results/` (`results/levers/summary.json`, `more.json`, `more2.json`, `more3.json`, and the `*_2425.json` checks).

Sections: Promising, Smaller but real, Known and confirmed, Dead ends, Empty or unusable, What to feed a model. Items added in the later sweeps on 29 Oct are marked **(sweep 2)**, **(sweep 3)**, **(sweep 4)**. Sweep 2 is `more.py` (set plays, discipline, foul trouble, bench windows, hot hand, icing, timeouts, clutch clock). Sweep 3 is `more2.py` (who rebounds, blocks, missed free throws, rest, clutch free throws, home whistle, referees). Sweep 4 is `more3.py` (hustle, missed-free-throw rebounds, momentum, letdown games, fouling up three).

Full-season rerun (`scoreboard.py`, then `levers.py`, `more.py`, `more2.py`, `more3.py`) is what the numbers below use, except the 2024–25 checks and the hustle boxes, which the backfill did not pull.

---

## Promising

Ranked by how much they move the score and how well they held up.

### 1. Crash the offensive glass

The biggest one. Sending more bodies within 3.5 feet of a miss buys rebounds and points, and does not show up as fast-break points the other way.

On 2,444 team-games this season, within a team, one extra body per miss:

- Offensive rebound rate +0.24
- Offensive rating +17.3 per 100 possessions
- Defensive rating −0.9 ± 1.8 (a wash)
- Opponent fast-break points +0.4 ± 1.0 per 100 (a wash)
- Net rating +18.1 ± 2.4. The two halves were +15.6 and +20.7

A one-standard-deviation swing inside a team is 0.13 bodies per miss, about 2 points a game. Across all 30 teams, both seasons, more offensive-rebound chances track second-chance points at about 0.9. The "crashing teams also defend better" pattern from this season's team table did not show up last season. Treat the defense as a wash.

Last season's game sample (within team) was even larger on offense: net +22 ± 8, opponent fast breaks −1.5 ± 3.5.

Who should go:

- The closest body is a big 60% of the time, a forward 27%, a guard 12%.
- Bigs, forwards, and guards each convert chances into rebounds. Bigs convert best.
- On misses at the rim, the offensive rebound goes to a big 44% of the time. On missed threes it splits evenly: guards 34%, bigs 32%, forwards 32%.
- A guard as the second body near the ball did not hurt the defense (−0.05 ± 0.89) and added a little to the rebound rate.
- More guard crashes do not show a defensive cost on the full season (+3.0 ± 3.6, and the two halves go opposite ways, +8.6 and −1.9). The earlier cuts pointed the wrong way for guards (+7.4 ± 5.7 on the first 388 games, +14.6 ± 11.9 last season) but none of them clear the noise.

Teams furthest below the top quarter of big-and-forward crashing, in the first 388 games: Oklahoma City, Sacramento, Indiana, Boston, Washington, Milwaukee. Already there: Houston, Portland, Charlotte, Utah.

**The missed free throw is the cheapest crash. (sweep 4)** 4.5 last-of-trip free throws a game are missed (end-game deliberate misses dropped). On the full season the shooting team gets the rebound 13.6% of the time (halves 14.3% and 12.9%). Teams differ and the difference is stable (split-half correlation 0.69): Charlotte 27%, Portland 26%, Houston 21%, Minnesota 20%, Phoenix 19%; New Orleans 7%, Cleveland 8%, Denver 9%, Atlanta 9%, Oklahoma City 9%. The 1.15 points scored after the rebound, and the defensive ranking (Minnesota and Houston 8%, Clippers 20%), were measured on the first 665 games and not rerun. It is the same trait as the general crash (correlation 0.65 with team offensive-rebound rate on that sample). Going from 8% to 26% on your own 2.25 misses a game is about 0.45 points a game, and there is no transition cost: pushing after a missed-free-throw rebound scores −0.05 ± 0.05 on the full season.

### 2. Push after a steal. Do not push after they score.

"Push" means the first action of the possession comes within 8 seconds.

| how the possession starts | this season, full | last season |
|---|---|---|
| Steal | +0.21 (1.39 vs 1.18). Halves +0.19 and +0.24. | +0.33 (1.42 vs 1.10) |
| Opponent scores | −0.12 (1.01 vs 1.13). Halves −0.15 and −0.10. | −0.17 (0.95 vs 1.11) |
| Rebound of a miss from far out | +0.03. Both halves small and positive. | +0.01 |
| Rebound of a miss inside 10 feet | +0.10. Halves +0.12 and +0.07. | +0.02. A tie. Do not treat as a rule. |

The early layup is the shot that pays: 1.73 after a steal (same as last season), 1.59 after a short miss. A three after a steal is 1.36. A floater or pull-up two after a steal is 1.28 against 1.18 for walking it up, both halves the same direction, and it does not clear the noise. It is not a rule, and it is not worse.

Some of the value is a foul, not a shot. Possessions that reach the free-throw line within 8 seconds score 1.62.

Teams already push after 73% of steals, and only 7% of possessions after a make (about 3 of those a game). Stopping the bad push is worth about 0.4 points a game.

The steal-push gain stays large in the 4th quarter (+0.19, push rate 70%; +0.39 last season). After a defensive rebound the 4th-quarter push is +0.09 on the full season and was slightly negative last season, so no claim there.

Pushing is not more valuable against teams that crash a lot. The gain is about the same at every opponent crash rate.

### 3. After your own miss, do not reset the offense

The shot clock only goes back to 14 seconds, so "run a play" means shooting late. What the rest of the possession scores, both seasons:

| what they do next | 2025–26 | 2024–25 | share this season |
|---|---|---|---|
| Put it back within 3 seconds | 1.52 | 1.45 | 26% |
| Another rim shot within 8 seconds | 1.50 | 1.52 | 4% |
| Kick-out three within 8 seconds | 1.36 | 1.33 | 19% |
| Quick floater or mid-range two | 1.17 | 1.12 | 9% |
| Reset, more than 8 seconds | 1.17 | 1.12 | 10% |

The late part of a reset is the bad part: shots with 12 to 16 seconds used score 0.96 this season and 0.86 last season. A kick-out three at a normal 37% beats resetting by about 0.2 points. The full season did not move these.

Teams that reset most: Charlotte 15%, Golden State 13%, Boston 13%, Houston 12%, Clippers 12%. Least: Washington 5%, Utah 8%, New Orleans 8%. Worth roughly 0.1 to 0.3 points a game.

### 4. Take the two-for-one

Possessions that start with 28 to 36 seconds left in the first three quarters. Teams go quick (first action within 8 seconds) only 27% of the time.

Rest of the quarter, their points minus the opponent's:

- This season, full: +0.68 if they go quick, +0.43 if they hold it (+0.25 ± 0.09, 2,267 chances). Both halves are clear: +0.23 and +0.28. The weak half from the first 665 games filled in.
- Last season: +0.65 vs +0.26 (+0.39, 279 chances).

The held possession itself is a bad shot: 1.11 points this season, 0.94 last season, against 1.34 and 1.04 for the quick one. About one chance per team per game, so roughly 0.25 points a game for a team that never takes it. Oklahoma City takes it 50% of the time. Charlotte 12%, Milwaukee 14%, Indiana 16%, Lakers 19%.

### 5. A tougher defender takes a star's shots, not the team's points

On 5,624 regular-season matchup pairs with at least 30 possessions. Tracked matchup time covers about 37% of minutes, and "minutes guarded" and "possessions attacked" are the same split.

Against a defender who usually gives up less, a star shoots less and his own line drops. A normal gap between defenders is about 3 points on his line over a game he plays. A very large gap is about 9. His shooting percentage stays where it usually is. The team's points on those same trips do not follow him down. Someone else takes the shot. Medium scorers move less. Role players barely move.

The defender explains about 10% of the variance in a star's points and 18% of the variance in his attempt rate, after his own season rate.

### 6. Free throws in the last 30 seconds of a close game go in 10 points less often **(sweep 3, full season)**

Last two minutes of the 4th or overtime, score within 3, looked like a 7-point drop on the first 665 games. On the full season the drop is real and it is the last 30 seconds, not the whole two minutes. Each free throw is compared with the same shooter's rate on all his other free throws (players with 20 or more).

- Last two minutes, within 3: −5.6 ± 1.2 (1,212 attempts). Halves −5.4 and −5.7.
- Split by time left: under 10 seconds −10.2 ± 2.2 (421), 10 to 30 seconds −9.1 ± 2.7 (258), 30 to 60 seconds −0.9 ± 2.9, 60 to 120 seconds +0.4 ± 2.1. The first 90 seconds of the two-minute window are normal. The last 30 seconds are not.
- Last season sample: −3.6 ± 4.9 (84 attempts). Same direction, too few to say more.
- It is not deliberate missing by the trailing team. The leading team drops the most: −7.4 ± 1.7 (673 attempts). The trailing team is −5.0 ± 2.0.
- Every tier of shooter drops: under 70% −6, 70–80% −5, 80–90% −5, over 90% −11.
- Placebos, full season: the same players in quarters 1–3 are at +0.1 ± 0.2. Last two minutes of a blowout, −0.1 ± 1.4 (839 attempts). Three to five minutes left, within 5, 0.0 ± 1.1 (1,272 attempts).

About 4 of these free throws in each of the 24% of games that get there, and a bit over half of them are in the last 30 seconds. If you are fouling while trailing inside 30 seconds, take 10 points off the shooter's rate. The player-by-player list from the first 665 games (Harden, Gilgeous-Alexander down; George, White steady) was 12 to 22 attempts each and was not rerun.

---

## Smaller, real, not a priority

**The bonus.** Being in the penalty is worth +0.028 points per possession on the full season (halves +0.038 and +0.018), down from +0.05 on the first 665 games. Last season was +0.05. Free-throw trips go from 11.5% of possessions to 18.0%. Teams spend about 6 minutes a game with the opponent in the penalty (New York 4.6 on offense, Dallas 7.5). Within a team, each extra minute is still +0.16 ± 0.07 net rating. Getting there early is worth maybe half a point a game at the extremes. Washington puts itself in the penalty most on defense (8.0 minutes). Dallas least (4.2).

**Challenges exist and win.** 1,739 coach challenges in 1,228 games, 0.71 per team-game, won 65% of the time, 60% in the 4th quarter. The play-by-play does not say which team challenged (`team_id` is 0 on the replay rows), so per-team use cannot be measured.

**Sitting a starter with two first-quarter fouls buys nothing. (sweep 2)** 1,021 cases, 0.83 a game. Coaches pull him within 90 seconds 68% of the time. Those who sat: fouled out 4.5%, reached five fouls 19.7%, played 27.2 minutes. Those who stayed: 3.6%, 17.9%, 29.1 minutes. The foul-out gap is 0.8 ± 1.3 points and the minutes gap is 2.0 ± 0.5. Last season sample (95 cases): sat 1.7% fouled out, stayed 0%, 27.9 against 31.1 minutes. Sitting costs about 2 starter minutes a game it happens and prevents no measurable foul-outs. Caveat: coaches sit the players they are most worried about, which should make sitting look better, not worse.

**The first possession of a quarter is a bad one. (sweep 2)** On the full season: Q1 1.02, Q2 1.07, Q3 1.06, Q4 1.09, against 1.13 for other dead-ball starts. Last season: Q1 0.99, Q2 0.98, Q3 1.14, Q4 1.04. More turnovers (17.1% against 14.8% on other dead balls). Four of these a game per team, so about 0.3 points of offense a game is left on the table, by both teams. Team samples are still too small to rank anyone.

**Plays drawn up after a timeout do not beat flow offense. (sweep 2)** Possessions after the team's own timeout score 1.08 against 1.13 for other dead-ball starts (6,786 of them; halves 1.10 and 1.07). Last season 1.07. Team after-timeout numbers from the first 665 games sat on about 130 possessions each and were not rankable.

**Discipline points are real but did not track winning. (sweep 2)** Per team-game, teams give away about 2.0 points on things that are not basketball: technical free throws 0.44 a game (82% made), fouls on three-point shooters 0.38, shot-clock violations 0.67, backcourt/8-second/5-second 0.17, defensive three seconds 0.13, flagrants 0.10. Within a team, games with more points given away were games with a better net rating (+0.46 ± 0.21 per point). Something about leading causes these, so the raw count is not a lever as measured. The Detroit 2.8 / Miami 1.2 ranking was the first 665 games and was not rerun.

**Unused timeouts. (sweep 2)** The number in the timeout description counts timeouts used, and includes challenges. In close games (final margin 5 or less) teams finish with 0.82 unused on average; 24% of close team-games and 13% of close losses end with two or more unused. Houston 1.6 a game, Utah 1.4, Portland 1.2; Denver and Sacramento 0.3, Oklahoma City 0.3. An observation. There is no measured cost of the unused timeout.

---

## Known already, confirmed here, should be in the model

**Rest. (sweep 3)** Full-season game logs, within team, controlling for home court. Own back-to-back: −1.7 ± 0.8 points this season, −3.1 ± 0.9 last season. Opponent on a back-to-back: +2.0 ± 0.8 and +2.7 ± 0.9. Each day of rest edge (capped at 3): +0.8 ± 0.4 and +1.3 ± 0.3. Home court is worth 1.7 points this season and 1.9 last season. Raw margins by days of rest this season: one day −1.2, two +0.2, three +0.9. The fatigue shows up as offense (offensive rating −1.5 ± 0.8 on a back-to-back), not as pace (−0.2 ± 0.3), distance run, deflections, contested shots, or rebounding, all of which are flat. Halves disagree on size this season (+0.3 and −3.7), so lean on the two-season picture: about 2 points for a back-to-back.

**Hustle stats, sorted into real and mechanical. (sweep 4)** 766 team-games with hustle boxes, within team, per 100 possessions, effect of one within-team standard deviation on net rating:

- Contested shots: +1.7 ± 0.5 (halves +2.6, +1.0). This one is real defense. Contesting 7 more points of the opponent's shots (share of their attempts contested, league average 45%) lowers their effective field-goal percentage by 1.7 points and defensive rating by 2.4 ± 0.4. That is the clearest effort lever in the tracking data.
- Deflections: +2.1 ± 0.5 (both halves +1.9, +2.3), but it is all turnovers. One standard deviation (5 deflections) is 1.3 points of opponent turnover rate, and after controlling for turnovers the defensive rating effect is +0.1 ± 0.2. A deflection is a turnover that was counted twice, not a hidden lever.
- Defensive box-outs: +1.2 ± 0.5 raw, 0.1 ± 0.2 after controlling for the opponent's shooting, and no effect at all on opponent offensive-rebound rate (0.000 ± 0.001). Box-outs are counted on misses, so more misses means more box-outs. Mechanical.
- Charges drawn: +0.9 ± 0.5 per 0.6 charges. A charge is a turnover. Mechanical.
- Screen assists: offense +1.1 ± 0.4, defense also +1.0 ± 0.4 worse. Net zero.
- Loose balls recovered, passes, touches, offensive box-outs: nothing (all within ±0.7 and halves disagree).

---

## Dead ends

Checked, and they do not move the score, or the data cannot answer them.

**Who guards whom, beyond shot volume.** See item 5 above. It changes the star's attempts. It does not change the trip.

**Switching, help defense, playoff matchups.** The columns came back empty. `switches_on` is all zeros. There is no playoff matchup table. Synergy defensive rows for cuts, putbacks, transition, and misc came back empty. The win-probability endpoint returns an empty body.

**Defender stint length.** Looked like longer stints were worse defense. It was transition possessions mixed in. At the team level, how stable the assignments are has about no relationship with defensive rating.

**Lineups built around the opponent's stopper.** Reconstructed the floor from starters and substitutions (name matching had to strip accents; team rebounds were being counted as players). A star is worth about the same with the opponent's best defender on or off: +0.068 vs +0.087 points per possession on offense. The difference is inside the noise. Star-plus-defenders versus star-plus-scorers was too noisy to call.

**Pushing after a short miss as a standing rule.** +0.12 this season, +0.02 last season. The layup is good both years. The rest of the push was not, last year.

**"One guard always stays back."** See item 1. The second body being a guard had no measurable cost. The rate regression points the wrong way for guards but does not clear the noise.

**Timeouts to stop a run.** After the opponent outscored them by 8 or more over the previous 6 possessions, a timeout did nothing to the next 6: −0.26 vs +0.13 this season, +0.44 vs −0.08 last season. Both inside the noise. Only 503 such runs this season, 112 last season.

**Slowing down with a 4th-quarter lead.** A first cut said slow leaders bleed 2.3 points more than fast ones. That was the opponent scoring: a made basket forces a slow inbound, so "slow" was just "the other team is making shots." Measured only on the leader's own defensive-rebound starts, pace changes the rest of the lead by 0.01 ± 0.12 points per second, and the win rate is 91% either way.

**Take fouls.** 910 in 1,228 games, 0.74 per game. Of the 198 that line up with a possession, 58% are followed by free throws and that possession is worth 1.18, about what the break was worth anyway. Not enough, outside the penalty, to say whether the foul helped.

**Heaves.** The 2025–26 rule change already fixed the old habit of letting the clock run out. 89% of possessions that start with 1.5 to 5 seconds left in a quarter get a shot. Shots from 40 feet and out go in 2.2% of the time (25 of 1,148). The teams that pass one up most still take about 82% of them. There is not much left to gain.

**Fouling to get into, or stay out of, the bonus as a big edge.** The bonus is real and small. See above. It is not a hidden point a game.

**Early offense after a long rebound, and early floaters.** Ties. Already in item 2.

**The hot hand. (sweep 2)** A three after two straight made threes goes in 2.3 ± 0.5 points less often than the shooter's own rate on the full season (7,673 threes), 2.5 ± 1.5 less last season. After two misses: +0.7 ± 0.4 this season, +3.2 ± 0.9 last season. Do not feed the hot shooter. Do not pull the cold one.

**Icing the free-throw shooter. (sweep 2)** First free throws preceded by a timeout: −1.6 ± 1.4 points against the shooter's own rate (992 iced), +2.1 ± 3.3 last season. Nothing there.

**Milking the clock with a late lead. (sweep 2)** Last three minutes, up 1 to 5. Possessions that take 18 or more seconds to the first shot score 1.02 against 1.22 for quicker ones, and the team wins 80.5% against 80.0%. Per second slower: +0.002 ± 0.001 win probability, with the two halves on opposite sides of zero. The points lost and the clock gained cancel. For the trailing team, speed does not matter either (0.000 ± 0.001 per second).

**Clutch offense in general. (sweep 2)** Last three minutes within 5: 1.15 points a possession against 1.16 normally (4,121 possessions). Trailing teams shoot sooner (median 12 s against 18 s for leaders) and score more (1.18 against 1.11), which is the leader's clock-milking, not a lesson.

**Bench windows. (sweep 2)** First three minutes of the 2nd and 4th quarters, when starters sit. League offense drops to 1.10 and 1.14 from 1.16. The spread of team net scores in those windows (0.7 to 0.85 points a game across teams) is the same size as sampling noise at 44 games. Nothing can be said about any team yet. Rerun when the season is loaded.

**Who grabs the defensive rebound. (sweep 3)** Guards take 36% of them, forwards 29%, bigs 27%. The push after a guard's rebound is +0.08 ± 0.02 on the full season (halves +0.10 and +0.06) and was −0.02 ± 0.04 last season. Not a rule until it shows up in a second season.

**Pushing after a block. (sweep 3)** 8% of defensive-rebound starts follow a block. They score +0.03 ± 0.02 against other defensive-rebound starts on the full season, and the two halves disagree (+0.08 and −0.01). Last season was 0.00 ± 0.05. Not a separate rule from the short-miss push, which itself did not hold up last season.

**Pushing after a missed free throw. (sweep 3)** Teams push only 17% of these. When they do: −0.09 ± 0.06 this season, −0.04 ± 0.14 last season. The defense is already set. Lean against it; too noisy to call.

**The home whistle. (sweep 3)** Fouls called on the home team 16.7 a game, on the visitors 17.0. Shooting fouls 10.3 against 10.5. Last five minutes of close games 2.13 against 2.19. There is no whistle. The home edge in points is 1.7 ± 0.5 a game this season and 1.9 last season. A late-close home scoring edge of +0.09 ± 0.03 a possession on the first 665 games was −0.08 ± 0.08 last season, so no claim.

**Referees. (sweep 3)** The calling official is named on every foul row, so this can be measured. 80 referees, 73 with 10 or more games. Individual call rates run from 8.8 to 15.4 personal fouls a game and are stable across halves of the schedule (r = 0.61). That is mostly the share of the crew's calls, not the crew's total. The sum of the three officials' leave-one-out averages predicts +0.24 ± 0.13 actual fouls per expected foul (the crews only differ by 1.4 expected fouls), +0.40 ± 0.24 free throws, and +0.09 ± 0.42 points. No official is reliably home-friendly (split-half r = 0.02 on home share). Referee assignment does not move the score.

**Momentum. (sweep 4)** After a team outscores the other by 8 or more over 6 possessions (not late in the 4th, lead 15 or less), its next 6 possessions net +0.07 ± 0.08 on the full season (1,570 runs) and +0.14 ± 0.20 last season. Controlling for the margin, the run itself is +0.07 ± 0.08. Runs do not continue.

**Letdown and bounce-back games. (sweep 4)** Within team, controlling for home court and back-to-backs, in the full-season logs. After a win by 20 or more: +2.7 ± 1.0 this season, −0.1 ± 1.1 last season. After a loss by 20 or more: +1.4 ± 1.0 and −1.8 ± 1.1. Per point of the previous game's margin: 0.005 ± 0.02 and 0.027 ± 0.02. After three straight losses: +0.3 ± 0.9 and −0.7 ± 0.9. The last game tells you nothing about the next one once you know the team.

**Fouling when up three in the last 24 seconds. (sweep 4)** 108 situations in 1,228 games; the leader fouled first in 29 of them. Defended: leader won 97%, the trailer got a three off 71% of the time. Fouled: won 90%, went to overtime 14%. Still too few. Noted so nobody expects an answer from one season.

---

## Empty or unusable

Recorded so nobody re-pulls them expecting an answer.

- Win probability
- Switches on the matchup box
- Help defense
- Playoff player-versus-defender matchups
- Synergy defense for cuts, putbacks, transition, misc
- All-Star game logs
- Which team issued a coach challenge
- 5-man lineup tables are the API's top 2,000 lineups, not every lineup
- There is no team usage endpoint
- Play-by-play covers 1,228 of 1,230 games. League tables are the full season.
- "Possessions after the opponent's timeout" are fragments. They show 28% free-throw trips and 27% turnovers because the timeout splits a possession in two (called before free throws, or on a dead ball mid-possession). Do not use `o_ppp_after_opp_timeout` or the "opp" row in `set_plays`. (sweep 2)
- Free-throw rows have an empty `shot_result`; a miss is a description starting with "MISS". (sweep 3)
- The number in a timeout description, "(Full 4 Short 0)" or "(Reg.4 Short 0)", counts timeouts used so far, including challenges, not timeouts left. (sweep 2)
- Hustle boxes exist for 766 team-games only; the backfilled games (marked `partial`) have no hustle, defensive, or matchup boxes, so the hustle section was not rerun. `speed` is empty and `distance` is in mixed units in `team_games.csv`. `tov_pct` and `opp_tov_pct` are the same column there. (sweep 4)
- Foul rows name the calling official at the end of the description, e.g. "(J.Goble)". (sweep 3)

---

## What to feed a model, in order

1. Offensive-rebound chances per miss, split by the crasher's position, and the miss being a two or a three.
2. How the possession started (steal, made shot, long miss, short miss) and whether the first action came within 8 seconds, plus the shot zone if it did.
3. What happened in the first 8 seconds after an offensive rebound: putback, kick-out three, floater, or reset.
4. Whether the possession started with 28 to 36 seconds left in the quarter, and whether they went quick.
5. Both teams' rest: back-to-back flags and the rest-day edge. About 2 points a game for a back-to-back.
6. Free-throw percentage in the last 30 seconds of a close game: take 10 points off the shooter's rate. The prior 90 seconds of the two-minute window are normal.
7. Share of opponent shots contested (the one hustle number that is real defense rather than a recount of turnovers or misses).
8. Offensive-rebound rate on the team's own missed free throws (stable team trait, same family as item 1).
9. Minutes the opponent has spent in the penalty.
10. A flag for the first possession of a quarter (scores about 0.1 less).

Do not feed matchup identity, lineup composition, timeout calls, 4th-quarter pace, the hot hand, who grabbed the defensive rebound, whether the rebound followed a block, the referee crew, home/away foul differences, scoring runs, the previous game's result, deflections or box-outs on top of turnovers and misses. Those were checked and did not move the score on their own.
