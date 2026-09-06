# Data Sources and Snapshot Rules

This project is designed around **reproducible pregame information**. A data field is not safe merely because it is labeled "pregame"; it must also have been available at the exact forecast timestamp being replayed.

## 1. nflverse / nflreadpy

Primary open historical data backbone.

Python package: `nflreadpy`

Core datasets to use:

| Dataset | Planned use | Historical notes |
|---|---|---|
| Play-by-play | EPA, success rate, turnovers, sacks, explosives, pace, situational efficiency | 1999+ |
| Schedules | games, results, rest, moneylines, spread, total, QBs, coaches, venue/weather fields | all available seasons |
| Team stats | cross-check / supplemental weekly summaries | varies by field |
| Player stats | player usage and performance | varies by field |
| Weekly rosters | roster continuity and availability context | 2002+ |
| Depth charts | starter/backup role and replacement context | 2001+; timestamp handling changes after 2024 |
| Snap counts | player importance / role weighting | historical coverage varies |
| Next Gen Stats | QB/receiver/rusher advanced candidates | modern seasons |
| ESPN QBR | QB challenger features | historical coverage varies |
| PFR advanced stats | pass/rush/receive/defense challenger features | modern seasons |
| FTN charting | advanced charting challenger features | licensing/coverage dependent |
| Injury reports | practice and game-status history | 2009-2024 currently available through nflverse |

## 2. Schedule / market fields

The nflverse schedule includes useful pregame/game context such as:

- home/away team and score
- game date/time
- rest days
- home/away moneyline
- spread and total
- division indicator
- roof
- surface
- temperature
- wind
- starting QB IDs/names
- coaches
- referee
- stadium

### Critical market rule

The historical `spread_line` is a **closing line**. It is therefore:

- a required benchmark;
- valid as a feature only for a forecast snapshot that truly occurs at/near the close;
- **not valid** for an early-week replay if the close occurred after the user's pick deadline.

The same principle applies to any moneyline field whose snapshot timing cannot be proven.

The project will keep two concepts separate:

1. `market_close_*` = benchmark / close-time information;
2. `market_asof_*` = line actually available at the modeled prediction timestamp.

A future odds-history source can populate `market_asof_*` for exact historical replay. Until then, market-close performance should be treated as a demanding benchmark rather than silently leaked into an earlier forecast.

## 3. Injury and availability data

### Historical training

Use nflverse injury/practice reports from 2009-2024. Relevant fields include:

- player/team/week
- position
- primary/secondary injury
- practice status
- game report status
- modification timestamp

The model should learn the *effect of lost player value*, not count injured players equally.

### Current-season inference

The nflverse injury feed has no current post-2024 data at present. Live forecasts should therefore ingest the official NFL injury report (or another source that can be legally and reproducibly archived), normalize it to the historical schema, and save the raw snapshot with an `as_of` timestamp.

For each player, preserve:

- source
- team
- player ID/name mapping
- position
- injury
- practice participation
- game designation
- source update time
- ingestion time

### Player importance

Candidate importance score:

`availability_value = position_weight * role_weight * snap_share * player_quality * availability_probability`

Where:

- QB is modeled separately;
- `role_weight` is based on starter/depth-chart context;
- `snap_share` is from recent healthy games;
- `player_quality` uses only pregame historical performance/priors;
- `availability_probability` is mapped from practice/report status and may later be learned/calibrated.

## 4. Quarterback data

QB identity needs its own snapshot because generic injury aggregation is not enough.

Candidate sources/features:

- schedule starting-QB identifiers
- play-by-play EPA/dropback
- CPOE
- sack rate
- turnover rate
- depth of target / air-yards profile
- ESPN QBR
- Next Gen Stats
- starter announcement / depth-chart movement

Historical replay must use the QB believed/announced to be starting *at that timestamp*, not the starter known after kickoff.

## 5. Weather

### Historical model training

The nflverse schedule contains observed game weather fields such as roof, temperature, and wind. These are useful for discovering whether weather interactions are worth carrying forward, but observed game-time weather must not masquerade as a forecast if the historical prediction was made days earlier.

### Live / forecast snapshots

Production should ingest a real forecast valid for the stadium and kickoff time and archive it at each pick snapshot.

Candidate variables:

- temperature
- sustained wind
- gust
- precipitation probability/type/rate
- humidity/dew point when useful
- roof expected open/closed

Weather should primarily be tested through football interactions rather than assumed universal effects.

## 6. Travel / geography

Derived reproducibly from stadium/team locations and schedule:

- great-circle travel distance
- time-zone change
- local kickoff/body-clock time
- consecutive road games
- international travel
- neutral-site flag
- altitude
- days of rest

These are deterministic and can be reconstructed historically without leakage.

## 7. Coaching / scheme continuity

Candidate fields:

- head coach
- offensive/defensive coordinator when a reliable history is assembled
- first year in system
- coach continuity
- fourth-down aggressiveness
- pace
- pass rate over expectation

Promotion requires out-of-sample value; coaching narratives alone are not features.

## 8. Officials

Referee/crew information is available historically and can be tested for penalty/style interactions. Because crew effects are likely small and noisy, this remains a challenger feature group and must pass validation before production use.

## 9. Transactions, suspensions, and roster changes

Potential additions:

- IR/PUP/return designations
- suspensions
- trades
- releases/signings
- roster activation

Convert these into measurable changes in expected player availability/role rather than raw news counts.

## 10. Data provenance requirement

Every production row should eventually be traceable to:

```text
game_id
snapshot_type
as_of_timestamp
source_name
source_updated_at
ingested_at
raw_snapshot_path
feature_version
```

If a historical field cannot be assigned to a defensible pregame timestamp, it is excluded from timestamp-sensitive model training until that provenance problem is solved.
