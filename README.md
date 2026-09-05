# NFL Pick'em Forecasting System

Research-first NFL straight-up winner prediction system for weekly pick'em leagues.

## Primary objective

Maximize **out-of-sample straight-up winner accuracy** for every NFL game while also producing calibrated win probabilities. The system must be judged against strong baselines, especially the sportsbook market favorite, and must never use information that would not have been available at the historical prediction timestamp.

This is not an against-the-spread betting model. The target is the game winner.

## League lock rule

The pick'em league does **not** lock an entire week at one time. Each game can be selected any time before that individual game's kickoff.

That means the production system should optimize a **game-specific final forecast** rather than a single weekly forecast. For example:

- Thursday night game: keep updating that game until shortly before Thursday kickoff.
- Sunday early games: keep updating those games through Sunday morning and as close to their kickoffs as practical.
- Sunday late games: continue refreshing those games after the early slate has locked if new information arrives.
- Sunday night and Monday night games: continue updating those individual games until shortly before kickoff.
- International/Saturday/holiday games follow the same rule: each game's own kickoff is its lock time.

This is a major advantage for the model because later injury, inactive, quarterback, weather, and market information can be used for later games without forcing earlier games to wait or locking the entire slate too soon.

## Core philosophy

The most powerful model is not necessarily the most complicated model. At the game level, the NFL provides only a few hundred labeled games per season, so model complexity must earn its place through true future-season performance.

The production forecast will be a **stacked ensemble** whose candidate components include:

1. Market prior from de-vigged moneyline / point spread
2. Dynamic team strength rating (Elo-style and opponent-adjusted variants)
3. Quarterback strength / starter-change model
4. Leak-free rolling team efficiency features from play-by-play data
5. Roster continuity and player availability / injury-value layer
6. Situational features: rest, bye, travel, time zone, international/neutral site, divisional familiarity, coaching changes, etc.
7. Weather / venue interaction layer
8. Gradient-boosted models (CatBoost and LightGBM)
9. Regularized logistic regression for a stable, interpretable counterweight
10. A calibrated stacking/meta-model that combines only models that improve future-season validation

## Data backbone

Primary historical source: **nflverse / nflreadpy**.

Planned data families:

- Schedules/results and market lines
- Play-by-play (1999+)
- Team and player weekly stats
- Weekly rosters
- Depth charts
- Snap counts
- Next Gen Stats where available
- ESPN QBR / quarterback metrics where available
- PFR advanced stats where available
- FTN charting where licensing/availability permits
- Historical injury/practice reports (2009-2024 from nflverse)
- Current official NFL injury reports for live-season inference
- Venue/roof/surface/temperature/wind
- Rest and schedule context

### Injury-data note

The nflverse public injury feed currently has no post-2024 data. The model will therefore:

- learn historical injury effects from the 2009-2024 injury archive;
- build player-value estimates from actual usage/performance/snap data rather than treating all injuries equally;
- ingest current official NFL injury reports into a normalized live snapshot for current-season forecasts;
- retain the exact timestamp of every injury snapshot to prevent hindsight leakage.

## Prediction snapshots

The system will support multiple forecast timestamps, but the **authoritative pick for each game is the latest valid snapshot generated before that game's kickoff**.

Useful snapshot classes:

- `early_week`: first useful forecast for planning and identifying uncertain games.
- `practice_update`: refreshed after meaningful injury/practice/QB news.
- `game_day`: refreshed on the day of the game with current weather, market, roster, and injury information.
- `final_pre_kick`: latest practical forecast before the individual game's kickoff. This is the official pick'em recommendation.

Every feature must have an `as_of` timestamp. No post-kickoff data, later injury knowledge, later line movement, later weather observation, or later depth-chart update may leak into an earlier snapshot.

For historical validation, each replayed game should be scored using only information that would have been available by the corresponding pre-kickoff snapshot time. The model must never backtest a Thursday prediction using information that became known on Sunday, and it must never force a Sunday night prediction to use stale Thursday information when newer pre-kickoff data was available.

## Live refresh logic

The production pipeline should be event-aware rather than simply running once per week.

For each scheduled game:

1. Create an early forecast.
2. Refresh when meaningful injury/QB/depth-chart information changes.
3. Refresh market and weather information as game day approaches.
4. Create a game-day forecast.
5. Create the latest practical `final_pre_kick` snapshot before kickoff.
6. Freeze/archive that game's final recommendation once kickoff occurs.
7. Continue refreshing all later games on the slate until their own kickoffs.

This allows Thursday, Sunday early, Sunday late, Sunday night, Monday night, international, Saturday, and holiday games to use different information cutoffs while remaining fully reproducible.

## Feature families

### Market

- Home and away moneyline
- Vig-free market win probability
- Point spread
- Total
- Market-implied margin / win probability
- Opening-to-current movement when a historically reproducible source is available

### Team efficiency

All rolling statistics are shifted so the current game is excluded.

- EPA/play offense and defense
- Dropback EPA and rush EPA
- Success rate
- Early-down EPA/success
- Explosive-play rate allowed/created
- Sack rate and pressure proxies
- Turnover rate and turnover-regression features
- Red-zone and goal-to-go efficiency
- Third/fourth-down efficiency with regression toward more stable early-down metrics
- Pace / play volume
- Pass rate over expectation where reproducible
- Opponent-adjusted versions of the most stable efficiency metrics
- Recent-form and longer-window estimates blended with preseason priors

### Team strength / ratings

- Elo-style dynamic rating
- Offense/defense split ratings
- Margin-aware rating with capped blowout influence
- Opponent adjustment
- Home-field advantage allowed to vary by era/team/venue only if validated
- Offseason regression and roster/QB continuity adjustment

### Quarterbacks

Quarterback status receives its own explicit layer because a starter change can alter team strength much more than a generic injury count.

Candidate features:

- starter identity
- rolling QB EPA/dropback
- CPOE and accuracy metrics
- sack avoidance
- turnover rate
- QBR / NGS features when available
- career prior + recent-form shrinkage
- starter-to-backup value difference
- first-start / rookie / return-from-injury flags only if they add validated signal

### Injuries / availability

Do **not** use raw injury counts as the main signal.

Build a player value / availability score using:

- position
- starter/depth-chart role
- recent snap share
- recent participation
- player performance/value estimate
- replacement quality
- practice participation trend
- game designation (out/doubtful/questionable/etc.)

Aggregate into matchup features such as:

- QB availability delta
- offensive line availability/value delta
- pass-catcher availability/value delta
- front-seven availability/value delta
- secondary availability/value delta
- total weighted availability delta

### Schedule / travel / situational

- home / away / neutral site
- rest-day differential
- short week
- bye-week advantage
- consecutive road games
- travel distance
- time-zone change / body-clock interaction
- international games
- divisional game
- postseason game
- altitude
- surface change
- coaching continuity/change

### Weather / venue

Weather is expected to matter more through interactions than as a simple universal adjustment.

- roof state
- surface
- temperature
- wind sustained/gust where available
- precipitation type/rate where a reproducible archive is added
- humidity / heat stress only if supported

Interactions to test:

- wind x passing explosiveness
- wind x QB depth-of-target / accuracy profile
- precipitation x turnover/fumble tendencies
- extreme heat/cold x team/climate/travel context
- weather x market total

## Candidate models

### Baselines (must always be reported)

- Home team every game
- Better current win percentage
- Elo favorite
- Point-spread favorite
- De-vigged moneyline favorite

### Machine-learning candidates

1. Elastic-net logistic regression
2. CatBoost classifier
3. LightGBM classifier
4. Optional XGBoost challenger
5. Dynamic rating / Bradley-Terry style model
6. QB-adjusted rating model
7. Market-only probability model

### Final ensemble

Use **out-of-fold, time-ordered predictions** from the base models as inputs to a simple meta-model. Never train the stacker on in-sample base-model predictions.

Calibrate the final probability using only prior validation data (sigmoid / isotonic / beta calibration as appropriate).

## Validation: the most important part of the project

Random train/test splits are prohibited for final model selection.

Use expanding walk-forward validation, for example:

- train through 2016 -> test 2017
- train through 2017 -> test 2018
- ...
- continue through the latest fully completed season

A stricter weekly/game-level replay mode will recreate what the model would have known before every historical game's own kickoff.

Primary metric:

- straight-up accuracy

Secondary metrics:

- log loss
- Brier score
- calibration error / reliability curves
- accuracy by confidence bucket
- accuracy in games where model agrees/disagrees with market
- upset identification without sacrificing total accuracy

Statistical checks:

- bootstrap confidence intervals by season/game
- paired comparison against market favorite
- McNemar test on games where systems differ
- sensitivity to season windows and hyperparameters

## Model-selection gates

A feature or model is **not** promoted because it sounds football-smart or improves training score.

It should satisfy most of the following:

1. Improves walk-forward straight-up accuracy and/or probability score.
2. Improvement persists across multiple seasons rather than one outlier year.
3. Does not depend on hindsight or unavailable timestamps.
4. Does not collapse when the closing market line is removed.
5. Adds information beyond a market-only baseline when included alongside the market.
6. Has plausible football interpretation or stable statistical evidence.

The production model may ultimately choose to follow the market favorite for many games and override it only when the ensemble has demonstrated a sufficiently reliable historical edge. That override threshold will be tuned only on past validation folds.

## Planned repository layout

```text
nfl-pick-em/
  README.md
  docs/
    MODEL_DESIGN.md
    DATA_SOURCES.md
    VALIDATION.md
  src/nfl_pickem/
    data/
    features/
    ratings/
    injuries/
    weather/
    models/
    validation/
  scripts/
    build_historical_dataset.py
    backtest.py
    train.py
    predict_week.py
  data/
    raw/          # gitignored
    interim/      # gitignored
    processed/    # gitignored
  artifacts/      # model files / validation summaries
  outputs/        # weekly picks
```

## Phased build

### Phase 1 - trustworthy baseline

- Ingest schedules/results/market data
- Build target and leak-safe game table
- Add Elo and core rolling EPA/success features
- Train logistic + CatBoost/LightGBM candidates
- Walk-forward test against market favorite

### Phase 2 - quarterback and player availability

- Starter identification
- QB value model
- Snap/depth-chart weighted player values
- Historical injury features
- Current official injury snapshot ingestion

### Phase 3 - advanced context

- Opponent-adjusted efficiencies
- NGS/PFR/FTN candidates
- travel/time-zone/rest
- coaching continuity
- weather interactions

### Phase 4 - ensemble and calibration

- out-of-fold stacker
- calibration
- market-override decision rule
- confidence tiers

### Phase 5 - per-game live automation

- scheduled/event-driven data refresh
- injury/QB/depth-chart refresh
- game-specific weather refresh
- current market refresh
- early-week planning forecasts
- game-day forecasts
- final pre-kick forecast for each individual game
- freeze/archive each game independently at kickoff
- continue updating later games after earlier games have locked
- automatic grading after games
- season dashboard / CSV for pick'em entry

## Non-negotiable reproducibility rules

- Every official pick is archived before that individual game's kickoff.
- Every data snapshot used for a prediction is timestamped.
- Each game's latest eligible pre-kick snapshot is identifiable and reproducible.
- Final scores never enter pregame feature generation.
- Market information is only used if it existed by that snapshot time.
- Injury/QB/weather information is only used if it existed by that snapshot time.
- Historical backtests are rerunnable from code.
- We publish both wins and misses.

## Immediate next milestone

Build the Phase 1 historical game table and establish the true benchmark: how often the market favorite wins straight up, then determine whether football-derived features improve future-season accuracy or calibration beyond that benchmark.
