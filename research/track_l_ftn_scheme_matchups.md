# Track L — FTN charted schemes, opponent interactions and player data

## Source, limitations, guardrails

**Research only**. No production 2026 model, official pick or grader changes.
Source attribution: **FTN Data via nflverse**, CC-BY-SA 4.0.
Field documentation: https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html

Public FTN charting from 2022 onward contains motion, play action,
screens, RPOs, shotgun location and number of blitzers.
It does **not** provide robust man/zone labels, named coverage assignments,
blocking assignments for an OL/EDGE duel, or the current projected starters.
This study measures *team-level schematic matchups*, not literal 1v1 players.

## Fixed pre-test hypotheses

Research offense-versus-opponent-defense relationships, always over
previously completed games:
- Offensive EPA versus blitz × opposing blitz rate.
- Play-action EPA × opposing play-action EPA allowed.
- Screen EPA × opposing screen EPA allowed.
- RPO EPA × opposing RPO EPA allowed.
- Motion EPA × opposing motion EPA allowed.

Usage rates and conditional EPA are separately ablated; matchup products
are compared to the main effects instead of evaluated alone.

## Data discipline

- FTN game/play IDs are merged with nflverse PBP after duplicate checks.
  Reject conflicting duplicate chart rows or matched-row share below 90%.
- Every team-game statistic is shifted at least one game before rolling,
  with 4- and 8-game windows. Target-game outcomes never become features
  for the same game.
- Conditional EPA requires at least 3 charted qualifying plays per game.
  Empty/insufficient conditions remain missing, **not zero**.
- Historical FTN charting is published *after* games: it cannot
  reconstruct the forecast available before its own game's kickoff.
- Historical schedule odds are closing/near-closing, **not** archived
  exact-time final-entry bookmaker data.
- Report join match rate, team-game coverage, and feature missingness.

## Comparative methodology

Compare the 2024 and 2025 previously studied seasons in the
market-underdog domain where market favorite probability is 52.5% to
below 80%, using the same frozen-style variance CatBoost specialist.

Models:
1. Market favorite.
2. Full-history (2009 onward) original matchup-logistic plus variance
   consensus.
3. 2022+ same-training-era team-only matchup plus variance consensus.
4. Same-era team plus scheme usage rates.
5. Same-era team plus usage and EPA effects.
6. Same-era team plus usage, effects, and explicit scheme products.

All scheme variants use fixed regularization C=0.015, without selection
on tested seasons. The original incumbent uses its established
2009+ training history. Unequal lengths are a known comparison confound.
Scores: winner accuracy, changed consensus decisions, Brier/log loss,
paired exact tests, feature availability and by-game forecasts.

Both 2024 and 2025 results have been repeatedly studied across
previous research. **These are exploratory diagnostics, not fresh
confirmatory holdouts.** Any apparent edge would require
a separately frozen prospective shadow test and an exact same-timestamp
comparison to the full production 2026 decision strategy.

## Existing player study

Track F already examined QB pressure, receiver production, pass rush,
secondary, and snap continuity with the historical upset architecture.
Its all-player consensus lost five correct picks versus the incumbent
on the earlier 2022–2025 diagnostic. Avoid retuning that same holdout
without new data and a specific measurable hypothesis.

## Player availability as-of research schema

A future live row per observed availability report should include:
event_id, ingested_at_utc, source_published_at_utc, source_uri,
source_player_id, canonical_player_id, team, opponent, game_id, position,
report_type, practice_status, game_status, active_status, as_of_utc,
verification_status and raw_source_digest.

Require **source_published_at_utc <= as_of_utc < kickoff_utc**;
never treat a final postgame starter/snap total as an earlier prediction.
Confirm individual player IDs, roster and game assignment, status,
backup and role estimates without inventing unknown active/inactive
states. Preserve dated original sources/hashes and never leak secrets
into a public research branch.

## Run

PYTHONPATH=scripts python -m pytest -q tests/test_track_l_ftn_scheme_matchups.py

python scripts/track_l_ftn_scheme_matchups.py

GitHub Actions stores per-game predictions, consensus accuracy, paired
changes and coverage under outputs/track_l_scheme_matchups.
