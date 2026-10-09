# Track M: participation-based coverage and pass-pressure matchup research

## Source and time caveats

Historical nflverse participation: **NFL Next Gen Stats via nflverse**
through 2022; **FTN Data via nflverse** starting in 2023
(CC-BY-SA 4.0, attribute FTN Data). See
https://nflreadr.nflverse.com/articles/dictionary_participation.html

Important: 2023 onward participation is **not available during the
corresponding NFL season**, but released after the playoffs.
This historical study must NOT be used as if the 2026 participation
charting were a live feature source.

Coverage fields are the charted man/zone and coverage-shell labels,
not a named cornerback-to-receiver assignment.
On-field player IDs are who participated in the *past* play, not proof
of who will start or cover a specific player at a later kickoff.

## Hypotheses and model

Can a team's prior offense vs man/zone performance and pressure response,
contrasted with the opposing defense's man/zone tendency, coverage shell
and pressure frequency, improve NFL straight-up picks?

Proposed pre-specified features:
- Offensive EPA on man, zone and pressured passing plays.
- Offensive primary receiver deep-route share.
- Defensive man/zone usage, man/zone EPA allowed.
- Defensive pressure rate and two-deep shell tendency.
- Underdog-versus-favorite historical differential in each.
- Four offense-vs-defense interactions: man, zone, pressure and
  deep-route-versus-two-deep shell.

All chart/PBP joined by verified game and play identifiers. Fail on
conflicting duplicate game/play IDs or <88% matched chart rows.
Conditional game EPA requires at least three plays.
Missing fields remain NA instead of incorrectly encoding zero.
Team game features shift **one game** before 4/8-game rolling;
target game's EPA and labels never enter its own predictions.

## Validation

- Entire historical team-strength and variance-specialist history 2009–2025.
- Participation data 2016–2025; audit per-season source coverage
  and provider differences. No same-game postgame info leaked.
- Evaluation cohort 2023–2025, within market favorite
  probability 52.5% to under 80%.
- Market and original full-history matchup-logistic + variance CatBoost
  consensus as primary baselines.
- Same-era team-only model, coverage-feature model, and coverage +
  interaction model trained on 2016+ prior-year charting era.
- Fixed ridge C=0.02, **not** tuned on evaluation seasons.
- Compare straight-up correct picks, disagreements, exact paired
  test, missingness and individual-year source coverage.

All 2023–2025 games were used by prior research tracks.
These are **reused historical diagnostics, not independent confirmation**.
Differences between 2009+ full-history incumbent and 2016+
coverage-training variants are a confound, not an excuse to select
the model with the best in-sample results.
Provider transitions must be audited before comparing rates across eras.

## Next steps if reproducible and prospectively promising

1. Validate availability and documented publication date for each feature.
2. Develop projected pre-kickoff depth, individual players' past
   cover/route performance and replacement value from immutable
   timestamped official injury/inactive observations.
3. Do not infer a 1-v-1 receiver/DB or OL/EDGE matchup simply
   because both players were on the same historical snap.
4. Archive prospective challenger decisions beside each pre-kickoff
   market quote and frozen official 2026 pick.
5. Require new independent outcomes, stable season performance,
   probability calibration and meaningful disagreement sample
   before any separately versioned production change.

The frozen 2026 operational model, snapshots and grader stay unchanged.

## Running

PYTHONPATH=scripts python -m pytest -q tests/test_track_m_participation_matchups.py

python scripts/track_m_participation_matchups.py

Output artifacts are stored under outputs/track_m_participation/
and preserved via Actions. No files are committed to main.
