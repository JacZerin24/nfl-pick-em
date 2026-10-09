# NFL Pick'em — Expanded feature-research registry (proposal)

This is a test roadmap, **not** a list of validated predictors. Preserve the
frozen 2026 official model throughout the existing prospective season.

## Design requirements

1. Maintain a versioned feature registry: source, source release timestamp,
   earliest defensible as-of availability, historical coverage, missingness,
   computation version, and allowed use (historical development, prospective
   shadow, or production).
2. Store prediction-time source snapshots. Historical final lines, actual
   starters, actual observed weather, and the target game's play-by-play
   cannot masquerade as earlier available inputs.
3. Treat each feature family as an independent *challenger* and record all
   attempted hypotheses, not just the ones that win on past seasons.
4. Compare against the market and prospective-v1-frozen-2025 on identical
   game/time cutoffs. Separate improvements in win probability calibration
   from changes that actually improve straight-up pick selection.
5. Avoid post-hoc promotion from reused 2019–2025 test seasons. Pre-register
   future confirmation criteria, prospective dates, and minimum disagreement
   counts. Account for multiple hypothesis/model searches.
6. Archive every challenger game prediction and version pre-kickoff; validate
   no post-kickoff updates can retroactively change prior pick decisions.

## Proposed separate tracks

| Track | Theme | Candidate features | Source / timestamp issue | Next test |
|---|---|---|---|---|
| I | Injury × opponent-unit exploitation | QB/OL/WR/DB/front gap × opposing pressure, passing, rushing strengths | Final weekly injuries end 2024; no 2026 source yet | Implemented exploratory code in this branch |
| J | Primetime & game-slot effect | TNF/SNF/MNF, late afternoon, international morning, flex changes, local body clock, QB/coach era residuals | Kickoffs available; historical broadcast classification/flex provenance must be verified | Design pre-registered baseline with partial pooling |
| K | Expanded play-level situation | EPA/success split by down, distance, red zone, early downs, 4th-down, script, personnel | Existing historical PBP likely supports much; audit completeness | Candidate vs base/consensus ablation |
| L | Advanced opponent-adjusted matchups | pass scheme vs coverage, blitz vs QB, OL vs DL, personnel vs personnel | Participation, tracking, charting coverage varies | Only test if source continuity sufficient |
| M | Coaching and roster change | coordinator changes, scheme continuity, coaching era, transaction/injury transitions | Reliable historical as-of rosters and coordinator roles needed | Evaluate as isolated contextual layer |
| N | News timing / repricing | Time of confirmed inactives, QB announcements, cross-book movement, market response | Must collect prospective event-time feeds | Link Tracks B and H to official reports |
| O | Environment / weather interactions | pre-kick forecast change, stadium wind/gust, precipitation, roof decision, passing matchup | Observed historical weather is not a forecast | Forecast-time snapshots vs market moves |
| P | Opponent-quality and short-window performance | opponent-adjusted rolling EPA, explosive, pass/rush strengths; recent-vs-season form | Prior game stats available; opponent correction must avoid future season knowledge | Nested expanding-season train |

## Primetime-specific protocol proposal

- Establish the actual pregame television slot (TNF, SNF, MNF, other),
  originally scheduled slot, and whether flexed, with an independently
  verifiable source. Date/time alone is an incomplete proxy.
- Model the residual of team results versus *pregame implied probabilities*,
  not raw win rate. Strong teams are disproportionately selected for primetime.
- Separate current QB, head coach, roster and team franchise era with
  partial pooling. Minimum sample thresholds and shrinkage are required.
- Test TNF short rest/travel separately from SNF/MNF, and explicitly compare
  to the existing Track E rest/travel/body-clock results.
- Assess consistency across seasons, new team/QB/coach combinations, and
  whether game-slot variables ever produce profitable (for pick'em: correct)
  changes relative to the frozen official selection.

## Follow-up after Track I

Track I's position-group gaps are not individual direct matchups. If the
group-level interaction shows genuine prospective signal, attempt:

- Left tackle/guard/center expected absence × opposing EDGE/DT depth and
  pressure profile, weighting expected snaps and identified replacement.
- WR/TE expected role loss × opponent man/zone usage and healthy cover players.
- Defensive back absence × opponent QB target distribution and receiver matchups.
- Front-seven losses × opposing rushing gap/zone usage and pass protection.
- Joint missing-player combinations and backup cascade, with a low-dimensional
  penalty to prevent rare combinations from overfitting.

Only give interactions a learned model weight if time-ordered OOS evidence
demonstrates incremental value after the main injury effect, opponent strength,
the market, and the frozen system are already included.

## Reporting and promotion

Every track reports source coverage, skipped games, year-by-year accuracy,
head-to-head corrected disagreements, Brier/log loss, probability calibration,
bootstrap uncertainty with seasonal dependence, negative controls when feasible,
and a plain-English failure analysis. Failed tracks remain documented.
Passing exploratory research only authorizes **prospective shadow testing**,
not a retroactive change to frozen live picks.
