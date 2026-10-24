# Track I — Injury vulnerability × opposing strength (research only)

## Hypothesis

An injury can change win probability depending on who is missing, backup capability,
and what the opposing team does well. Losing a high-snap offensive lineman could be
more damaging against a productive pass rush than against a weak rush, for example.
Our previous Track C evaluated raw counts and replacement gaps but not these
specific **cross-team multiplicative interactions**.

## What this experiment actually tests

Using the prior Track C historical injury + snap-count builders and the existing
rolling team efficiency model, compare four pick systems on exactly the same
walk-forward games:

1. The closing-market favorite.
2. Market-logit plus broad injury controls (Track C-type ridge).
3. The same controls plus QB/OL/SKILL/FRONT/DB severe replacement-gap main effects.
4. All of those plus seven specified gap × opposing strength interactions.

Fixed interaction families:

| Injured unit | Opponent metric | Football intuition |
|---|---|---|
| QB | Defensive sack rate | Pressure can magnify QB replacement costs |
| QB | Defensive takeaway rate | Risky replacement play vs turnover-producing defense |
| Offensive line | Defensive sack rate | Pass protection weakness vs pressure |
| Skill positions | Defensive pass EPA allowed, sign-reversed | Injured targets vs stronger pass coverage |
| Defensive front | Offensive rush EPA | Weaker front vs powerful rushing offense |
| Defensive front | Offensive pass EPA | Reduced rush/front depth vs effective passing |
| Defensive backs | Offensive pass EPA | Missing coverage contributors vs effective passing |

Interactions are home/away symmetric: away injury gap × home unit strength minus
home injury gap × away unit strength. Features are standardized inside each
training fold. Both raw position gaps and aggregate injury burden stay in the
model so interactions do not simply proxy general injury counts.

## Data and leakage warnings

- Sample: regular seasons 2012–2024 (public historical injury/snap coverage).
- Expanding season training; testing 2016–2024; earlier seasons never use later results.
- The full 2019–2024 outcome window is **previously examined** in Track C.
  It is **NOT untouched validation**. Treat this historical result as an exploratory
  diagnostic, irrespective of its significance.
- Team strengths use prior-game four/eight-game metrics (Track I interactions use
  eight-game windows). The current game does not update its own pregame strength.
- Individual player snap-value estimation uses snap counts strictly prior to the
  target week, reusing Track C's player-value routine.
- Injury designations are *final weekly reports*, not archived as-of actual entry
  timestamps; historically reconstructing an injury report is not enough to
  demonstrate a real-time advantage.
- The historical market is a closing-line benchmark and should not be interpreted
  as an early-week market timestamp.
- Teams without matching injury-report rows are represented by zero gaps **and**
  explicit coverage flags. Both-reports-present games are separately diagnosed.
- The missing-player-gap estimates approximate player value; they do not yet
  resolve specific player matchups (e.g., absent left tackle vs a named EDGE).

## Statistical comparison

Fixed penalized-logistic specification (C=0.08) from earlier Track C is used in
all challenger variants; no new hyperparameter search uses 2019–2024 results.
Record accuracy, Brier/log loss, season-level results, disagreement wins/losses,
exact paired McNemar p-values, and season-block bootstrap 95% CIs. Analyze all
games, material-injury games, and games with both team injury rows observed.

The **primary incremental diagnostic** is *full interaction* versus *position-
only*, not full interaction versus an unadjusted market. An apparent positive
result on the reused historical window is not proof of a future edge. Correct
for the multiple track/hypothesis experiments before making global claims.

## Requirements before any production promotion

1. Build timestamped current player availability/inactives feeds legally and
   reproducibly (player IDs, report time, game assignment, depth/role).
2. Confirm snap-based replacement-gap calculations can run at each individual
   pre-kickoff deadline without using actual starters or post-kick snaps.
3. Test new injury × opposing-unit inputs on genuinely **unseen** future NFL games,
   alongside an unchanged frozen prospective-v1-frozen-2025 benchmark and
   contemporary market at the same cutoffs.
4. Demonstrate positive incremental **straight-up correct pick** lift on the
   changed games, robust probability quality, stable seasons/weeks, and paired
   uncertainty that reasonably excludes noise. Ablate position main effects and
   interactions independently.
5. Only then propose a separately versioned challenger and a reviewable PR.
   **Do not modify or silently retune the official 2026 model.**

## Future feature expansions (separate tests, not bundled into this experiment)

- Exact offensive-line gap by left tackle/center/right tackle vs opponent EDGE/DT.
- Missing receiver/TE speed, route profile, separation vs healthy DB coverage.
- Missing corner/safety vs opponent targeted receiver and QB passing depth.
- Scheme dependence: blitz, man/zone, motion, no-huddle, run concept.
- Specific primetime type (TNF/SNF/MNF), kickoff local/body clock, time-slot
  familiarity, coaching-era and QB-era residuals vs market expectations.
- Opponent-adjusted situational EPA, early downs, red zone, third/fourth down.
- Market consensus/late repricing following confirmed inactives, linked to Tracks B/H.

## Execution

Unit tests:

    PYTHONPATH=scripts python -m pytest -q tests/test_track_i_injury_matchup.py

Historical exploratory experiment:

    python scripts/track_i_injury_matchup_interactions.py

Outputs: outputs/track_i_injury_matchup/{predictions.csv,by_season.csv,paired_comparisons.csv,diagnostics.csv,features.txt,summary.md}.

## Completed exploratory result (GitHub Actions run 37882214302)

The 2009-2025 era material is not used here; this experiment specifically
used historical injury and snap data through 2024. In the reused 2019-2024
holdout of **1,594** regular-season games:

| Research variant | Correct | Net versus broad injury |
|---|---:|---:|
| Closing market | 1,061 | -13 |
| Broad injury control | **1,074** | 0 |
| Broad + position gap | 1,056 | -18 |
| Broad + position gap + opponent-strength interaction | 1,056 | -18 |

The additional interaction features changed 26 picks relative to the
position-gap model; the interaction version won **13** and lost **13**.
Thus this first group-level opponent vulnerability implementation showed
**zero incremental correct picks** beyond the position model and performed
worse than the simpler broad injury controls.

These are **exploratory**, previously examined historical games; there is
no evidence here to promote Track I. A more granular true player-vs-player
injury/coverage/pressure hypothesis requires timestamped future data and an
independently frozen prospective test. Research output was preserved as
GitHub Actions artifact on run 37882214302. Production remains unchanged.
