# Track K — Comprehensive team efficiency and matchup research

## Status and safeguard

**New research-only challenger; never changes frozen official 2026 model.**
The models use expanding-season walk-forward evaluation on regular-season
historical games, but the 2019–2025 seasons have already been extensively
explored by this project. Any observed Track K lift on them is an
**exploratory retrospective diagnostic, not independent validation**.

Historical nflverse odds are closing/near-closing market reference, not
an early-week, timestamp-matched betting feed. Actual pregame availability
must be confirmed independently for any operational implementation.

## New features

Track K reuses the existing 12 core EPA/success/sack/turnover metrics in
4- and 8-game rolling windows, Elo, spread, total, rest, divisional and
neutral-site context. It adds six offensive and corresponding defensive
statistics (12 per-game input families):

- First/second-down EPA and success
- Third-down EPA and success
- Red-zone EPA within the opposing 20
- Early-down pass rate

These are calculated from scrimmage play-by-play with a one-game shift:
the target game's events never contribute to its own pregame features.

Opponent-relative efficiency includes:

- Realized offensive EPA minus the opponent's previous defensive EPA-allowed
  baseline (computed from prior completed games)
- Realized defensive EPA allowed minus the opponent's prior offensive EPA
  baseline

These game results enter only subsequent games after a one-game shift.
This is a simple prior-opponent baseline correction, NOT a fully converged
schedule-adjusted ranking.

Six interactions, each over 4- and 8-game windows:

- Passing efficiency times opposing pass EPA allowed
- Rushing efficiency times opposing rush EPA allowed
- Early-down EPA times opposing early-down EPA allowed
- Third-down EPA times opposing third-down EPA allowed
- Red-zone EPA times opposing red-zone EPA allowed
- Sack exposure times opposing defensive sack generation

Interactions are home-versus-away differences of opposing products, with
explicit opposite sign for sack susceptibility. This is an initial
low-dimensional study, NOT direct individual player matchup modeling.

## Fixed comparison

All candidates use penalized-logistic *market offsets* and a common penalty
fixed at 0.1. They train on identical chronological data:

1. Market only
2. Core established team metrics plus market
3. Core plus situational EPA/success/pass rate
4. Core plus opponent-relative efficiency
5. Core plus new offense-defense interactions
6. Core plus all features above

Report per-season winner accuracy, Brier, log loss, paired market and
core disagreements, exact paired McNemar, season-blocked bootstrap
intervals, feature registry and missingness.

**Core is a new numerical research baseline, not the frozen 2026 model.**
A separate experiment must compare any promising candidates with the
historical reconstructed incumbent's close residual and two-specialist
upset consensus before claiming improvement.

## Coverage and methodology

- Regular-season historical window: 2009–2025.
- Walk-forward test seasons: 2016–2025.
- Previously inspected diagnostic holdout: 2019–2025.
- Missing early-season features use training-fold-only imputation.
- This is an initial tractable batch, not all possible NFL statistics.
- Weather, QB, player availability, coaching, primetime broadcasts/flexing,
  officials, schemes, and quote-time market effects remain other tracks.
- Numerous previous experiments introduce selection bias. No retrospective
  apparent improvement is sufficient for automatic promotion.

## Promotion gates

A challenger must pass source availability and reproducibility checks;
time-ordered evaluation versus an exact incumbent reconstruction plus
the contemporary market on identical games and cutoffs; stability,
ablation and selection-bias controls; a preregistered future shadow
sample sufficiently large for changed decisions; and calibrated
probabilities before any new production version is considered.

**Frozen official 2026 operational picks remain unchanged.**

## Run locally or in GitHub Actions

    PYTHONPATH=scripts python -m pytest -q tests/test_track_k_comprehensive_matchups.py
    python scripts/track_k_comprehensive_matchups.py

Results go to outputs/track_k_comprehensive_matchups and a GitHub Actions
artifact. No output commits, production retuning, or live updates occur.
