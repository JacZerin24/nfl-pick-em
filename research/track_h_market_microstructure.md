# Track H: Multi-book market microstructure

## Status

Prospective research infrastructure for the 2026 NFL season. **Research-only.** Nothing in Track H changes `prospective-v1-frozen-2025`, the 52.5% close-game threshold, the 80% market-anchor threshold, the residual selector, or the validated upset-consensus rule.

## Why this track exists

The production model already treats the market as the default anchor, but the nflverse schedule table provides only one market snapshot rather than the full cross-book state. Track H asks whether the *shape of the market* contains incremental straight-up information that a single market probability misses.

The live source is The Odds API v4. The collector requests a fixed group of up to 10 books so the request counts as one region-equivalent. The default group is:

- Pinnacle
- DraftKings
- FanDuel
- BetMGM
- BetRivers
- BetOnline.ag
- Bovada
- BetUS
- LowVig.ag
- MyBookie.ag

Some books may be absent from an individual response. Pinnacle is treated only as a **reference-book field**, not as ground truth; its public feed may be delayed.

## Collection cadence

Two prospective snapshot roles are frozen before Week 1 results:

1. `SCHEDULED_FULL`
   - Runs at 02:17, 10:17, and 18:17 UTC during September-January.
   - Collects `h2h`, `spreads`, and `totals`.
   - Three markets x one <=10-book group = approximately 3 API credits per snapshot.

2. `NEAR_KICKOFF_H2H`
   - Triggered only after a successful `2026 Final Entry Pick'em` workflow.
   - Before making an API call, the collector checks `live_archive` and requires an unstarted game within 90 minutes.
   - Collects moneyline only, approximately 1 credit per qualifying snapshot.

This design is intended to fit within the provider's 500-credit monthly free tier while preserving both slow market evolution and near-kickoff microstructure.

## Immutable archive

Live calls write:

`market_archive/2026/snapshots/<UTC timestamp>/`

with:

- `raw.json` - untouched provider response
- `bookmakers.csv` - one normalized row per game/book
- `consensus.csv` - one cross-book summary row per game
- `metadata.json` - collection role, requested markets/books, and API quota headers

Season-level `latest_*` files are convenience pointers. The timestamped snapshot directory is the source of truth.

## Bookmaker-level fields

For every book/game snapshot Track H stores:

- provider/book key and last-update timestamp
- age of the book quote at collection time
- home and away American moneyline prices
- raw implied probabilities
- two-way overround/hold
- no-vig home win probability
- that book's straight-up favorite
- home/away spread and price when collected
- total and over/under prices when collected

## Cross-book fields

For every game/snapshot Track H calculates:

- number of books with a usable moneyline
- mean and median no-vig home probability
- standard deviation, IQR, and full probability range across books
- consensus favorite and consensus favorite probability
- number of books favoring each team
- favorite split and majority margin
- median/oldest quote age and count of quotes older than five minutes
- Pinnacle reference probability
- non-Pinnacle retail median
- Pinnacle-minus-retail probability gap
- median/range of home spread
- median/range of total
- current production `p_home_market`, market favorite, and final pick
- consensus-minus-production probability gap
- whether cross-book consensus favors the opposite team from the production market feed

## Pre-registered 2026 hypotheses

These are frozen before completed 2026 games are used to judge them.

### H1: Cross-book consensus baseline

The median no-vig probability across books may be better calibrated and/or more accurate than the single production market feed.

Primary comparison: latest strictly pre-kickoff consensus favorite vs the production market favorite on the same games.

### H2: Cross-book disagreement

Conditional on consensus favorite probability, wider book dispersion may identify games with more true uncertainty and a higher upset rate.

Primary dispersion measures: no-vig probability range, IQR, and favorite split. These are context variables first, not automatic dog triggers.

### H3: Reference-vs-retail disagreement

A meaningful Pinnacle-vs-retail probability difference may contain information about which side of the market is moving first.

No threshold is being promoted now. Direction and magnitude are collected prospectively and evaluated only after sufficient games exist.

### H4: Consensus-vs-production disagreement

Games where the multi-book consensus favorite differs from the production market favorite are the cleanest candidate set for a future market-only override.

Promotion would require enough disagreements, a positive time-ordered record against the production market favorite, multi-week stability, and bootstrap evidence. A handful of Week 1 wins is not enough.

### H5: Market convergence and late movement

Changes from early snapshots to the latest strictly pre-kickoff snapshot may distinguish stable information from temporary book-specific noise.

Primary movement measures are consensus probability change, consensus favorite flips, cross-book dispersion change, and reference-vs-retail convergence/divergence.

## Frozen evaluation windows

When enough 2026 outcomes exist, analysis will select snapshots without looking at the result:

- **Early:** latest available snapshot at least 72 hours before kickoff.
- **Day-before:** latest available snapshot at least 24 hours before kickoff.
- **Final:** latest available snapshot strictly before kickoff.

If a window has no qualifying observation, that game is excluded from that window rather than backfilled with post-cutoff information.

## Promotion standard

Track H cannot alter official picks unless a pre-specified rule demonstrates:

1. strictly pre-kickoff inputs and no leakage;
2. improvement over the relevant market-favorite benchmark;
3. enough independent games and disagreement cases;
4. stability across multiple weeks/months rather than one hot stretch;
5. an understandable, frozen decision rule;
6. paired uncertainty/bootstrapping that supports positive lift; and
7. no retuning on 2026 after the rule is frozen.

Until then the data are explanatory and research-only.

## One-time activation

The workflow can validate itself from the committed fixture without credentials. Live collection requires a repository Actions secret named exactly:

`THE_ODDS_API_KEY`

No code change is required after that secret exists.
