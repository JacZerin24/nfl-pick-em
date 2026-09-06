# Track F: Player-Level Matchup Enhancements

## Status

Complete historical research track. **No change to `prospective-v1-frozen-2025`.**

Track F tested whether prior player-level and position-unit performance can improve the existing upset-matchup specialist, especially in pressure, receiving/coverage, explosive-play, and lineup-continuity matchups.

The answer from this historical test is **no**. The pre-specified all-player enhancement underperformed the current matchup specialist by itself and also underperformed when substituted into the frozen matchup-logistic + variance-CatBoost consensus architecture.

## Why this track was tested

The existing matchup specialist is deliberately team-level. It already uses market probability, rest, Elo, total, rolling offensive and defensive EPA/success/sack/turnover measures, and team-level matchup interactions such as pass, rush, pressure, and turnover-pressure edges.

Track F asked whether more granular information adds something the team-level model misses, such as:

- quarterback efficiency and pressure susceptibility
- individual/front-seven pressure production and concentration
- receiving efficiency and explosive-play production
- defensive-back coverage efficiency and tackling
- offensive-line, skill-position, front-seven, and defensive-back continuity
- direct QB-pressure, receiver-coverage, explosive-tackling, and protection-front matchup interactions

## Data and leakage guardrails

Player/PFR advanced weekly data needed for this study is available from 2018 onward.

Protocol:

- Team-level historical foundation: **2009-2025**.
- Player-level feature history: **2018-2025**.
- Development seasons for player-model regularization: **2020-2021**.
- Untouched Track F holdout: **2022-2025**.
- Decision domain: market favorite probability **52.5% to <80%**, matching the existing upset-specialist regime.
- Every player/unit performance feature is shifted before its rolling calculation. The target game never contributes to its own input state.
- Continuity is based on overlap among recent high-snap players and is also strictly prior-only.
- Hyperparameter selection occurs on 2020-2021 before 2022-2025 is scored.
- The current matchup baseline is trained with its normal longer historical team-level history, so the experiment does not weaken the incumbent just to make the player model look better.

## Player-level feature families

### Quarterback

- QB EPA per dropback
- CPOE
- pressure percentage
- bad-throw percentage

### Pass rush / front

- pressure rate
- top pass-rusher share of unit pressures
- sack + QB-hit rate

### Receiving / skill positions

- receiving EPA per target
- YAC per reception
- explosive reception rate
- top-1 target share
- top-3 target share

### Defensive backs / coverage

- yards allowed per target
- passer rating allowed
- interception + pass-defense rate
- missed-tackle rate

### Lineup / unit continuity

- offensive-line continuity
- skill-position continuity
- front-seven continuity
- defensive-back continuity

### Explicit player/unit matchup interactions

- QB pressure susceptibility versus opponent front pressure
- receiving efficiency versus opponent coverage efficiency
- receiving explosiveness versus opponent DB missed tackles
- offensive-line continuity versus opponent sack/hit production
- overall continuity edge

## Feature coverage

On the 2022-2025 holdout, the rolling performance features had essentially complete home/away coverage. QB, pass-rush, receiving, and coverage features were non-null for **100%** of holdout team-game sides after rolling/imputation preparation.

Continuity features had approximately **81.75%** raw non-null coverage per side, largely because early-season games do not yet have enough same-season prior lineup observations.

This matters because the negative result is not well explained by widespread missing performance data.

## Stage 1: matchup specialist by itself

All player variants selected strong regularization on the 2020-2021 development period. The selected setting for each pre-specified family was `C=0.03`, `l1_ratio=0.50`.

On the untouched 2022-2025 holdout, 915 games fell in the 52.5%-<80% favorite domain:

| Model | Correct | Accuracy | Upset calls | Upset-call wins | Net vs market | Net vs current matchup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market | 608/915 | 66.45% | 0 | 0 | 0 | +6 |
| Continuity-enhanced | 604/915 | 66.01% | 56 | 26 | -4 | +2 |
| Current full-history matchup | 602/915 | 65.79% | 76 | 35 | -6 | 0 |
| QB/pressure-enhanced | 601/915 | 65.68% | 59 | 26 | -7 | -1 |
| All-player enhanced | 599/915 | 65.46% | 55 | 23 | -9 | -3 |
| Receiving/coverage-enhanced | 598/915 | 65.36% | 52 | 21 | -10 | -4 |
| Team-window control | 595/915 | 65.03% | 63 | 25 | -13 | -7 |

For the pre-specified all-player enhancement versus the current matchup specialist:

- paired accuracy lift: approximately **-0.328 percentage points**
- bootstrap 95% CI: approximately **[-1.967, +1.311] percentage points**
- P(all-player lift > 0): approximately **32.7%**
- all-player and current matchup calls disagreed on **59** games

This was already a negative screening result. However, the current production matchup specialist was never justified as a standalone picker; its historical value comes from requiring agreement with the independent variance specialist. Track F therefore also tested the actual consensus architecture before closing the research track.

## Stage 2: frozen-consensus architecture test

The existing variance CatBoost specialist was held fixed. The only experimental change was replacing the current matchup-logistic leg with the pre-specified all-player matchup leg.

No thresholds, variance features, CatBoost settings, market fallback rule, or 2026 production code were changed.

### Headline result

Across the same 915-game 2022-2025 holdout:

| Final decision architecture | Correct | Accuracy | Consensus upset calls | Upset wins | Net vs market |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Current frozen-style consensus** | **610/915** | **66.67%** | 18 | 10 | **+2** |
| Market | 608/915 | 66.45% | 0 | 0 | 0 |
| Receiving/coverage consensus | 606/915 | 66.23% | 10 | 4 | -2 |
| Team-window consensus | 606/915 | 66.23% | 14 | 6 | -2 |
| QB/pressure consensus | 606/915 | 66.23% | 14 | 6 | -2 |
| Continuity consensus | 606/915 | 66.23% | 14 | 6 | -2 |
| **All-player consensus** | **605/915** | **66.12%** | 11 | 4 | **-3** |

The pre-specified all-player replacement lost **5 correct picks** relative to the current consensus.

Paired all-player versus current-consensus result:

- accuracy lift: approximately **-0.546 percentage points**
- bootstrap 95% CI: approximately **[-1.311, +0.219] percentage points**
- P(all-player lift > 0): approximately **6.0%**

The all-player replacement therefore did not merely fail to prove improvement. The observed direction was meaningfully unfavorable to promotion.

## Audit of changed consensus decisions

The all-player model changed the final consensus decision on **13** holdout games.

### Added upset calls

It added **3** upset calls that the current consensus would not have made:

- 2022 Week 1 NE @ MIA
- 2022 Week 1 TB @ DAL
- 2022 Week 7 NO @ ARI

The underdog lost **all 3**.

### Removed existing upset calls

It removed **10** current-consensus upset calls. The underdog actually won **6 of those 10**.

Those removed calls included successful current-consensus upsets such as:

- 2022 Week 14 BAL @ PIT
- 2023 Week 6 WAS @ ATL
- 2024 Week 3 PHI @ NO
- 2024 Week 14 JAX @ TEN
- 2025 Week 1 BAL @ BUF
- 2025 Week 18 WAS @ PHI

The player model did correctly suppress some current-consensus misses, but not enough to offset the successful upsets it removed, and its three newly added upset calls were all wrong.

Overall, the 13 changed decisions produced a **-5** net result versus the existing consensus.

## Season robustness

All-player minus current consensus by holdout season:

- 2022: **-4**
- 2023: **+1**
- 2024: **0**
- 2025: **-2**

Leave-one-season-out all-player minus current:

- excluding 2022: **-1**
- excluding 2023: **-6**
- excluding 2024: **-5**
- excluding 2025: **-3**

The conclusion is therefore not being driven by one isolated bad season. Every leave-one-season-out sample remains negative versus the current consensus.

## Interpretation

The player variables are intuitively football-relevant, but this version does not add reliable straight-up winner information beyond the market, team-level matchup model, and variance specialist.

Several likely reasons remain plausible:

1. **The market already incorporates much of this information.** Recent QB efficiency, pressure performance, receiving production, and defensive quality are highly visible inputs.
2. **Team-level efficiency may summarize the useful portion more robustly.** Player data adds dimensionality and noise to a relatively small upset-call sample.
3. **Historical on-field production is not the same as current expected personnel.** A truly live player matchup layer may need confirmed starters, availability, substitutions, and projected snap roles rather than only trailing player performance.
4. **Matchup effects may be more useful for score/prop modeling than straight-up winner flips.** A real schematic mismatch can matter without being strong enough to overturn a market favorite.
5. **The consensus architecture benefits from specialist diversity.** Adding player variables to the matchup leg may make it noisier or reduce the complementary relationship that currently exists with the variance model.

## Production decision

**Do not promote Track F player-level matchup features into the official 2026 pick system.**

Keep the current frozen matchup-logistic + variance-CatBoost consensus unchanged.

Player-level information can still be useful as a manually inspectable **context layer**, especially for unusual pressure/protection mismatches or materially changed personnel. It should not silently alter official picks without a new prospectively justified model version.

No Track F code changes `operational_pickem.py`, frozen thresholds, official model weights, grader, dashboard decision logic, or live production workflows.
