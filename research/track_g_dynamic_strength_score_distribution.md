# Track G: Dynamic Team Strength + Score Distribution

## Status

Complete historical research track. **No change to `prospective-v1-frozen-2025`.**

Track G tested whether a dynamic latent offense/defense state model plus a score/margin distribution forecast could add straight-up winner value beyond the closing market and, more importantly, beyond the existing frozen 2019-2025 integrated system.

The answer from this implementation is **no**. The standalone latent score model materially underperformed the market. A strongly market-anchored score-residual version effectively collapsed back toward the market and gained only one correct pick across 1,865 holdout games. Every pre-specified way of inserting that score model into the existing frozen decision architecture reduced accuracy.

## Why this track was tested

The existing production system directly models win probabilities and uses specialized close-game and upset decision layers. Track G tested a structurally different idea:

1. estimate each team's latent offensive and defensive scoring strength as it changes through time;
2. carry uncertainty around those latent states rather than only point estimates;
3. predict expected home and away points before each game;
4. derive an expected margin and margin uncertainty;
5. convert the resulting score distribution into a home-win probability;
6. optionally use the market's spread/total implied team scores as a prior and learn only small residual point corrections.

This is intentionally different from simply adding more features to CatBoost.

## Leakage-safe staged protocol

The research splits were fixed before inspecting the final holdout:

- **2009-2011:** warmup/history for team states
- **2012-2013:** tune dynamic state parameters
- **2014-2015:** tune the market-anchored score-residual Ridge model and probability scaling
- **2016-2018:** clean development/diagnostic period
- **2019-2025:** untouched final holdout

No 2019-2025 outcome was used to choose first-pass state parameters, Ridge strength, probability scaling, or integration rules.

The dynamic state for a target game is recorded **before** that game's score updates the teams. Therefore the target result cannot leak into its own pregame forecast.

## Dynamic latent-state model

Each team carries four evolving values:

- offensive scoring strength
- defensive scoring strength
- offensive-state uncertainty
- defensive-state uncertainty

Pregame expected scores are built from league scoring level, home-field advantage, the offense's latent state, and the opponent defense's latent state.

After a completed game, home and away score residuals update the relevant offense/defense states using Kalman-style gains. State uncertainty increases between games through process noise and is partially regressed/reset between seasons.

The first-pass pre-holdout state grid tested:

- process noise `q`: 0.25, 0.75, 1.5, 3.0
- score observation variance `r`: 64, 100, 144, 196
- offseason persistence: 0.55, 0.70, 0.82, 0.90
- home-field advantage: 1.5, 2.0, 2.5, 3.0 points

## Market-anchored score model

For games with spread and total, the market-implied team scores are:

- home points = `(total + spread) / 2`
- away points = `(total - spread) / 2`

Track G then learns strongly regularized residual corrections rather than replacing those market expectations. Inputs include latent-vs-market point/margin/total gaps, offense/defense state estimates, state uncertainty, rest differential, divisional status, and neutral-site status.

Historical 2019-2025 spread/total coverage in this experiment was **100%**.

## First-pass tuning

Selected before the 2019-2025 holdout:

- state process noise `q`: **0.25**
- score observation variance `r`: **144**
- offseason persistence: **0.55**
- home-field advantage: **3.0 points**
- score-residual Ridge alpha: **10,000**
- latent probability scale: **0.70**
- score-margin sigma: approximately **13.71 points**

Several selections were on a search boundary, especially low process noise/persistence and the strongest available Ridge penalty. That was treated as a methodological warning and triggered the separate boundary-extension sensitivity described below rather than being ignored.

## First-pass results

### Development: 2016-2018

| Model | Correct | Accuracy |
| --- | ---: | ---: |
| Market | 515/764 | 67.41% |
| Market-anchored score distribution | 511/764 | 66.88% |
| Latent score distribution | 491/764 | 64.27% |

The candidate did not beat the market even on the clean 2016-2018 diagnostic period.

### Untouched holdout: 2019-2025

| Model | Correct | Accuracy | Net vs market |
| --- | ---: | ---: | ---: |
| Market-anchored score distribution | 1,239/1,865 | 66.43% | **+1** |
| Market | 1,238/1,865 | 66.38% | 0 |
| Latent score distribution | 1,158/1,865 | 62.09% | **-80** |

The pure dynamic latent model was clearly inferior to the betting market.

The anchored score model was essentially a market clone. It disagreed with the market on only **9 of 1,865 games** and went **5-4** on those disagreements, producing the single extra correct pick.

### Probability quality

2019-2025:

| Model | Log loss | Brier |
| --- | ---: | ---: |
| Market | **0.608393** | **0.210547** |
| Market-anchored score distribution | 0.610290 | 0.211287 |
| Latent score distribution | 0.647442 | 0.227867 |

Despite the +1 winner count, the market-anchored score model was slightly **worse** than the market on both proper probability scores.

### Paired bootstrap vs market

Market-anchored score distribution:

- accuracy lift: **+0.054 percentage points**
- 95% bootstrap CI: **[-0.268, +0.375] percentage points**
- P(lift > 0): **56.5%**

Latent score distribution:

- accuracy lift: **-4.290 percentage points**
- 95% CI: **[-6.220, -2.359] percentage points**
- P(lift > 0): effectively **0%**

## First-pass score-model season stability

Market-anchored score model minus market:

- 2019: **0**
- 2020: **-1**
- 2021: **0**
- 2022: **+2**
- 2023: **+1**
- 2024: **-1**
- 2025: **0**

There is no sign of a large hidden winner-selection edge.

## Audit of the nine score-model market disagreements

The market-anchored score distribution flipped only nine games on the final holdout. It won five and lost four. Several were exact or near 50/50 market games. The net contribution was therefore **+1**, not a broad systematic improvement.

## Boundary-extension sensitivity

Because the first pass selected multiple grid edges, Track G was rerun with a wider pre-holdout search:

- process noise `q`: 0.05, 0.10, 0.25, 0.75, 1.5, 3.0
- score variance `r`: 64, 100, 144, 196
- offseason persistence: 0.30, 0.45, 0.55, 0.70, 0.82, 0.90
- HFA: 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
- Ridge alpha: 1,000 through 1,000,000

The selected extended configuration was:

- `q`: **0.75**
- `r`: **196**
- offseason persistence: **0.45**
- HFA: **3.0**
- Ridge alpha: **10,000**
- latent probability scale: **0.70**

The wider search therefore moved the latent-state parameters, but importantly **did not ask for more score-residual freedom**. The same 10,000 Ridge penalty remained preferred despite allowing much weaker and much stronger shrinkage.

### Extended sensitivity holdout

| Model | Correct | Accuracy | Net vs market |
| --- | ---: | ---: | ---: |
| Market-anchored score distribution | 1,239/1,865 | 66.43% | **+1** |
| Market | 1,238/1,865 | 66.38% | 0 |
| Latent score distribution | 1,167/1,865 | 62.57% | **-71** |

The anchored score result was **identical** to the first pass.

Extended anchored-score bootstrap:

- lift: **+0.054 percentage points**
- 95% CI: **[-0.268, +0.375]**
- P(lift > 0): **56.3%**

The extended latent model improved slightly from -80 to -71 correct versus market, but remained decisively inferior.

## Test inside the actual frozen decision architecture

The more important question was whether the score model could add information to the existing system even if it did not beat the market by itself.

The exact current frozen 2019-2025 system was rebuilt from scratch. The reconstruction matched the archived integrated decision logic with **0 reconstruction differences** and reproduced:

- market: **1,238/1,865 = 66.38%**
- current frozen system: **1,254/1,865 = 67.24%**, **+16 versus market**

Four integration rules were specified **before** inspecting Track G's holdout outcomes and all four are reported:

1. `score_close_replace`: Track G controls the <52.5% close zone; current upset consensus remains.
2. `close_blend_50_50`: average the current residual and Track G score probabilities in the close zone.
3. `close_agreement_gate`: change a close-game market pick only when the current residual and Track G agree.
4. `triple_upset_confirmation`: retain the current close model but require Track G also to support the dog before an existing consensus upset is taken.

### Integration results

| Strategy | Correct | Accuracy | Net vs market | Net vs current |
| --- | ---: | ---: | ---: | ---: |
| **Current frozen** | **1,254** | **67.24%** | **+16** | 0 |
| Close agreement gate | 1,249 | 66.97% | +11 | **-5** |
| Close 50/50 blend | 1,247 | 66.86% | +9 | **-7** |
| Score close replacement | 1,247 | 66.86% | +9 | **-7** |
| Triple upset confirmation | 1,246 | 66.81% | +8 | **-8** |
| Market | 1,238 | 66.38% | 0 | -16 |

The boundary-extension sensitivity produced **the same integration table**.

### Paired comparison against current frozen system

- score close replacement: **-0.375 pp**, 95% CI **[-0.804, +0.054]**, P(improvement) **3.0%**
- close 50/50 blend: **-0.375 pp**, 95% CI **[-0.804, +0.054]**, P(improvement) **3.1%**
- close agreement gate: **-0.268 pp**, 95% CI **[-0.643, +0.107]**, P(improvement) **6.1%**
- triple upset confirmation: **-0.429 pp**, 95% CI **[-1.019, +0.161]**, P(improvement) **5.9%**

### What changed

For the close-zone replacement/blend, Track G differed from the current system on **17 games**. The current system won those disagreements **12-5**, explaining the **-7** net result.

The agreement gate changed 13 decisions and lost **9-4** to the current system, or **-5**.

The triple-confirmation rule changed 30 decisions and lost **19-11**, or **-8**.

A particularly revealing diagnostic is that Track G confirmed **0 of the 30 existing consensus upset calls**. In other words, the market-anchored score model was so conservative that requiring it as a third upset vote would have deleted every validated consensus upset on the holdout.

## Interpretation

Track G provides several useful lessons:

1. **Dynamic offense/defense scoring strength is not enough to beat the NFL closing market.** The pure latent score model was substantially worse.
2. **The market's spread and total already summarize scoring expectations extremely well.** Once Track G was anchored to those expectations, strong regularization pushed it to make only tiny corrections.
3. **A +1 winner count is not meaningful evidence.** Proper probability scores were slightly worse, the bootstrap interval was centered near zero, and the model changed only nine market picks.
4. **The existing close-game residual contains more useful pick'em information.** Replacing or blending it with Track G lost seven correct picks.
5. **The existing upset specialists capture information that a conservative score model does not.** Requiring Track G confirmation removed all 30 consensus upset calls and cost eight correct picks.
6. **The negative result survived a materially wider pre-holdout parameter search.** The anchored score result and integration outcomes were unchanged.

This does not mean score-distribution models are useless generally. They may be valuable for score, spread, total, or prop forecasting. This experiment specifically shows that this dynamic-state/market-anchored implementation does not improve **straight-up winner selection** beyond the current system.

## Production decision

**Do not promote Track G into the official 2026 pick system.**

Keep `prospective-v1-frozen-2025`, its <52.5% close residual, and its 52.5%-<80% two-specialist upset consensus unchanged.

No Track G file modifies `operational_pickem.py`, production thresholds, learned production weights, live pick logic, grader behavior, dashboard decision logic, or production workflows.
