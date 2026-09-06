# Track D: Weather Interaction Research

## Status

Complete historical research track. **No change to `prospective-v1-frozen-2025`.**

This track tested whether specific weather conditions interact with team style in a way that improves straight-up NFL winner selection beyond the historical closing market.

## Protocol

- Historical window: **2009-2025**.
- Walk-forward OOS development: **2016-2018**.
- Official diagnostic holdout: **2019-2025**.
- Model family: strongly regularized market-anchored logistic offset with fixed penalty.
- Generic weather comparison uses wind/cold/freezing/heat intensity and reported-weather context.
- Primary interaction candidate adds a small pre-specified set of football mechanisms using only prior team performance:
  - wind x passing dependence
  - wind x explosive passing
  - wind x sack exposure
  - wind x pass-defense profile
  - cold x pass-vs-run dependence
  - cold x rushing efficiency
  - freezing x turnover tendency
  - heat x offensive success rate
  - heat x defensive success allowed
- Historical nflverse temperature/wind are near-kick/final game context, not a timestamped forecast source. Any future live use would require a forecast snapshot available before the pick deadline.

## Holdout results: all games

- Games: **1,865**.
- Market: **1,238/1,865 (66.38%)**.
- Generic weather: **1,242/1,865 (66.60%)**, **+4** correct versus market.
- Weather interactions: **1,245/1,865 (66.76%)**, **+7** versus market and **+3** versus generic weather.
- Interaction paired lift versus market: **+0.375 percentage points**.
- 50,000-sample bootstrap 95% CI: **[-0.214, +0.965] percentage points**.
- P(lift > 0): **88.2%**.

The interaction model's probability metrics were slightly worse than the market despite the small accuracy gain:

- Market log loss: **0.608393**; Brier: **0.210547**.
- Weather-interaction log loss: **0.608711**; Brier: **0.210727**.

That is an important warning against treating the +7 correct-pick result as a robust probability-model improvement.

## Weather-exposed games

Outdoor/open games with weather reported:

- Games: **1,129**.
- Market: **750/1,129 (66.43%)**.
- Generic weather: **752/1,129 (66.61%)**, **+2** versus market.
- Weather interactions: **755/1,129 (66.87%)**, **+5** versus market and **+3** versus generic weather.
- Interaction paired lift versus market: **+0.443 percentage points**.
- Bootstrap 95% CI: **[-0.266, +1.240] percentage points**.
- P(lift > 0): **85.0%**.

## Condition subsets

### High wind: 15+ mph

- Games: **136**.
- Market: **93 correct (68.38%)**.
- Generic weather: **94 correct (69.12%)**.
- Weather interactions: **95 correct (69.85%)**.
- Net: **+2 versus market**, **+1 versus generic weather**.

### Cold: 40 F or colder

- Games: **199**.
- Market: **139 correct (69.85%)**.
- Weather interactions: **140 correct (70.35%)**.
- Net: **+1 versus market**.

### Freezing: 32 F or colder

- Games: **65**.
- All three approaches finished with **47 correct (72.31%)**.

### Hot: 80 F or warmer

- Games: **140**.
- All three approaches finished with **84 correct (60.00%)**.

The lack of improvement in the freezing and hot subsets argues against broad rules such as "cold favors X" or "heat favors Y."

## Disagreement and stability diagnostics

- Weather interactions disagreed with the market on only **31** holdout games.
- On those 31 games, interaction picks won **19-12** versus the market.
- They disagreed with generic weather on **17** games and won those disagreements **10-7**.
- Leave-one-holdout-season-out net advantage versus market remains positive, ranging from **+2 to +9** correct picks.

### By season

| Season | Market | Generic weather | Weather interactions | Interaction net vs market |
| --- | ---: | ---: | ---: | ---: |
| 2019 | 164 | 165 | 168 | +4 |
| 2020 | 173 | 173 | 173 | 0 |
| 2021 | 169 | 168 | 169 | 0 |
| 2022 | 176 | 180 | 181 | +5 |
| 2023 | 184 | 186 | 185 | +1 |
| 2024 | 195 | 194 | 194 | -1 |
| 2025 | 177 | 176 | 175 | -2 |

## Conclusion

Weather x team-style interactions are **more promising than generic weather alone**, but the evidence is not strong enough to alter official picks:

1. The total gain is only **+7 correct picks across 1,865 holdout games**.
2. The paired bootstrap interval crosses zero.
3. Probability calibration did not improve over the market.
4. High-wind and cold subsets show only small gains; freezing and hot subsets show none.
5. 2024 and 2025 were negative versus market.
6. Historical weather is not timestamped pre-kick forecast data.
7. The frozen 2026 system should not be opportunistically retuned.

Track D should remain a **context/research layer**. A stronger future test would archive timestamped forecast conditions alongside the existing Track B line-movement snapshots and evaluate whether late weather changes create a short-lived market inefficiency.

## Production decision

**No Track D result changes official 2026 picks, thresholds, model weights, grader, or dashboard decision logic.**

The next planned research track is **travel/body-clock effects**, followed by **player-level matchup enhancements**.
