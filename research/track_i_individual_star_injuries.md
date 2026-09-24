# Track I: Individualized Star-Player Injury Value

## Status

Complete historical research track. **Do not promote into `prospective-v1-frozen-2025`.**

Track I tested whether individualized player value can turn injury information into a reliable straight-up winner edge beyond the closing market. The design intentionally avoids a subjective "star" list: player impact is estimated only from information available before the target game.

## Primary design

Historical window: **2018-2024**.

Development seasons used for regularization only: **2020-2021**.

Untouched walk-forward holdout for the primary test: **2022-2024**.

Player value combines prior snap role with position-appropriate prior production:

- QB: passing EPA/dropback plus a small rushing component.
- Skill positions: receiving/rushing EPA per opportunity.
- Front seven: pressure, sack, and QB-hit production.
- Defensive backs: interception/pass-defense production with missed-tackle penalty.
- Offensive line / special teams / other: role and usage because equivalent individual production measures are not available in the same historical feed.

Injury context includes OUT, DOUBTFUL, QUESTIONABLE, severe OUT+DOUBTFUL burden, highest-impact injured player, top-two injured-player burden, and position-group-specific impact.

Models compared:

1. Closing market.
2. Broad injury burden.
3. Individualized star-player injury features.
4. Broad injury burden + individualized star-player features.

The 70% straight-up target is reported by season but is **not** used to tune the holdout.

## Primary holdout result: 2022-2024

| Model | Correct | Games | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Market | 555 | 813 | 68.27% | 0.607792 | 0.209894 |
| Broad injury | 553 | 813 | 68.02% | 0.607425 | 0.209846 |
| Individualized star | 539 | 813 | 66.30% | 0.618641 | 0.214835 |
| Broad + individualized star | 541 | 813 | 66.54% | 0.618141 | 0.214597 |

For the primary broad + individualized-star model versus market:

- Accuracy lift: **-1.722 percentage points**.
- Bootstrap 95% CI: **[-3.936, +0.492] percentage points**.
- P(lift > 0): **5.9%**.

The observed direction is unfavorable.

## 70% target by season

Primary broad + individualized-star model:

| Season | Correct | Games | Accuracy | 70% reached? |
| --- | ---: | ---: | ---: | --- |
| 2022 | 173 | 269 | 64.31% | No |
| 2023 | 174 | 272 | 63.97% | No |
| 2024 | 194 | 272 | 71.32% | Yes |

Closing market for reference:

| Season | Correct | Games | Accuracy | 70% reached? |
| --- | ---: | ---: | ---: | --- |
| 2022 | 176 | 269 | 65.43% | No |
| 2023 | 184 | 272 | 67.65% | No |
| 2024 | 195 | 272 | 71.69% | Yes |

To reach 70% on every game in those seasons, a system would have needed at least **189 correct in 2022** and **191 in 2023 and 2024**. Relative to the market, that means finding approximately **13 additional correct picks in 2022** and **7 in 2023** while not giving back the already-above-70% 2024 result.

## Stage 2: position-group diagnostics

Because the full individualized model failed, a post-primary exploratory analysis tested whether specific position groups were hiding useful signal. This reuses the already-touched 2022-2024 sample and therefore is diagnostic, not a fresh validation set.

| Variant | Correct | Accuracy |
| --- | ---: | ---: |
| Market | 555 | 68.27% |
| Broad + OL | 553 | 68.02% |
| Broad only | 553 | 68.02% |
| Broad + QB | 553 | 68.02% |
| Broad + doubtful | 552 | 67.90% |
| Broad + front | 551 | 67.77% |
| Broad + offense | 546 | 67.16% |
| Broad + severe stars | 545 | 67.04% |
| Broad + skill | 544 | 66.91% |
| Broad + questionable | 544 | 66.91% |
| Broad + OUT stars | 541 | 66.54% |
| Broad + DB | 541 | 66.54% |
| Broad + defense | 540 | 66.42% |

No position-specific individualized injury variant beat the closing market in straight-up accuracy.

## Descriptive absence signal

For high-impact OUT/DOUBTFUL player absences on the 2022-2024 holdout, the injured team's actual win rate minus its closing-market expected win rate averaged approximately:

- QB: **-3.31 percentage points** across 85 team-games.
- Skill: **-3.05 points** across 246.
- DB: **-2.95 points** across 401.
- OL: **+0.37 points** across 353.
- Front seven: **+1.14 points** across 270.

These are descriptive associations, not causal player-value estimates. Multiple injuries, team quality, opponent quality, and the market's own injury adjustment all confound these comparisons.

Individual-player samples are especially unstable. Repeated absences for recognizable players can show large positive or negative residuals, but the samples are typically only a handful of games and often contradict one another. They should not be turned into fixed "Player X is worth Y points" rules.

## Interpretation

The primary individualized star-player hypothesis failed. Adding more specific player-value information to final weekly injury reports made the straight-up model worse, not better.

This reinforces the conclusion from Tracks A, C, and F: much of the readily observable personnel information appears to be incorporated efficiently by the closing market, and increasing feature granularity can add noise faster than it adds usable straight-up signal.

The simpler broad injury model remains more defensible than the individualized version, but it also failed to beat the market on the 2022-2024 primary window and should not be promoted on this evidence.

## Most promising next injury question

The remaining injury hypothesis is **timing rather than static closing-line information**:

> When a high-value player's availability changes, is there a measurable interval before the market fully reprices the news?

A future prospective study should join timestamped injury/status/inactive events to the existing timestamped market archive and measure:

- pre-news market probability,
- 15/30/60/120-minute post-news probability,
- closing probability,
- player position and estimated prior-only value,
- replacement quality,
- whether the market continues moving after the first response,
- straight-up result.

This would test for a genuine information-timing edge rather than asking a model to beat a closing price that already contains most known injury information.

## Production decision

**No Track I result changes the frozen 2026 production model, thresholds, official picks, grader, or dashboard.**

The Track I code and workflows remain isolated on `research/individual-star-injuries`.
