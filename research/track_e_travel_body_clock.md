# Track E: Travel / Body-Clock Research

## Status

Complete historical research track. **No change to `prospective-v1-frozen-2025`.**

This track tested whether travel burden, time-zone changes, road sequencing, short rest, and kickoff timing relative to a team's home body clock add straight-up winner value beyond the historical closing market.

## Protocol

- Historical window: **2009-2025**.
- Development OOS seasons: **2016-2018**.
- Diagnostic holdout: **2019-2025**.
- The closing market is the probability anchor.
- Travel models are strongly regularized logistic offsets rather than standalone winner models.
- Development seasons choose the regularization strength before the 2019-2025 holdout is scored.
- Inputs are pregame-safe schedule/context variables:
  - approximate stadium-to-stadium travel distance
  - time zones crossed
  - eastward versus westward travel
  - kickoff hour on each team's home body clock
  - early body-clock deficit
  - short-rest burden
  - prior road-game streak
  - neutral-site and international-game flags
  - small, pre-specified interactions such as eastward travel x early kickoff and long travel x short rest

## First-pass result

The original penalty grid was **10, 30, 100, 300, 1000**. Both the main-effects and interaction models selected the strongest available penalty, **1000**.

On the 2019-2025 holdout:

- Games: **1,865**.
- Market: **1,238/1,865 (66.38%)**.
- Travel main effects: **1,247/1,865 (66.86%)**, **+9** correct versus market.
- Travel/body-clock interactions: **1,247/1,865 (66.86%)**, **+9** versus market.
- Interactions disagreed with the market on **19** games and won those disagreements **14-5**.
- Paired accuracy lift: approximately **+0.483 percentage points**.
- Bootstrap 95% CI: approximately **[-0.107, +1.126] percentage points**.
- P(lift > 0): approximately **94.1%**.
- Leave-one-holdout-season-out net advantage remained between **+7 and +11** correct picks.

### First-pass condition subsets

The apparent edge was not concentrated strongly in one obvious travel regime:

- early road-team body-clock kickoff: **+2** versus market across 157 games
- 2+ time zones crossed: **+2** across 238 games
- 2+ zones eastward: **+1** across 114 games
- 2+ zones westward: **+1** across 124 games
- 1,500+ mile road trip: **+2** across 242 games
- short rest plus 1,000+ mile travel: **0** across 24 games
- consecutive road-game context: **+6** across 798 games
- international games: **+1** across 69 games

The +9 headline was interesting, but selecting the largest value in the original penalty grid was a methodological warning: development was asking the model to stay even closer to the market than the search allowed.

## Extended shrinkage sensitivity

Before accepting the +9 result, the regularization grid was extended to **300, 1000, 3000, 10000, 30000, 100000** while preserving the same 2016-2018 development / 2019-2025 holdout split.

Both the main-effects and interaction models again selected the **largest available penalty, 100000**. Development log loss continued to improve as the model was shrunk more aggressively toward the market.

Under that stricter specification:

- Holdout net advantage fell from **+9 to +4 correct picks**.
- The model disagreed with the market on only **6 of 1,865 games**.
- It won those disagreements **5-1**.
- Paired accuracy lift was only about **+0.214 percentage points**.
- Bootstrap 95% CI was approximately **[-0.161, +0.643] percentage points**.
- P(lift > 0) was approximately **78.7%**.
- Leave-one-holdout-season-out net advantage was only **+2 to +4** correct picks.

The repeated boundary selection is important. The development data is effectively saying that the safest travel model is one that makes only extremely tiny adjustments to the closing market.

## Audit of the six surviving disagreements

The six remaining flips under the 100000-penalty model were:

| Game | Market home probability | Tiny travel-adjusted direction | Winner |
| --- | ---: | --- | --- |
| 2022 Week 7 CHI @ NE | exactly 50.0% | away CHI | CHI |
| 2022 Week 15 NYG @ WAS | exactly 50.0% | away NYG | NYG |
| 2022 Week 16 LV @ PIT | exactly 50.0% | away LV | PIT |
| 2023 Week 14 TEN @ MIA | exactly 50.0% | away TEN | TEN |
| 2024 Week 12 TB @ NYG | exactly 50.0% | away TB | TB |
| 2025 Week 17 ARI @ CIN | exactly 50.0% | away ARI | ARI |

All six had **exactly 0.500000 market home probability**. The travel model moved them only trivially below 0.500, by roughly **0.01 to 0.12 percentage points**, which changed the deterministic market pick from the home team to the road team. Road teams happened to go **5-1** in those six games.

This is not convincing evidence of an independent travel/body-clock edge. It is primarily an alternative tie-break on exact 50/50 market games.

That distinction matters because the existing frozen system already has a separately validated **close-residual layer** for market tossups. Historical project research previously found that the arbitrary market-home tie-break on exact 50/50 games was weak, while the close-residual model handled those games better. Track E therefore does not establish a distinct reason to add another travel-based flip rule on top of the close-game system.

## Interpretation

Travel and circadian context may still matter physically, and the first-pass +9 result was worth investigating. However, the stricter audit changes the conclusion:

1. Both development searches selected the strongest available shrinkage value.
2. Extending the grid reduced the apparent edge from **+9 to +4**.
3. Only **6** holdout picks survived as disagreements under extreme shrinkage.
4. Every one of those six games was an **exact 50/50 market game**.
5. The adjustments were effectively infinitesimal tie-breaks, not meaningful travel probability shifts.
6. The project already contains a validated close-residual system for the same tossup regime.
7. The bootstrap interval still crosses zero and P(lift > 0) weakened to about **78.7%** under the stricter specification.

## Production decision

**Do not promote Track E into the official 2026 pick system.**

Travel/body-clock variables can remain useful as a **context layer** or future research descriptor, especially for unusual international, short-rest, long-haul, or extreme body-clock situations. But this study did not demonstrate independent straight-up pick value beyond the closing market and the existing close-game architecture.

No Track E result changes official 2026 picks, thresholds, model weights, grader, dashboard decision logic, or live workflow behavior.

The next planned research track is **player-level matchup enhancements**.
