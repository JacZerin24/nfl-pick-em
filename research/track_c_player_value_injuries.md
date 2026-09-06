# Track C: Player-Value Injury Research

## Status

Complete historical research track. **No change to `prospective-v1-frozen-2025`.**

This track tested whether injury information could add straight-up winner value beyond the historical closing market, with special emphasis on identifying which players mattered and how difficult they were to replace.

## Data and protocol

- Historical injury and snap-count window: **2012-2024**.
- Walk-forward out-of-sample testing begins in **2016**.
- Diagnostic holdout: **2019-2024**.
- Historical injury data uses the final weekly injury/practice report.
- Every snap-role, core-player, unit-depth, and replacement-capacity feature uses only games strictly before the target week.
- Historical betting information is the closing market, so this is a demanding test after the market had time to incorporate most injury information.
- Public nflverse historical injury data currently ends after 2024.
- Prior-role matching succeeded for **89.9%** of injury rows.

## Primary hypothesis: player value and replacement difficulty

The primary pre-specified model was a strongly regularized market-anchored ridge model using richer injury context:

- prior snap role
- empirically inferred core-unit status
- position group
- multiple simultaneous core-player losses
- prior-only backup capacity
- estimated uncovered replacement gap
- OUT / DOUBTFUL / QUESTIONABLE and practice-status context

### 2019-2024 holdout: all games

- Games: **1,594**
- Market: **1,061/1,594 (66.56%)**
- Player-value ridge: **1,050/1,594 (65.87%)**
- Net correct versus market: **-11**
- Disagreements: **137**
- Player-value model / market wins on disagreements: **63 / 74**
- Paired accuracy lift: **-0.690 percentage points**
- Bootstrap 95% CI: **[-2.133, +0.753] percentage points**
- P(lift > 0): **16.3%**

### 2019-2024 holdout: material-injury games

A material-injury game is one in which at least one team has a sufficiently large prior-role/core-player loss or estimated replacement gap under the pre-specified Track C definition.

- Games: **1,327**
- Market accuracy: **66.54%**
- Player-value ridge accuracy: **65.79%**
- Net correct versus market: **-10**
- Disagreements: **118**
- Player-value model / market wins on disagreements: **54 / 64**
- Paired accuracy lift: **-0.754 percentage points**
- Bootstrap 95% CI: **[-2.336, +0.829] percentage points**
- P(lift > 0): **16.6%**

### Primary conclusion

The richer player-value / replacement-gap hypothesis **did not beat the closing market** and should not be promoted into the official pick system. More detailed football-aware injury features were not automatically more predictive and may add noise or overfit information the market already prices effectively.

This does **not** imply injuries are unimportant. It means the final weekly injury information available in this historical dataset did not provide a reliable straight-up edge after the closing market had incorporated it.

## Secondary candidate: broad market-anchored injury ridge

Track C also included a pre-specified comparison model using the closing market anchor plus simpler injury information:

- counts of OUT / DOUBTFUL / QUESTIONABLE players
- DNP / LIMITED practice counts
- prior-role-weighted injury burden

This simpler model was unexpectedly more encouraging than the richer player-value model.

### 2019-2024 holdout

- Games: **1,594**
- Market: **1,061/1,594 (66.56%)**
- Broad injury ridge: **1,075/1,594 (67.44%)**
- Net correct versus market: **+14**
- Disagreements: **58**
- Broad model / market wins on disagreements: **36 / 22**
- Paired lift: **+0.878 percentage points**
- 50,000-sample bootstrap 95% CI: **[-0.063, +1.819] percentage points**
- P(lift > 0): **96.3%**

### Material-injury subset

- Games: **1,327**
- Net correct versus market: **+14**
- Disagreements: **48**
- Broad model / market wins on disagreements: **31 / 17**
- Paired lift: **+1.055 percentage points**
- Bootstrap 95% CI: **[+0.075, +2.110] percentage points**
- P(lift > 0): **97.6%**

### Stability check

Leaving any single 2019-2024 holdout season out still leaves the broad model between **+9 and +17 correct picks versus market** over the remaining seasons.

### Why this is not being promoted

The broad injury result is an **encouraging secondary candidate, not a validated production edge**:

1. The primary Track C hypothesis was the richer player-value model, which failed.
2. Multiple injury formulations have now been examined in this project, increasing the risk of selection effects.
3. The all-game bootstrap interval still slightly crosses zero.
4. The historical feed represents final weekly injury reports rather than timestamped late-breaking inactive information.
5. The public historical injury feed ends after 2024, so a current-season data source would be required for prospective use.
6. The frozen 2026 model is already serving as a prospective benchmark and should not be opportunistically retuned.

The broad injury ridge should therefore remain a **research/context signal**. A stronger future test would pair timestamped injury/inactive changes with the Track B line-movement archive to determine whether there is an exploitable interval before the market fully adjusts.

## Production decision

**No Track C result changes the official 2026 picks, thresholds, model weights, grader, or dashboard decision logic.**

The next planned research track is **weather interactions**, followed by **travel/body-clock effects** and then **player-level matchup enhancements**.
