# Track A: QB Starter-Change / Replacement-Value Findings

## Status

Complete historical ceiling test. Research-only; no change to the frozen 2026 production decision rule.

## Protocol

- Historical seasons loaded: 2009-2025.
- Time-ordered out-of-sample evaluation: 2016-2025.
- Current-game historical starter identity is treated as an oracle label because the schedule identifies the QB who actually started/played.
- Every QB performance feature uses only prior-game history.
- Closing market probability remains the benchmark.
- The focused model is trained and evaluated on games where at least one team changed starting QBs from its previous game.

## Results

- All OOS games: **2,629**.
- Games with at least one starter change: **468 (17.8%)**.
- All games, market: **1,753/2,629 (66.68%)**.
- All games, market-anchored change-feature model: **1,749/2,629 (66.53%)**, **-4** correct versus market.
- Starter-change games, market: **324/468 (69.23%)**.
- Starter-change games, change-only model: **314/468 (67.09%)**, **-10** correct versus market.
- Pick disagreements on starter-change games: **50**.
- On disagreements: model **20**, market **30**.
- Paired accuracy lift versus market: **-2.137 percentage points**.
- 30,000-sample bootstrap 95% CI: **[-5.128, +0.855] percentage points**.
- P(lift > 0): **6.8%**.

## Conclusion

Generic starter-change and replacement-value features do **not** justify a new official pick rule. Even with oracle knowledge of the eventual starter, the closing market performed better. This strengthens the hypothesis that any useful QB edge is more likely to come from **timing**: a surprise or late starter change before the market has fully adjusted.

That timing question should be studied prospectively by linking future QB/start-status events to the Track B timestamped line-movement archive. No Track A result should alter `prospective-v1-frozen-2025` during the 2026 prospective season.
