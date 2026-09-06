# Track B: 2026 Prospective Line-Movement Collection

## Status

Collector and analysis pipeline are operational. Context-only; no change to the frozen 2026 production decision rule.

## Source of truth

The immutable timestamped files under `live_archive/2026/week_*/snapshots/*/picks.csv` remain the raw source of truth. Track B derives reproducible analysis tables from those snapshots and never rewrites them.

## Derived outputs

For each distinct pre-kick game snapshot, the panel records:

- snapshot timestamp and minutes to kickoff
- home/away moneylines
- market home probability and favorite probability
- spread and total
- market favorite
- official model pick and decision type
- movement from the first archived observation
- favorite flips
- official model-pick flips
- decision-type changes

The per-game summary records the first and latest pre-kick observations, total snapshot count, maximum probability move, spread movement, favorite flips, model-pick flips, and final pre-kick lead time.

## Initial Week 1 status

The first validated run found:

- **16 games** represented
- **48 distinct pre-kick rows**
- **3 archived observations per game**
- **0 market-favorite flips**
- **0 official-model pick flips**
- **0.00 percentage-point market probability movement** across the initial smoke-test snapshots

That is expected because the first three archived observations were generated close together during system smoke testing. The dataset becomes meaningfully longitudinal when the Monday early-look and later game-day / near-kick snapshots begin accumulating.

## Prospective research guardrail

Movement features and any future evaluation rule should be specified before completed 2026 outcomes are used to judge them. Track B is initially a data-collection and context layer only. It must not silently alter `prospective-v1-frozen-2025`.

After the Track B workflow is on the default branch, it is configured to rebuild the movement panel automatically after a successful `2026 Live NFL Pick'em` workflow run.
