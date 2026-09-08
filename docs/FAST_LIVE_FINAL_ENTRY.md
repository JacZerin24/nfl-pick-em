# Fast Live Runner and Final-Entry Infrastructure

## Purpose

This infrastructure change makes the frozen 2026 NFL straight-up Pick'em system fast enough to create an actionable recommendation close to kickoff without changing the forecasting model itself.

The production model remains `prospective-v1-frozen-2025`. The frozen decision thresholds and specialist logic are unchanged.

## What changed

### Versioned frozen artifacts

The learned 2009-2025 model components are exported once under:

`frozen_artifacts/prospective-v1-frozen-2025/`

The artifact manifest records the model version, training window, package versions, frozen 52.5% close-game threshold, frozen 80% favorite threshold, and SHA-256 hashes for every artifact file.

The live runner verifies the manifest and every artifact hash before making predictions.

### Fast operational runner

`scripts/fast_operational_pickem.py` loads the frozen models and frozen historical state tails instead of refitting the full 2009-2025 system on every live run.

Only current schedule/market inputs and available current-season state are refreshed. The official pick logic is copied from the frozen production rules and was parity-tested against the legacy operational runner.

### Final-entry controller

`.github/workflows/final-entry-2026.yml` provides a kickoff-aware submission layer.

The controller runs twice per hour as a lightweight scheduling layer. When a kickoff cluster is within the controller window it waits inside the job until approximately T-12 minutes, runs the fast model, and writes a `FINAL_ENTRY` snapshot.

If the primary attempt cannot be verified, the workflow retries around T-5 and writes a `FALLBACK_ENTRY` snapshot.

Routine updates remain useful for monitoring, but once an official entry snapshot exists it is treated as the actionable source of truth for that game.

### Grader and dashboard semantics

The grader prefers `FINAL_ENTRY` or `FALLBACK_ENTRY` when one exists. Historical games without those roles retain the original newest-pre-kickoff fallback behavior.

The Pages payload and dashboard recognize official-entry status so the user can distinguish an updating recommendation from the final submission recommendation.

Track B line-movement collection also runs after successful final-entry workflows so the near-kick market state becomes part of the prospective movement panel.

## Validation

Validation was performed in GitHub Actions run `34031409072` on the Week 1 2026 slate at the same fixed as-of timestamp for both runners.

- Games compared: 16
- Market-pick mismatches: 0
- Market-underdog mismatches: 0
- Official-pick mismatches: 0
- Decision-type mismatches: 0
- Maximum probability difference: floating-point noise only, about `5.55e-16`
- Required tolerance: `1e-7`
- Parity result: PASS

The legacy prediction path took roughly 38 seconds in the validation job. The frozen-artifact fast path took roughly 2.4 seconds.

A branch smoke run of the actual `2026 Live NFL Pick'em` workflow also completed successfully using the fast runner.

The `2026 Final Entry Pick'em` workflow was smoke-tested when no game was inside the 75-minute controller window. It correctly returned `no_game_in_controller_window` and created no false official-entry snapshot.

## Guardrails

This change does **not** retune or replace the forecasting model.

It does not change:

- `prospective-v1-frozen-2025`
- the 52.5% close-game threshold
- the 80% favorite threshold
- the residual close-game selector rule
- the two-specialist true-upset consensus rule
- any learned coefficient or CatBoost tree

Future research remains separate from production until it earns promotion through the project's existing time-ordered out-of-sample validation standards.
