# Track J — Primetime, game-slot and scheduling research

## Status

Research-only exploratory challenger. No change to frozen official 2026
straight-up picks. This is an isolated test of a new hypothesis, NOT a
production adjustment that awards primetime teams extra points.

## Hypotheses

1. After controlling for pregame sportsbook strength, existing pregame
   Elo and rest, certain game windows may be systematically different
   (Thursday night, Sunday night, Monday night, Sunday morning,
   early/late Sunday, Saturday, Friday).
2. Some teams may display repeatable residual *performance relative to
   pregame market expectations* within particular kickoff windows.
3. Short rest may interact with Thursday night more than with other
   windows; experience and market residual effects may depend on the
   specific slot.

## Sources and interpretation

NFL schedule date/time, historical scores, home/away moneylines, Elo
from prior games, rest days, total and divisional status. For this first
screen the slot is inferred from weekday plus **Eastern clock time**,
NOT independently verified network assignment.

Labels THU_NIGHT_PROXY, SUN_NIGHT_PROXY, MON_NIGHT_PROXY are NOT
proof a game aired on Amazon/NBC/ESPN or was branded TNF/SNF/MNF.
Flex changes, Thanksgiving windows, multiple MNF games, special
broadcasts and time-zone adjustments need authoritative historical
sources in a subsequent study.

Past team-slot performance is computed as the team's cumulative,
prior-game wins-minus-market-expectations, shrunken by 20 pseudo-games,
less the team's similarly shrunken all-slot performance. This removes
some confounding due to selecting already strong teams for television.
It does not establish causal TV or psychology effects. Current-game
outcomes never contribute to current-game history.

Historical betting lines are closing/near-closing, not timestamped
early-week market quotes. Actual playing QBs and head coaches known
only retrospectively are not included as current QB/coach game-slot
features in this test. A future era-specific study must use verified
pregame QB/coaching identity at the relevant deadline.

## Chronological comparison

Expanding-season testing 2016–2025, with 2019–2025 as a **reused**
exploratory diagnostic. Fixed market-anchored penalized logistic
corrections (common penalty 0.1) compare:

- Market
- Control (Elo, rest, total, divisional)
- Control plus kickoff-slot indicators
- Control plus slots and prior team-slot residual/experience
- Control plus slots and rest × Thursday / primetime history interactions
- All of those combined

Evaluate identical games and report per-season and per-slot Brier,
log loss, straight-up correct picks, paired changed picks, exact
McNemar p-values and season-block bootstrap intervals.

Even a significant result on the reused 2019–2025 games is only
exploratory because they were repeatedly examined in other tracks.
A promising challenger must be tested against the **exact frozen
2026 incumbent**, with validated broadcasts and prospectively archived
odds/picks, before any new model is promoted.

## Next expansions if defensible

- Independently verify historic TNF/SNF/MNF television tags,
  flex changes and original vs revised kickoff slots.
- Opponent strength/selection-adjusted hierarchical franchise,
  current coach and projected QB-era partial pooling.
- Travel/rest/body-clock interactions, linked to previously negative
  Track E findings, with repeated-season robustness checks.
- Third-party network reach/audience only if measurable and reliable,
  and never assume larger audience produces superior play.
- Pregame injury timing and opponent-specific vulnerabilities when
  repeated across time slots (e.g., short-rest OL issues against pass rush).
- True prospective calibration and decision lift on non-overlapping
  future games, with all attempted hypotheses tracked.

## Running

    PYTHONPATH=scripts python -m pytest -q tests/test_track_j_primetime_slots.py
    python scripts/track_j_primetime_slots.py

Artifacts: outputs/track_j_primetime/ (predictions, per-slot metrics,
feature registry, paired comparisons and summary).

## Completed historical diagnostic — October 9, 2026

[Successful GitHub Actions research run](https://github.com/JacZerin24/nfl-pick-em/actions/runs/37884721885)
completed 4 unit tests and expanding-season backtests. The previously
examined 2019–2025 holdout contains **1,865 games**:

| Variant | Correct | Net vs market | Net vs control |
|---|---:|---:|---:|
| Market | 1,238 | 0 | -7 |
| Market + Elo/rest/total/division control | **1,245** | +7 | 0 |
| Control + kickoff-slot indicators | 1,238 | 0 | -7 |
| Control + team slot history | 1,236 | -2 | -9 |
| Control + slot/rest interactions | 1,240 | +2 | -5 |
| Control + all slot features | 1,238 | 0 | -7 |

Relative to the control model, the slot-only challenger won **6** and
lost **13** on changed historical decisions (exact paired McNemar
p = 0.167). The all-features challenger won **10** and lost **17**
(p = 0.248), also worse. These data do **not** justify adding
primetime- or slot-based pick weights to the 2026 production model.

The control's +7 advantage over the market itself is small
(McNemar p = 0.189) and not independent confirmation. Neither this
control nor any slot challenger is the frozen production system.
A true branded-TNF/SNF/MNF study requires validated broadcast tags,
original/flexed schedule provenance, and a separate, untouched
prospective comparison. Results remain research-only.
