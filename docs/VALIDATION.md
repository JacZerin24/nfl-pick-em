# Validation Protocol

The model is optimized for one thing: **future straight-up NFL winner accuracy**. All model/feature decisions must therefore be made under future-like validation.

## 1. No random train/test split for model selection

NFL games are a time series. Random splitting allows future team/player information and era effects to contaminate earlier examples.

Primary validation is expanding walk-forward:

```text
train <= 2015 -> test 2016
train <= 2016 -> test 2017
train <= 2017 -> test 2018
...
train <= prior season -> test next season
```

Later, the strictest replay will update training/features week by week and generate each prediction from an archived `as_of` snapshot.

## 2. Primary benchmark

The system must beat or add measurable value beyond the simplest strong rule:

**pick the market favorite**.

Report these baselines every time:

- home team
- better record
- Elo favorite
- point-spread favorite
- de-vigged moneyline favorite

A 67% model is not impressive if the market favorite was 68% on the exact same games.

## 3. Metrics

### Primary

- straight-up accuracy

### Probability quality

- log loss
- Brier score
- calibration curve
- expected calibration error

### Decision diagnostics

- accuracy when model agrees with market
- accuracy when model disagrees with market
- number of market overrides
- accuracy of overrides
- average probability edge on overrides
- upset recall
- favorite false-override rate

### Stability

- per-season accuracy
- per-week accuracy
- home/away split
- divisional/non-divisional
- favorite-strength buckets
- early-season vs late-season
- QB-change vs QB-stable games
- indoor/outdoor and weather buckets

## 4. Model comparison tests

Because models predict the same games, use paired tests.

Planned diagnostics:

- paired bootstrap confidence intervals for accuracy difference
- McNemar test for winner-pick disagreements
- paired bootstrap for Brier/log-loss difference
- season-block bootstrap to reduce overconfidence from game-level dependence

Do not promote a challenger based only on a tiny aggregate improvement with no stability.

## 5. Hyperparameter tuning

Hyperparameters are tuned only inside historical training data.

For a test season `Y`:

1. outer test = season `Y`
2. training pool = seasons `< Y`
3. inner folds = earlier expanding-season folds inside training pool
4. choose hyperparameters using inner folds
5. refit on all data `< Y`
6. evaluate once on `Y`

The outer test season is never used for early stopping, feature selection, calibration fitting, threshold selection, or stack weights.

## 6. Stacking

The production stacker may use predictions from:

- market model
- Elo / dynamic rating model
- QB-adjusted rating model
- regularized logistic regression
- CatBoost
- LightGBM
- other validated challengers

The stacker must train on **out-of-fold historical predictions** from each base model.

Never train a meta-model on base predictions generated from rows that the base model already saw during fitting.

## 7. Calibration

Calibration choices are treated as model components and validated in time order.

Candidates:

- no calibration
- sigmoid/Platt calibration
- isotonic regression
- beta calibration

A calibration method must improve future Brier/log loss without reducing winner accuracy materially.

## 8. Market override policy

A likely high-accuracy production strategy is not "always trust our model". It may be:

1. use the market as a strong prior;
2. let football-derived models modify the probability;
3. override a market favorite only when validated disagreement is large/reliable enough.

Possible rule:

```text
if model_pick == market_pick:
    pick model/market consensus
else:
    override only when validated edge >= threshold
```

The threshold must be learned only from prior folds and tested on untouched future folds.

Also compare this rule with a fully probabilistic stacked model. Whichever is more accurate/stable wins.

## 9. Feature ablation

Every major feature family gets an ablation test:

- market
- Elo/team strength
- rolling EPA
- QB
- injuries
- depth/roster continuity
- travel/rest
- weather
- coaching
- advanced stats

Measure performance with and without each block across the same walk-forward games.

This prevents a large model from carrying dozens of impressive-sounding but useless variables.

## 10. Leakage audit

Before a feature enters production, answer:

- Was this value known before the pick deadline?
- Was it computed only from prior games/events?
- Does a rolling statistic explicitly shift before rolling?
- Is an injury status from the proper report timestamp?
- Is starting QB knowledge historically valid at this snapshot?
- Is weather a forecast or a later observed value?
- Is market information the line from that timestamp or the later closing line?
- Is any target-derived statistic hidden inside a source table?

If any answer is unclear, quarantine the feature until proven safe.

## 11. Model promotion gate

A challenger is promoted only if it demonstrates a repeatable improvement such as:

- higher pooled walk-forward accuracy;
- better Brier/log loss;
- improvement across multiple test seasons;
- better performance specifically on games where it changes the final pick;
- no material calibration degradation;
- no dependence on leaked/later information.

Small unstable gains are not enough.

## 12. Final live audit

Every weekly production run should archive:

- model version / git commit
- feature version
- raw data snapshot IDs
- prediction timestamp
- pick deadline assumed
- home/away win probabilities
- final pick
- market pick at that same timestamp
- confidence tier

After the game, append outcome and grade without altering the original prediction.

The season record should always be reproducible from immutable pregame prediction files.
