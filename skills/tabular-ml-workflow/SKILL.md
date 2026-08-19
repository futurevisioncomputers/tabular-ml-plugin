---
name: tabular-ml-workflow
description: End-to-end workflow for supervised machine learning on tabular data — Kaggle competitions, business prediction tasks, or any CSV-with-a-target-column problem. Use this skill whenever the user wants to build a predictive model from tabular/spreadsheet data, mentions a Kaggle competition, asks about a train/test CSV setup, wants to improve a model score, or asks "where do I start" with a dataset. Also use when the user needs to decide what to do next in an ML project, or asks whether a change actually improved their model. Routes to tabular-data-profiling, tabular-validation, tabular-model-selection, and tabular-foundation-models for each stage.
---

# Tabular ML Workflow

The orchestrator for supervised learning on tabular data. This skill covers the
overall sequence and the discipline that makes results trustworthy; four
companion skills handle the stages in depth.

## The core problem this workflow solves

Most tabular ML effort is wasted on changes that look like improvements but
aren't. A single cross-validation run has enough random variation that a
genuinely useless change will appear to help roughly half the time. Teams then
adopt it, build on top of it, and never find out.

Everything below is organized around one principle: **a change is only real if
it survives multiple random validation splits.** Getting this right matters more
than model choice, more than feature engineering, more than hyperparameters.

## Sequence

Work in this order. Later stages depend on decisions made earlier.

1. **Understand the data and the metric** — what does each column mean, what is
   being scored, what does missing mean here? → `tabular-data-profiling`
2. **Establish validation before modeling** — set up the CV harness and the
   experiment log first, so every later change can be measured.
   → `tabular-validation`
3. **Submit or score a trivial baseline immediately** — median/mode prediction,
   or a plain linear model. This validates the whole pipeline end-to-end
   (data → preprocess → fit → predict → output format) before any cleverness.
   Bugs found here are cheap; bugs found after feature engineering are not.
4. **Clean, using documented per-column rules** → `tabular-data-profiling`
5. **Compare models on identical folds** → `tabular-model-selection`. On
   datasets under roughly 10k rows, include a tabular foundation model in the
   comparison → `tabular-foundation-models`
6. **Engineer features, testing each one** → `tabular-validation` (the multi-seed
   harness is the tool for this)
7. **Blend or ensemble** → `tabular-model-selection`
8. **For classification, check calibration and the decision threshold** →
   `tabular-validation`. Skip it when the output only ever ranks.
9. **Tune hyperparameters last** — it usually yields less than features or
   blending, and tuning on noise is easy. Reserve seeds from the search before
   starting → `tabular-model-selection`

## Non-negotiables

**Fit preprocessing inside the fold, never before the split.** Any transform
that learns from data — imputers, encoders, scalers, target encoding, feature
selection — must be fit on the training fold only and applied to the validation
fold. Fitting on the full dataset before splitting leaks validation information
into training and produces optimistic scores that collapse on real held-out data.

**Match the transform to the metric.** If the metric is RMSE on log(y), train on
log(y) and evaluate on log(y). Convert predictions back only at the final output
step. Getting this backwards silently optimizes the wrong objective.

**Log every experiment.** One row per attempt: what changed, the CV score, the
held-out or leaderboard score, and the conclusion. Without this you will re-test
the same idea and won't notice when a "win" fails to replicate. Negative results
are as valuable as positive ones — record them explicitly so they don't get
retried.

**Trust validation over intuition, and over a single run.** When CV and a public
leaderboard disagree, prefer CV — a public leaderboard is often scored on a small
subset and is noisier than it looks.

## Interpreting a validation/holdout gap

A persistent gap between CV score and leaderboard/holdout score is common and
does not automatically mean something is broken. Diagnose before chasing:

- **Gap is stable across very different models** → likely structural: different
  data distribution, or a small scoring subset. Not a pipeline bug.
- **Gap appeared after a specific change** → suspect that change for leakage.
- **Gap size is comparable to across-seed variation** → it's noise, not signal.
- **CV improves but holdout consistently worsens** → real overfitting to CV;
  simplify, or increase regularization.

Compute across-seed standard deviation early (see `tabular-validation`) so you
have a noise floor to compare gaps against. Without that number, every gap looks
meaningful.

## Deciding what to work on next

Rough expected payoff, largest first, for a typical tabular problem:

1. **Correct validation setup** — prevents wasted effort on everything else
2. **Feature engineering** — usually the biggest genuine modeling gain
3. **Blending several diverse models** — reliable, modest, low-risk gain
4. **Better cleaning / encoding decisions** — matters most for linear models
5. **Hyperparameter tuning** — real but usually smallest; do it last

If the user asks "should I try a neural network / time series model / deep
learning", check the shape of the problem before agreeing. Sequence models
(ARIMA, LSTM for forecasting) only apply when rows are ordered observations of
the same entity over time AND the split is chronological — a random train/test
split over independent records is not a forecasting problem. Neural networks
**trained from scratch** rarely beat gradient boosting or regularized linear
models on small tabular datasets (under ~10k rows), though they can add useful
diversity to a blend.

**Pre-trained tabular foundation models are the exception** and the argument
above does not apply to them: under ~10k rows is their target range rather than
their weakness. On a small dataset they belong in step 5 rather than being
treated as exotic. They carry constraints no other model here has — a context
row cap, a separate license on the weights (several are non-commercial), and
their own preprocessing path. → `tabular-foundation-models`

## Honesty about ceilings

When a user wants a specific score or rank, say plainly what is achievable.
If a public leaderboard shows implausible perfect scores, that usually means the
target values are obtainable from a public source rather than that those
competitors modeled better — explain this rather than trying to match it.
Set expectations from the noise floor: if across-seed std is 0.0008, a proposed
change that might yield 0.0002 is not worth days of work.

## Reference

`scripts/project_scaffold.py` creates a standard project layout (data/,
notebooks/, src/, submissions/, experiments/) with an experiment log ready to
use. Run it at the start of a new project:

```bash
python scripts/project_scaffold.py --name my-project --target SalePrice --metric rmse_log
```
