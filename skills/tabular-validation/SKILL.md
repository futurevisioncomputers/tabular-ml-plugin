---
name: tabular-validation
description: Set up trustworthy cross-validation for tabular ML regression or classification, and determine whether a change is a real improvement or random noise. Use this skill whenever the user asks if a feature/model/parameter change actually helped, reports that CV and leaderboard scores disagree, wants to test new features, sees a small score improvement and wants to know if it's meaningful, or is setting up cross-validation. Also use when the user is about to adopt a change based on a single validation run, or asks about data leakage, fold-safe preprocessing, or experiment tracking.
---

# Tabular Validation

Tools and discipline for knowing whether a change actually improved a model.

## Why single-run cross-validation misleads

A 5-fold CV score is one sample from a distribution. Change the random seed that
determines the fold assignment and the score moves — often by more than the
effect you're trying to measure.

Concretely, on a 1,458-row dataset, across-seed standard deviation was ~0.0008
while typical candidate features produced effects of ~0.0005. Under those
conditions a single CV run has close to coin-flip odds of ranking a useless
feature above the baseline. Two changes that improved single-seed CV in one real
project — collapsing rare categories, and log-transforming skewed features —
both turned out to be noise or actively harmful when tested properly.

**Establish the noise floor first.** Run the baseline across several seeds and
compute the standard deviation. Every subsequent result gets compared against
that number. Without it there's no way to tell a real effect from a fluctuation.

## The adoption bar

A change is adopted when it **improves on most seeds** (4 of 5, or 5 of 5), not
when it improves the mean. Consistency across seeds is much harder to achieve by
chance than a favorable mean, and it's the stronger evidence.

Report three things for any candidate:
- mean score across seeds
- delta versus baseline mean
- how many seeds improved (e.g. "5/5")

A small delta with 5/5 consistency beats a larger delta with 3/5. Say so
explicitly when the delta is smaller than the noise floor — it may still be real
if consistency is perfect, but the user should know the magnitude is marginal.

## Fold-safe preprocessing

Anything that learns from data must be fit inside the fold:

| Transform | Leaks if fit on full data? |
|---|---|
| Mean/median/mode imputation | Yes |
| Scaling / standardization | Yes |
| Target/mean encoding | Yes — worst offender |
| Feature selection by target correlation | Yes |
| One-hot encoding (categories only) | Mild — align columns instead |
| Fixed rules (log transform, ratios, sums) | No — safe anywhere |

Target encoding deserves special care: computing category means from the whole
training set before splitting puts the validation rows' own targets into their
features. This produces dramatically optimistic CV that collapses on real data.
Compute the encoding from the training fold only, and apply it to the validation
fold.

For imputation that references another column's grouping (e.g. filling a numeric
column by group median), pass the training fold as an explicit reference so the
group statistics come from training rows only.

## Choosing the splitter and metric

- **Regression, independent rows** → `KFold(shuffle=True)`, metric `rmse` / `mae`
- **Classification** → `StratifiedKFold`, metric `auc` / `logloss` / `accuracy` / `f1`
- **Grouped data** (multiple rows per customer/patient/store) → `GroupKFold`,
  or the same entity appears in both train and validation and the score is
  inflated
- **Genuinely temporal data with a chronological split** → `TimeSeriesSplit`

Match the CV structure to how the real held-out data was separated. If the
competition or production split is chronological, a random KFold will overstate
performance.

**Probability metrics need probabilities.** `auc` and `logloss` must be scored
against `predict_proba` output, not hard labels — scoring 0/1 predictions with
AUC silently produces a wrong, plausible-looking number. This matters even more
for blending: averaging hard labels from several classifiers is close to
meaningless, while averaging predicted probabilities is the standard and correct
approach. The scripts here handle this via `predict_for_metric`; if you write a
custom loop, handle it yourself.

**Check the metric direction.** Lower is better for `rmse`, `mae`, and
`logloss`; higher is better for `accuracy`, `auc`, and `f1`. Sorting or
win-counting with the wrong direction inverts every conclusion while still
producing output that looks fine.

## Scripts

### `scripts/multiseed_cv.py`

Core harness. Scores a model (or a feature variant) across several CV seeds and
reports mean, per-seed values, standard deviation, and win counts against a
baseline.

```bash
# regression
python scripts/multiseed_cv.py --data train.csv --target SalePrice \
    --log-target --seeds 42 7 2024 99 555

# classification (stratified folds are selected automatically)
python scripts/multiseed_cv.py --data train.csv --target churned --metric auc
```

Supported metrics: `rmse`, `mae` (regression); `auc`, `logloss`, `accuracy`,
`f1` (classification). The splitter defaults to stratified for classification
metrics and KFold otherwise.

Import it for custom experiments:

```python
from multiseed_cv import multiseed_score, compare_to_baseline
```

### `scripts/feature_screen.py`

Tests candidate features one at a time against a baseline across seeds, then
prints an adopt/reject table. Register each candidate as a function that takes a
DataFrame and returns a DataFrame with the new column(s), then run the screen.
Test features individually before combining — a bundle that helps on average may
contain one strong feature and two harmful ones, and only individual testing
reveals that.

### `scripts/calibration_report.py`

Brier score, ECE, a reliability table, and out-of-fold threshold selection for
binary classifiers.

```bash
python scripts/calibration_report.py --data train.csv --target churned --objective f1
```

Use it whenever the predicted probability is used **as a probability** — compared
to a threshold, multiplied by a value, or shown to a person as a percentage. When
the output only ever ranks, calibration does not matter and this can be skipped.

### `scripts/oof_cache.py`

Computes out-of-fold predictions once per model per seed and caches them, so any
number of blend weightings or ensemble combinations can be evaluated instantly
without refitting. Use this whenever comparing more than two or three ensemble
configurations — refitting for each is needlessly slow.

## Calibration and decision thresholds

AUC measures ranking only. A model can rank perfectly and still be
systematically overconfident — predicting 0.9 on cases that occur 60% of the
time — and no ranking metric shows it.

**Where the threshold came from is a leakage question.** A threshold swept
against the data the score is then reported on is fitted to that data. Accuracy
and F1 come out optimistic, often substantially so on imbalanced problems. Tune
it on out-of-fold or validation predictions, then apply the fixed number to the
test set once.

Which metrics that affects is the useful diagnostic:

| Moves with the threshold | Independent of it |
|---|---|
| accuracy, F1, precision, recall | AUC, log loss, Brier, ECE |

So strong AUC alongside accuracy that collapses on new data is the signature of
a threshold fitted to the evaluation set — the first column broke and the second
did not.

Two more things worth checking:

- **0.5 chosen or inherited?** On a 5% positive rate, 0.5 often predicts the
  majority class nearly everywhere and produces high accuracy that means
  nothing. On one synthetic 17%-positive dataset, moving from 0.5 to an
  out-of-fold-tuned 0.234 raised F1 from 0.392 to 0.544 while accuracy *fell*
  from 0.848 to 0.799 — which of those is the improvement depends entirely on
  what the model is for.
- **Resampling shifts the probability scale by construction.** SMOTE,
  undersampling and `class_weight="balanced"` all move predicted probabilities
  off the true base rate. The fix is recalibration on natural-rate data, not
  removing the rebalancing.

Any calibrator — Platt for small validation sets and roughly sigmoid distortion,
isotonic when there is more data — must be fit on data the model did not train
on, and evaluated on data neither the model nor the calibrator saw. A calibrator
fitted on in-fold predictions leaks exactly the way target encoding does.

**Check whether the threshold is a real decision at all.** If the objective is
flat across a wide band of thresholds, picking one to four decimal places is
fitting noise. `calibration_report.py` reports the near-optimal band width.

## Experiment logging

Keep a CSV with one row per experiment: date, id, description, model, feature
notes, CV score, holdout/leaderboard score, output file, and conclusion.

Two habits make it worth maintaining:

**Record negative results explicitly.** "Collapsing rare categories: −0.0002
CV but +0.0002 on holdout, 0/5 seeds, rejected" prevents someone retrying it in
three weeks. Most experiments fail; a log of only successes is a log that lies.

**Write the conclusion, not just the number.** The score alone doesn't say
whether the change was adopted or why.

When writing the log programmatically, build rows as lists and use `csv.writer`
rather than string-joining — description fields contain commas and will corrupt
the file otherwise. Assert every row has the same field count as the header
before writing.
