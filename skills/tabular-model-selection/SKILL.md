---
name: tabular-model-selection
description: Compare models, build blends and ensembles, and tune hyperparameters for tabular regression or classification. Use this skill whenever the user asks which model to use, wants to try gradient boosting/XGBoost/LightGBM/random forest, asks about ensembling, stacking, or blending, wants to tune hyperparameters, asks whether a neural network or deep learning would help on their tabular data, or asks how to improve a model score once a baseline exists. Also use when comparing model families or deciding whether more model complexity is worth it.
---

# Tabular Model Selection

Choosing, combining, and tuning models — in the order that actually pays off.

## Order of operations

1. Trivial baseline (mean/median/mode) to validate the pipeline
2. Simple model baseline (Ridge/Lasso, or logistic regression)
3. Model family comparison on identical folds
4. Blending across diverse families
5. Hyperparameter tuning

Tuning is last deliberately. It typically yields less than feature engineering
or blending, and it's the easiest place to overfit validation noise.

## Comparing model families

Run every candidate through **identical fold indices** — same splitter, same
seed. Comparing models that saw different random splits measures split variation
as much as model quality.

Candidates worth trying on most tabular problems:

- **Regularized linear** (Ridge, Lasso, ElasticNet) — strong on small datasets,
  especially after good feature engineering; often underrated
- **Gradient boosting** (LightGBM, XGBoost, HistGradientBoosting) — usually
  strongest on medium/large tabular data
- **Random forest** — rarely wins but adds diversity
- **CatBoost** — worth trying when high-cardinality categoricals dominate; pass
  categorical column indices rather than one-hot columns, or you have disabled
  the mechanism that makes it worth running
- **A tabular foundation model** — on datasets under roughly 10k rows this is a
  real candidate rather than an exotic one; see `tabular-foundation-models`

**Don't assume complexity wins.** On small datasets (a few thousand rows),
untuned gradient boosting frequently loses to a regularized linear model. In one
real project with ~1,450 rows, Ridge beat LightGBM, XGBoost, GradientBoosting,
and RandomForest individually. Report what the comparison actually shows rather
than what the model hierarchy suggests it should.

## Neural networks and sequence models

**Networks trained from scratch** rarely beat gradient boosting or regularized
linear models on tabular data under roughly 10k rows — there isn't enough data
to learn representations that beat well-engineered features. They can still add
diversity to a blend even when individually weaker.

**Pre-trained tabular foundation models are the exception**, and the distinction
matters. These are transformers pre-trained on large collections of tabular
tasks that do in-context learning over the training rows rather than fitting
from scratch, so the small-data argument against neural networks does not apply
to them — small data is precisely their target range. TabPFN's authors published
results in Nature under "Accurate predictions on small data with a tabular
foundation model". Treat them as serious candidates on datasets in the low
thousands of rows, and measure them with the same multi-seed protocol as
everything else rather than assuming either that they win or that they lose.

The family is now larger than TabPFN alone — TabPFN v1 through v3, TabICL v1 and
v2, TabDPT, Google's TabFM, Mitra, LimiX — and they differ in row cap, feature
cap, task support, and weight licensing. **Use the `tabular-foundation-models`
skill** for selection, licensing, and evaluation; `scripts/tabpfn_model.py` here
remains the TabPFN-specific path and blend integration.

Four things to check before recommending any of them:

- **Capacity**: each has a row cap (1,000 to 1,000,000) and sometimes a feature
  cap, and the two do not move together. Exceeding a cap subsamples silently
  rather than erroring.
- **Gated weights**: first use may need license acceptance via browser or a
  `TABPFN_TOKEN`, and some checkpoints also gate on HuggingFace.
- **Licensing**: several checkpoints — TabFM v1.0.0, TabPFN v2.5 and v3 — are
  released under NON-COMMERCIAL licenses on the weights, independently of the
  source license. TabPFN v2's weights use Apache 2.0 with an attribution
  requirement and are the commercially usable option in that line. Fine for
  competitions and research; check terms before shipping anything commercial.
- **Different preprocessing**: their docs say do NOT one-hot encode or scale,
  and they handle nulls natively. They need their own data path — cleaning and
  engineered features yes, encoding and imputation no. Feeding one the same
  matrix as a linear model is using it wrong. Feature engineering does still
  help.

**Time series models** (ARIMA, Prophet, sequence RNNs) require that rows be
ordered observations of the same entity over time AND that the evaluation split
be chronological. A dataset of independent records with a random train/test split
is not a forecasting problem, even when it contains date columns. Using date
fields as ordinary features (age, elapsed time, market-condition indicators) is
feature engineering, not time series modeling — and it's usually the right move.

## Blending

Averaging predictions from diverse models is one of the most reliable gains
available. Two separate questions, with very different confidence levels:

**Does blending beat the best solo model?** Usually yes, robustly. Verify across
seeds — a real blending gain holds on every seed and is typically several times
the across-seed noise.

**Which exact weighting is best?** Often not a real decision. Differences among
reasonable weightings are frequently smaller than across-seed variation, meaning
picking "the best" one is fitting noise. In one project the top four weightings
spanned 0.00027 while across-seed std was 0.0005 — and a single-seed run picked
a weighting that multi-seed testing then overturned.

Practical approach:
- Average in the same space the metric uses (log space for log-scale metrics),
  converting back once at the end
- Prefer a weighting that wins on most seeds over the one with the best mean
- Weight stronger models higher; equal weighting is often measurably worse
- Use `tabular-validation`'s `oof_cache.py` to test many weightings cheaply

**Stacking** (a meta-model learning the weights from out-of-fold predictions)
can beat fixed weights, but the meta-model must be trained on out-of-fold
predictions only. Training it on in-fold predictions leaks badly.

## Hyperparameter tuning

Do this after features and blending are settled.

- **Search strategy**: random search over sensible ranges beats grid search per
  unit of compute; Optuna/Bayesian search beats both when the budget is large
- **Guard against noise**: a tuning run that evaluates hundreds of configurations
  on one CV split will find configurations that fit that split's noise. The bias
  *grows* with trial count — a bigger search buys a more confident wrong answer.
- **Reserve seeds before starting.** Split them into search seeds and held-out
  seeds up front. Reserving afterwards does not work: once a seed has helped
  select the winner, re-scoring on it measures how well the winner fits that
  seed, which is what selection already optimized.
- **Report both scores.** The gap between the search score and the held-out
  score is the search's optimism, and a large gap means the budget was too big
  for the data.
- **Watch the effect size**: if tuning moves the *held-out* score less than the
  across-seed standard deviation, it hasn't found anything. That is a common and
  legitimate outcome — log it and keep the defaults.

`scripts/tune_search.py` implements this order; prefer it over a fresh harness.

```bash
python scripts/tune_search.py --data train.csv --target SalePrice \
    --model lightgbm --trials 100 --log-target
```

A flat top of the results table — the top ten configurations within noise of
each other — means the parameter choice is not a real decision. The script says
so; take the simplest configuration and stop.

Parameters that usually matter most:
- Linear: regularization strength (`alpha`, `C`)
- Tree ensembles: learning rate + n_estimators together, max_depth/num_leaves,
  min_child_samples, subsample, colsample_bytree, regularization terms

For small datasets, favor stronger regularization and shallower trees than
defaults — defaults are generally tuned for larger data.

## Reporting results

Show the comparison table with mean and standard deviation, and state plainly
when differences fall inside the noise. "XGBoost scored 0.1155 vs Ridge 0.1147,
a difference smaller than the across-seed variation of 0.0008 — these are
effectively tied" is more useful than presenting a ranked list that implies
precision the data doesn't support.

Include fit and prediction time in the table. Score is one axis; a model that
wins inside the noise floor and predicts forty times slower is a different
recommendation than the ranking alone suggests, and the user is the one who
should weigh that.

When comparing across several datasets rather than one, rank by **average rank**
rather than average raw score. Raw scores aren't commensurable across datasets —
whichever dataset has the widest score range dominates the mean, and the
conclusion becomes an artifact of that scale.

## Scripts

- `scripts/compare_models.py` — runs several model families on identical folds
  across seeds and prints a comparison with noise-aware verdicts
- `scripts/blend_search.py` — evaluates blend weightings using cached OOF
  predictions and reports which weightings are meaningfully different
- `scripts/tabpfn_model.py` — TabPFN evaluation and OOF generation, with its
  own preprocessing path and an environment/licensing preflight check
- `scripts/tune_search.py` — random search with seeds reserved from the search,
  then re-validation of the winner on those held-out seeds

```bash
# regression (defaults to rmse)
python scripts/compare_models.py --data train.csv --target SalePrice --log-target

# classification (defaults to auc, stratified folds)
python scripts/compare_models.py --data train.csv --target churned --task classification
```

Both scripts take a `metric` argument covering `rmse`/`mae` for regression and
`auc`/`logloss`/`accuracy`/`f1` for classification. For classification blending,
use a probability metric (`auc` or `logloss`) — the scripts then blend predicted
probabilities rather than hard labels, which is the only version that makes
sense.
