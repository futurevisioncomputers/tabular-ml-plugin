"""
Cache out-of-fold predictions so many ensemble configurations can be evaluated
without refitting.

Naive approach: for each blend weighting, run full CV refitting every model.
Testing 5 weightings x 5 seeds x 3 models x 5 folds = 375 model fits, and it
times out.

Better: compute OOF predictions once per (model, seed) — 75 fits — then evaluate
any number of weightings instantly as weighted sums of cached arrays. Adding a
sixth weighting then costs nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from multiseed_cv import (
    METRICS, LOWER_IS_BETTER, DEFAULT_SEEDS,
    predict_for_metric, is_classification,
)
from sklearn.model_selection import StratifiedKFold


def compute_oof(X, y, model_fns, seed=42, n_splits=5, metric="rmse"):
    """Out-of-fold predictions for each model at one seed.

    Every model sees identical folds, so downstream comparisons reflect real
    differences rather than different random splits.

    For classification, predictions are collected in the form the metric needs.
    This matters for blending specifically: averaging hard 0/1 labels from
    several models is close to meaningless, while averaging predicted
    probabilities is the standard and correct way to blend classifiers. Pass a
    probability metric (auc, logloss) to blend properly.

    Returns {model_name: np.ndarray of OOF predictions}.
    """
    if is_classification(metric):
        # keep class balance stable across folds
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = {name: np.zeros(len(X), dtype=float) for name in model_fns}

    for tr_idx, va_idx in splitter.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        for name, fn in model_fns.items():
            model = fn()
            model.fit(X_tr, y_tr)
            oof[name][va_idx] = predict_for_metric(model, X_va, metric)
    return oof


def evaluate_weightings(oof, y, weightings, metric="rmse"):
    """Score blend weightings against cached OOF predictions.

    weightings: {label: {model_name: weight}} — weights should sum to 1.
    """
    score_fn = METRICS[metric]
    y_arr = np.asarray(y)
    out = {}
    for label, weights in weightings.items():
        blended = sum(w * oof[name] for name, w in weights.items() if w)
        out[label] = score_fn(y_arr, blended)
    return out


def multiseed_blend_study(X, y, model_fns, weightings, seeds=None,
                          n_splits=5, metric="rmse", verbose=True):
    """Evaluate solo models and blend weightings across several seeds.

    Two distinct questions get answered here, and they deserve different levels
    of confidence:

      1. Does blending beat the best solo model? — usually a large, robust
         effect that holds on every seed.
      2. Which exact weighting is best? — often a difference smaller than the
         across-seed noise, i.e. not a real decision. Prefer a weighting that
         wins on most seeds, and don't over-invest in tuning weights.
    """
    seeds = list(seeds or DEFAULT_SEEDS)
    rows = {}

    for seed in seeds:
        oof = compute_oof(X, y, model_fns, seed=seed, n_splits=n_splits,
                          metric=metric)
        col = {f"[solo] {n}": METRICS[metric](np.asarray(y), oof[n]) for n in model_fns}
        col.update(evaluate_weightings(oof, y, weightings, metric=metric))
        rows[f"seed{seed}"] = col
        if verbose:
            print(f"seed {seed} done")

    df = pd.DataFrame(rows).T
    ascending = LOWER_IS_BETTER[metric]

    if verbose:
        print(f"\n=== {metric} by seed ===")
        print(df.round(5).to_string())
        print("\n=== mean across seeds ===")
        print(df.mean().sort_values(ascending=ascending).round(5).to_string())
        print("\n=== across-seed std ===")
        print(df.std().round(5).to_string())

        blend_cols = [c for c in df.columns if not c.startswith("[solo]")]
        if blend_cols:
            winners = df[blend_cols].idxmin(axis=1) if ascending else df[blend_cols].idxmax(axis=1)
            print("\n=== winning weighting per seed ===")
            print(winners.value_counts().to_string())

            solo_cols = [c for c in df.columns if c.startswith("[solo]")]
            if solo_cols:
                best_blend = df[blend_cols].mean().min() if ascending else df[blend_cols].mean().max()
                best_solo = df[solo_cols].mean().min() if ascending else df[solo_cols].mean().max()
                gap = abs(best_solo - best_blend)
                noise = float(df[blend_cols].std().mean())
                print(f"\nbest blend vs best solo: {gap:.5f}  "
                      f"(~{gap/noise:.1f}x the across-seed noise)"
                      if noise else "")
                spread = abs(df[blend_cols].mean().max() - df[blend_cols].mean().min())
                print(f"spread among weightings: {spread:.5f}"
                      + ("  — smaller than noise, so the exact weights barely matter"
                         if spread < noise else ""))

    return df
