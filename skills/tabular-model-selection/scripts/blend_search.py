"""
Search blend weightings using cached out-of-fold predictions.

Refitting every model for every candidate weighting is wasteful and slow.
Instead compute OOF predictions once per (model, seed), then evaluate any
number of weightings as weighted sums of cached arrays.

The important output is not "which weighting won" but whether the weightings
differ by more than the across-seed noise. Frequently they don't, in which case
the choice among them is arbitrary and shouldn't be agonized over.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import sys, os
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "tabular-validation", "scripts"))
from multiseed_cv import (  # noqa: E402
    METRICS, LOWER_IS_BETTER, predict_for_metric, is_classification,
)

DEFAULT_SEEDS = [42, 7, 2024, 99, 555]


def compute_oof(X, y, model_fns, seed=42, n_splits=5, metric="rmse"):
    """OOF predictions per model at one seed, all sharing identical folds.

    For classification, use a probability metric (auc, logloss) so predictions
    are probabilities. Blending hard 0/1 labels is close to meaningless;
    averaging probabilities is the correct way to blend classifiers.
    """
    if is_classification(metric):
        sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {name: np.zeros(len(X), dtype=float) for name in model_fns}
    for tr, va in sp.split(X, y):
        for name, fn in model_fns.items():
            m = fn()
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[name][va] = predict_for_metric(m, X.iloc[va], metric)
    return oof


def generate_weightings(model_names, step=0.1, max_models=4):
    """All weightings on a simplex grid summing to 1.

    With many models this explodes; keep the step coarse. Fine-grained weight
    search is usually fitting noise anyway.
    """
    if len(model_names) > max_models:
        raise ValueError(f"too many models for grid search ({len(model_names)})")
    n_steps = int(round(1 / step))
    out = {}
    for combo in itertools.product(range(n_steps + 1), repeat=len(model_names)):
        if sum(combo) != n_steps:
            continue
        weights = {n: c * step for n, c in zip(model_names, combo)}
        label = "/".join(f"{n[:4]}{w:.1f}" for n, w in weights.items() if w)
        out[label] = weights
    return out


def search(X, y, model_fns, weightings=None, seeds=None, n_splits=5,
           top_n=10, metric="rmse", verbose=True):
    seeds = list(seeds or DEFAULT_SEEDS)
    weightings = weightings or generate_weightings(list(model_fns), step=0.2)
    score_fn = METRICS[metric]
    lower_better = LOWER_IS_BETTER[metric]

    rows = {}
    for seed in seeds:
        oof = compute_oof(X, y, model_fns, seed=seed, n_splits=n_splits,
                          metric=metric)
        col = {f"[solo] {n}": score_fn(y, oof[n]) for n in model_fns}
        for label, w in weightings.items():
            blended = sum(wt * oof[n] for n, wt in w.items() if wt)
            col[label] = score_fn(y, blended)
        rows[f"seed{seed}"] = col
        if verbose:
            print(f"seed {seed} done ({len(weightings)} weightings evaluated free)")

    df = pd.DataFrame(rows).T
    means = df.mean().sort_values(ascending=lower_better)
    noise = float(df.std().mean())

    if verbose:
        print(f"\n=== top {top_n} by mean {metric} across seeds ===")
        print(means.head(top_n).round(5).to_string())
        print(f"\nacross-seed noise floor: {noise:.5f}")

        blend_cols = [c for c in df.columns if not c.startswith("[solo]")]
        solo_cols = [c for c in df.columns if c.startswith("[solo]")]

        if blend_cols and solo_cols:
            if lower_better:
                best_blend = df[blend_cols].mean().min()
                best_solo = df[solo_cols].mean().min()
                gap = best_solo - best_blend
            else:
                best_blend = df[blend_cols].mean().max()
                best_solo = df[solo_cols].mean().max()
                gap = best_blend - best_solo
            print(f"\nbest blend beats best solo by {gap:.5f}"
                  f" ({gap/noise:.1f}x noise)" if noise else "")
            if gap > noise:
                print("  -> blending is a real gain here")
            else:
                print("  -> blending gain is within noise; may not be worth the complexity")

            top = means[[c for c in means.index if c in blend_cols]].head(top_n)
            spread = float(top.max() - top.min())
            print(f"\nspread among top {len(top)} weightings: {spread:.5f}")
            if spread < noise:
                print("  -> these weightings are effectively tied. Pick one that "
                      "wins on most seeds; don't tune weights further.")

            winners = (df[blend_cols].idxmin(axis=1) if lower_better
                       else df[blend_cols].idxmax(axis=1))
            print("\nwinning weighting per seed:")
            print(winners.value_counts().to_string())

    return df
