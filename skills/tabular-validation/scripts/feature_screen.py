"""
Screen candidate features one at a time with multi-seed CV.

Test features INDIVIDUALLY before combining. A bundle of three features that
helps on average may contain one strong feature and two harmful ones — only
individual testing separates them. In one real project, a three-ratio bundle
scored -0.00051; testing individually showed one ratio contributed -0.00038,
another -0.00016, and the third was actively useless. Dropping the third
improved the result.

Usage:

    from feature_screen import screen_features

    def add_area_ratio(df):
        df = df.copy()
        df["area_per_room"] = df["area"] / df["rooms"].replace(0, np.nan)
        df["area_per_room"] = df["area_per_room"].fillna(df["area_per_room"].median())
        return df

    screen_features(
        train_df=df,
        target="SalePrice",
        candidates={"area_ratio": add_area_ratio},
        prepare_fn=my_pipeline,   # optional: standard cleaning applied first
        log_target=True,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from multiseed_cv import (
    DEFAULT_SEEDS, multiseed_score, compare_to_baseline, LOWER_IS_BETTER,
)


def _default_prepare(df):
    return df


def _to_matrix(df, target, log_target, drop_cols):
    y = df[target]
    if log_target:
        y = np.log1p(y)
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns],
                errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))
    return X.reset_index(drop=True), y.reset_index(drop=True)


def screen_features(train_df, target, candidates, model_fn=None,
                    prepare_fn=None, log_target=False, seeds=None,
                    metric="rmse", drop_cols=("Id", "id", "ID"),
                    n_splits=5, verbose=True):
    """Score each candidate feature function against a no-candidate baseline.

    candidates: {name: fn(df) -> df}  — each adds column(s) and returns the df
    prepare_fn: standard cleaning/feature pipeline applied BEFORE the candidate
    model_fn:   returns a fresh model; defaults to Ridge (fast screening model)

    Returns a DataFrame sorted best-first, and prints an adopt/reject table.

    Screening with a single fast model is deliberate — it keeps the loop quick
    enough to test many ideas. Re-confirm the winners with the full
    model/ensemble afterwards, since a feature that helps a linear model does
    not always help a tree ensemble.
    """
    seeds = list(seeds or DEFAULT_SEEDS)
    prepare_fn = prepare_fn or _default_prepare
    if model_fn is None:
        from sklearn.linear_model import Ridge
        model_fn = lambda: Ridge(alpha=10.0)

    base_df = prepare_fn(train_df)
    X_base, y_base = _to_matrix(base_df, target, log_target, drop_cols)
    baseline = multiseed_score(X_base, y_base, model_fn, seeds=seeds,
                               n_splits=n_splits, metric=metric)

    if verbose:
        print(f"BASELINE {metric}: {baseline['mean']:.5f}")
        print(f"  per-seed: {[round(v,5) for v in baseline['per_seed'].values()]}")
        print(f"  NOISE FLOOR (across-seed std): {baseline['std']:.5f}\n")

    rows = []
    for name, fn in candidates.items():
        try:
            cand_df = fn(prepare_fn(train_df))
            X, y = _to_matrix(cand_df, target, log_target, drop_cols)
        except Exception as exc:
            if verbose:
                print(f"{name:28s} FAILED: {exc}")
            continue

        result = multiseed_score(X, y, model_fn, seeds=seeds,
                                 n_splits=n_splits, metric=metric)
        cmp = compare_to_baseline(baseline, result, metric=metric)

        rows.append({
            "feature": name,
            "mean": result["mean"],
            "delta": cmp["delta"],
            "consistency": cmp["consistency"],
            "exceeds_noise": cmp["delta_exceeds_noise"],
            "verdict": cmp["verdict"],
            "n_cols": X.shape[1],
        })
        if verbose:
            flag = "" if cmp["delta_exceeds_noise"] else "  (< noise)"
            print(f"{name:28s} {result['mean']:.5f}  delta={cmp['delta']:+.5f}  "
                  f"wins={cmp['consistency']}  {cmp['verdict']}{flag}")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        "mean", ascending=LOWER_IS_BETTER[metric]
    ).reset_index(drop=True)

    if verbose:
        adopt = out[out["verdict"] == "ADOPT"]
        print("\n=== ADOPT ===")
        print(adopt.to_string(index=False) if len(adopt) else "(none cleared the bar)")
        print("\n=== REJECT ===")
        rej = out[out["verdict"] != "ADOPT"]
        print(rej.to_string(index=False) if len(rej) else "(none)")
        print("\nRecord the rejects in the experiment log too — otherwise they "
              "get retried later.")

    return out


def decompose_bundle(train_df, target, bundle_parts, **kwargs):
    """Test each part of a multi-feature bundle separately.

    bundle_parts: {name: fn} for each individual piece.
    Run this whenever a bundle is adopted — some parts usually aren't pulling
    their weight, and removing them tends to improve the result further.
    """
    return screen_features(train_df, target, bundle_parts, **kwargs)
