"""
Multi-seed cross-validation harness.

The point: a single CV run's score is one draw from a distribution. Comparing
two options on one seed frequently ranks them wrong. This module scores across
several seeds and reports how consistently one option beats another.

Usable as a library or from the command line.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score,
    roc_auc_score, log_loss, f1_score,
)

DEFAULT_SEEDS = [42, 7, 2024, 99, 555]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def auc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))


def logloss(y_true, y_prob):
    return float(log_loss(y_true, y_prob))


def f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="binary"))


METRICS = {"rmse": rmse, "mae": mae, "accuracy": accuracy,
           "auc": auc, "logloss": logloss, "f1": f1}

# For these, a LOWER score is better. Used to decide win/loss direction and
# sort order — getting this wrong silently inverts every ranking.
LOWER_IS_BETTER = {"rmse": True, "mae": True, "accuracy": False,
                   "auc": False, "logloss": True, "f1": False}

# Metrics scored against predicted PROBABILITIES rather than hard labels.
# Using predict() for these would score 0/1 labels and give wrong answers —
# and for blending, averaging hard labels is meaningless where averaging
# probabilities is correct.
NEEDS_PROBA = {"auc", "logloss"}

CLASSIFICATION_METRICS = {"accuracy", "auc", "logloss", "f1"}


def is_classification(metric: str) -> bool:
    return metric in CLASSIFICATION_METRICS


def predict_for_metric(model, X, metric: str):
    """Get predictions in the form the metric expects.

    Probability metrics need predict_proba; label metrics need predict.
    Falls back to decision_function, then predict, when a model lacks
    predict_proba.
    """
    if metric in NEEDS_PROBA:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
    return model.predict(X)


# --------------------------------------------------------------------------
# splitters
# --------------------------------------------------------------------------

def make_splitter(kind: str, n_splits: int, seed: int):
    """Pick a splitter matching the data structure.

    Using the wrong one silently inflates scores — e.g. random KFold on grouped
    data puts the same entity in both train and validation.
    """
    if kind == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if kind == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if kind == "group":
        # GroupKFold is deterministic; seed has no effect. Multi-seed testing is
        # therefore less informative here — vary n_splits or resample instead.
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"unknown splitter: {kind}")


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------

def cv_score(X, y, model_fn, seed=42, n_splits=5, metric="rmse",
             splitter="kfold", groups=None):
    """One CV run at one seed. Returns the mean fold score.

    model_fn must return a FRESH unfitted model each call — reusing a fitted
    model across folds leaks information between them.
    """
    score_fn = METRICS[metric]
    sp = make_splitter(splitter, n_splits, seed)
    split_args = (X, y, groups) if splitter == "group" else (X, y)

    fold_scores = []
    for tr_idx, va_idx in sp.split(*split_args):
        model = model_fn()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = predict_for_metric(model, X.iloc[va_idx], metric)
        fold_scores.append(score_fn(y.iloc[va_idx], preds))
    return float(np.mean(fold_scores))


def multiseed_score(X, y, model_fn, seeds=None, n_splits=5, metric="rmse",
                    splitter="kfold", groups=None):
    """Score across several seeds.

    Returns {"per_seed": {seed: score}, "mean": float, "std": float}.
    The std is the noise floor — any effect smaller than it needs strong
    seed-consistency before it should be believed.
    """
    seeds = seeds or DEFAULT_SEEDS
    per_seed = {
        s: cv_score(X, y, model_fn, seed=s, n_splits=n_splits,
                    metric=metric, splitter=splitter, groups=groups)
        for s in seeds
    }
    values = list(per_seed.values())
    return {"per_seed": per_seed, "mean": float(np.mean(values)),
            "std": float(np.std(values))}


def compare_to_baseline(baseline_result, candidate_result, metric="rmse"):
    """Compare two multiseed_score outputs on the SAME seeds.

    'wins' — how many seeds the candidate beat the baseline on — is the number
    that matters most. A candidate that wins on every seed is credible even
    when the mean delta is small; a candidate that wins on 3 of 5 is a coin flip
    regardless of its mean.
    """
    lower_better = LOWER_IS_BETTER[metric]
    seeds = sorted(set(baseline_result["per_seed"]) & set(candidate_result["per_seed"]))
    if not seeds:
        raise ValueError("baseline and candidate share no seeds — compare like with like")

    wins = 0
    for s in seeds:
        b, c = baseline_result["per_seed"][s], candidate_result["per_seed"][s]
        if (c < b) if lower_better else (c > b):
            wins += 1

    delta = candidate_result["mean"] - baseline_result["mean"]
    improved = (delta < 0) if lower_better else (delta > 0)

    return {
        "delta": delta,
        "improved_mean": improved,
        "wins": wins,
        "n_seeds": len(seeds),
        "consistency": f"{wins}/{len(seeds)}",
        "noise_floor": baseline_result["std"],
        "delta_exceeds_noise": abs(delta) > baseline_result["std"],
        # The adoption rule: consistency across seeds, not mean improvement.
        "verdict": "ADOPT" if wins >= max(4, len(seeds) - 1) and improved else "REJECT",
    }


def format_comparison(name, comparison):
    flag = "" if comparison["delta_exceeds_noise"] else "  (delta < noise floor)"
    return (f"{name:28s} delta={comparison['delta']:+.5f}  "
            f"wins={comparison['consistency']}  "
            f"{comparison['verdict']}{flag}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_matrix(df, target, log_target, drop_cols):
    y = df[target]
    if log_target:
        y = np.log1p(y)
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns],
                errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))
    return X.reset_index(drop=True), y.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Multi-seed CV baseline scorer")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--drop", nargs="*", default=["Id", "id", "ID"])
    p.add_argument("--log-target", action="store_true",
                   help="train/score on log1p(target); use when the metric is on log scale")
    p.add_argument("--metric", default="rmse", choices=list(METRICS))
    p.add_argument("--splitter", default=None,
                   choices=["kfold", "stratified", "group"],
                   help="defaults to stratified for classification metrics, kfold otherwise")
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    clf = is_classification(args.metric)
    # Stratified splits keep class balance stable across folds; using plain
    # KFold on imbalanced data makes fold scores swing for the wrong reason.
    splitter = args.splitter or ("stratified" if clf else "kfold")

    if args.log_target and clf:
        p.error("--log-target is a regression option; it makes no sense with a "
                "classification metric")

    df = pd.read_csv(args.data)
    X, y = _build_matrix(df, args.target, args.log_target, args.drop)
    print(f"matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"task: {'classification' if clf else 'regression'}  "
          f"metric: {args.metric}  splitter: {splitter}")

    if clf:
        from sklearn.linear_model import LogisticRegression
        model_fn = lambda: LogisticRegression(max_iter=5000)
    else:
        from sklearn.linear_model import Ridge
        model_fn = lambda: Ridge(alpha=10.0)

    res = multiseed_score(X, y, model_fn, seeds=args.seeds,
                          n_splits=args.n_splits, metric=args.metric,
                          splitter=splitter)

    print(f"\nbaseline mean {args.metric}: {res['mean']:.5f}")
    print(f"per-seed: {[round(v, 5) for v in res['per_seed'].values()]}")
    print(f"across-seed std (NOISE FLOOR): {res['std']:.5f}")
    print(f"\nTreat any future change smaller than {res['std']:.5f} as suspect "
          f"unless it improves on nearly every seed.")


if __name__ == "__main__":
    main()
