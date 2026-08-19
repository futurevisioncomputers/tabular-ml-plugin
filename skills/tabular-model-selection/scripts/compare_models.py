"""
Compare model families on identical folds across multiple seeds.

Four design choices matter here:

1. Every model sees the SAME fold indices — otherwise the comparison partly
   measures split variation rather than model quality.
2. Results are reported against the across-seed noise floor, so differences too
   small to be real are labelled as ties instead of ranked.
3. Fit and predict time are measured alongside score. A ranking on score alone
   recommends models nobody would actually choose — a win inside the noise
   floor that costs forty times more to predict is not obviously a win, and the
   user can only weigh that with both numbers in front of them.
4. Models are also ranked by AVERAGE RANK across seeds, not only by mean score.
   Mean score is dominated by whichever seed happened to produce the widest
   spread; average rank is not. Where the two orderings disagree, the mean is
   being driven by one seed and the disagreement is itself the finding.
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
import sys, os
from sklearn.model_selection import KFold, StratifiedKFold

# reuse the metric registry so regression and classification behave identically
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "tabular-validation", "scripts"))
from multiseed_cv import (  # noqa: E402
    METRICS, LOWER_IS_BETTER, predict_for_metric, is_classification,
)

DEFAULT_SEEDS = [42, 7, 2024, 99, 555]


def build_model_zoo(task="regression", small_data=True):
    """Sensible starting configurations.

    Tree params are deliberately conservative (shallow, regularized) — defaults
    are tuned for larger datasets and overfit small ones.
    """
    zoo = {}
    if task == "regression":
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        zoo["Ridge"] = lambda: Ridge(alpha=20.0 if small_data else 1.0)
        zoo["Lasso"] = lambda: Lasso(alpha=0.0005, max_iter=5000)
        zoo["RandomForest"] = lambda: RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1)
        zoo["GradientBoosting"] = lambda: GradientBoostingRegressor(random_state=42)
        try:
            from lightgbm import LGBMRegressor
            zoo["LightGBM"] = lambda: LGBMRegressor(
                n_estimators=2000, learning_rate=0.01, num_leaves=8, max_depth=4,
                min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=-1)
        except ImportError:
            pass
        try:
            from xgboost import XGBRegressor
            zoo["XGBoost"] = lambda: XGBRegressor(
                n_estimators=2000, learning_rate=0.01, max_depth=3,
                min_child_weight=3, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbosity=0)
        except ImportError:
            pass
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        zoo["LogisticRegression"] = lambda: LogisticRegression(max_iter=2000)
        zoo["RandomForest"] = lambda: RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1)
        try:
            from lightgbm import LGBMClassifier
            zoo["LightGBM"] = lambda: LGBMClassifier(
                n_estimators=1000, learning_rate=0.02, num_leaves=15,
                random_state=42, verbosity=-1)
        except ImportError:
            pass
    return zoo


def compare(X, y, model_fns, seeds=None, n_splits=5, metric="rmse", verbose=True):
    """Score every model on identical folds across seeds.

    Returns {"scores": per-seed score frame, "summary": per-model summary frame,
    "noise": average across-seed std}. The summary carries mean score,
    across-seed std, average rank, and mean fit/predict time per fold.
    """
    seeds = list(seeds or DEFAULT_SEEDS)
    score_fn = METRICS[metric]
    lower_better = LOWER_IS_BETTER[metric]
    rows = {}
    fit_times = {name: [] for name in model_fns}
    predict_times = {name: [] for name in model_fns}

    for seed in seeds:
        if is_classification(metric):
            sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # cache fold indices so every model sees identical splits
        folds = list(sp.split(X, y))
        col = {}
        for name, fn in model_fns.items():
            scores = []
            for tr, va in folds:
                m = fn()
                # perf_counter, not time(): wall-clock resolution on Windows is
                # too coarse to measure a fast model on a small fold.
                t0 = time.perf_counter()
                m.fit(X.iloc[tr], y.iloc[tr])
                t1 = time.perf_counter()
                preds = predict_for_metric(m, X.iloc[va], metric)
                t2 = time.perf_counter()
                fit_times[name].append(t1 - t0)
                predict_times[name].append(t2 - t1)
                scores.append(score_fn(y.iloc[va], preds))
            col[name] = float(np.mean(scores))
        rows[f"seed{seed}"] = col
        if verbose:
            print(f"seed {seed} done")

    df = pd.DataFrame(rows).T
    # sort so the BEST model is first, whichever direction the metric runs
    means = df.mean().sort_values(ascending=lower_better)
    stds = df.std()
    noise = float(stds.mean())

    # Rank within each seed, then average. `ascending=lower_better` makes rank 1
    # the best model regardless of which way the metric runs — getting this
    # backwards silently inverts the whole table.
    per_seed_ranks = df.rank(axis=1, ascending=lower_better)
    avg_rank = per_seed_ranks.mean()

    summary = pd.DataFrame({
        f"mean_{metric}": means,
        "std": stds.reindex(means.index),
        "avg_rank": avg_rank.reindex(means.index),
        "fit_s": pd.Series({n: float(np.mean(v)) for n, v in fit_times.items()}
                           ).reindex(means.index),
        "predict_s": pd.Series({n: float(np.mean(v)) for n, v in predict_times.items()}
                               ).reindex(means.index),
    })

    if verbose:
        print("\n=== per-seed scores ===")
        print(df.round(5).to_string())
        print(f"\n=== summary (best mean {metric} first) ===")
        print(summary.round(5).to_string())
        print(f"\naverage across-seed std (noise floor): {noise:.5f}")
        print("fit_s / predict_s are seconds per fold, averaged over all "
              "folds and seeds.")

        best_name = means.index[0]
        best = means.iloc[0]
        rank_best = avg_rank.idxmin()
        if rank_best != best_name:
            print(f"\nNOTE: best mean is {best_name} but best average rank is "
                  f"{rank_best}. The mean is being driven by one seed — treat "
                  f"the ordering as unresolved rather than picking either.")

        print(f"\n=== verdict vs best ({best_name}) ===")
        fastest_predict = summary["predict_s"].idxmin()
        for name, score in means.items():
            if name == best_name:
                continue
            gap = (score - best) if lower_better else (best - score)
            tag = "tied (within noise)" if gap < noise else f"worse by {gap:.5f}"
            # Cost only earns a mention where the model is otherwise a
            # contender; a slower model that also loses on score is just worse.
            cost = ""
            if gap < noise:
                ratio = (summary.loc[best_name, "predict_s"]
                         / max(summary.loc[name, "predict_s"], 1e-9))
                if ratio > 2:
                    cost = f" — and predicts {ratio:.0f}x faster than {best_name}"
            print(f"  {name:20s} {tag}{cost}")

        print("\nModels labelled 'tied' should be treated as equally good on "
              "score -- prefer the simpler or faster one, or use both in a "
              "blend for diversity.")
        print(f"fastest to predict: {fastest_predict} "
              f"({summary.loc[fastest_predict, 'predict_s']:.4f}s per fold)")

    return {"scores": df, "summary": summary, "noise": noise}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--drop", nargs="*", default=["Id", "id", "ID"])
    p.add_argument("--log-target", action="store_true")
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--metric", default=None, choices=list(METRICS),
                   help="defaults to rmse for regression, auc for classification")
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    args = p.parse_args()

    metric = args.metric or ("auc" if args.task == "classification" else "rmse")
    if (args.task == "classification") != is_classification(metric):
        p.error(f"--metric {metric} does not match --task {args.task}")
    if args.log_target and args.task == "classification":
        p.error("--log-target is a regression option")

    df = pd.read_csv(args.data)
    y = df[args.target]
    if args.log_target:
        y = np.log1p(y)
    X = df.drop(columns=[args.target] + [c for c in args.drop if c in df.columns],
                errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))

    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    print(f"matrix: {X.shape[0]} rows x {X.shape[1]} features\n")

    zoo = build_model_zoo(args.task, small_data=len(X) < 5000)
    print(f"task: {args.task}  metric: {metric}")
    print(f"models: {list(zoo)}\n")
    compare(X, y, zoo, seeds=args.seeds, metric=metric)


if __name__ == "__main__":
    main()
