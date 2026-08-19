"""
Hyperparameter search with held-out-seed re-validation.

THE PROBLEM THIS SOLVES

A search evaluating 300 configurations against one cross-validation split will
find configurations that fit that split's particular fold assignment. The
winner's reported score is the maximum of 300 noisy draws, which is biased
upward by construction -- and the bias GROWS with the trial count. A bigger
search buys a more confident wrong answer.

THE FIX, AND WHY IT HAS TO BE THIS ORDER

Seeds are split before the search starts: some for searching, the rest held back
and never seen by it. The held-out seeds are the only unbiased estimate of the
winner's value.

Reserving them afterwards does not work and the failure is not obvious. Once a
seed has contributed to selecting the winner, re-scoring on it measures how well
the winner fits that seed -- which is what selection already optimized. That is
why `split_seeds` runs before `search` here rather than being a reporting step
at the end.

WHAT TO DO WITH THE GAP

Report the search score and the held-out score separately. The difference is the
search's optimism, and it is information in its own right: a large gap means the
budget was too big for the data. Compare the HELD-OUT improvement, never the
search improvement, against the noise floor.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "tabular-validation", "scripts"))
from multiseed_cv import (  # noqa: E402
    multiseed_score, compare_to_baseline, LOWER_IS_BETTER, METRICS,
    is_classification, DEFAULT_SEEDS,
)


# --------------------------------------------------------------------------
# seed discipline
# --------------------------------------------------------------------------

def split_seeds(seeds=None, n_holdout: int = 2):
    """Split seeds into (search, holdout). Call this BEFORE searching.

    Leaves at least two seeds on each side -- one held-out seed cannot
    distinguish a real gain from that seed's own luck, which is the failure the
    whole protocol exists to avoid.
    """
    seeds = list(seeds or DEFAULT_SEEDS)
    if len(seeds) < 4:
        raise ValueError(
            f"need at least 4 seeds to split search from holdout, got "
            f"{len(seeds)}. Fewer means either a search you cannot trust or a "
            "holdout that cannot detect anything.")
    n_holdout = max(2, min(n_holdout, len(seeds) - 2))
    return seeds[:-n_holdout], seeds[-n_holdout:]


# --------------------------------------------------------------------------
# search spaces
# --------------------------------------------------------------------------

def sample_space(space: dict, n_trials: int, rng):
    """Random search over a dict of {param: list-of-values}.

    Random beats grid per unit of compute: grid spends most of its budget
    varying parameters that do not matter, while random covers the ones that do
    at every trial.
    """
    keys = list(space)
    seen = set()
    out = []
    # Exhaust the space rather than resampling forever when it is smaller than
    # the requested budget.
    total = int(np.prod([len(space[k]) for k in keys])) if keys else 0
    if total <= n_trials:
        for combo in itertools.product(*(space[k] for k in keys)):
            out.append(dict(zip(keys, combo)))
        return out

    while len(out) < n_trials:
        cfg = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        sig = tuple(sorted(cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


def default_space(model: str, small_data: bool = True):
    """Ranges centred on stronger regularization than library defaults.

    Library defaults are tuned for larger datasets. On a few thousand rows they
    overfit, so a search centred on them spends its budget in the wrong region.
    """
    if model == "ridge":
        return {"alpha": [0.1, 1.0, 3.0, 10.0, 20.0, 50.0, 100.0, 300.0]}
    if model == "lasso":
        return {"alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]}
    if model == "logreg":
        return {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    if model in ("lightgbm", "lgbm"):
        return {
            # learning_rate and n_estimators are searched together on purpose:
            # they trade off directly, and tuning either alone finds a compromise
            # that is optimal for neither.
            "learning_rate": [0.005, 0.01, 0.02, 0.05],
            "n_estimators": [500, 1000, 2000, 3000],
            "num_leaves": [4, 8, 15, 31] if not small_data else [4, 8, 15],
            "max_depth": [3, 4, 5, -1],
            "min_child_samples": [5, 10, 20, 40],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.0, 0.5, 1.0, 5.0],
        }
    raise ValueError(f"no default space for {model!r}")


def default_factory(model: str, task: str):
    """Return fn(config) -> fresh unfitted estimator."""
    if model == "ridge":
        from sklearn.linear_model import Ridge
        return lambda cfg: Ridge(**cfg)
    if model == "lasso":
        from sklearn.linear_model import Lasso
        return lambda cfg: Lasso(max_iter=5000, **cfg)
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        return lambda cfg: LogisticRegression(max_iter=5000, **cfg)
    if model in ("lightgbm", "lgbm"):
        if task == "classification":
            from lightgbm import LGBMClassifier as L
        else:
            from lightgbm import LGBMRegressor as L
        return lambda cfg: L(random_state=42, verbosity=-1, **cfg)
    raise ValueError(f"no default factory for {model!r}")


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def search(X, y, factory, space, search_seeds, n_trials=50, n_splits=5,
           metric="rmse", splitter=None, rng_seed=0, verbose=True):
    """Random search scored on the SEARCH seeds only.

    Returns a frame of every configuration tried with its search score, best
    first. Returning all of them rather than only the winner is deliberate: a
    flat top of the table means the parameter choice is not a real decision, and
    that is worth seeing.
    """
    lower_better = LOWER_IS_BETTER[metric]
    splitter = splitter or ("stratified" if is_classification(metric) else "kfold")
    rng = np.random.default_rng(rng_seed)
    configs = sample_space(space, n_trials, rng)

    rows = []
    for i, cfg in enumerate(configs, 1):
        res = multiseed_score(X, y, lambda: factory(cfg), seeds=search_seeds,
                              n_splits=n_splits, metric=metric,
                              splitter=splitter)
        rows.append({**cfg, "search_score": res["mean"],
                     "search_std": res["std"]})
        if verbose and (i % 10 == 0 or i == len(configs)):
            print(f"  {i}/{len(configs)} trials")

    df = pd.DataFrame(rows).sort_values("search_score", ascending=lower_better)
    return df.reset_index(drop=True)


def revalidate(X, y, factory, config, holdout_seeds, baseline_config=None,
               n_splits=5, metric="rmse", splitter=None):
    """Score a configuration on seeds the search never saw.

    When `baseline_config` is given, the same holdout seeds score it too, so the
    comparison is paired and `wins` is meaningful.
    """
    splitter = splitter or ("stratified" if is_classification(metric) else "kfold")
    tuned = multiseed_score(X, y, lambda: factory(config), seeds=holdout_seeds,
                            n_splits=n_splits, metric=metric, splitter=splitter)
    base = None
    if baseline_config is not None:
        base = multiseed_score(X, y, lambda: factory(baseline_config),
                               seeds=holdout_seeds, n_splits=n_splits,
                               metric=metric, splitter=splitter)
    return tuned, base


def run(X, y, model="ridge", task="regression", metric="rmse", seeds=None,
        n_trials=50, n_splits=5, n_holdout=2, space=None, factory=None,
        baseline_config=None, verbose=True):
    """Full protocol: split seeds, search, re-validate, report a verdict."""
    search_seeds, holdout_seeds = split_seeds(seeds, n_holdout=n_holdout)
    space = space or default_space(model, small_data=len(X) < 5000)
    factory = factory or default_factory(model, task)
    baseline_config = baseline_config if baseline_config is not None else {}

    if verbose:
        print(f"model: {model}   metric: {metric}   trials: {n_trials}")
        print(f"search seeds:  {search_seeds}")
        print(f"holdout seeds: {holdout_seeds}   (never seen by the search)")
        print(f"space: { {k: len(v) for k, v in space.items()} }\n")

    table = search(X, y, factory, space, search_seeds, n_trials=n_trials,
                   n_splits=n_splits, metric=metric, verbose=verbose)
    param_cols = [c for c in table.columns if not c.startswith("search_")]
    best = {k: table.iloc[0][k] for k in param_cols}
    # numpy scalars back to python so the config prints and logs cleanly
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}

    tuned, base = revalidate(X, y, factory, best, holdout_seeds,
                             baseline_config=baseline_config,
                             n_splits=n_splits, metric=metric)

    optimism = tuned["mean"] - table.iloc[0]["search_score"]
    if not LOWER_IS_BETTER[metric]:
        optimism = -optimism

    verdict = None
    if base is not None:
        verdict = compare_to_baseline(base, tuned, metric=metric)
        # compare_to_baseline's ADOPT rule requires `wins >= max(4, n-1)`,
        # which was written for a 5-seed screen. A 2- or 3-seed holdout can
        # never reach 4, so its 'verdict' field would read REJECT no matter
        # what the numbers say. Decide here instead: unanimous across the
        # holdout seeds AND larger than the noise floor.
        unanimous = verdict["wins"] == verdict["n_seeds"]
        verdict["verdict"] = (
            "ADOPT" if unanimous and verdict["delta_exceeds_noise"] else "REJECT")
        verdict["rule"] = (f"unanimous across {verdict['n_seeds']} holdout "
                           "seeds and larger than the noise floor")

    if verbose:
        print(f"\n=== top configurations (search seeds) ===")
        print(table.head(8).round(6).to_string(index=False))

        spread = abs(table.iloc[0]["search_score"]
                     - table.iloc[min(len(table) - 1, 9)]["search_score"])
        noise = float(table.iloc[0]["search_std"])
        if spread < noise:
            print(f"\nThe top 10 configurations span {spread:.6f}, inside the "
                  f"across-seed std of {noise:.6f}. The parameter choice is "
                  "not a real decision here -- take the simplest of them.")

        print(f"\n=== winner ===")
        print(f"  config:        {best}")
        print(f"  search score:  {table.iloc[0]['search_score']:.6f}")
        print(f"  holdout score: {tuned['mean']:.6f} "
              f"(std {tuned['std']:.6f}, seeds {holdout_seeds})")
        print(f"  search optimism: {optimism:+.6f}  "
              f"(positive = the search score flattered the winner)")
        if optimism > tuned["std"]:
            print("  The search score is optimistic by more than the holdout "
                  "noise -- the budget was large relative to the data. Report "
                  "the holdout number, not the search number.")
        elif optimism < -tuned["std"]:
            print("  The holdout scored BETTER than the search, so the search "
                  "was not overfitting its seeds. That is a property of these "
                  "particular seeds rather than evidence the winner is good -- "
                  "the verdict below is still what decides.")

        if verdict is not None:
            print(f"\n=== tuned vs defaults, on holdout seeds ===")
            print(f"  baseline: {base['mean']:.6f}   tuned: {tuned['mean']:.6f}")
            print(f"  delta:    {verdict['delta']:+.6f}   "
                  f"wins: {verdict['consistency']}")
            print(f"  noise floor: {verdict['noise_floor']:.6f}")
            # Identical per-seed scores mean the search landed back on the
            # baseline configuration. Comparing the config dicts would not
            # catch it -- the baseline is usually `{}` (library defaults) while
            # the winner names the same values explicitly.
            if base["per_seed"] == tuned["per_seed"]:
                print("  The search selected the baseline configuration. "
                      "There is nothing to adopt, and a zero delta here is the "
                      "correct result rather than a failed run.")
            elif not verdict["delta_exceeds_noise"]:
                print("  VERDICT: REJECT -- tuning found nothing. The held-out "
                      "gain is inside the noise floor. Keep the defaults and "
                      "log the negative result.")
            else:
                print(f"  VERDICT: {verdict['verdict']}  ({verdict['rule']})")

        print("\nLog: model, trial count, objective metric, search seeds, "
              "holdout seeds, and the held-out delta -- including when it is "
              "negative. A tuning result missing those cannot be reproduced.")

    return {"table": table, "best": best, "tuned": tuned, "baseline": base,
            "verdict": verdict, "optimism": optimism,
            "search_seeds": search_seeds, "holdout_seeds": holdout_seeds}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Hyperparameter search with held-out-seed re-validation")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--drop", nargs="*", default=["Id", "id", "ID"])
    p.add_argument("--log-target", action="store_true")
    p.add_argument("--model", default="ridge",
                   choices=["ridge", "lasso", "logreg", "lightgbm"])
    p.add_argument("--task", default="regression",
                   choices=["regression", "classification"])
    p.add_argument("--metric", default=None, choices=list(METRICS))
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--n-holdout", type=int, default=2)
    p.add_argument("--n-splits", type=int, default=5)
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

    run(X, y, model=args.model, task=args.task, metric=metric,
        seeds=args.seeds, n_trials=args.trials, n_splits=args.n_splits,
        n_holdout=args.n_holdout)


if __name__ == "__main__":
    main()
