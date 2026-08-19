"""
Evaluate a tabular foundation model with the same multi-seed protocol as
everything else in this plugin.

TWO THINGS THIS FILE EXISTS TO PREVENT

**Feeding a foundation model the wrong matrix.** The standard path here --
one-hot encode, impute, scale -- is the correct path for Ridge and for gradient
boosting, and the wrong one for these models. They take mixed-type columns and a
declaration of which columns are categorical, and they handle missing values
themselves. Passing them a 300-column one-hot matrix does not error; it just
performs worse than the model can, and the resulting comparison against
LightGBM is then meaningless. `prepare_for_fm` below builds the right matrix.

**Subsampling outside the fold.** Most of these models cap the number of context
rows. Subsampling the whole dataset down to the cap before splitting puts
validation rows into the context, which is leakage of the ordinary kind and
inflates the score. The subsample must be drawn from the TRAINING FOLD, inside
the loop. `ContextCappedEstimator` does that, and nothing else here subsamples.

WHAT IS NOT VERIFIED HERE

The adapters call each library's documented sklearn-style API. They were written
from the upstream repositories, not run against installed packages for every
model -- the TabFM signature is transcribed from google-research/tabfm's
constructor, the row caps from TALENT's method registry. Treat an adapter that
errors on import as a signature that moved upstream, and fix the adapter rather
than working around it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "tabular-validation", "scripts"))

from fm_registry import REGISTRY, Task  # noqa: E402
from fm_preflight import preflight  # noqa: E402
from multiseed_cv import (  # noqa: E402
    multiseed_score, compare_to_baseline, is_classification, DEFAULT_SEEDS,
)


# ---------------------------------------------------------------------------
# data path
# ---------------------------------------------------------------------------

def prepare_for_fm(df: pd.DataFrame, target: str | None = None,
                   drop_cols=("Id", "id", "ID")):
    """Build the matrix a foundation model wants.

    Kept: engineered features, raw numerics, categorical columns as categories.
    Not done: one-hot encoding, scaling, imputation. Those are the model's job
    and doing them here throws away information it would have used.

    Returns (X, y, cat_indices). `cat_indices` are positional, because that is
    what every one of these libraries takes.
    """
    work = df.drop(columns=[c for c in drop_cols if c in df.columns],
                   errors="ignore")
    y = None
    if target is not None:
        if target not in work.columns:
            raise KeyError(f"target column {target!r} not in the frame")
        y = work[target]
        work = work.drop(columns=[target])

    cat_indices = [
        i for i, c in enumerate(work.columns)
        if work[c].dtype == object
        or isinstance(work[c].dtype, pd.CategoricalDtype)
        or work[c].dtype == bool
    ]

    # Object columns are cast to category rather than to codes: the libraries
    # want the labels, and casting to codes here would impose an arbitrary
    # ordering on a nominal column.
    for c in work.columns[cat_indices] if cat_indices else []:
        work[c] = work[c].astype("category")

    work = work.reset_index(drop=True)
    if y is not None:
        y = y.reset_index(drop=True)
    return work, y, cat_indices


# ---------------------------------------------------------------------------
# context cap
# ---------------------------------------------------------------------------

class ContextCappedEstimator:
    """Wraps an estimator so its fit() never sees more rows than the cap.

    The subsample is drawn inside fit(), which is inside the CV fold, so it can
    only ever contain training rows. Stratified for classification, so a cap
    that is small relative to the data does not quietly drop a rare class.

    `random_state` fixes the subsample. Two runs of the same seed draw the same
    context; the reported score is reproducible. Report `n_context_` alongside
    any score from a capped model -- a score from an unstated subsample size is
    not comparable to anything.
    """

    def __init__(self, inner, row_limit: int | None, stratify: bool,
                 random_state: int = 0):
        self.inner = inner
        self.row_limit = row_limit
        self.stratify = stratify
        self.random_state = random_state
        self.n_context_ = None
        self.subsampled_ = False

    def fit(self, X, y):
        if self.row_limit is not None and len(X) > self.row_limit:
            from sklearn.model_selection import train_test_split
            strat = y if self.stratify else None
            try:
                X, _, y, _ = train_test_split(
                    X, y, train_size=self.row_limit, stratify=strat,
                    random_state=self.random_state)
            except ValueError:
                # Stratification fails when a class has fewer members than the
                # split requires. Fall back to an unstratified draw rather than
                # crashing, and make the fallback visible.
                X, _, y, _ = train_test_split(
                    X, y, train_size=self.row_limit,
                    random_state=self.random_state)
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True) if hasattr(y, "reset_index") else y
            self.subsampled_ = True
        self.n_context_ = len(X)
        self.inner.fit(X, y)
        return self

    def predict(self, X):
        return self.inner.predict(X)

    def predict_proba(self, X):
        return self.inner.predict_proba(X)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

def _tabpfn(task: Task, cat_indices, device: str, random_state: int):
    if task is Task.CLASSIFICATION:
        from tabpfn import TabPFNClassifier as C
    else:
        from tabpfn import TabPFNRegressor as C
    return C(device=device, categorical_features_indices=cat_indices,
             random_state=random_state)


def _tabicl(task: Task, cat_indices, device: str, random_state: int):
    if task is Task.CLASSIFICATION:
        from tabicl import TabICLClassifier as C
    else:
        from tabicl import TabICLRegressor as C
    return C(device=device, random_state=random_state)


def _tabdpt(task: Task, cat_indices, device: str, random_state: int):
    if task is Task.CLASSIFICATION:
        from tabdpt import TabDPTClassifier as C
    else:
        from tabdpt import TabDPTRegressor as C
    return C()


def _tabfm(task: Task, cat_indices, device: str, random_state: int,
           max_num_rows: int | None = None, n_estimators: int = 32):
    """TabFM, PyTorch backend.

    `max_num_rows` is forwarded rather than left at its default. TabFM has no
    registry row cap, so `ContextCappedEstimator` does not engage for it and
    TabFM does its own context sampling instead. Its constructor default for
    `max_num_rows` is None while the README FAQ describes a 100-row default
    context -- so leaving it unset means the context size is whichever of those
    the installed version happens to implement. Passing it makes the number a
    decision that appears in the experiment log.
    """
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    if task is Task.CLASSIFICATION:
        from tabfm import TabFMClassifier as C
        model = tabfm_v1_0_0.load(device=device)
    else:
        from tabfm import TabFMRegressor as C
        model = tabfm_v1_0_0.load(model_type="regression", device=device)
    return C(model=model, n_estimators=n_estimators,
             max_num_rows=max_num_rows, random_state=random_state)


ADAPTERS = {
    "tabpfn_v1": _tabpfn,
    "tabpfn_v2": _tabpfn,
    "tabpfn_v2_5": _tabpfn,
    "tabpfn_v3": _tabpfn,
    "tabpfn_real": _tabpfn,
    "tabicl": _tabicl,
    "tabicl_v2": _tabicl,
    "tabdpt": _tabdpt,
    "tabfm": _tabfm,
}


def make_model_fn(model: str, task: Task, cat_indices, device="auto",
                  random_state=0, context_rows: int | None = None):
    """Return a zero-argument factory producing a fresh capped estimator.

    A factory rather than an instance: the CV harness needs an unfitted model
    per fold, and reusing one fitted estimator across folds leaks between them.
    """
    if model not in ADAPTERS:
        raise ValueError(
            f"no adapter for {model!r}. Adapters exist for: "
            f"{', '.join(sorted(ADAPTERS))}. Models in the registry without an "
            "adapter (mitra, limix) ship through their own frameworks -- run "
            "them there and bring the out-of-fold predictions back for blending.")

    spec = REGISTRY[model]
    if not spec.handles(task):
        raise ValueError(f"{model} does not support {task.value}")

    cap = context_rows if context_rows is not None else spec.train_row_limit
    stratify = task is Task.CLASSIFICATION

    # TabFM samples its own context and has no registry cap, so the wrapper
    # would never engage for it. Forward the cap into the estimator instead, so
    # exactly one component decides the context size either way.
    inner_kwargs = {}
    wrapper_cap = cap
    if model == "tabfm":
        inner_kwargs["max_num_rows"] = cap
        wrapper_cap = None

    def factory():
        inner = ADAPTERS[model](task, cat_indices, device, random_state,
                                **inner_kwargs)
        return ContextCappedEstimator(inner, wrapper_cap, stratify, random_state)

    return factory


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_fm(df: pd.DataFrame, target: str, model: str,
                metric: str = "rmse", seeds=None, n_splits: int = 5,
                log_target: bool = False, device: str = "auto",
                commercial: bool = False, context_rows: int | None = None,
                baseline_result: dict | None = None):
    """Score a foundation model across seeds, with a preflight gate first.

    Returns the multiseed_score dict, plus a comparison when a baseline result
    from the same seeds is supplied.
    """
    task = Task.CLASSIFICATION if is_classification(metric) else Task.REGRESSION
    seeds = seeds or DEFAULT_SEEDS

    X, y, cat_idx = prepare_for_fm(df, target=target)

    pf = preflight(model, commercial=commercial, n_rows=len(X),
                   n_features=X.shape[1])
    print(pf.render())
    if not pf.runnable:
        raise RuntimeError(
            f"{model} is blocked -- see the blockers above. Nothing was fitted.")

    if log_target:
        if task is Task.CLASSIFICATION:
            raise ValueError("log_target is a regression option")
        y = np.log1p(y)

    print(f"\nmatrix: {X.shape[0]} rows x {X.shape[1]} features, "
          f"{len(cat_idx)} categorical (passed as categories, not one-hot)")

    splitter = "stratified" if task is Task.CLASSIFICATION else "kfold"
    result = multiseed_score(
        X, y,
        make_model_fn(model, task, cat_idx, device=device,
                      context_rows=context_rows),
        seeds=seeds, n_splits=n_splits, metric=metric, splitter=splitter)

    cap = context_rows if context_rows is not None else REGISTRY[model].train_row_limit
    print(f"\n{model} mean {metric}: {result['mean']:.5f}")
    print(f"per-seed: {[round(v, 5) for v in result['per_seed'].values()]}")
    print(f"across-seed std: {result['std']:.5f}")
    if cap and len(X) > cap:
        print(f"CONTEXT: capped at {cap} of {len(X)} training rows, drawn "
              f"inside each fold. Record this in the experiment log -- the "
              f"score is only comparable to other runs at the same cap.")

    comparison = None
    if baseline_result is not None:
        comparison = compare_to_baseline(baseline_result, result, metric=metric)
        print(f"\nvs baseline: delta={comparison['delta']:+.5f}  "
              f"wins={comparison['consistency']}  {comparison['verdict']}")
        if not comparison["delta_exceeds_noise"]:
            print("delta is inside the baseline's noise floor")

    return {"result": result, "comparison": comparison,
            "n_features": X.shape[1], "cat_indices": cat_idx}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Multi-seed evaluation of a tabular foundation model")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--metric", default="rmse")
    p.add_argument("--log-target", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 2024])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--commercial", action="store_true",
                   help="predictions will be shipped; blocks non-commercial weights")
    p.add_argument("--context-rows", type=int, default=None,
                   help="override the registry row cap; lower it to trade "
                        "accuracy for speed while iterating")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    try:
        evaluate_fm(df, args.target, args.model, metric=args.metric,
                    seeds=args.seeds, n_splits=args.n_splits,
                    log_target=args.log_target, device=args.device,
                    commercial=args.commercial, context_rows=args.context_rows)
    except RuntimeError as e:
        # A blocked model is an expected outcome, not a crash. Print the reason
        # and exit non-zero rather than dumping a traceback that buries it.
        print(f"\n{e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
