"""
TabPFN integration for tabular ML pipelines.

TabPFN is a pre-trained transformer for tabular data. Unlike gradient boosting
or linear models it is not fit from scratch on your data — it does in-context
learning over the training rows at predict time. It is specifically strong on
small datasets, which is exactly where tree ensembles and linear models are
usually the safe choice.

VERIFICATION STATUS: the API calls here were written against tabpfn 8.3.0 with
the package installed, so parameter names are correct. The model itself was NOT
benchmarked when this script was written — model weights could not be
downloaded in that environment. Treat the accuracy claims in TabPFN's own
documentation as unverified here, and measure on your own data with the
multi-seed harness before adopting it.

REQUIREMENTS
    pip install tabpfn            # Python 3.10+
    GPU strongly recommended. On CPU only small datasets (roughly <=1000 rows)
    are practical; beyond that set TABPFN_ALLOW_CPU_LARGE_DATASET=true and
    expect it to be very slow.

MODEL ACCESS
    Weights are gated. On first use TabPFN opens a browser to accept the
    license, or set TABPFN_TOKEN (from https://ux.priorlabs.ai) for headless
    environments. Some checkpoints also require accepting terms on HuggingFace.

LICENSING — CHECK THIS BEFORE COMMERCIAL USE
    TabPFN-3, 2.6 and 2.5 weights are released under NON-COMMERCIAL licenses.
    TabPFN-2 weights and the code are under the Prior Labs License (Apache 2.0
    plus an attribution requirement). Kaggle and research use is fine; shipping
    predictions in a commercial product is not, for the non-commercial
    checkpoints. Verify current terms at https://docs.priorlabs.ai/models
    rather than trusting this comment.

THE PREPROCESSING CONFLICT — IMPORTANT
    TabPFN's documentation says NOT to one-hot encode or scale features, and it
    handles missing values natively. That directly conflicts with the standard
    pipeline in this plugin, which one-hot encodes categoricals and imputes
    nulls before modeling.

    So TabPFN needs its own data path: cleaning and engineered features yes,
    one-hot encoding and scaling no. `prepare_for_tabpfn` below does that, and
    passes categorical column positions via `categorical_features_indices` so
    the model can handle them itself.

    Do not feed TabPFN the same matrix you feed Ridge. Blending them means
    maintaining two feature paths.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
        return True
    except ImportError:
        return False


def check_environment(n_rows: int) -> list[str]:
    """Return a list of warnings about running TabPFN in this environment."""
    warnings = []
    if not tabpfn_available():
        return ["tabpfn is not installed (pip install tabpfn, needs Python 3.10+)"]

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if not has_gpu:
        warnings.append(
            "No CUDA GPU detected. TabPFN is slow on CPU; the documented "
            "practical CPU limit is around 1000 rows."
        )
        if n_rows > 1000 and os.environ.get("TABPFN_ALLOW_CPU_LARGE_DATASET") != "true":
            warnings.append(
                f"{n_rows} rows exceeds the CPU guardrail. Set "
                "TABPFN_ALLOW_CPU_LARGE_DATASET=true to override — expect it "
                "to be very slow."
            )
    return warnings


def prepare_for_tabpfn(df: pd.DataFrame, target: str = None,
                        drop_cols=("Id", "id", "ID")):
    """Build a TabPFN-appropriate matrix.

    Deliberately does NOT one-hot encode, scale, or impute — TabPFN's docs say
    those hurt, and it handles missing values natively. Categoricals are label-
    coded only so they fit in a numeric array, and their positions are returned
    so they can be declared to the model.

    Returns (X, y_or_None, categorical_indices).
    """
    df = df.copy()
    y = None
    if target and target in df.columns:
        y = df[target]
        df = df.drop(columns=[target])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    cat_indices = []
    for i, col in enumerate(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            # label-code to a numeric array; TabPFN is told which columns these
            # are, so it does not treat the codes as magnitudes
            df[col] = pd.Categorical(df[col]).codes
            cat_indices.append(i)

    return df, y, cat_indices


def make_tabpfn_regressor(cat_indices=None, device="auto", random_state=0,
                          allow_large=False, model_version=None):
    """Construct a TabPFNRegressor.

    model_version: None uses the current default checkpoint. Pass "V2" to use
    the Apache-2.0-licensed weights instead, which matters if the work is
    commercial — see the licensing note at the top of this file.
    """
    from tabpfn import TabPFNRegressor

    kwargs = dict(
        categorical_features_indices=cat_indices,
        device=device,
        random_state=random_state,
        ignore_pretraining_limits=allow_large,
    )
    if model_version:
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(
            getattr(ModelVersion, model_version), **kwargs
        )
    return TabPFNRegressor(**kwargs)


def make_tabpfn_classifier(cat_indices=None, device="auto", random_state=0,
                            allow_large=False, model_version=None):
    from tabpfn import TabPFNClassifier

    kwargs = dict(
        categorical_features_indices=cat_indices,
        device=device,
        random_state=random_state,
        ignore_pretraining_limits=allow_large,
    )
    if model_version:
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(
            getattr(ModelVersion, model_version), **kwargs
        )
    return TabPFNClassifier(**kwargs)


def evaluate_tabpfn(train_df, target, task="regression", log_target=False,
                    seeds=(42, 7, 2024), n_splits=5, metric=None,
                    prepare_fn=None, device="auto", verbose=True):
    """Score TabPFN with the same multi-seed protocol as every other model.

    prepare_fn: optional cleaning/feature function applied BEFORE the
    TabPFN-specific encoding. Pass your pipeline's cleaning step, but NOT its
    one-hot encoding step.

    Returns {"per_seed": {...}, "mean": float, "std": float}.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import mean_squared_error, roc_auc_score

    metric = metric or ("rmse" if task == "regression" else "auc")

    for w in check_environment(len(train_df)):
        print(f"WARNING: {w}")

    df = prepare_fn(train_df) if prepare_fn else train_df
    X, y, cat_idx = prepare_for_tabpfn(df, target=target)
    if y is None:
        raise ValueError(f"target column '{target}' not found")
    if log_target:
        y = np.log1p(y)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    if verbose:
        print(f"TabPFN input: {X.shape[0]} rows x {X.shape[1]} features "
              f"({len(cat_idx)} categorical, no one-hot, nulls kept)")

    per_seed = {}
    for seed in seeds:
        if task == "regression":
            sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_scores = []
        for tr, va in sp.split(X, y):
            if task == "regression":
                m = make_tabpfn_regressor(cat_idx, device=device, random_state=seed)
                m.fit(X.iloc[tr], y.iloc[tr])
                preds = m.predict(X.iloc[va])
                fold_scores.append(np.sqrt(mean_squared_error(y.iloc[va], preds)))
            else:
                m = make_tabpfn_classifier(cat_idx, device=device, random_state=seed)
                m.fit(X.iloc[tr], y.iloc[tr])
                preds = m.predict_proba(X.iloc[va])[:, 1]
                fold_scores.append(roc_auc_score(y.iloc[va], preds))

        per_seed[seed] = float(np.mean(fold_scores))
        if verbose:
            print(f"  seed {seed}: {metric}={per_seed[seed]:.5f}")

    values = list(per_seed.values())
    result = {"per_seed": per_seed, "mean": float(np.mean(values)),
              "std": float(np.std(values))}

    if verbose:
        print(f"\nTabPFN mean {metric}: {result['mean']:.5f} "
              f"(across-seed std {result['std']:.5f})")
        print("\nCompare against your existing baseline on the SAME seeds "
              "before adopting. TabPFN being a foundation model is not "
              "evidence that it wins on your data.")

    return result


def tabpfn_oof_predictions(train_df, test_df, target, log_target=False,
                            seed=42, n_splits=5, prepare_fn=None,
                            device="auto"):
    """Out-of-fold train predictions plus test predictions, for blending.

    Returns (oof_array, test_pred_array). Both are in the same space as y
    (log space if log_target=True) — convert back only at submission time.

    Blend these with your other models' OOF predictions using the existing
    blend search, so TabPFN's weight is chosen by the same evidence standard
    as everything else.
    """
    from sklearn.model_selection import KFold

    tr_df = prepare_fn(train_df) if prepare_fn else train_df
    te_df = prepare_fn(test_df) if prepare_fn else test_df

    X, y, cat_idx = prepare_for_tabpfn(tr_df, target=target)
    X_test, _, _ = prepare_for_tabpfn(te_df, target=target)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    if log_target:
        y = np.log1p(y)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    oof = np.zeros(len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(X):
        m = make_tabpfn_regressor(cat_idx, device=device, random_state=seed)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])

    final = make_tabpfn_regressor(cat_idx, device=device, random_state=seed)
    final.fit(X, y)
    test_pred = final.predict(X_test)

    return oof, test_pred


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate TabPFN with multi-seed CV")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--log-target", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 2024])
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    evaluate_tabpfn(df, args.target, task=args.task,
                    log_target=args.log_target, seeds=args.seeds,
                    device=args.device)
