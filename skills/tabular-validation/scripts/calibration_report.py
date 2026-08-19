"""
Probability calibration and decision-threshold selection for classifiers.

AUC measures ranking. It says nothing about whether a predicted 0.9 corresponds
to a 90% chance. A model can rank perfectly and still be systematically
overconfident, and that gap is invisible in every ranking metric.

It stops being invisible the moment the output is compared to a threshold,
multiplied by a value, or shown to a person as a percentage.

TWO THINGS THIS FILE ENFORCES

**The threshold is chosen out-of-fold and applied once.** Sweeping thresholds
against the data you then report the score on fits the threshold to that data.
Accuracy and F1 come out optimistic, often substantially so when classes are
imbalanced. `tune_threshold` here takes out-of-fold predictions and returns a
number; applying it to the test set is a separate step that happens once.

**The calibrator is fitted on data the model did not train on.** A Platt or
isotonic fit on in-fold predictions leaks exactly the way target encoding does,
and produces a reliability curve that looks excellent and means nothing.

WHICH METRICS MOVE WITH THE THRESHOLD

Accuracy, F1, precision and recall do. AUC, log loss, Brier and ECE do not --
they read the probabilities directly. That split is also the diagnostic: strong
AUC alongside accuracy that collapses on new data is the signature of a
threshold fitted to the evaluation set.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, log_loss, f1_score, accuracy_score,
    precision_score, recall_score,
)


# --------------------------------------------------------------------------
# calibration metrics
# --------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins: int = 10,
                               strategy: str = "uniform"):
    """Weighted mean gap between predicted probability and observed frequency.

    `strategy="uniform"` bins by probability value; `"quantile"` bins by equal
    counts. Uniform bins are easier to read but can leave near-empty bins
    dominating nothing; quantile bins give every bin the same weight. Report
    which you used -- ECE is not comparable across binning schemes, and a model
    can look better or worse purely by the choice.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    if len(edges) < 2:
        return 0.0, pd.DataFrame()

    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0,
                  len(edges) - 2)

    rows = []
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(y_prob[mask].mean())
        freq = float(y_true[mask].mean())
        ece += (count / len(y_prob)) * abs(conf - freq)
        rows.append({"bin": f"[{edges[b]:.2f},{edges[b+1]:.2f})",
                     "n": count, "mean_predicted": conf,
                     "observed_rate": freq, "gap": conf - freq})

    return float(ece), pd.DataFrame(rows)


def calibration_metrics(y_true, y_prob):
    """Threshold-independent metrics only. None of these move with a threshold."""
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob)[0],
        "base_rate": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_prob)),
    }


def calibration_direction(y_true, y_prob) -> str:
    """Which way the model is wrong, in one line.

    Mean predicted probability above the base rate means the model is
    overconfident in the positive class -- the usual outcome for boosted trees
    and for anything trained with class weights or on resampled data.
    """
    gap = float(np.mean(y_prob)) - float(np.mean(y_true))
    if abs(gap) < 0.01:
        return f"well centred (mean predicted - base rate = {gap:+.4f})"
    word = "over" if gap > 0 else "under"
    return (f"{word}-predicts the positive class by {abs(gap):.4f} on average "
            f"(mean predicted {np.mean(y_prob):.4f} vs base rate "
            f"{np.mean(y_true):.4f})")


# --------------------------------------------------------------------------
# threshold
# --------------------------------------------------------------------------

THRESHOLD_METRICS = {
    "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
    "accuracy": accuracy_score,
    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
}


def tune_threshold(y_true, y_prob, objective: str = "f1", n_steps: int = 200):
    """Pick a decision threshold from OUT-OF-FOLD predictions.

    Pass out-of-fold or validation predictions. Passing test predictions here
    and then reporting the resulting accuracy is the leak this module exists to
    prevent -- the function cannot tell the difference, so the caller must.

    Returns (threshold, score_at_threshold, sweep_frame).
    """
    if objective not in THRESHOLD_METRICS:
        raise ValueError(f"objective must be one of {list(THRESHOLD_METRICS)}")
    fn = THRESHOLD_METRICS[objective]

    lo, hi = float(np.min(y_prob)), float(np.max(y_prob))
    if hi - lo < 1e-12:
        raise ValueError("predictions are constant; no threshold to choose")

    grid = np.linspace(lo, hi, n_steps)
    rows = [{"threshold": float(t),
             objective: float(fn(y_true, (y_prob >= t).astype(int)))}
            for t in grid]
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep[objective].idxmax()]
    return float(best["threshold"]), float(best[objective]), sweep


def threshold_metrics(y_true, y_prob, threshold: float):
    """Everything that moves with the threshold, at one fixed threshold."""
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {name: float(fn(y_true, pred)) for name, fn in THRESHOLD_METRICS.items()}


def threshold_sensitivity(sweep: pd.DataFrame, objective: str,
                          tolerance: float = 0.01):
    """How wide the near-optimal threshold band is.

    A flat objective curve means the threshold is not really a decision, and a
    threshold picked to three decimal places is fitting noise. A sharp peak
    means the choice matters and deserves the out-of-fold discipline.
    """
    best = sweep[objective].max()
    band = sweep[sweep[objective] >= best - tolerance * abs(best)]
    width = float(band["threshold"].max() - band["threshold"].min())
    return {"best": float(best), "band_width": width,
            "band_low": float(band["threshold"].min()),
            "band_high": float(band["threshold"].max()),
            "flat": width > 0.15}


# --------------------------------------------------------------------------
# out-of-fold generation
# --------------------------------------------------------------------------

def oof_probabilities(X, y, model_fn, n_splits: int = 5, seed: int = 42):
    """Out-of-fold predicted probabilities for the positive class.

    Every row is predicted by a model that never saw it. This is the only input
    a threshold or a calibrator may legitimately be fitted on.
    """
    oof = np.zeros(len(X), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        if not hasattr(m, "predict_proba"):
            raise TypeError(
                f"{type(m).__name__} has no predict_proba; calibration is only "
                "meaningful for models that emit probabilities")
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def calibration_report(y_true, y_prob, objective: str = "f1", n_bins: int = 10,
                       verbose: bool = True):
    """Full report from out-of-fold predictions."""
    metrics = calibration_metrics(y_true, y_prob)
    ece, table = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    thr, thr_score, sweep = tune_threshold(y_true, y_prob, objective=objective)
    sens = threshold_sensitivity(sweep, objective)
    at_half = threshold_metrics(y_true, y_prob, 0.5)
    at_best = threshold_metrics(y_true, y_prob, thr)

    if verbose:
        print("=== threshold-independent metrics ===")
        for k, v in metrics.items():
            print(f"  {k:16s} {v:.5f}")
        print(f"\n  direction: {calibration_direction(y_true, y_prob)}")

        print(f"\n=== reliability ({n_bins} uniform bins) ===")
        print(table.round(4).to_string(index=False))
        print("\n  gap = mean predicted - observed. Positive means "
              "overconfident in that bin.")

        print(f"\n=== threshold, chosen on these (out-of-fold) predictions ===")
        print(f"  objective:        {objective}")
        print(f"  chosen threshold: {thr:.4f}  ({objective}={thr_score:.5f})")
        print(f"  near-optimal band: [{sens['band_low']:.3f}, "
              f"{sens['band_high']:.3f}]  width {sens['band_width']:.3f}")
        if sens["flat"]:
            print("  The objective curve is FLAT across that band -- the exact "
                  "threshold is not a real decision. Pick a round number and "
                  "move on rather than reporting four decimal places.")

        print("\n=== threshold-dependent metrics ===")
        print(f"  {'metric':12s} {'at 0.5':>10s} {'at ' + format(thr, '.3f'):>10s}")
        for k in THRESHOLD_METRICS:
            print(f"  {k:12s} {at_half[k]:10.5f} {at_best[k]:10.5f}")

        print("\nApply the chosen threshold to the test set ONCE. Re-sweeping "
              "it there turns the reported accuracy and F1 into fitted numbers "
              "rather than held-out ones -- AUC, log loss, Brier and ECE above "
              "are unaffected either way.")

    return {"metrics": metrics, "ece": ece, "reliability": table,
            "threshold": thr, "threshold_score": thr_score,
            "sensitivity": sens, "at_half": at_half, "at_threshold": at_best}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Calibration and decision-threshold report for a binary classifier")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--drop", nargs="*", default=["Id", "id", "ID"])
    p.add_argument("--objective", default="f1", choices=list(THRESHOLD_METRICS))
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.target]
    if y.nunique() != 2:
        p.error(f"target has {y.nunique()} distinct values; this report is "
                "binary-only")
    X = df.drop(columns=[args.target] + [c for c in args.drop if c in df.columns],
                errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    print(f"matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"base rate: {y.mean():.4f}\n")

    from sklearn.linear_model import LogisticRegression
    oof = oof_probabilities(X, y, lambda: LogisticRegression(max_iter=5000),
                            n_splits=args.n_splits, seed=args.seed)
    calibration_report(y, oof, objective=args.objective, n_bins=args.bins)


if __name__ == "__main__":
    main()
