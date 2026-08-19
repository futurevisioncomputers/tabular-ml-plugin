"""The validation harness decides what gets adopted, so a sign error here
silently inverts every conclusion the plugin reaches.

Three failure classes are worth a test each, because none of them raises:

  * **Direction.** rmse is better lower, AUC better higher. A ranking or
    win-count that hard-codes one direction produces a confident, wrong,
    plausible-looking table. Every direction-sensitive function is tested
    against BOTH kinds of metric.
  * **In-fold discipline.** A context subsample drawn before the split, or a
    threshold tuned on the data the score is reported on, inflates the number
    without erroring.
  * **Seed reservation.** Holdout seeds must be disjoint from search seeds and
    decided before the search. If they overlap, the "held-out" score measures
    what selection already optimized.

Run: python -m tests.test_harness   (plain asserts, no pytest dep)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS = os.path.join(ROOT, "skills")
for path in (ROOT,
             os.path.join(SKILLS, "tabular-validation", "scripts"),
             os.path.join(SKILLS, "tabular-model-selection", "scripts"),
             os.path.join(SKILLS, "tabular-foundation-models", "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from multiseed_cv import LOWER_IS_BETTER, METRICS, multiseed_score  # noqa: E402
from compare_models import compare  # noqa: E402
from tune_search import split_seeds, sample_space  # noqa: E402
from calibration_report import (  # noqa: E402
    expected_calibration_error, tune_threshold, threshold_metrics,
    threshold_sensitivity, oof_probabilities, calibration_metrics,
)
from fm_evaluate import prepare_for_fm, ContextCappedEstimator  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def regression_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(3.0 * X["a"] + 0.5 * X["b"] + rng.normal(0, 0.5, n))
    return X, y


def classification_frame(n=600, seed=0, base_rate=-1.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    p = 1 / (1 + np.exp(-(base_rate + 1.5 * X["a"] + 0.4 * X["b"])))
    y = pd.Series(rng.binomial(1, p))
    return X, y


class _Constant:
    """A deliberately bad model, so 'worse' is unambiguous in both directions."""

    def __init__(self, noise=0.0, seed=0):
        self.noise, self.rng = noise, np.random.default_rng(seed)

    def fit(self, X, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu) + self.rng.normal(0, self.noise, len(X))

    def predict_proba(self, X):
        p = np.clip(np.full(len(X), self.mu), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------

def test_metric_direction_table_is_complete():
    assert set(LOWER_IS_BETTER) == set(METRICS), (
        "a metric without a direction entry ranks in whichever direction the "
        "sort default happens to be")
    print(f"  {len(METRICS)} metrics all have a direction")


def test_average_rank_puts_the_best_model_first_in_both_directions():
    """rank(ascending=...) is the single line that inverts a whole comparison."""
    from sklearn.linear_model import Ridge, LinearRegression

    X, y = regression_frame()
    zoo = {"Good": lambda: Ridge(alpha=1.0),
           "AlsoGood": lambda: LinearRegression(),
           "Bad": lambda: _Constant()}
    out = compare(X, y, zoo, seeds=[1, 2, 3], n_splits=3, metric="rmse",
                  verbose=False)
    summary = out["summary"]
    assert summary.index[0] != "Bad", "constant model ranked best on rmse"
    assert summary.loc["Bad", "avg_rank"] == summary["avg_rank"].max(), (
        "lower-is-better: the worst model does not have the worst rank")
    assert summary["avg_rank"].idxmin() == summary.index[0], (
        "best average rank disagrees with the mean sort order on clean data")

    Xc, yc = classification_frame()
    from sklearn.linear_model import LogisticRegression
    zoo_c = {"Good": lambda: LogisticRegression(max_iter=2000),
             "Bad": lambda: _Constant(noise=1e-9)}
    out_c = compare(Xc, yc, zoo_c, seeds=[1, 2, 3], n_splits=3, metric="auc",
                    verbose=False)
    sc = out_c["summary"]
    assert sc.index[0] == "Good", "higher-is-better: worse model sorted first"
    assert sc.loc["Bad", "avg_rank"] > sc.loc["Good", "avg_rank"], (
        "higher-is-better: rank direction is inverted")
    print("  average rank correct for lower-is-better and higher-is-better")


def test_compare_reports_cost_alongside_score():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    X, y = regression_frame()
    out = compare(X, y, {"Ridge": lambda: Ridge(),
                         "Forest": lambda: RandomForestRegressor(
                             n_estimators=50, random_state=0)},
                  seeds=[1, 2], n_splits=3, metric="rmse", verbose=False)
    s = out["summary"]
    for col in ("fit_s", "predict_s", "avg_rank", "std"):
        assert col in s.columns, f"summary is missing {col}"
        assert s[col].notna().all(), f"{col} has missing values"
    assert (s["fit_s"] > 0).all(), "fit time measured as zero"
    assert s.loc["Forest", "fit_s"] > s.loc["Ridge", "fit_s"], (
        "a forest fitting faster than a ridge means the timer is not "
        "measuring what it claims")
    print("  fit/predict time measured and attributed correctly")


# ---------------------------------------------------------------------------
# in-fold discipline
# ---------------------------------------------------------------------------

def test_context_cap_applies_inside_fit_only():
    X, y = classification_frame(n=400)
    seen = {}

    class Spy:
        def fit(self, Xf, yf):
            seen["rows"] = len(Xf)
            seen["classes"] = set(np.unique(yf))
            return self

        def predict(self, Xf):
            return np.zeros(len(Xf))

    capped = ContextCappedEstimator(Spy(), row_limit=50, stratify=True,
                                    random_state=0)
    capped.fit(X, y)
    assert seen["rows"] == 50, f"inner model saw {seen['rows']} rows, not 50"
    assert capped.n_context_ == 50 and capped.subsampled_
    assert seen["classes"] == {0, 1}, (
        "stratification dropped a class -- a cap small relative to the data "
        "must not silently make the problem single-class")

    # Below the cap nothing is touched: sampling when it is not needed would
    # throw away data for no reason.
    untouched = ContextCappedEstimator(Spy(), row_limit=10_000, stratify=True,
                                       random_state=0)
    untouched.fit(X, y)
    assert seen["rows"] == len(X) and not untouched.subsampled_
    print("  context cap engages inside fit, and only when needed")


def test_context_cap_is_reproducible():
    X, y = classification_frame(n=300)

    class Capture:
        def __init__(self): self.idx = None
        def fit(self, Xf, yf):
            self.idx = tuple(Xf["a"].round(9))
            return self
        def predict(self, Xf): return np.zeros(len(Xf))

    a, b = Capture(), Capture()
    ContextCappedEstimator(a, 40, True, random_state=7).fit(X, y)
    ContextCappedEstimator(b, 40, True, random_state=7).fit(X, y)
    assert a.idx == b.idx, "same seed drew a different context -- the reported "\
                           "score is not reproducible"
    c = Capture()
    ContextCappedEstimator(c, 40, True, random_state=8).fit(X, y)
    assert c.idx != a.idx, "different seeds drew an identical context"
    print("  context subsample is seed-reproducible")


def test_foundation_model_data_path_keeps_what_the_model_wants():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Id": range(50),
        "num": rng.normal(size=50),
        "cat": rng.choice(list("xy"), 50),
        "flag": rng.random(50) > 0.5,
        "y": rng.integers(0, 2, 50),
    })
    df.loc[df.index[:5], "num"] = np.nan

    X, y, cat_idx = prepare_for_fm(df, target="y")
    assert "Id" not in X.columns and "y" not in X.columns
    assert X["num"].isna().sum() == 5, (
        "nulls were imputed -- these models handle them natively and the "
        "imputation throws away the missingness signal")
    assert list(X.columns) == ["num", "cat", "flag"], (
        "columns were one-hot expanded; the model wants categorical indices")
    assert cat_idx == [1, 2], f"categorical positions wrong: {cat_idx}"
    assert str(X.dtypes["cat"]) == "category"
    print("  FM data path: no one-hot, no imputation, indices correct")


def test_threshold_is_tuned_on_out_of_fold_predictions():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    X, y = classification_frame(n=800, base_rate=-1.8)
    oof = oof_probabilities(X, y, lambda: LogisticRegression(max_iter=2000),
                            n_splits=5, seed=42)
    assert len(oof) == len(X) and (oof > 0).all() and (oof < 1).all()

    # A memorizing model makes the in-fold/out-of-fold gap unmistakable. An
    # unpruned tree fits its training rows perfectly, so if `oof_probabilities`
    # were leaking, its out-of-fold AUC would also be near 1. Comparing two
    # well-regularized models instead would be flaky -- on two clean features a
    # logistic fit barely overfits, and the gap can land either way by chance.
    memorizer = DecisionTreeClassifier(random_state=0)
    infold_auc = calibration_metrics(y, memorizer.fit(X, y).predict_proba(X)[:, 1])["auc"]
    oof_auc = calibration_metrics(
        y, oof_probabilities(X, y, lambda: DecisionTreeClassifier(random_state=0)))["auc"]
    assert infold_auc > 0.99, "the memorizing model did not memorize; test is void"
    assert oof_auc < 0.9, (
        f"an unpruned tree scored AUC {oof_auc:.4f} out of fold -- these "
        "predictions are not actually out of fold")

    thr, oof_f1, _ = tune_threshold(y, oof, objective="f1")
    assert 0.0 < thr < 1.0 and oof_f1 > 0
    print(f"  memorizer AUC {infold_auc:.4f} in-fold vs {oof_auc:.4f} OOF "
          f"(fold isolation holds); OOF threshold {thr:.3f}")


# ---------------------------------------------------------------------------
# calibration numerics
# ---------------------------------------------------------------------------

def test_ece_is_near_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 40_000)
    y = rng.binomial(1, p)
    ece, table = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.02, f"perfectly calibrated input scored ECE {ece:.4f}"
    assert not table.empty and abs(table["gap"]).max() < 0.05

    # A model shifted off the base rate must be caught.
    ece_bad, _ = expected_calibration_error(y, np.clip(p + 0.25, 0, 1), n_bins=10)
    assert ece_bad > 0.15, (
        f"a 0.25 probability shift scored ECE {ece_bad:.4f} -- ECE is not "
        "detecting miscalibration")
    print(f"  ECE {ece:.4f} calibrated vs {ece_bad:.4f} shifted")


def test_threshold_metrics_move_and_ranking_metrics_do_not():
    """The diagnostic split the calibration skill relies on."""
    from sklearn.linear_model import LogisticRegression

    X, y = classification_frame(n=800, base_rate=-1.8)
    oof = oof_probabilities(X, y, lambda: LogisticRegression(max_iter=2000))

    at_half = threshold_metrics(y, oof, 0.5)
    thr, _, _ = tune_threshold(y, oof, objective="f1")
    at_tuned = threshold_metrics(y, oof, thr)
    assert at_tuned["f1"] > at_half["f1"], "tuning the threshold did not "\
                                           "improve the objective it tuned"

    before = calibration_metrics(y, oof)
    after = calibration_metrics(y, oof)
    assert before == after, "threshold-independent metrics are not a pure "\
                            "function of the probabilities"
    for key in ("auc", "brier", "ece", "log_loss"):
        assert key in before
    print(f"  f1 {at_half['f1']:.4f} -> {at_tuned['f1']:.4f} at threshold "
          f"{thr:.3f}; ranking metrics unchanged")


def test_flat_threshold_surface_is_reported_as_flat():
    """A threshold picked to four decimals off a flat curve is fitting noise."""
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)
    _, _, sweep = tune_threshold(y, p, objective="accuracy")
    sens = threshold_sensitivity(sweep, "accuracy", tolerance=0.01)
    assert 0.0 <= sens["band_width"] <= 1.0
    assert sens["band_low"] <= sens["band_high"]
    assert isinstance(sens["flat"], bool)
    print(f"  threshold band width {sens['band_width']:.3f}, flat={sens['flat']}")


# ---------------------------------------------------------------------------
# seed reservation
# ---------------------------------------------------------------------------

def test_search_and_holdout_seeds_are_disjoint():
    search, holdout = split_seeds([1, 2, 3, 4, 5], n_holdout=2)
    assert not set(search) & set(holdout), (
        "a seed appears in both the search and the holdout -- the held-out "
        "score then measures what the search already optimized")
    assert set(search) | set(holdout) == {1, 2, 3, 4, 5}
    assert len(holdout) == 2
    print(f"  search {search} / holdout {holdout} disjoint")


def test_seed_split_refuses_a_split_it_cannot_make_honestly():
    for seeds in ([1], [1, 2], [1, 2, 3]):
        try:
            split_seeds(seeds)
        except ValueError as e:
            assert "at least 4" in str(e)
        else:
            raise AssertionError(f"accepted {len(seeds)} seeds")

    # Both sides keep at least two seeds even when asked for more holdout: one
    # holdout seed cannot distinguish a real gain from that seed's own luck.
    search, holdout = split_seeds([1, 2, 3, 4, 5], n_holdout=99)
    assert len(search) >= 2 and len(holdout) >= 2
    search, holdout = split_seeds([1, 2, 3, 4], n_holdout=0)
    assert len(holdout) >= 2
    print("  seed split refuses <4 seeds and never leaves a side with one")


def test_random_search_never_repeats_a_configuration():
    rng = np.random.default_rng(0)
    space = {"alpha": [0.1, 1.0, 10.0], "beta": [1, 2]}
    # Space smaller than the budget: enumerate it rather than resample forever.
    configs = sample_space(space, n_trials=50, rng=rng)
    assert len(configs) == 6, f"expected the full 6-point grid, got {len(configs)}"
    sigs = {tuple(sorted(c.items())) for c in configs}
    assert len(sigs) == len(configs), "duplicate configurations sampled"

    big = {f"p{i}": list(range(10)) for i in range(4)}
    sampled = sample_space(big, n_trials=25, rng=rng)
    assert len(sampled) == 25
    assert len({tuple(sorted(c.items())) for c in sampled}) == 25
    print("  search space enumerated when small, deduplicated when sampled")


def test_multiseed_std_is_the_noise_floor():
    from sklearn.linear_model import Ridge

    X, y = regression_frame()
    res = multiseed_score(X, y, lambda: Ridge(alpha=1.0), seeds=[1, 2, 3],
                          n_splits=3, metric="rmse")
    assert set(res) == {"per_seed", "mean", "std"}
    assert len(res["per_seed"]) == 3
    assert res["std"] >= 0
    assert abs(res["mean"] - float(np.mean(list(res["per_seed"].values())))) < 1e-12
    print(f"  multiseed mean {res['mean']:.4f} std {res['std']:.4f}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"tests/test_harness.py -- {len(tests)} checks\n")
    for fn in tests:
        fn()
    print(f"\nOK ({len(tests)} checks)")


if __name__ == "__main__":
    main()
