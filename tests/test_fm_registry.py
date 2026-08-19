"""The foundation-model registry is data, and wrong data here is expensive.

Two of its fields carry consequences beyond a bad score. A missing or wrong
`license_class` can put non-commercial weights into a shipping product. A wrong
`train_row_limit` silently changes how much of the training set the model
actually saw, which makes the reported number incomparable to every other run.

These tests check the invariants that keep the filter honest: that commercial
work never sees non-commercial weights, that an unverified licence is treated
as unsafe rather than permissive, and that the adapter table and the registry
have not drifted apart.

Run: python -m tests.test_fm_registry   (plain asserts, no pytest dep)
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "skills", "tabular-foundation-models", "scripts")
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

from fm_registry import (  # noqa: E402
    REGISTRY, FMSpec, LicenseClass, Task, check_fit, shortlist, format_report,
)
from fm_preflight import IMPORT_NAME, preflight  # noqa: E402
import fm_evaluate  # noqa: E402


def test_registry_entries_are_well_formed():
    assert REGISTRY, "registry is empty"
    for key, spec in REGISTRY.items():
        assert isinstance(spec, FMSpec)
        assert spec.name == key, f"{key}: name field is {spec.name!r}"
        assert spec.tasks, f"{key}: supports no task at all"
        assert spec.upstream.startswith("http"), f"{key}: no upstream URL"
        assert spec.license_note, f"{key}: no licence note"
        for cap in (spec.train_row_limit, spec.feature_limit):
            assert cap is None or cap > 0, f"{key}: non-positive cap {cap}"
    print(f"  {len(REGISTRY)} registry entries well formed")


def test_unverified_licence_is_never_treated_as_permissive():
    """CHECK means 'nobody read the terms', which is not the same as 'allowed'.

    Getting this backwards is the one registry error that has consequences
    outside the experiment.
    """
    for key, spec in REGISTRY.items():
        if spec.license_class is LicenseClass.PERMISSIVE:
            assert "commercial use permitted" in spec.license_note.lower(), (
                f"{key}: marked PERMISSIVE but the note does not say so. Do "
                "not assert a licence you have not read.")
    permissive = [k for k, s in REGISTRY.items()
                  if s.license_class is LicenseClass.PERMISSIVE]
    print(f"  permissive entries justified in-note: {permissive}")


def test_commercial_filter_excludes_unsafe_weights():
    results = {e.spec.name: e for e in
               check_fit(1000, 20, Task.CLASSIFICATION, commercial=True)}
    for key, spec in REGISTRY.items():
        if spec.license_class in (LicenseClass.NON_COMMERCIAL, LicenseClass.CHECK):
            assert not results[key].eligible, (
                f"{key} is {spec.license_class.value} but survived a "
                "commercial filter")
    eligible = [n for n, e in results.items() if e.eligible]
    assert eligible, "commercial filter excluded everything -- no usable path"
    print(f"  commercial filter leaves only: {eligible}")


def test_non_commercial_filter_is_wider():
    """The filter must actually loosen when the work is not commercial."""
    strict = len(shortlist(1000, 20, Task.CLASSIFICATION, commercial=True))
    loose = len(shortlist(1000, 20, Task.CLASSIFICATION, commercial=False))
    assert loose > strict, (
        f"non-commercial shortlist ({loose}) is not wider than commercial "
        f"({strict}) -- the licence dimension is doing nothing")
    print(f"  shortlist: {strict} commercial vs {loose} non-commercial")


def test_row_cap_excludes_and_reports_why():
    tiny = {e.spec.name: e for e in
            check_fit(500, 10, Task.CLASSIFICATION, commercial=False)}
    huge = {e.spec.name: e for e in
            check_fit(5_000_000, 10, Task.CLASSIFICATION, commercial=False)}
    capped = [k for k, s in REGISTRY.items() if s.train_row_limit]
    for key in capped:
        assert huge[key].eligible is False, f"{key}: 5M rows passed its cap"
        assert any("row cap" in r for r in huge[key].reasons), (
            f"{key}: excluded on size but the reason does not say so")
    # and the same models are fine at 500 rows, licence permitting
    assert any(tiny[k].eligible for k in capped), (
        "no capped model is eligible at 500 rows -- the cap logic is inverted")
    print(f"  {len(capped)} capped models excluded at 5M rows, with reasons")


def test_task_filter():
    clf_only = [k for k, s in REGISTRY.items() if not s.handles(Task.REGRESSION)]
    reg = {e.spec.name: e for e in
           check_fit(500, 10, Task.REGRESSION, commercial=False)}
    for key in clf_only:
        assert not reg[key].eligible, f"{key} is classification-only but "\
                                      "survived a regression filter"
    print(f"  classification-only models excluded from regression: {clf_only}")


def test_excluded_models_always_carry_a_reason():
    """A bare shortlist is not actionable; the exclusions are the useful half."""
    for e in check_fit(50_000, 900, Task.REGRESSION, commercial=True):
        if not e.eligible:
            assert e.reasons, f"{e.spec.name}: excluded with no reason given"
        else:
            assert not e.reasons, f"{e.spec.name}: eligible but carries reasons"
    print("  every exclusion carries a reason")


def test_eligible_models_sort_first():
    results = check_fit(1000, 20, Task.CLASSIFICATION, commercial=False)
    seen_ineligible = False
    for e in results:
        if not e.eligible:
            seen_ineligible = True
        else:
            assert not seen_ineligible, "an eligible model sorted after an "\
                                        "ineligible one"
    print("  eligible models sort first")


def test_preflight_covers_every_registry_entry():
    for key in REGISTRY:
        assert key in IMPORT_NAME, (
            f"{key}: no IMPORT_NAME entry, so preflight probes the wrong "
            "package name and reports it missing when it is installed")
    print(f"  IMPORT_NAME covers all {len(REGISTRY)} entries")


def test_preflight_blocks_rather_than_crashes_on_missing_packages():
    """Nothing here may raise on a machine without the model installed."""
    for key in REGISTRY:
        res = preflight(key, commercial=False, n_rows=1000, n_features=10)
        assert isinstance(res.runnable, bool)
        assert res.render()
        if not res.runnable:
            assert res.blockers, f"{key}: not runnable but no blocker listed"
    res = preflight("no-such-model")
    assert not res.runnable and res.blockers
    print("  preflight degrades to blockers, never raises")


def test_adapters_match_the_registry():
    for key in fm_evaluate.ADAPTERS:
        assert key in REGISTRY, f"adapter {key!r} has no registry entry"
    missing = sorted(set(REGISTRY) - set(fm_evaluate.ADAPTERS))
    # Models without an adapter are a deliberate choice, not an oversight --
    # they ship through their own frameworks. The error message must say so.
    try:
        fm_evaluate.make_model_fn(missing[0], Task.CLASSIFICATION, [])
    except ValueError as e:
        assert "no adapter" in str(e).lower()
    else:
        raise AssertionError(f"{missing[0]} has no adapter but did not raise")
    print(f"  adapters {sorted(fm_evaluate.ADAPTERS)} all in registry; "
          f"no-adapter models ({missing}) raise clearly")


def test_task_support_is_enforced_by_make_model_fn():
    clf_only = [k for k, s in REGISTRY.items()
                if not s.handles(Task.REGRESSION) and k in fm_evaluate.ADAPTERS]
    for key in clf_only:
        try:
            fm_evaluate.make_model_fn(key, Task.REGRESSION, [])
        except ValueError as e:
            assert "regression" in str(e)
        else:
            raise AssertionError(f"{key}: accepted a regression task")
    print(f"  make_model_fn rejects unsupported tasks: {clf_only}")


def test_report_renders():
    text = format_report(check_fit(1450, 78, Task.REGRESSION, commercial=True))
    assert "ELIGIBLE" in text and "EXCLUDED" in text
    # ASCII only: the console codepage on Windows mangles em dashes into
    # replacement characters, which makes the licence text hard to read at
    # exactly the moment it matters.
    assert text.isascii(), "report contains non-ASCII characters"
    print("  report renders, ASCII-clean")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"tests/test_fm_registry.py -- {len(tests)} checks\n")
    for fn in tests:
        fn()
    print(f"\nOK ({len(tests)} checks)")


if __name__ == "__main__":
    main()
