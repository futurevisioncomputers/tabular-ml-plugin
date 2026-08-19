"""
Registry of tabular foundation models: capacity limits, licensing, and the
preprocessing contract each one requires.

WHY THIS EXISTS

Foundation models are not interchangeable with the rest of the model zoo and
they are not interchangeable with each other. Three things differ per model and
all three break silently:

  1. A context/row cap. Feeding more rows than the cap does not error -- the
     wrapper subsamples, or the model degrades, and you get a number that looks
     fine and means something different than you think.
  2. A license on the WEIGHTS that is separate from the license on the code.
     Several of these are non-commercial. The code being Apache-2.0 says nothing
     about whether you can ship predictions from the default checkpoint.
  3. A preprocessing contract. These models want raw-ish columns with
     categorical positions declared. One-hot encoding and scaling them -- the
     standard path for every other model in this plugin -- is using them wrong.

SOURCE OF THE NUMBERS

Row caps and HPO support are transcribed from TALENT's `method_registry.py`
(github.com/LAMDA-Tabular/TALENT), which documents them as derived from the
actual `__init__` asserts in each method wrapper rather than from paper claims.
TabFM's estimator defaults are transcribed from the constructor signature in
google-research/tabfm `tabfm/src/classifier_and_regressor.py`.

VERIFICATION STATUS

Row caps, HPO flags, install commands and preprocessing contracts were read from
those two repositories. Licence fields are asserted ONLY where a repository
states the terms explicitly; everything else is `LicenseClass.CHECK` with a URL,
because guessing a license is worse than admitting you have not read it. None of
these models were benchmarked when this file was written -- no accuracy claim
here is measured. Run them through the multi-seed harness like anything else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LicenseClass(str, Enum):
    """What the WEIGHTS permit -- not what the source code permits."""

    #: Commercial use permitted, possibly with an attribution requirement.
    PERMISSIVE = "permissive"
    #: Weights explicitly forbid commercial or production use.
    NON_COMMERCIAL = "non_commercial"
    #: Not verified here. Read the upstream terms before shipping anything.
    CHECK = "check"


class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass(frozen=True)
class FMSpec:
    name: str
    install: str
    #: Soft cap on training rows used as context. None means no documented cap
    #: (retrieval-based or streaming context), NOT "unlimited in practice".
    train_row_limit: Optional[int]
    #: Soft cap on feature count, where the model documents one.
    feature_limit: Optional[int]
    tasks: tuple
    license_class: LicenseClass
    license_note: str
    supports_hpo: bool
    gpu_required: bool
    upstream: str
    notes: str = ""

    def handles(self, task: Task) -> bool:
        return task in self.tasks


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_BOTH = (Task.CLASSIFICATION, Task.REGRESSION)
_CLF = (Task.CLASSIFICATION,)

REGISTRY: dict[str, FMSpec] = {
    "tabpfn_v1": FMSpec(
        name="tabpfn_v1",
        install="pip install -U tabpfn",
        train_row_limit=1_000,
        feature_limit=100,
        tasks=_CLF,
        license_class=LicenseClass.CHECK,
        license_note="Superseded by v2+. Check current Prior Labs terms.",
        supports_hpo=True,
        gpu_required=False,
        upstream="https://github.com/PriorLabs/TabPFN",
        notes="Original 2022 model. Classification only, 1k rows. Kept for "
              "reproducing older results; prefer v2 or later for new work.",
    ),
    "tabpfn_v2": FMSpec(
        name="tabpfn_v2",
        install="pip install -U tabpfn",
        train_row_limit=10_000,
        feature_limit=500,
        tasks=_BOTH,
        license_class=LicenseClass.PERMISSIVE,
        license_note="Prior Labs License -- Apache 2.0 plus an attribution "
                     "requirement. Commercial use permitted with attribution. "
                     "This is the one commercially-usable checkpoint in the "
                     "TabPFN line; v2.5 and v3 are not.",
        supports_hpo=False,
        gpu_required=False,
        upstream="https://github.com/PriorLabs/TabPFN",
        notes="Nature 2025. The default choice on small data when the work is "
              "commercial, because later checkpoints are not licensed for it.",
    ),
    "tabpfn_v2_5": FMSpec(
        name="tabpfn_v2_5",
        install="pip install -U 'tabpfn>=8.0.0'",
        train_row_limit=50_000,
        feature_limit=2_000,
        tasks=_BOTH,
        license_class=LicenseClass.NON_COMMERCIAL,
        license_note="Non-commercial weights. Research and competitions only.",
        supports_hpo=False,
        gpu_required=True,
        upstream="https://arxiv.org/abs/2511.08667",
        notes="Nov 2025 intermediate release between v2 and v3.",
    ),
    "tabpfn_v3": FMSpec(
        name="tabpfn_v3",
        install="pip install -U 'tabpfn>=8.0.0'",
        train_row_limit=1_000_000,
        feature_limit=200,
        tasks=_BOTH,
        license_class=LicenseClass.NON_COMMERCIAL,
        license_note="Non-commercial weights. Research and competitions only.",
        supports_hpo=False,
        gpu_required=True,
        upstream="https://github.com/PriorLabs/TabPFN",
        notes="~1M-row context, 160-class support, optional thinking mode. "
              "Note the feature cap is LOWER than v2.5 despite the far larger "
              "row cap -- wide tables may need v2.5 or feature selection.",
    ),
    "tabpfn_real": FMSpec(
        name="tabpfn_real",
        install="see upstream",
        train_row_limit=10_000,
        feature_limit=500,
        tasks=_CLF,
        license_class=LicenseClass.CHECK,
        license_note="Derived from TabPFN v2 by continued pre-training on real "
                     "datasets; terms not verified here.",
        supports_hpo=False,
        gpu_required=True,
        upstream="https://arxiv.org/abs/2507.03971",
        notes="Real-TabPFN. Classification.",
    ),
    "tabicl": FMSpec(
        name="tabicl",
        install="pip install -U tabicl",
        train_row_limit=500_000,
        feature_limit=None,
        tasks=_CLF,
        license_class=LicenseClass.CHECK,
        license_note="Not verified here -- read the soda-inria repo terms.",
        supports_hpo=True,
        gpu_required=True,
        upstream="https://github.com/soda-inria/tabicl",
        notes="TabICL v1 (ICML 2025). Classification only -- use tabicl_v2 for "
              "regression.",
    ),
    "tabicl_v2": FMSpec(
        name="tabicl_v2",
        install="pip install -U 'tabicl>=2.0.0'",
        train_row_limit=1_000_000,
        feature_limit=None,
        tasks=_BOTH,
        license_class=LicenseClass.CHECK,
        license_note="Not verified here -- read the soda-inria repo terms.",
        supports_hpo=False,
        gpu_required=True,
        upstream="https://github.com/soda-inria/tabicl",
        notes="ICML 2026. Adds regression with native quantile regression.",
    ),
    "tabdpt": FMSpec(
        name="tabdpt",
        install="pip install -U tabdpt",
        train_row_limit=None,
        feature_limit=None,
        tasks=_BOTH,
        license_class=LicenseClass.CHECK,
        license_note="Not verified here -- read the Layer 6 AI repo terms.",
        supports_hpo=True,
        gpu_required=True,
        upstream="https://github.com/layer6ai-labs/TabDPT-inference",
        notes="In-context learning plus retrieval, so there is no fixed "
              "context cap. Absence of a cap is not a promise of accuracy at "
              "any size -- measure it.",
    ),
    "tabfm": FMSpec(
        name="tabfm",
        install="pip install -U 'tabfm[pytorch]'   # or tabfm[jax] / tabfm[jax,cuda]",
        train_row_limit=None,
        feature_limit=500,
        tasks=_BOTH,
        license_class=LicenseClass.NON_COMMERCIAL,
        license_note="Source is Apache-2.0 but the default pretrained weights "
                     "downloaded by tabfm_v1_0_0.load() are under "
                     "'tabfm-non-commercial-v1.0'. The README states commercial "
                     "or production use of those weights is NOT permitted.",
        supports_hpo=False,
        gpu_required=False,
        upstream="https://github.com/google-research/tabfm",
        notes="Google Research zero-shot ICL model, sklearn-compatible. "
              "Python >= 3.11. Estimator defaults: n_estimators=32, "
              "max_num_features=500, max_num_rows=None, batch_size=1. "
              "The README FAQ describes a 100-row default context while the "
              "constructor default for max_num_rows is None -- do not rely on "
              "either, pass max_num_rows explicitly so the context size is a "
              "decision you made rather than one you inherited.",
    ),
    "mitra": FMSpec(
        name="mitra",
        install="see upstream (ships via AutoGluon)",
        train_row_limit=10_000,
        feature_limit=None,
        tasks=_BOTH,
        license_class=LicenseClass.CHECK,
        license_note="Not verified here -- read the Amazon Science / AutoGluon "
                     "terms.",
        supports_hpo=False,
        gpu_required=True,
        upstream="https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models",
        notes="Cross-attention ICL with O(n_train * n_test) memory -- memory "
              "grows with the TEST set too, unlike most models here. Batch "
              "predictions on large test sets.",
    ),
    "limix": FMSpec(
        name="limix",
        install="see upstream",
        train_row_limit=None,
        feature_limit=None,
        tasks=_BOTH,
        license_class=LicenseClass.CHECK,
        license_note="Not verified here -- read the LimiX repo terms.",
        supports_hpo=True,
        gpu_required=True,
        upstream="https://arxiv.org/abs/2509.03505",
        notes="Unified architecture also covering imputation and causal "
              "inference, not only prediction.",
    ),
}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass
class Eligibility:
    spec: FMSpec
    eligible: bool
    reasons: list = field(default_factory=list)


def check_fit(n_rows: int, n_features: int, task: Task,
              commercial: bool, gpu_available: bool = True) -> list:
    """Score every registry entry against a concrete problem.

    Returns Eligibility records for ALL models, eligible ones first, each
    carrying the reasons it was excluded. Reporting the exclusions matters as
    much as the shortlist -- "TabPFN v3 was excluded because the work is
    commercial" is actionable, while a bare shortlist is not.

    `commercial` should reflect whether predictions will be shipped in a
    product. When in doubt, pass True: the cost of a false positive here is a
    license violation, and the cost of a false negative is a slightly shorter
    shortlist.
    """
    out = []
    for spec in REGISTRY.values():
        reasons = []

        if not spec.handles(task):
            reasons.append(f"does not support {task.value}")

        if commercial and spec.license_class is LicenseClass.NON_COMMERCIAL:
            reasons.append(f"weights are non-commercial: {spec.license_note}")
        elif commercial and spec.license_class is LicenseClass.CHECK:
            reasons.append(
                "license not verified -- read the upstream terms before "
                f"commercial use: {spec.upstream}")

        if spec.train_row_limit is not None and n_rows > spec.train_row_limit:
            over = n_rows / spec.train_row_limit
            reasons.append(
                f"{n_rows} rows exceeds the {spec.train_row_limit} row cap "
                f"({over:.1f}x) -- usable only on a subsample")

        if spec.feature_limit is not None and n_features > spec.feature_limit:
            reasons.append(
                f"{n_features} features exceeds the {spec.feature_limit} "
                f"feature cap -- needs feature selection first")

        if spec.gpu_required and not gpu_available:
            reasons.append("needs a GPU in practice; CPU will be impractically slow")

        out.append(Eligibility(spec=spec, eligible=not reasons, reasons=reasons))

    out.sort(key=lambda e: (not e.eligible, len(e.reasons), e.spec.name))
    return out


def shortlist(n_rows: int, n_features: int, task: Task, commercial: bool,
              gpu_available: bool = True) -> list:
    """Just the eligible specs, for callers that want the names only."""
    return [e.spec for e in check_fit(n_rows, n_features, task, commercial,
                                      gpu_available) if e.eligible]


def format_report(results: list) -> str:
    lines = []
    ok = [e for e in results if e.eligible]
    no = [e for e in results if not e.eligible]

    lines.append(f"ELIGIBLE ({len(ok)})")
    if not ok:
        lines.append("  none -- see exclusions below")
    for e in ok:
        cap = e.spec.train_row_limit
        cap_s = f"{cap:,} rows" if cap else "no documented row cap"
        lines.append(f"  {e.spec.name:14s} {cap_s:26s} "
                     f"hpo={'yes' if e.spec.supports_hpo else 'no':3s} "
                     f"license={e.spec.license_class.value}")

    lines.append(f"\nEXCLUDED ({len(no)})")
    for e in no:
        lines.append(f"  {e.spec.name}")
        for r in e.reasons:
            lines.append(f"      - {r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Which tabular foundation models fit this problem?")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--features", type=int, required=True)
    p.add_argument("--task", choices=["classification", "regression"],
                   required=True)
    p.add_argument("--commercial", action="store_true",
                   help="predictions will be shipped in a product; excludes "
                        "non-commercial and unverified weights")
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()

    results = check_fit(
        n_rows=args.rows,
        n_features=args.features,
        task=Task(args.task),
        commercial=args.commercial,
        gpu_available=not args.no_gpu,
    )
    print(format_report(results))
    print("\nEligibility is a filter, not a ranking. Nothing here has been "
          "measured on your data -- run the shortlist through the multi-seed "
          "harness before believing any of it.")


if __name__ == "__main__":
    main()
