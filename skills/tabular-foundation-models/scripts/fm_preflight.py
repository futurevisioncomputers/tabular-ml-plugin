"""
Preflight checks for tabular foundation models.

Every failure mode here costs real time when it surfaces halfway through an
experiment instead of before it: a missing package, a weight download that wants
a browser, a CPU-only box asked to run a GPU model, or a non-commercial
checkpoint discovered after the model is already in a shipping pipeline.

This module answers "can I run this, and am I allowed to" before anything is
fitted. It never installs, downloads, or accepts a license on the user's behalf.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
from dataclasses import dataclass, field

from fm_registry import REGISTRY, LicenseClass


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def gpu_available() -> bool:
    """True if a CUDA device is visible to torch.

    Returns False rather than raising when torch is absent -- a machine without
    torch is a machine without a usable GPU for these models.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def python_at_least(major: int, minor: int) -> bool:
    import sys
    return sys.version_info >= (major, minor)


def package_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


#: Which import name to probe for each registry entry. Several models ship
#: under a package name that differs from the registry key.
IMPORT_NAME = {
    "tabpfn_v1": "tabpfn",
    "tabpfn_v2": "tabpfn",
    "tabpfn_v2_5": "tabpfn",
    "tabpfn_v3": "tabpfn",
    "tabpfn_real": "tabpfn",
    "tabicl": "tabicl",
    "tabicl_v2": "tabicl",
    "tabdpt": "tabdpt",
    "tabfm": "tabfm",
    "mitra": "autogluon",
    "limix": "limix",
}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class Preflight:
    model: str
    runnable: bool
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)

    def render(self) -> str:
        head = f"{self.model}: {'READY' if self.runnable else 'BLOCKED'}"
        lines = [head]
        for b in self.blockers:
            lines.append(f"  BLOCKER  {b}")
        for w in self.warnings:
            lines.append(f"  WARNING  {w}")
        for i in self.info:
            lines.append(f"  info     {i}")
        return "\n".join(lines)


def preflight(model: str, commercial: bool = False,
              n_rows: int | None = None,
              n_features: int | None = None) -> Preflight:
    """Check one model end to end. Does not import the model's weights."""
    if model not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        return Preflight(model=model, runnable=False,
                         blockers=[f"unknown model. Known: {known}"])

    spec = REGISTRY[model]
    res = Preflight(model=model, runnable=True)

    # --- licence. Checked first, because a licence blocker makes every other
    # --- check irrelevant.
    if commercial and spec.license_class is LicenseClass.NON_COMMERCIAL:
        res.blockers.append(f"licence: {spec.license_note}")
    elif commercial and spec.license_class is LicenseClass.CHECK:
        res.blockers.append(
            f"licence not verified for commercial use. Read {spec.upstream} "
            "and re-run with the answer, rather than assuming.")
    elif spec.license_class is LicenseClass.NON_COMMERCIAL:
        res.warnings.append(
            "non-commercial weights -- fine for research and competitions, not "
            "for anything shipped. " + spec.license_note)
    elif spec.license_class is LicenseClass.CHECK:
        res.info.append(f"licence unverified here; upstream: {spec.upstream}")
    else:
        res.info.append(f"licence: {spec.license_note}")

    # --- package
    import_name = IMPORT_NAME.get(model, model)
    if not package_installed(import_name):
        res.blockers.append(
            f"package `{import_name}` not installed. Install with: {spec.install}")
    else:
        ver = installed_version(import_name)
        res.info.append(f"`{import_name}` installed"
                        + (f" (version {ver})" if ver else ""))

    # --- python floor. TabFM pins >= 3.11; the rest are looser.
    if model == "tabfm" and not python_at_least(3, 11):
        res.blockers.append("TabFM requires Python >= 3.11")

    # --- hardware
    has_gpu = gpu_available()
    if spec.gpu_required and not has_gpu:
        res.warnings.append(
            "no CUDA device visible. This model is GPU-bound in practice; "
            "expect it to be slow enough that the multi-seed protocol becomes "
            "impractical.")
    elif not has_gpu:
        res.info.append("no CUDA device -- usable on CPU at small sizes")

    # --- capacity
    if n_rows is not None and spec.train_row_limit and n_rows > spec.train_row_limit:
        res.warnings.append(
            f"{n_rows} rows exceeds the {spec.train_row_limit} row cap. The "
            "context will be subsampled. Fix the subsample seed and report the "
            "sample size alongside the score -- a score from an unstated "
            "subsample is not reproducible.")
    if n_features is not None and spec.feature_limit and n_features > spec.feature_limit:
        res.warnings.append(
            f"{n_features} features exceeds the {spec.feature_limit} feature "
            "cap. Select features before fitting; letting the wrapper choose "
            "silently makes the comparison against other models unfair.")

    # --- weight gating
    if import_name == "tabpfn" and not os.environ.get("TABPFN_TOKEN"):
        res.info.append(
            "TABPFN_TOKEN is not set. First use may open a browser to accept "
            "the licence, which fails in headless environments. Set "
            "TABPFN_TOKEN from https://ux.priorlabs.ai to avoid that.")
    if model == "tabfm":
        res.info.append(
            "tabfm_v1_0_0.load() downloads weights from Hugging Face on first "
            "use; needs network access and disk space in the HF cache.")
        if shutil.which("bazel") is None:
            res.info.append("bazel not found -- only needed to run TabFM's own "
                            "test suite, not to use the model")

    res.runnable = not res.blockers
    return res


def preflight_all(models: list, **kw) -> list:
    return [preflight(m, **kw) for m in models]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Check whether a tabular foundation model can and may be run")
    p.add_argument("models", nargs="*", default=sorted(REGISTRY),
                   help="model names; defaults to all")
    p.add_argument("--commercial", action="store_true",
                   help="predictions will be shipped; turns licence warnings "
                        "into blockers")
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--features", type=int, default=None)
    args = p.parse_args()

    results = preflight_all(args.models, commercial=args.commercial,
                            n_rows=args.rows, n_features=args.features)
    for r in results:
        print(r.render())
        print()

    ready = [r.model for r in results if r.runnable]
    print(f"runnable now: {', '.join(ready) if ready else 'none'}")


if __name__ == "__main__":
    main()
