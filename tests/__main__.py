"""Run every test module in one pass: `python -m tests`

Plain asserts and no pytest dependency, matching the rest of the repo. Each
module is also runnable on its own (`python -m tests.test_manifest`) so a
failure can be re-run in isolation without waiting for the others.

Exits non-zero on the first failure so this is usable as a pre-commit gate.
"""

from __future__ import annotations

import importlib
import sys
import traceback

MODULES = [
    "tests.test_manifest",     # docs/config vs code -- fastest, fails first
    "tests.test_fm_registry",  # registry invariants, no model packages needed
    "tests.test_harness",      # numerics; needs sklearn/pandas
]


def main() -> int:
    failed = []
    for name in MODULES:
        print(f"{'=' * 70}\n{name}\n{'=' * 70}")
        try:
            importlib.import_module(name).main()
        except AssertionError as e:
            # An assertion is an expected failure mode: print the message the
            # test author wrote rather than a wall of frames.
            print(f"\nFAILED: {e}\n")
            failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
        print()

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"ALL PASS ({len(MODULES)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
