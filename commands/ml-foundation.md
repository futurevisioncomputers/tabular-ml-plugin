---
description: Shortlist and evaluate tabular foundation models for this dataset
argument-hint: "[model name, e.g. tabpfn_v2 or tabfm]"
---

Work out which tabular foundation models fit this problem, and measure the ones
that do.

Model: `$ARGUMENTS` if given; otherwise shortlist from the dataset's shape.

Use the `foundation-model-advisor` agent if subagents are available; otherwise
follow the `tabular-foundation-models` skill directly.

Establish two things before running anything:

1. **Will predictions be shipped commercially?** TabFM v1.0.0, TabPFN v2.5 and
   TabPFN v3 have non-commercial weights regardless of their source license. Ask
   rather than assume, and treat an unclear answer as commercial.
2. **Row and feature count.** Caps range from 1,000 to 1,000,000 rows and do not
   move together with feature caps.

Then:

- `scripts/fm_registry.py` — shortlist, with the reason each model was excluded
- `scripts/fm_preflight.py` — packages, Python version, GPU, weight gating,
  license verdict. Report blockers instead of working around them.
- `scripts/fm_evaluate.py` — multi-seed scoring on the foundation-model data
  path (raw categoricals, no scaling, no imputation), with the context
  subsample drawn inside each fold

Report against the existing baseline in the usual terms — mean, across-seed
std, seeds improved — plus three things a score alone hides: the context size
actually used, the license verdict, and prediction cost relative to the current
model.

Do not substitute a different model when the requested one is blocked. Say what
would unblock it.
