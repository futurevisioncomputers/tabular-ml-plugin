---
description: Audit the pipeline for data leakage and validation errors
---

Review the current tabular ML pipeline for leakage before its scores are trusted.

Use the `leakage-auditor` agent if subagents are available; otherwise follow the
leakage section of the `tabular-validation` skill.

Trace the actual order of operations in the feature engineering, CV, and
prediction code — not what the comments or function names claim. Check:

- Every learned transform (imputers, scalers, encoders, target encoding, feature
  selection) is fit inside the fold, not before the split
- Target encoding statistics come from the training fold only
- The target transform matches the metric and is inverted exactly once
- Train and test pass through identical logic with train-derived statistics
- Row-dropping (outlier removal) happens on train only
- The splitter matches the data structure — GroupKFold for grouped data, and a
  chronological split where the real evaluation is chronological

Report findings by severity (LEAK / RISK / NOTE) with location, what actually
happens, and a suggested fix. Describe fixes; do not apply them.

If the pipeline is clean, say so and list what was checked. Do not invent
findings to appear thorough — and distinguish a genuine leak from an ordinary
CV-to-holdout gap, which is usually distribution difference rather than leakage.
