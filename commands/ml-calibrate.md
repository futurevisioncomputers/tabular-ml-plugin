---
description: Audit classifier probability calibration and decision thresholds
---

Check whether this classifier's predicted probabilities mean what they claim,
and whether its decision threshold was chosen legitimately.

Use the `calibration-auditor` agent if subagents are available; otherwise follow
the `tabular-validation` skill and its `scripts/calibration_report.py`.

First establish whether the output is used as a probability or only as a
ranking. If it only ever sorts — a shortlist, a queue, an AUC-scored
submission — say calibration does not matter here and stop.

Otherwise check:

- **Where the decision threshold came from.** A threshold swept against the test
  set is fitted to it, and the reported accuracy and F1 are not held-out
  numbers. Report as `LEAK`. The legitimate version tunes on a validation split
  or out-of-fold predictions, then applies the fixed threshold to test once.
- **Which metrics that affects.** Accuracy, F1, precision and recall move with
  the threshold; AUC, log loss, Brier and ECE do not. Strong AUC alongside
  accuracy that collapses on new data is the signature.
- **Whether 0.5 was chosen or inherited**, particularly on imbalanced data.
- **Brier score, ECE, and a reliability table** binning predicted probability
  against observed frequency. Report the direction of any miscalibration.
- **Whether resampling or class weighting explains it** — SMOTE, undersampling
  and `class_weight="balanced"` shift the probability scale off the base rate by
  construction.

Report findings by severity (LEAK / RISK / NOTE) with location, what happens,
and a suggested fix — described, not applied. Report the calibration numbers
even when there is no finding.

Any recommended calibrator (Platt or isotonic) must be fit on data the model did
not train on, and evaluated on data neither the model nor the calibrator saw.
