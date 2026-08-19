---
description: Scaffold a new tabular ML project and profile the dataset
argument-hint: "[path-to-train.csv] [target-column]"
---

Set up a new tabular ML project and produce a first read of the data.

Arguments: `$ARGUMENTS` — path to the training CSV, then the target column name.
If either is missing, ask before proceeding rather than guessing.

Steps:

1. Scaffold the project layout with `project_scaffold.py` from the
   `tabular-ml-workflow` skill (data/, notebooks/, src/, submissions/,
   experiments/ plus an experiment log).
2. Profile the dataset with `profile_dataset.py` from `tabular-data-profiling`.
   Report shape, missingness split by train/test, near-constant columns,
   train/test category mismatches, and target skew.
3. Establish the noise floor: run `multiseed_cv.py` from `tabular-validation`
   for a baseline across several seeds, and report the across-seed standard
   deviation. Every later change gets compared to this number, so get it early.
4. Summarize what needs human input — which columns need the data dictionary to
   classify, and which nulls are ambiguous between structural and genuine.

Do not start cleaning or modeling in this command. The goal is an oriented
starting point and an honest list of what is still unknown.
