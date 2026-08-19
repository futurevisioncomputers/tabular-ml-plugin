---
description: Tune hyperparameters and report whether the gain survives held-out seeds
argument-hint: "[model to tune, e.g. lightgbm or ridge]"
---

Run a hyperparameter search and report which part of the result is real.

Model: `$ARGUMENTS`. If not given, tune whichever model currently leads the
comparison, and say which that is.

Use the `hyperparameter-tuner` agent if subagents are available; otherwise
follow the `tabular-model-selection` skill and its `scripts/tune_search.py`,
which enforces the seed split.

Before searching:

- Confirm features and blending are settled — tuning yields less than either and
  has to be redone when they change
- Confirm the model has a meaningful search space. Foundation models from TabPFN
  v2 onward, TabICL v2, Mitra and TabFM are used as-is; searching over them is
  wasted compute.
- State the objective metric explicitly. Tuning for accuracy and for AUC select
  different configurations.
- **Reserve seeds the search never sees.** They are the only honest estimate of
  the winner, and they cannot be reserved afterwards.

Then run a random or Optuna search over ranges centered on stronger
regularization than the library defaults if the dataset is small.

Report:

- The search's best score and the held-out-seed score **separately** — the gap
  is the search's optimism
- The held-out improvement against the noise floor. If it is smaller, say
  tuning found nothing and recommend keeping the defaults.
- The top several configurations, not only the winner, so a flat surface is
  visible
- The search budget, objective metric, search seeds, and held-out seeds

Prefer a configuration winning on most held-out seeds over the best mean, and
the simpler configuration when results tie.

Finish with an experiment log row, including when the result is negative.
