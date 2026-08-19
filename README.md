# Tabular ML Workflow

A disciplined end-to-end workflow for supervised learning on tabular data.

## The problem it addresses

Most tabular ML effort goes into changes that look like improvements but aren't.
A single cross-validation run has enough random variation that a useless change
appears to help roughly half the time. Teams adopt it, build on top of it, and
never find out.

Everything here is organized around one rule: **a change is real only if it
survives multiple random validation splits.**

## Components

**Skills** (work in Claude Code and claude.ai)

| Skill | Covers |
|---|---|
| `tabular-ml-workflow` | Sequence, non-negotiables, diagnosing CV/holdout gaps, what to work on next |
| `tabular-data-profiling` | Profiling, missingness classification, encoding decisions, rulebook generation |
| `tabular-validation` | Multi-seed testing, feature screening, leakage prevention, OOF caching |
| `tabular-model-selection` | Model comparison, blending, tuning, honest answers on NNs and time-series |
| `tabular-foundation-models` | TabPFN/TabICL/TabDPT/TabFM selection, context caps, weight licensing, their separate data path |

**Agents** (Claude Code only)

| Agent | Use |
|---|---|
| `leakage-auditor` | Read-only audit of a pipeline for leakage; independent second pair of eyes |
| `feature-tester` | Multi-seed screening of candidate features, adopt/reject with numbers |
| `data-profiler` | Profiles a new dataset, produces a per-column rulebook |
| `model-comparator` | Model comparison on identical folds, blend weight search |
| `foundation-model-advisor` | Shortlists and measures foundation models; enforces row caps and weight licensing |
| `preprocessing-router` | Works out which models need which matrix; catches one-hot reaching CatBoost or TabPFN |
| `calibration-auditor` | Brier/ECE, reliability, and whether the decision threshold was chosen on test data |
| `hyperparameter-tuner` | Searches, then re-validates the winner on seeds the search never saw |

**Commands** (Claude Code only)

- `/ml-start <train.csv> <target>` — scaffold, profile, establish the noise floor
- `/ml-test-feature <description>` — screen candidate features
- `/ml-audit` — leakage audit
- `/ml-foundation [model]` — shortlist and evaluate tabular foundation models
- `/ml-calibrate` — probability calibration and threshold audit
- `/ml-tune [model]` — hyperparameter search with held-out-seed re-validation

## Key scripts

- `multiseed_cv.py` — score across seeds; reports the noise floor
- `feature_screen.py` — test candidates individually, adopt on seed-consistency
- `oof_cache.py` / `blend_search.py` — evaluate many ensemble weightings cheaply
- `profile_dataset.py` — dataset profile plus rulebook skeleton
- `verify_rulebook.py` — catch documented-vs-implemented drift
- `project_scaffold.py` — standard project layout with experiment log
- `compare_models.py` — identical folds across seeds, with fit/predict time and
  average rank alongside score
- `calibration_report.py` — Brier, ECE, reliability, out-of-fold threshold
- `tune_search.py` — search with seeds reserved from it, then re-validated
- `fm_registry.py` — which foundation models fit this shape, licence, and task
- `fm_preflight.py` — packages, GPU, weight gating and licence, before fitting
- `fm_evaluate.py` — foundation-model data path with in-fold context capping

## Findings baked in

These came from real runs and are written into the skills as warnings:

- Log-transforming skewed features is standard advice that **hurt** on a real
  dataset (won 1 of 5 seeds)
- Collapsing rare categories improved single-seed CV but **worsened** the
  holdout score
- Untuned gradient boosting **lost** to a regularized linear model on ~1,450 rows
- Blending beat every solo model on every seed, but the exact weighting was a
  difference smaller than the noise

## Grounded in

The foundation-model and evaluation material is transcribed from two upstream
projects rather than recalled:

- **[TALENT](https://github.com/LAMDA-Tabular/TALENT)** (LAMDA, JMLR 2025) — a
  benchmark toolbox covering 45+ methods over 300 datasets. Three ideas are
  borrowed: preprocessing as a **per-method contract** rather than a project-wide
  one, a **registry** as the single source of truth for context caps and
  tunability, and ranking by **average rank plus cost** rather than raw score.
  Context caps and HPO support in `fm_registry.py` come from its
  `method_registry.py`.
- **[TabFM](https://github.com/google-research/tabfm)** (Google Research) — the
  zero-shot in-context model. Its estimator defaults and licence terms are
  transcribed from the repository.

Neither is vendored. This plugin stays dependency-free; the registry copies
facts, not code.

## Licensing warning

Several tabular foundation models license their **weights** separately from
their source code, and the weight licence is the one that governs whether you
can ship predictions:

- **TabFM v1.0.0** — Apache-2.0 code, `tabfm-non-commercial-v1.0` weights.
  Commercial and production use of the default weights is not permitted.
- **TabPFN v2.5 and v3** — non-commercial weights.
- **TabPFN v2** — Apache 2.0 plus attribution. The commercially usable
  checkpoint in that line.
- **TabICL, TabDPT, Mitra, LimiX, Real-TabPFN** — not verified here. Read the
  upstream terms; an unverified licence is not a permissive one.

`fm_preflight.py --commercial` turns these from warnings into blockers.

## Tests

Plain asserts, no pytest dependency, no foundation-model packages required.
Needs `pyyaml`, `pandas`, `numpy`, `scikit-learn`.

```bash
python -m tests                 # all three modules
python -m tests.test_manifest   # or one at a time
```

| Module | Guards against |
|---|---|
| `test_manifest.py` | Frontmatter that fails to parse (loads with *empty* metadata), an `argument-hint` that parses as a list instead of a string, an agent naming a skill that does not exist, a SKILL.md advertising a script nobody wrote — and the reverse |
| `test_fm_registry.py` | Non-commercial or unverified weights surviving a commercial filter, row/feature caps not excluding, adapters drifting from the registry, preflight raising instead of reporting a blocker |
| `test_harness.py` | A rank or win-count direction that inverts on `auc` vs `rmse`, a context subsample drawn outside the fold, a threshold tuned on non-out-of-fold predictions, search and holdout seeds overlapping |

Each assertion was negative-tested — reintroducing the original `argument-hint`
bug and flipping `rank(ascending=...)` both fail the suite.

## Install

```bash
claude plugin validate ./tabular-ml-plugin
```

Then install via a marketplace, `--plugin-dir`, or copy into `~/.claude/skills/`
to load as `tabular-ml@skills-dir`.
