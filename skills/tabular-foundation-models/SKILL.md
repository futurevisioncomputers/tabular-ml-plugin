---
name: tabular-foundation-models
description: Choose, run, and license pre-trained tabular foundation models — TabPFN v1-v3, TabICL, TabDPT, TabFM, Mitra, LimiX. Use this skill when the user asks about TabPFN or any tabular foundation model, asks whether a pre-trained model would beat gradient boosting, has a small dataset and wants the strongest available model, asks about zero-shot or in-context learning on tables, hits a row or context limit, or needs to know whether a model's weights may be used commercially. Also use when a foundation model needs its own preprocessing path separate from the rest of the pipeline.
---

# Tabular Foundation Models

Pre-trained transformers that do in-context learning over your training rows
instead of fitting parameters from scratch. They are the one class of neural
model that reliably competes on small tabular data, and they come with three
constraints that the rest of the model zoo does not have.

## Why they are a special case

The standard argument against neural networks on tabular data — not enough rows
to learn a good representation — does not apply, because the representation was
learned during pre-training on millions of synthetic tasks. Small data is the
target range rather than the weakness.

That changes the default advice. On a few thousand rows, "try gradient boosting
and a regularized linear model" is incomplete: a foundation model belongs in the
comparison, measured on the same folds with the same seeds.

What does not change: nothing here is assumed to win. Every claim in the papers
is a claim about benchmark averages, not about your dataset. Run it through
`tabular-validation`'s multi-seed harness like any other candidate.

## The three constraints

### 1. A context cap

Each model has a maximum number of training rows it can hold as context. The
caps span three orders of magnitude:

| Model | Row cap | Feature cap | Tasks | Tunable |
|---|---|---|---|---|
| `tabpfn_v1` | 1,000 | 100 | classification | yes |
| `tabpfn_v2` | 10,000 | 500 | both | no |
| `tabpfn_v2_5` | 50,000 | 2,000 | both | no |
| `tabpfn_v3` | 1,000,000 | 200 | both | no |
| `tabpfn_real` | 10,000 | 500 | classification | no |
| `tabicl` | 500,000 | — | classification | yes |
| `tabicl_v2` | 1,000,000 | — | both | no |
| `tabdpt` | none documented | — | both | yes |
| `tabfm` | none documented | 500 | both | no |
| `mitra` | 10,000 | — | both | no |
| `limix` | none documented | — | both | yes |

Caps and tunability are transcribed from TALENT's `method_registry.py`, which
documents them from each wrapper's actual assertions rather than from paper
claims. Note that TabPFN v3 has a *lower* feature cap than v2.5 despite a
twenty-fold larger row cap — wide tables and tall tables push toward different
checkpoints.

Exceeding a cap does not raise. The wrapper subsamples, and you get a plausible
number computed on data you did not choose. Two rules follow:

- **Subsample inside the fold, never before splitting.** A subsample drawn from
  the full dataset puts validation rows into the model's context. That is
  ordinary leakage and it inflates the score.
- **Report the context size with the score.** A capped run at 10,000 rows and an
  uncapped run are not comparable, and six weeks later nobody remembers which
  was which.

`scripts/fm_evaluate.py` enforces both — `ContextCappedEstimator` draws its
sample inside `fit()`, stratified for classification, with a fixed seed.

"No documented cap" means retrieval-based or streaming context, not unlimited
accuracy. Measure at your size.

### 2. A license on the weights, separate from the code

This is the constraint that causes real damage, because it surfaces after the
model is already in a pipeline.

- **TabFM v1.0.0** — code Apache-2.0, weights `tabfm-non-commercial-v1.0`. The
  README states commercial or production use of the default weights is **not
  permitted**.
- **TabPFN v2.5 and v3** — non-commercial weights.
- **TabPFN v2** — Prior Labs License: Apache 2.0 plus attribution. Commercial
  use permitted. It is the commercially usable checkpoint in that line, which
  is why it stays worth running even though later versions score better.
- **TabICL, TabDPT, Mitra, LimiX, Real-TabPFN** — not verified in this skill.
  Read the upstream terms before shipping. An unverified license is not a
  permissive one.

Ask whether predictions will be shipped **before** shortlisting, not after. For
a Kaggle competition or research the answer is no and the whole zoo is open; for
anything that reaches a customer it narrows sharply. When the answer is unclear,
treat it as commercial — the cost of guessing wrong in that direction is a
shorter shortlist, and in the other direction it is a license violation.

### 3. A different preprocessing path

These models want mixed-type columns and a list of which column positions are
categorical. They handle missing values themselves.

| Step | Standard pipeline | Foundation model |
|---|---|---|
| One-hot encoding | yes | **no** — pass categorical indices |
| Scaling / standardization | yes | **no** |
| Imputation | yes | **no** — handled natively |
| Engineered features | yes | **yes** — these still help |
| Outlier removal on train | optional | optional |

Feeding a foundation model the same one-hot, imputed, scaled matrix you feed
Ridge does not error. It just performs below what the model can do, and the
comparison against LightGBM you then draw is wrong in the model's disfavor.

The practical consequence: **blending a foundation model into an ensemble means
maintaining two feature paths.** That is a real maintenance cost and it is worth
saying out loud before starting, not after. `prepare_for_fm` in
`scripts/fm_evaluate.py` builds the foundation-model path; the rest of the
plugin builds the other one.

TALENT encodes the same contract as a per-method assertion — its TabFM wrapper
asserts `normalization="none"`, `cat_policy="indices"`, `num_policy="none"`.
Treating preprocessing as a property of the model rather than of the project is
the idea worth stealing.

## Choosing one

Filter, then measure. `scripts/fm_registry.py` does the filtering:

```bash
python scripts/fm_registry.py --rows 1450 --features 78 \
    --task regression --commercial
```

It prints the eligible models and, for every excluded one, why — over the row
cap, wrong task, non-commercial weights, needs a GPU. The exclusions are the
useful half of the output.

Rough guidance once the filter has run:

- **Under ~10k rows, non-commercial acceptable** — TabPFN v3 or v2.5 first,
  TabFM as a second opinion. This is the range where they are strongest.
- **Under ~10k rows, commercial** — TabPFN v2 is the checkpoint you can ship.
- **10k–100k rows** — TabPFN v2.5, TabICL, or TabDPT. Gradient boosting is a
  serious competitor here rather than an afterthought; run both.
- **Above ~500k rows** — the caps stop being the binding constraint and
  wall-clock time starts to be. Gradient boosting usually wins on cost even when
  it loses slightly on score. Say so rather than burning a day proving it.
- **Wide tables (>500 features)** — most caps bind. Select features first, and
  note that the selection then has to be fold-safe or it leaks.

## Running one

```bash
# check the environment and the licence before fitting anything
python scripts/fm_preflight.py tabfm --commercial --rows 5000 --features 40

# multi-seed evaluation on the foundation-model data path
python scripts/fm_evaluate.py --data train.csv --target SalePrice \
    --model tabpfn_v2 --metric rmse --log-target --seeds 42 7 2024
```

`fm_preflight.py` checks package presence, Python version, CUDA visibility,
weight-gating (`TABPFN_TOKEN`, Hugging Face download), and the license class
against whether the work is commercial. It never installs anything or accepts a
license on the user's behalf. A blocked model reports why and nothing is fitted.

Three seeds rather than five is a reasonable default here — these models are
slow enough that five seeds on a large dataset is an afternoon. Three is enough
to see whether a difference is consistent; say which you used.

## Blending them in

A foundation model earns its place in a blend the same way any model does, via
`tabular-model-selection`'s out-of-fold caching and blend search. Two specifics:

- Generate out-of-fold predictions on the **foundation-model data path**, then
  blend the prediction vectors. The two paths never need to agree on features,
  only on row order.
- Diversity is the point. A foundation model that scores slightly below
  LightGBM solo can still improve the blend substantially, because its errors
  are uncorrelated with a tree ensemble's. Check the blend before dismissing it
  on the solo score.

## What to tell the user honestly

When a foundation model wins, say by how much relative to the noise floor, and
whether the win survives the license and speed constraints. A model that scores
0.3% better, cannot be shipped, and takes forty times longer to predict is not
obviously the right choice — that is the user's trade-off to make, and they can
only make it if all three numbers are on the table.

When it loses, that is a normal result and worth logging. "TabPFN v2 lost to
LightGBM by 0.004, 1/3 seeds" saves the next person a day.

## Scripts

- `scripts/fm_registry.py` — capacity, license, and task filter over the model
  zoo; prints eligible models and the reason each other one was excluded
- `scripts/fm_preflight.py` — environment, weight-gating and license check
  before anything is fitted
- `scripts/fm_evaluate.py` — foundation-model data path, in-fold context
  capping, and multi-seed scoring through the shared harness

## Sources

Row caps, HPO support and preprocessing contracts are transcribed from
[TALENT](https://github.com/LAMDA-Tabular/TALENT) (`method_registry.py`, JMLR
2025). TabFM's estimator defaults and license terms are transcribed from
[google-research/tabfm](https://github.com/google-research/tabfm). Neither the
caps nor the accuracy claims were benchmarked when this skill was written.
