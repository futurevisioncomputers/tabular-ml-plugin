---
name: tabular-data-profiling
description: Profile a tabular dataset and produce a per-column cleaning and encoding rulebook covering missing-value semantics, ordinal vs nominal classification, and skew. Use this skill whenever the user has a new CSV or dataset and wants to explore it, asks how to clean or handle missing values, asks how to encode categorical columns, wants EDA, asks "what do these columns mean", or is deciding between one-hot, ordinal, and target encoding. Also use before modeling on any unfamiliar tabular data, and when a data dictionary or column documentation is available to cross-reference.
---

# Tabular Data Profiling

Turn an unfamiliar dataset into a documented, per-column set of cleaning and
encoding decisions.

## Why a written rulebook

Cleaning decisions get made repeatedly and inconsistently otherwise — once in a
notebook, again in a script, differently the third time. A rulebook fixes each
decision once, records the reasoning, and makes the pipeline auditable.

It also exposes a failure mode that is otherwise invisible: **the documented rule
and the implemented code drifting apart.** In one real project the rulebook
correctly identified eight columns as ordinal, but the code never implemented
their mappings — they silently fell through to one-hot encoding, discarding the
ordering. Nothing errored. Generate the rulebook, then verify the code matches it.

## Profile first

Compute, from the actual data rather than assumption:

- shape, dtypes, memory
- null count and percentage per column, **for train and test separately** (a
  column null only in test signals something different from one null in both)
- cardinality per column
- for numerics: describe + skew
- for categoricals: value counts, and which levels are rare
- target distribution, and whether a transform normalizes it
- **categories present in one split but not the other**

`scripts/profile_dataset.py` produces all of this.

Then check for near-constant columns: if one value covers >99% of rows, the
column usually carries no signal and can be dropped. Verify across train and test
combined before dropping.

## Classify missingness — the highest-value step

Not all nulls mean the same thing, and treating them uniformly destroys
information.

**Structural missing** — the feature doesn't exist for that row. A null pool
quality means "no pool", not "quality unrecorded". Fill with an explicit
sentinel: `"None"` for categoricals, `0` for numerics. These often carry real
signal.

**Genuine missing** — the value exists but wasn't recorded. Impute: median (or
group median) for numerics, mode for categoricals.

The data dictionary is the authority. When it lists `NA` as a valid documented
level for a column, that's structural. Never infer this from null percentage
alone — a 99% null column can be structural (rare feature) and a 2% null column
can be genuine.

Two traps worth knowing:

- **Sentinel fills that create impossible values.** Filling a "year built" column
  with 0 produces year 0, which distorts any derived age feature and any scaling.
  Fill with a sensible related value (e.g. the main construction year) instead.
- **Group-aware imputation.** When a numeric correlates strongly with a
  categorical (frontage with neighborhood, income with job title), the
  per-group median beats the global median. Compute groups from training data
  only.

## Classify each column for encoding

| Type | Signal | Encoding |
|---|---|---|
| Ordinal | documented order (Poor < Fair < Good) | integer map preserving order |
| Nominal, low cardinality | no order, few levels | one-hot |
| Nominal, high cardinality | no order, many levels | one-hot baseline; try target encoding (fold-safe) |
| Binary | two values | map to 0/1 |
| Numeric | continuous | usually none |
| Identifier | unique per row | drop from features |

**Ordinal columns are the ones most often mishandled.** One-hot encoding an
ordered scale throws away the ordering the data dictionary documented. Watch for
ordinal columns that don't share a common scale — a quality scale (Po/Fa/TA/Gd/Ex)
and a finish scale (Unf/RFn/Fin) need separate maps, and a single shared mapping
dictionary will silently miss the second kind.

Numeric-looking codes (class or type identifiers like 20, 30, 60) are usually
nominal categoricals — the numbers aren't magnitudes. Cast to string before
encoding.

## Verify implementation against the rulebook

After generating the rulebook and writing the cleaning code, check:

- every column marked ordinal actually has a mapping applied
- no column produces unexpected nulls after cleaning
- train and test go through identical logic, with train-derived statistics
- feature count after encoding is sane relative to row count

`scripts/verify_rulebook.py` checks the first two automatically.

Watch the rows-to-features ratio after one-hot encoding. Many features relative
to rows raises overfitting risk, particularly for linear models — but treat this
as a flag to investigate, not an automatic fix. Collapsing rare categories is the
usual response and it does not always help; test it rather than assuming.

## Transformations: evaluate, don't apply by default

Skew correction (log1p on right-skewed positives) is standard advice and often
helps linear models. It is not automatic. In one real project, log-transforming
all features with skew > 0.75 made the model measurably worse and won on only
1 of 5 validation seeds — despite being recommended by the project's own written
pipeline document.

Same for outlier removal: investigate rather than delete on a rule. Plot the
suspicious points, check whether they're genuine anomalies or just large valid
values, and confirm any removal rule catches only what it should.

Test every transformation through the multi-seed harness in `tabular-validation`.

## Scripts

- `scripts/profile_dataset.py` — full profile; writes a rulebook CSV skeleton
  with computed nulls, cardinality, skew, and a suggested type per column
- `scripts/verify_rulebook.py` — checks the cleaning code against the rulebook
  and reports mismatches

```bash
python scripts/profile_dataset.py --train train.csv --test test.csv \
    --target SalePrice --out rulebook.csv
```

Fill in the `description`, `na_meaning`, and `cleaning_rule` columns by reading
the data dictionary. The script computes what's measurable; the semantics need a
human or a documentation source.

When writing the rulebook CSV programmatically, use `csv.writer` with row lists
rather than joining strings — rule descriptions contain commas and parentheses
that corrupt naively-written CSVs. Assert row field counts match the header.
