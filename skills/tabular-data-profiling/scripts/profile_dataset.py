"""
Profile a tabular dataset and emit a per-column rulebook skeleton.

Computes everything measurable (nulls, cardinality, skew, rare levels, train/test
category mismatches) and suggests a column type. The semantic fields —
what the column means, what a null means — need the data dictionary and are
left blank for a human to fill.
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import pandas as pd

ID_HINTS = ("id", "index", "key", "uuid")


def suggest_type(series, name, nunique, n_rows, target=None):
    """Best-effort type guess. The data dictionary overrides this — in
    particular, ordinal columns are indistinguishable from nominal ones without
    documentation, so anything categorical is suggested as nominal and must be
    reviewed."""
    if name == target:
        return "target"
    if name.lower() in ID_HINTS or (nunique == n_rows and nunique > 1):
        return "identifier"
    if pd.api.types.is_numeric_dtype(series):
        if nunique == 2:
            return "binary"
        # Small integer ranges are often encoded categories or ordinal ratings
        if nunique <= 15 and pd.api.types.is_integer_dtype(series):
            return "numeric_or_ordinal_CHECK_DICT"
        return "numeric"
    if nunique == 2:
        return "binary_categorical"
    return "nominal_categorical_CHECK_IF_ORDINAL"


def rare_levels(series, min_count=10):
    if pd.api.types.is_numeric_dtype(series):
        return ""
    counts = series.value_counts()
    rare = counts[counts < min_count]
    return f"{len(rare)} levels <{min_count} rows" if len(rare) else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test")
    p.add_argument("--target")
    p.add_argument("--out", default="rulebook.csv")
    p.add_argument("--min-count", type=int, default=10)
    args = p.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test) if args.test else None

    print(f"train: {train.shape}")
    if test is not None:
        print(f"test:  {test.shape}")

    # ---- target ----
    if args.target and args.target in train.columns:
        t = train[args.target]
        print(f"\n=== target: {args.target} ===")
        print(f"skew raw:   {t.skew():.3f}")
        if (t > 0).all():
            print(f"skew log1p: {np.log1p(t).skew():.3f}  "
                  f"({'log helps' if abs(np.log1p(t).skew()) < abs(t.skew()) else 'log does not help'})")

    # ---- near-constant columns ----
    print("\n=== near-constant columns (>99% one value) ===")
    flagged = []
    for c in train.columns:
        if c == args.target:
            continue
        top = train[c].value_counts(dropna=False)
        if len(top) and top.iloc[0] / len(train) > 0.99:
            flagged.append(c)
            print(f"  {c}: {top.index[0]!r} covers {top.iloc[0]/len(train)*100:.1f}%")
    if not flagged:
        print("  (none)")

    # ---- train/test category mismatches ----
    if test is not None:
        print("\n=== categories in one split only ===")
        found = False
        for c in train.select_dtypes(include=["object", "string"]).columns:
            if c not in test.columns:
                continue
            tr_c, te_c = set(train[c].dropna()), set(test[c].dropna())
            if tr_c - te_c or te_c - tr_c:
                found = True
                print(f"  {c}: train-only={sorted(tr_c-te_c)} test-only={sorted(te_c-tr_c)}")
        if not found:
            print("  (none)")

    # ---- rulebook ----
    header = ["column", "suggested_type", "description", "train_null_pct",
              "test_null_pct", "nunique", "skew", "rare_levels",
              "na_meaning", "cleaning_rule", "encoding_method"]
    rows = [header]

    for c in train.columns:
        s = train[c]
        nunique = int(s.nunique())
        skew = f"{s.skew():.2f}" if pd.api.types.is_numeric_dtype(s) else ""
        test_null = ""
        if test is not None and c in test.columns:
            test_null = f"{test[c].isnull().mean()*100:.1f}"

        rows.append([
            c,
            suggest_type(s, c, nunique, len(train), args.target),
            "",  # description — from data dictionary
            f"{s.isnull().mean()*100:.1f}",
            test_null,
            str(nunique),
            skew,
            rare_levels(s, args.min_count),
            "",  # na_meaning — structural vs genuine, from data dictionary
            "",  # cleaning_rule
            "",  # encoding_method
        ])

    for r in rows:
        assert len(r) == len(header), f"field count mismatch: {r[0]}"

    with open(args.out, "w", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_MINIMAL).writerows(rows)

    print(f"\nwrote {args.out} ({len(rows)-1} columns)")
    print("\nNext: fill in description / na_meaning / cleaning_rule from the data")
    print("dictionary. Pay special attention to columns marked CHECK_DICT or")
    print("CHECK_IF_ORDINAL — ordinal columns cannot be detected automatically,")
    print("and one-hot encoding them discards their ordering.")


if __name__ == "__main__":
    main()
