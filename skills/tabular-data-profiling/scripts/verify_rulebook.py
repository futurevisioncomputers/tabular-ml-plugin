"""
Check that cleaning code actually implements what the rulebook documents.

Motivated by a real failure: a rulebook correctly marked eight columns as
ordinal, but the pipeline never applied their mappings — they silently fell
through to one-hot encoding, discarding the ordering. Nothing raised an error.
The gap was only found by explicitly comparing documentation to output.

Usage:
    from verify_rulebook import verify
    verify(rulebook_path="rulebook.csv",
           cleaned_train=cleaned_df,
           cleaned_test=cleaned_test_df)
"""
from __future__ import annotations

import pandas as pd


def verify(rulebook_path, cleaned_train, cleaned_test=None, raw_train=None,
           verbose=True):
    """Compare a rulebook against a cleaned DataFrame.

    Reports:
      - ordinal columns still holding non-numeric values (mapping not applied)
      - columns marked 'drop' still present
      - any remaining nulls
      - train/test column mismatches
      - feature-count sanity after encoding
    """
    rb = pd.read_csv(rulebook_path)
    issues = []

    type_col = "suggested_type" if "suggested_type" in rb.columns else "category"
    enc_col = "encoding_method" if "encoding_method" in rb.columns else None

    # --- ordinal columns should be numeric after cleaning ---
    ordinal_mask = rb[type_col].astype(str).str.contains("ordinal", case=False, na=False)
    if enc_col:
        ordinal_mask |= rb[enc_col].astype(str).str.contains("ordinal", case=False, na=False)

    for col in rb.loc[ordinal_mask, "column"]:
        if col not in cleaned_train.columns:
            continue
        if not pd.api.types.is_numeric_dtype(cleaned_train[col]):
            issues.append(
                f"ORDINAL NOT APPLIED: '{col}' is documented ordinal but is still "
                f"{cleaned_train[col].dtype} after cleaning — it will be one-hot "
                f"encoded and lose its ordering"
            )

    # --- columns marked drop should be gone ---
    if enc_col:
        drop_mask = rb[enc_col].astype(str).str.strip().str.lower().eq("drop")
        for col in rb.loc[drop_mask, "column"]:
            if col in cleaned_train.columns:
                issues.append(f"NOT DROPPED: '{col}' is marked drop but is still present")

    # --- nulls ---
    n_null = int(cleaned_train.isnull().sum().sum())
    if n_null:
        cols = cleaned_train.isnull().sum()
        cols = cols[cols > 0].to_dict()
        issues.append(f"NULLS REMAIN in cleaned train: {cols}")
    if cleaned_test is not None:
        n_null_t = int(cleaned_test.isnull().sum().sum())
        if n_null_t:
            cols = cleaned_test.isnull().sum()
            cols = cols[cols > 0].to_dict()
            issues.append(f"NULLS REMAIN in cleaned test: {cols}")

    # --- train/test structure ---
    if cleaned_test is not None:
        only_train = set(cleaned_train.columns) - set(cleaned_test.columns)
        only_test = set(cleaned_test.columns) - set(cleaned_train.columns)
        # the target legitimately exists only in train
        only_train = {c for c in only_train
                      if c.lower() not in ("target", "y")}
        if len(only_train) > 1 or only_test:
            issues.append(
                f"COLUMN MISMATCH: train-only={sorted(only_train)} "
                f"test-only={sorted(only_test)} (one train-only column is "
                f"expected — the target)"
            )

    # --- dimensionality sanity ---
    encoded = pd.get_dummies(cleaned_train, drop_first=True)
    ratio = len(cleaned_train) / max(encoded.shape[1], 1)
    if verbose:
        print(f"after one-hot: {encoded.shape[1]} features for "
              f"{len(cleaned_train)} rows (ratio {ratio:.1f})")
        if ratio < 5:
            print("  note: few rows per feature — overfitting risk, especially "
                  "for linear models. Worth investigating, but collapsing rare "
                  "categories does not always help; test it rather than assuming.")

    if verbose:
        if issues:
            print(f"\n=== {len(issues)} ISSUE(S) ===")
            for i in issues:
                print(f"  - {i}")
        else:
            print("\nno mismatches found between rulebook and cleaned data")

    return issues
