---
description: Test candidate features with multi-seed CV and report adopt/reject
argument-hint: "[description of the feature(s) to test]"
---

Evaluate candidate engineered features and report which earn adoption.

Feature(s) to test: `$ARGUMENTS`. If none given, propose a handful of candidates
grounded in the dataset's columns and confirm before running.

Use the `feature-tester` agent if subagents are available; otherwise follow the
`tabular-validation` skill directly.

Requirements:

- Establish the baseline across the same seeds first, and report the noise floor
- Test each candidate individually, never only as a bundle
- Adopt on consistency across seeds (4/5 or 5/5), not on mean improvement alone
- Report rejects with their numbers, not just the adoptions

Finish by proposing experiment log rows for both the adoptions and the
rejections, so the negative results are recorded and not retried later.
