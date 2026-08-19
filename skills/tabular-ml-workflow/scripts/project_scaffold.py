"""
Create a standard tabular-ML project layout with an experiment log ready to use.

The layout separates raw data (never modified), scratch exploration, reusable
code, outputs, and the experiment log — so that reproducing a result later is
possible.
"""
from __future__ import annotations

import argparse
import csv
import os

README = """# {name}

## Target
`{target}` — metric: {metric}

## Layout
```
data/          raw inputs (never edit these)
notebooks/     exploration; messy is fine
src/           reusable code (cleaning, features, CV, models)
submissions/   output files, one per experiment
experiments/   experiment_log.csv — every attempt recorded
```

## Workflow
1. Profile the data and write a per-column rulebook before cleaning.
2. Set up cross-validation and the experiment log BEFORE modeling.
3. Score a trivial baseline immediately to validate the pipeline end-to-end.
4. Establish the noise floor: run the baseline across several CV seeds and
   record the standard deviation. Every later change gets compared to it.
5. Clean, then compare models on identical folds.
6. Engineer features, testing each individually across seeds.
7. Blend. Tune last.

## The rule
A change is real if it improves on MOST seeds — not if it improves the mean of
one run. Record negative results in the log so they don't get retried.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--target", default="target")
    p.add_argument("--metric", default="rmse")
    p.add_argument("--path", default=".")
    args = p.parse_args()

    root = os.path.join(args.path, args.name)
    for d in ("data", "notebooks", "src", "submissions", "experiments"):
        os.makedirs(os.path.join(root, d), exist_ok=True)

    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(README.format(name=args.name, target=args.target, metric=args.metric))

    log_path = os.path.join(root, "experiments", "experiment_log.csv")
    header = ["date", "exp_id", "description", "model", "features_notes",
              "cv_score", "cv_std_across_seeds", "holdout_score",
              "output_file", "verdict", "notes"]
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f, quoting=csv.QUOTE_MINIMAL).writerow(header)

    with open(os.path.join(root, "requirements.txt"), "w") as f:
        f.write("pandas\nnumpy\nscikit-learn\nlightgbm\nxgboost\n"
                "matplotlib\nseaborn\njupyter\n")

    print(f"created {root}/")
    print("  next: drop raw data into data/, then profile it before cleaning")
    print(f"  experiment log ready at experiments/experiment_log.csv")
    print("\n  when appending to the log, build rows as lists and use csv.writer —")
    print("  description fields contain commas and will corrupt naive writes")


if __name__ == "__main__":
    main()
