"""Paired alternative to analyze.py's stats_tests: each method against a control.

Usage (from assignment_2/):
    python src/analysis/paired_tests.py __data__/a2_experiments --out __data__/a2_experiments/figures
    python src/analysis/paired_tests.py __data__/a2_experiments --methods uniform blend neuron baseline

Every seed starts all methods from the same initial population, so runs are
compared per seed: diff = method - control (LOWER IS BETTER, so a negative
diff means the method beat the control on that seed).
Metrics:    final_best   = best fitness the run found
            average_best = its best-so-far averaged over all generations, so a
                           run that improves earlier scores better
Test:       two-sided Wilcoxon signed-rank on those diffs (no normality assumed).
Correction: Holm over all tests run here (never weaker than Bonferroni).
Needs >= 9 seeds for p_holm < 0.05 to be reachable with the default 8 tests
(4 methods x 2 metrics); 10 is a sensible minimum.
Writes paired_tests.csv and a readable paired_tests.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze import ALPHA, load_all, write_readable

# One number per run, from its per-generation best fitness.
METRICS: dict[str, str] = {
    "final_best": "min",     # how good it gets
    "average_best": "mean",  # how quickly it gets there (mean over generations)
}


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the order given."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * (len(p) - np.arange(len(p)))  # smallest p gets the largest factor
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted.tolist()


def paired_tests(df: pd.DataFrame, control: str, methods: list[str]) -> pd.DataFrame:
    """One row per metric and method: its per-seed difference to `control` and the test on it."""
    per_run = df.groupby(["config", "seed"])["best"].agg(**METRICS)
    records = []
    for metric in METRICS:
        values = per_run[metric].unstack("config")  # rows: seed, columns: method
        for method in methods:
            diff = (values[method] - values[control]).dropna()  # seeds both have run
            nonzero = diff[diff != 0]
            # With no difference on any seed there is nothing to test.
            p = wilcoxon(nonzero).pvalue if len(nonzero) else 1.0
            records.append({
                "metric": metric, "method": method, "control": control, "n_seeds": len(diff),
                "method_better": int((diff < 0).sum()), "control_better": int((diff > 0).sum()),
                "median_diff": diff.median(), "p_value": p,
            })
    table = pd.DataFrame(records)
    table["p_holm"] = holm(table["p_value"].tolist())
    winner = table["method"].where(table["median_diff"] < 0, table["control"])  # lower is better
    table["result"] = (winner + " better").where(table["p_holm"] < ALPHA, "no significant difference")
    return table


LEGEND = f"""
Wilcoxon signed-rank test (two-sided), paired per seed: each method against the control.
Every seed starts all methods from the same initial population. LOWER FITNESS IS BETTER.
  metric          final_best = best fitness of the run; average_best = best-so-far
                  averaged over all generations (rewards improving early)
  method_better   number of seeds where the method beat the control
  control_better  number of seeds where the control beat the method
  median_diff     median over seeds of (method - control); positive = method is worse
  p_value         chance of differences this consistent if both were equally good
  p_holm          p_value corrected (Holm) for running all the tests in this table
  result          significant when p_holm < {ALPHA}; names the better of the two
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="same folder you would pass to analyze.py")
    parser.add_argument("--out", default="figures/", help="output folder (default: figures/)")
    parser.add_argument("--control", default="none", help="method the others are compared to")
    parser.add_argument("--methods", nargs="+",
                        help="methods to compare to the control (default: all others)")
    args = parser.parse_args()

    df = load_all(args.results_dir)
    present = set(df["config"])
    methods = args.methods or sorted(present - {args.control})
    missing = {args.control, *methods} - present
    if missing:
        parser.error(f"no runs for {sorted(missing)}")

    table = paired_tests(df, args.control, methods)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "paired_tests.csv", index=False)
    write_readable(table, out / "paired_tests.txt", LEGEND)
    print(table.to_string(index=False, float_format="%.4f"))
    print(f"wrote {out / 'paired_tests.csv'}")


if __name__ == "__main__":
    main()
