"""Aggregate runs across seeds: fitness plot, summary table, statistical tests.

Usage (from assignment_2/):
    python src/analysis/analyze.py __data__/a2_experiments --out __data__/a2_experiments/figures
    python src/analysis/analyze.py __data__/a2_experiments/uniform_1 --out __data__/a2_experiments/uniform_1/figures
    python src/analysis/analyze.py __data__/a2_experiments --configs baseline uniform --tag main

Input:   main.py's __data__/a2_experiments/<method>_<iteration>/results.csv
         (one row per seed and generation). Pass one such folder, or their
         parent folder to compare the latest iteration of every method.
Fitness: LOWER IS BETTER, so best = min and worst = max.

Layer 1 (reader):  load_results_csv, load_all              -> tidy DataFrames
Layer 2 (outputs): plot_fitness, summary_table, stats_tests -> files in --out
                   (stats_tests.txt is the readable version of the .csv)
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu

# Make src/ importable (for config.py) when run as `python analysis/analyze.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NUM_SEED_PLOTS

ALPHA: float = 0.05  # significance level for the `result` column

# Fixed colour per method, so a method keeps its colour in every plot
# (also when --configs drops some). Baseline is a neutral grey reference.
CONFIG_COLORS: dict[str, str] = {
    "baseline": "#7f7f7f",
    "none": "#e34948",
    "uniform": "#eb6834",
    "blend": "#1baf7a",
    "neuron": "#4a3aa7",
}
# For any other method name, in fixed order.
EXTRA_COLORS: list[str] = ["#2a78d6", "#eda100", "#e87ba4", "#008300"]
CONFIG_ORDER: list[str] = list(CONFIG_COLORS)


# =============================================================================
# Layer 1: reading runs
# =============================================================================


def method_name(folder: Path) -> str:
    """'uniform_3' -> 'uniform'; any other folder name is used as is."""
    method, _, iteration = folder.name.rpartition("_")
    return method if method and iteration.isdigit() else folder.name


def load_results_csv(csv_path: Path, config: str) -> pd.DataFrame:
    """One results.csv written by main.py, labelled with its method name."""
    runs = pd.read_csv(csv_path)
    required = {"seed", "generation", "best", "mean", "worst"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if runs.empty:
        raise ValueError(f"{csv_path} has no results")
    runs.insert(0, "config", config)
    return runs


def load_all(results_dir: str | Path) -> pd.DataFrame:
    """Load one main.py execution folder, or the latest iteration of every method.

    `results_dir` is either one execution folder (uniform_1/) or their parent
    (a2_experiments/). Columns: config, seed, generation, best, mean, worst, ...
    """
    results_dir = Path(results_dir)
    main_csv = results_dir / "results.csv"
    if main_csv.exists():
        return load_results_csv(main_csv, method_name(results_dir))

    # Parent folder: <method>_<iteration>/results.csv per method.
    # Compare the methods using the latest iteration of each.
    latest: dict[str, tuple[int, Path]] = {}
    for csv_path in results_dir.glob("*_*/results.csv"):
        method, _, iteration = csv_path.parent.name.rpartition("_")
        if iteration.isdigit() and int(iteration) >= latest.get(method, (0, csv_path))[0]:
            latest[method] = (int(iteration), csv_path)
    if not latest:
        msg = f"no results.csv found in {results_dir} or its <method>_<n>/ folders"
        raise FileNotFoundError(msg)
    for method, (iteration, _) in sorted(latest.items()):
        print(f"using {method}_{iteration}")
    return pd.concat(
        [load_results_csv(path, method) for method, (_, path) in latest.items()],
        ignore_index=True,
    )


def ordered_configs(df: pd.DataFrame) -> list[str]:
    """Known configs in CONFIG_ORDER first, then any others alphabetically."""
    present = set(df["config"])
    known = [c for c in CONFIG_ORDER if c in present]
    return known + sorted(present - set(known))


def final_best(df: pd.DataFrame) -> pd.DataFrame:
    """Best fitness found in each run (min over generations), one row per run."""
    return df.groupby(["config", "seed"], as_index=False)["best"].min()


# =============================================================================
# Layer 2: outputs
# =============================================================================


def plot_fitness(df: pd.DataFrame, out_stem: Path) -> None:
    """One panel per config: mean of the per-generation best with a +-1 std band.

    Grey lines are the single runs of the first NUM_SEED_PLOTS seeds, to show
    what one run looks like next to the average. All panels share the y-axis.
    Sized for the full GECCO text width (7 in). Writes <out_stem>.png and .pdf.
    """
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    configs = ordered_configs(df)
    seeds = sorted(df["seed"].unique())[:NUM_SEED_PLOTS]
    fig, axes = plt.subplots(1, len(configs), figsize=(7.0, 1.9),
                             sharex=True, sharey=True, squeeze=False)

    extra = iter(EXTRA_COLORS)
    for ax, config in zip(axes[0], configs, strict=True):
        color = CONFIG_COLORS.get(config) or next(extra, "black")
        runs = df[df["config"] == config]
        best_mean = runs.groupby("generation")["best"].mean()
        best_std = runs.groupby("generation")["best"].std().fillna(0.0)

        ax.fill_between(best_mean.index, best_mean - best_std, best_mean + best_std,
                        color=color, alpha=0.25, linewidth=0)
        for seed in seeds:
            run = runs[runs["seed"] == seed]
            ax.plot(run["generation"], run["best"], color="#8c8c8c", linewidth=0.6)
        ax.plot(best_mean.index, best_mean, color=color, linewidth=1.4)

        ax.set_title(f"{config}\n{best_mean.iloc[-1]:.2f} $\\pm$ {best_std.iloc[-1]:.2f}", color=color)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Best fitness\n(lower is better)")
    axes[0, len(configs) // 2].set_xlabel("Generation")
    # One shared legend in neutral colours: every panel uses the same three elements.
    fig.legend(
        handles=[
            Line2D([], [], color="black", linewidth=1.4, label=f"mean of {df['seed'].nunique()} seeds"),
            Patch(facecolor="black", alpha=0.25, linewidth=0, label="$\\pm$ 1 std"),
            Line2D([], [], color="#8c8c8c", linewidth=0.6,
                   label=f"single runs (seeds {', '.join(map(str, seeds))})"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
        fontsize=7, handlelength=1.5, columnspacing=1.5,
    )

    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Final best fitness per config: mean / std / min / max over seeds, n_seeds."""
    finals = final_best(df)
    table = finals.groupby("config")["best"].agg(
        mean="mean", std="std", min="min", max="max", n_seeds="count",
    )
    return table.reindex(ordered_configs(df))


def stats_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise two-sided Mann-Whitney U tests on final best fitness.

    Non-parametric: the final fitness over seeds need not be normally distributed.
    p_bonferroni = p * number_of_pairs (capped at 1), to account for doing
    many comparisons at once.
    """
    finals = final_best(df)
    pairs = list(itertools.combinations(ordered_configs(df), 2))
    records = []
    for a, b in pairs:
        fa = finals.loc[finals["config"] == a, "best"]
        fb = finals.loc[finals["config"] == b, "best"]
        u, p = mannwhitneyu(fa, fb, alternative="two-sided")
        records.append({
            "config_a": a, "config_b": b,
            "median_a": fa.median(), "median_b": fb.median(),
            "n_a": len(fa), "n_b": len(fb),
            "U": u, "p_value": p, "p_bonferroni": min(1.0, p * len(pairs)),
        })
        better = a if fa.median() < fb.median() else b  # lower is better
        records[-1]["result"] = (f"{better} better" if records[-1]["p_bonferroni"] < ALPHA
                                 else "no significant difference")
    return pd.DataFrame(records, columns=[
        "config_a", "config_b", "median_a", "median_b",
        "n_a", "n_b", "U", "p_value", "p_bonferroni", "result",
    ])


def write_readable(table: pd.DataFrame, path: Path, legend: str) -> None:
    """Write `table` as aligned plain text below a short explanation of its columns."""
    text = table.to_string(index=False, float_format="%.4f")
    path.write_text(f"{legend.strip()}\n\n{text}\n", encoding="utf-8")


STATS_LEGEND = f"""
Mann-Whitney U test (two-sided) on the final best fitness of every pair of methods.
The seeds of the two methods are treated as two independent groups. LOWER FITNESS IS BETTER.
  median_a/b    median final best fitness of each method over its seeds
  n_a/b         number of seeds
  U             test statistic
  p_value       chance of a difference this large if the methods were equally good
  p_bonferroni  p_value * number of pairs (max 1): corrected for testing many pairs
  result        significant when p_bonferroni < {ALPHA}; names the method with the lower median
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="one <method>_<n>/ folder or their parent folder")
    parser.add_argument("--out", default="figures/", help="output folder (default: figures/)")
    parser.add_argument("--configs", nargs="+", help="only use these methods")
    parser.add_argument("--tag", default="", help="suffix for output file names, e.g. 'main'")
    args = parser.parse_args()

    df = load_all(args.results_dir)
    if args.configs:
        missing = set(args.configs) - set(df["config"])
        if missing:
            print(f"warning: no runs for {sorted(missing)}")
        df = df[df["config"].isin(args.configs)]
        if df.empty:
            parser.error("none of the requested configs have results")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    plot_fitness(df, out / f"fitness_over_generations{suffix}")

    table = summary_table(df)
    table.to_csv(out / f"summary_table{suffix}.csv")
    table.to_latex(
        out / f"summary_table{suffix}.tex",
        float_format="%.3f",
        caption="Final best fitness per configuration over seeds (lower is better).",
        label=f"tab:summary{suffix}",
        escape=True,  # escape LaTeX special characters such as '_'
    )

    tests = stats_tests(df)
    tests.to_csv(out / f"stats_tests{suffix}.csv", index=False)
    write_readable(tests, out / f"stats_tests{suffix}.txt", STATS_LEGEND)

    print(table.to_string(float_format="%.3f"))
    print(f"wrote outputs to {out}/")


if __name__ == "__main__":
    main()
