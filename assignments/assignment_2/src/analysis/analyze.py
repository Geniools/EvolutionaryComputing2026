"""Aggregate runs across seeds: fitness plot, summary table, statistical tests.

Usage (from assignment_2/):
    python src/analysis/analyze.py __data__/a2_experiments --out __data__/a2_experiments/figures
    python src/analysis/analyze.py __data__/a2_experiments/uniform_1 --out __data__/a2_experiments/uniform_1/figures
    python src/analysis/analyze.py results/ --out figures/
    python src/analysis/analyze.py results/ --out figures/ --configs ea_main baseline_random --tag main
    python src/analysis/analyze.py results/ --out figures/ --configs xover_uniform xover_blend xover_sbx xover_none --tag xover

Inputs:  __data__/a2_experiments/<crossover>_<iteration>/results.csv (main.py output)
         results/<config>/seed_<n>/database.db
         results/<config>/seed_<n>/evaluations.csv (random search)
Fitness: LOWER IS BETTER, so best = min and worst = max.
Choose the numbered execution folder to analyze; older results.csv files
directly inside a2_experiments can still be analyzed by passing that folder.

Layer 1 (reader):  load_run, load_evaluations_csv, load_all  -> tidy DataFrames
Layer 2 (outputs): plot_fitness, summary_table, stats_tests  -> files in --out
                   (stats_tests.txt is the readable version of the .csv)
"""

import argparse
import itertools
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu

# Make src/ importable (for config.py) when run as `python analysis/analyze.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NUM_SEED_PLOTS, POPULATION_SIZE

ALPHA: float = 0.05  # significance level for the `result` column

# Fixed colour per config, so a config keeps its colour in every plot
# (also when --configs drops some). Baseline is a neutral grey reference.
CONFIG_COLORS: dict[str, str] = {
    # main.py method names
    "baseline": "#7f7f7f",
    "none": "#e34948",
    "uniform": "#eb6834",
    "blend": "#1baf7a",
    "neuron": "#4a3aa7",
    # results/<config>/ layout names
    "baseline_random": "#7f7f7f",
    "ea_main": "#2a78d6",
    "xover_uniform": "#eb6834",
    "xover_blend": "#1baf7a",
    "xover_sbx": "#4a3aa7",
    "xover_none": "#e34948",
}
# For configs not listed above (e.g. pilot runs), in fixed order.
EXTRA_COLORS: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CONFIG_ORDER: list[str] = list(CONFIG_COLORS)


# =============================================================================
# Layer 1: reading runs
# =============================================================================


def load_run(db_path: str | Path) -> pd.DataFrame:
    """Per-generation best/mean/worst fitness from one ariel.ec SQLite DB.

    ariel stores every individual ever created as one row of table
    `individual`. EA._commit() sets `time_of_birth` the first time a row is
    saved and overwrites `time_of_death` with the current generation on
    every save while the individual is alive. So an individual belongs to
    the population that SURVIVED generation g when:
        time_of_birth <= g  and  (time_of_death > g  or  alive = 1)
    (individuals culled in generation g keep time_of_death = g and alive = 0).
    """
    with sqlite3.connect(db_path) as con:
        rows = pd.read_sql_query(
            "SELECT alive, time_of_birth, time_of_death, fitness_ FROM individual "
            "WHERE requires_eval = 0 AND fitness_ IS NOT NULL",
            con,
        )

    records = []
    first_gen = int(rows["time_of_birth"].min())
    last_gen = int(rows["time_of_death"].max())
    for g in range(first_gen, last_gen + 1):
        in_pop = (rows["time_of_birth"] <= g) & (
            (rows["time_of_death"] > g) | (rows["alive"] == 1)
        )
        fitness = rows.loc[in_pop, "fitness_"]
        if fitness.empty:
            continue
        records.append({
            "generation": g,
            "best": fitness.min(),  # lower is better
            "mean": fitness.mean(),
            "worst": fitness.max(),
        })
    return pd.DataFrame(records)


def load_evaluations_csv(csv_path: str | Path, evals_per_gen: int) -> pd.DataFrame:
    """Turn a random-search log (eval_index, fitness) into pseudo-generations.

    Random search has no generations. To share the EA's x-axis we cut its
    evaluations into chunks of `evals_per_gen` (= POPULATION_SIZE) and report,
    per chunk g: best = best-so-far over all evaluations up to the end of g,
    mean/worst = mean/max of the evaluations inside chunk g.
    """
    fitness = pd.read_csv(csv_path).sort_values("eval_index")["fitness"].to_numpy()
    n_gens = int(np.ceil(len(fitness) / evals_per_gen))
    records = []
    for g in range(n_gens):
        chunk = fitness[g * evals_per_gen:(g + 1) * evals_per_gen]
        records.append({
            "generation": g,
            "best": fitness[:(g + 1) * evals_per_gen].min(),  # best-so-far
            "mean": chunk.mean(),
            "worst": chunk.max(),
        })
    return pd.DataFrame(records)


def load_results_csv(csv_path: Path, config: str) -> pd.DataFrame:
    """One results.csv written by main.py, labelled with its config name."""
    runs = pd.read_csv(csv_path)
    required = {"seed", "generation", "best", "mean", "worst"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if runs.empty:
        raise ValueError(f"{csv_path} has no results")
    runs.insert(0, "config", config)
    return runs


def load_all(results_dir: str | Path, evals_per_gen: int = POPULATION_SIZE) -> pd.DataFrame:
    """Load main.py's CSV(s) or every results/<config>/seed_<n>/ run.

    `results_dir` may be one main.py execution folder (uniform_1/), their
    parent (a2_experiments/, latest iteration of every method), or results/.

    Columns: config, seed, generation, best, mean, worst.
    Nested runs use database.db when present, otherwise evaluations.csv.
    """
    results_dir = Path(results_dir)
    main_csv = results_dir / "results.csv"
    if main_csv.exists():
        # main.py already logged one row per generation and seed. Reading it
        # directly avoids reconstructing populations from the SQLite history.
        return load_results_csv(main_csv, "ea_main")

    # main.py's parent folder: <method>_<iteration>/results.csv per method.
    # Compare the methods using the latest iteration of each.
    latest: dict[str, tuple[int, Path]] = {}
    for csv_path in results_dir.glob("*_*/results.csv"):
        method, _, iteration = csv_path.parent.name.rpartition("_")
        if iteration.isdigit() and int(iteration) >= latest.get(method, (0, csv_path))[0]:
            latest[method] = (int(iteration), csv_path)
    if latest:
        for method, (iteration, _) in sorted(latest.items()):
            print(f"using {method}_{iteration}")
        return pd.concat(
            [load_results_csv(path, method) for method, (_, path) in latest.items()],
            ignore_index=True,
        )

    frames = []
    for seed_dir in sorted(results_dir.glob("*/seed_*")):
        config = seed_dir.parent.name
        seed = int(seed_dir.name.removeprefix("seed_"))
        if (seed_dir / "database.db").exists():
            run = load_run(seed_dir / "database.db")
        elif (seed_dir / "evaluations.csv").exists():
            run = load_evaluations_csv(seed_dir / "evaluations.csv", evals_per_gen)
        else:
            print(f"skipping {seed_dir}: no database.db or evaluations.csv")
            continue
        if config.startswith("baseline"):
            # Random search keeps nothing between "generations": report the
            # best found so far (no-op for the CSV path, which already does).
            run["best"] = run["best"].cummin()
        run.insert(0, "seed", seed)
        run.insert(0, "config", config)
        frames.append(run)
    if not frames:
        msg = f"no runs found under {results_dir}"
        raise FileNotFoundError(msg)
    return pd.concat(frames, ignore_index=True)


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

    Non-parametric because 5 seeds are too few to assume normality.
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
    parser.add_argument("results_dir", help="main.py output folder or folder containing <config>/seed_<n>/")
    parser.add_argument("--out", default="figures/", help="output folder (default: figures/)")
    parser.add_argument("--configs", nargs="+", help="only use these configs")
    parser.add_argument("--tag", default="", help="suffix for output file names, e.g. 'xover'")
    parser.add_argument("--evals-per-gen", type=int, default=POPULATION_SIZE,
                        help="evaluations per pseudo-generation for random search "
                             f"(default: POPULATION_SIZE={POPULATION_SIZE})")
    args = parser.parse_args()

    df = load_all(args.results_dir, args.evals_per_gen)
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
        escape=True,  # config names contain '_'
    )

    tests = stats_tests(df)
    tests.to_csv(out / f"stats_tests{suffix}.csv", index=False)
    write_readable(tests, out / f"stats_tests{suffix}.txt", STATS_LEGEND)

    print(table.to_string(float_format="%.3f"))
    print(f"wrote outputs to {out}/")


if __name__ == "__main__":
    main()
