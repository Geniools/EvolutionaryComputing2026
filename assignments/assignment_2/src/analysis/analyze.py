"""Aggregate runs across seeds: fitness plot, summary table, statistical tests.

Usage (from assignment_2/):
    python src/analysis/analyze.py __data__/a2_experiments --out __data__/a2_experiments/figures
    python src/analysis/analyze.py results/ --out figures/
    python src/analysis/analyze.py results/ --out figures/ --configs ea_main baseline_random --tag main
    python src/analysis/analyze.py results/ --out figures/ --configs xover_uniform xover_blend xover_sbx xover_none --tag xover

Inputs:  __data__/a2_experiments/results.csv (main.py output)
         results/<config>/seed_<n>/database.db
         results/<config>/seed_<n>/evaluations.csv (random search)
Fitness: LOWER IS BETTER, so best = min and worst = max.

Layer 1 (reader):  load_run, load_evaluations_csv, load_all  -> tidy DataFrames
Layer 2 (outputs): plot_fitness, summary_table, stats_tests  -> files in --out
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
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu

# Make src/ importable (for config.py) when run as `python analysis/analyze.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import POPULATION_SIZE

# Fixed colour per config, so a config keeps its colour in every plot
# (also when --configs drops some). Baseline is a neutral grey reference.
CONFIG_COLORS: dict[str, str] = {
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


def load_all(results_dir: str | Path, evals_per_gen: int = POPULATION_SIZE) -> pd.DataFrame:
    """Load main.py's CSV or every results/<config>/seed_<n>/ run.

    Columns: config, seed, generation, best, mean, worst.
    Nested runs use database.db when present, otherwise evaluations.csv.
    """
    results_dir = Path(results_dir)
    main_csv = results_dir / "results.csv"
    if main_csv.exists():
        # main.py already logged one row per generation and seed. Reading it
        # directly avoids reconstructing populations from the SQLite history.
        runs = pd.read_csv(main_csv)
        required = {"seed", "generation", "best", "mean", "worst"}
        missing = required - set(runs.columns)
        if missing:
            raise ValueError(f"{main_csv} is missing columns: {sorted(missing)}")
        if runs.empty:
            raise ValueError(f"{main_csv} has no results")
        runs.insert(0, "config", "ea_main")
        return runs

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


def plot_fitness(df: pd.DataFrame, out_stem: Path, show_mean: bool = False) -> None:
    """Mean (over seeds) of the per-generation best, with a +-1 std band.

    Sized for one GECCO column (3.33 in wide). Writes <out_stem>.png and .pdf.
    """
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    extra = iter(EXTRA_COLORS)
    for config in ordered_configs(df):
        color = CONFIG_COLORS.get(config) or next(extra, "black")
        per_gen = df[df["config"] == config].groupby("generation")
        best_mean = per_gen["best"].mean()
        best_std = per_gen["best"].std().fillna(0.0)  # std is NaN with 1 seed
        n_seeds = df.loc[df["config"] == config, "seed"].nunique()

        ax.plot(best_mean.index, best_mean, color=color, linewidth=1.5,
                label=f"{config} (n={n_seeds})")
        ax.fill_between(best_mean.index, best_mean - best_std, best_mean + best_std,
                        color=color, alpha=0.2, linewidth=0)
        if show_mean:
            pop_mean = per_gen["mean"].mean()
            ax.plot(pop_mean.index, pop_mean, color=color, linewidth=0.8, linestyle="--")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # no "2.5" generations
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    title = "mean $\\pm$ 1 std over seeds"
    if show_mean:
        title += "; dashed = population mean"
    ax.set_title(title, loc="left", color="#555555")
    # Legend above the axes so it never hides the curves.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=2, frameon=False,
              borderaxespad=0.0, handlelength=1.5, columnspacing=1.0)

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
    return pd.DataFrame(records, columns=[
        "config_a", "config_b", "median_a", "median_b",
        "n_a", "n_b", "U", "p_value", "p_bonferroni",
    ])


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
    parser.add_argument("--show-mean", action="store_true",
                        help="also draw the population mean as a dashed line")
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

    plot_fitness(df, out / f"fitness_over_generations{suffix}", show_mean=args.show_mean)

    table = summary_table(df)
    table.to_csv(out / f"summary_table{suffix}.csv")
    table.to_latex(
        out / f"summary_table{suffix}.tex",
        float_format="%.3f",
        caption="Final best fitness per configuration over seeds (lower is better).",
        label=f"tab:summary{suffix}",
        escape=True,  # config names contain '_'
    )

    stats_tests(df).to_csv(out / f"stats_tests{suffix}.csv", index=False)

    print(table.to_string(float_format="%.3f"))
    print(f"wrote outputs to {out}/")


if __name__ == "__main__":
    main()
