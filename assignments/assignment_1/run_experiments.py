"""Run the full experiment for the report.

Sweeps parent-selection strictness. A random-search baseline on the same evaluation budget is
included for comparison.

Each run of this script gets its own timestamped folder, so earlier results
are never overwritten:
assignment_1/__data__/A1_implementation_2026/experiments/<date_time>/
with settings.json, results.csv, convergence.png, final_vs_truncation.png,
diagnostics.png, the best body per setting, and one database per run.

    python run_experiments.py            # the real thing
    python run_experiments.py --pilot    # 1 run per setting, 10 generations, crash check
"""

# Standard library
import argparse
import csv
import time
from pathlib import Path

# Third-party libraries
import matplotlib

matplotlib.use("Agg")  # write files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

# Local scripts
from A1_implementation_2026 import (
    CULL_MODE,
    DATA,
    MUTATION_PROBABILITY,
    NUM_ELITES,
    NUM_OFFSPRING,
    NUM_STEPS,
    POPULATION_SIZE,
    base_settings,
    load_targets,
    new_run_folder,
    run_ea,
    run_random_search,
    show_body,
)

# Local libraries (ARIEL)
from ariel import console

# --- THE RESEARCH VARIABLE --- #
# The share of the population allowed to reproduce. 1.0 = everyone, 0.05 = only the best 5%.
TRUNCATION_FRACTIONS: list[float] = [1.0, 0.5, 0.25, 0.1, 0.05]

# Independent runs per setting. Each run gets its own seed (1, 2, 3, ...),
# so results can be averaged and their spread reported. They run one after
# another, not in parallel.
NUM_RUNS_PER_SETTING: int = 10
SEEDS: list[int] = list(range(1, NUM_RUNS_PER_SETTING + 1))

BASELINE = "random"

COLUMNS = [
    "variant",
    "truncation_fraction",
    "seed",
    "generation",
    "best",
    "mean",
    "worst",
    "fitness_std",
    "mean_modules",
    "modules_std",
    "population_size",
]


def variant_name(fraction: float) -> str:
    """Short name used in the CSV and in file paths."""
    return f"ea_{fraction:g}"


def variant_label(fraction: float) -> str:
    """Human readable name used in plot legends."""
    return f"EA, top {fraction:.0%} reproduce"


def build_variants(fractions: list[float]) -> dict[str, dict]:
    """One entry per EA setting, plus the random-search baseline."""
    variants = {
        variant_name(f): {"label": variant_label(f), "truncation_fraction": f}
        for f in fractions
    }
    variants[BASELINE] = {"label": "Random search", "truncation_fraction": None}
    return variants


def run_all(
        variants: dict[str, dict],
        out_dir: Path,
        *,
        seeds: list[int],
        num_steps: int,
        population_size: int,
        num_offspring: int,
) -> list[dict]:
    """Run every variant on every seed. One row per generation per run."""
    targets = load_targets()
    rows: list[dict] = []

    for name, settings in variants.items():
        fraction = settings["truncation_fraction"]
        for seed in seeds:
            run_name = f"{name}_seed{seed}"
            console.rule(f"[cyan]{run_name}")
            started = time.perf_counter()

            if fraction is None:
                log, best_body = run_random_search(
                        targets,
                        seed=seed,
                        population_size=population_size,
                        num_offspring=num_offspring,
                        num_steps=num_steps,
                )
            else:
                log, best_body = run_ea(
                        targets,
                        seed=seed,
                        truncation_fraction=fraction,
                        population_size=population_size,
                        num_offspring=num_offspring,
                        num_steps=num_steps,
                        mutation_probability=MUTATION_PROBABILITY,
                        cull_mode=CULL_MODE,
                        num_elites=NUM_ELITES,
                        db_file_path=out_dir / run_name / "database.db",
                        quiet=True,
                )

            for entry in log:
                rows.append({
                    "variant": name,
                    "truncation_fraction": fraction,
                    "seed": seed,
                    **entry,
                })

            elapsed = time.perf_counter() - started
            console.log(
                    f"{run_name}: best {log[-1]['best']:.4f}  "
                    f"({best_body.number_of_nodes()} modules, {elapsed:.1f}s)",
            )

            if seed == seeds[0]:
                # show_body saves relative to DATA, so point it into out_dir.
                image = (out_dir / f"best_{name}").relative_to(DATA)
                show_body(best_body, "frame", file_name=str(image))

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Save the raw results so plots can be redrawn without re-running."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    console.log(f"saved {path}")


def _curves(rows: list[dict], variant: str, column: str) -> np.ndarray:
    """A (seeds x generations) array of one column for one variant."""
    seeds = sorted({row["seed"] for row in rows if row["variant"] == variant})
    per_seed = [
        [
            row[column]
            for row in rows
            if row["variant"] == variant and row["seed"] == seed
        ]
        for seed in seeds
    ]
    return np.array(per_seed, dtype=float)


def _finals(rows: list[dict], variant: str) -> np.ndarray:
    """Final best fitness of each seed for one variant."""
    data = _curves(rows, variant, "best")
    return data[:, -1] if data.size else np.array([])


def _plot_band(axis, rows, variant, column, *, label, color) -> None:
    """Mean across seeds as a line, with a +/- 1 std band."""
    data = _curves(rows, variant, column)
    if data.size == 0:
        return
    generations = np.arange(data.shape[1])
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    axis.plot(generations, mean, label=label, color=color)
    axis.fill_between(
            generations,
            mean - std,
            mean + std,
            alpha=0.15,
            color=color,
    )


def _colours(fractions: list[float]) -> dict[str, str]:
    """Dark = strict, light = permissive, so the ordering is visible."""
    colormap = plt.get_cmap("viridis")
    ordered = sorted(fractions, reverse=True)
    colours = {
        variant_name(f): colormap(0.15 + 0.7 * i / max(1, len(ordered) - 1))
        for i, f in enumerate(ordered)
    }
    colours[BASELINE] = "black"
    return colours


def plot_convergence(
        rows: list[dict],
        variants: dict[str, dict],
        fractions: list[float],
        path: Path,
        num_seeds: int,
) -> None:
    """Required figure: fitness over generations, mean and spread."""
    colours = _colours(fractions)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, settings in variants.items():
        _plot_band(
                axis,
                rows,
                name,
                "best",
                label=settings["label"],
                color=colours[name],
        )
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best fitness (lower is better)")
    axis.set_title(f"Best fitness, mean +/- 1 std over {num_seeds} runs")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    console.log(f"saved {path}")


def plot_final_vs_truncation(
        rows: list[dict],
        fractions: list[float],
        path: Path,
) -> None:
    """The main result: does strictness help, and is there a sweet spot?"""
    xs, means, stds = [], [], []
    for fraction in sorted(fractions):
        finals = _finals(rows, variant_name(fraction))
        if finals.size:
            xs.append(fraction)
            means.append(finals.mean())
            stds.append(finals.std())

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.errorbar(xs, means, yerr=stds, marker="o", capsize=4, color="tab:blue")

    baseline = _finals(rows, BASELINE)
    if baseline.size:
        axis.axhline(
                baseline.mean(),
                color="black",
                linestyle="--",
                label="Random search",
        )
        axis.legend(fontsize=8)

    axis.set_xscale("log")
    axis.set_xticks(xs)
    axis.set_xticklabels([f"{x:.0%}" for x in xs])
    axis.set_xlabel("Share of population allowed to reproduce")
    axis.set_ylabel("Final best fitness (lower is better)")
    axis.set_title("Final fitness vs selection strictness")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    console.log(f"saved {path}")


def plot_diagnostics(
        rows: list[dict],
        variants: dict[str, dict],
        fractions: list[float],
        path: Path,
) -> None:
    """Diversity and body size, to explain the convergence curves."""
    colours = _colours(fractions)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        ("mean", "Mean fitness of population"),
        ("fitness_std", "Fitness spread (diversity)"),
        ("mean_modules", "Mean modules per body (bloat)"),
    ]
    for axis, (column, title) in zip(axes, panels, strict=True):
        for name, settings in variants.items():
            _plot_band(
                    axis,
                    rows,
                    name,
                    column,
                    label=settings["label"],
                    color=colours[name],
            )
        axis.set_xlabel("Generation")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    console.log(f"saved {path}")


def print_summary(rows: list[dict], fractions: list[float]) -> None:
    """Print the numbers that go into the report."""
    console.rule("[green]Final best fitness (mean +/- std over seeds)")

    scores: dict[float, np.ndarray] = {}
    for fraction in sorted(fractions, reverse=True):
        finals = _finals(rows, variant_name(fraction))
        if not finals.size:
            continue
        scores[fraction] = finals
        console.log(
                f"top {fraction:>5.0%}   {finals.mean():.4f} +/- "
                f"{finals.std():.4f}   (best run {finals.min():.4f})",
        )

    baseline = _finals(rows, BASELINE)
    if baseline.size:
        console.log(
                f"{'random':>9}   {baseline.mean():.4f} +/- "
                f"{baseline.std():.4f}   (best run {baseline.min():.4f})",
        )

    if len(scores) < 2:
        return

    # Mann-Whitney U: non-parametric, so it does not assume normal fitness.
    best_fraction = min(scores, key=lambda f: scores[f].mean())
    worst_fraction = max(scores, key=lambda f: scores[f].mean())
    statistic, p_value = mannwhitneyu(
            scores[best_fraction],
            scores[worst_fraction],
            alternative="two-sided",
    )
    console.rule("[green]Best vs worst setting")
    console.log(
            f"top {best_fraction:.0%} vs top {worst_fraction:.0%}: "
            f"Mann-Whitney U={statistic:.1f}, p={p_value:.4f} "
            f"(n={len(scores[best_fraction])} per group)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
            "--pilot",
            action="store_true",
            help="1 run per setting, 10 generations, just to check nothing crashes",
    )
    args = parser.parse_args()

    if args.pilot:
        kind, seeds, num_steps = "pilot", [1], 10
    else:
        kind, seeds, num_steps = "experiments", SEEDS, NUM_STEPS

    out_dir = new_run_folder(kind, {
        **base_settings(),
        "num_steps": num_steps,
        "truncation_fractions": TRUNCATION_FRACTIONS,
        "seeds": seeds,
        "evaluation_budget": POPULATION_SIZE + NUM_OFFSPRING * num_steps,
    })
    variants = build_variants(TRUNCATION_FRACTIONS)

    console.log(f"truncation  : {[f'{f:.0%}' for f in TRUNCATION_FRACTIONS]}")
    console.log(f"runs/setting: {len(seeds)}")
    console.log(f"generations : {num_steps}")
    console.log(f"budget/run  : {POPULATION_SIZE + NUM_OFFSPRING * num_steps}")
    console.log(f"runs        : {len(variants) * len(seeds)}")
    console.log(f"output      : {out_dir}")

    rows = run_all(
            variants,
            out_dir,
            seeds=seeds,
            num_steps=num_steps,
            population_size=POPULATION_SIZE,
            num_offspring=NUM_OFFSPRING,
    )

    write_csv(rows, out_dir / "results.csv")
    plot_convergence(
            rows, variants, TRUNCATION_FRACTIONS,
            out_dir / "convergence.png", len(seeds),
    )
    plot_final_vs_truncation(
            rows, TRUNCATION_FRACTIONS, out_dir / "final_vs_truncation.png",
    )
    plot_diagnostics(
            rows, variants, TRUNCATION_FRACTIONS, out_dir / "diagnostics.png",
    )
    print_summary(rows, TRUNCATION_FRACTIONS)
    console.log(f"results in  : {out_dir}")


if __name__ == "__main__":
    main()
