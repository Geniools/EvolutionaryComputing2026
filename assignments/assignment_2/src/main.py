"""Run the Assignment 2 EA for one or more seeds.

Use python main.py for seeds 1 through 5 by default, or pass --seeds in the
terminal to choose single or multiple different seeds. 
Each seed gets its own database, generation statistics go to one CSV.
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np

from ariel import console
from ariel.ec import EA, EAOperation, Population, config as ea_config, set_seed

from config import NUM_GENERATIONS, POPULATION_SIZE
from environment.simulation import get_io_sizes
from evolution.crossover import crossover
from evolution.evaluation import evaluate
from evolution.genotype import make_individual
from evolution.mutation import mutate
from evolution.selection import parent_selection, survivor_selection
from evolution.statistics import record_stats

# ARIEL/EA write to "__data__" in the working directory - land in
# assignment_2/__data__ regardless of where this script is invoked from.
os.chdir(Path(__file__).resolve().parent.parent)


def build_initial_population(input_size: int, output_size: int) -> Population:
    """Create the starting population of randomly initialised individuals."""
    return Population(
            [make_individual(input_size, output_size) for _ in range(POPULATION_SIZE)],
    )


def run_evolution(seed: int, db_path: Path) -> list[dict]:
    """Run one seed and return its generation statistics."""
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    input_size, output_size = get_io_sizes()
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")

    ea_config.target_population_size = POPULATION_SIZE

    initial = evaluate(build_initial_population(input_size, output_size))
    log: list[dict] = []
    record_stats(initial, log=log)  # Generation 0

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection, population_size=POPULATION_SIZE),
        EAOperation(record_stats, log=log),
    ]

    ea = EA(
        initial,
        ops,
        num_steps=NUM_GENERATIONS,
        is_maximisation=False,
        db_file_path=db_path,
        db_handling="delete",
    )
    ea.run()
    return log


def main() -> None:
    """Run the requested seeds and save one row per generation and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    seeds = parser.parse_args().seeds
    if len(seeds) != len(set(seeds)):
        parser.error("Each seed must be listed only once")

    output_dir = Path("__data__") / "a2_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed", "generation", "best", "mean", "worst",
                "fitness_std", "population_size",
            ],
        )
        writer.writeheader()

        for seed in seeds:
            console.log(f"running seed {seed}")
            log = run_evolution(seed, output_dir / f"seed_{seed}.db")
            writer.writerows({"seed": seed, **entry} for entry in log)
            file.flush()
            console.log(f"seed {seed}: final best fitness {log[-1]['best']:.4f}")

    console.log(f"saved {results_path}")


if __name__ == "__main__":
    main()
