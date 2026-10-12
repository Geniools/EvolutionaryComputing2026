"""Run the random-search baseline and every crossover variant, one after another.

Use python main.py for seeds 1 through 5 by default, or pass --seeds in the
terminal to choose single or multiple different seeds.

- Pass --methods to run only some of: baseline none uniform blend neuron.
- Pass --workers N to spread the simulations over N processes; results are
 the same, only faster.

Each execution gets a numbered folder per method, such as
__data__/a2_experiments/uniform_1/ or __data__/a2_experiments/baseline_1/.
It contains settings.json, results.csv, and per seed either a database (EA)
or seed_<n>_evaluations.csv with every evaluated genotype (baseline).
"""

import argparse
import csv
import json
import os
import random
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ariel import console
from ariel.ec import EA, EAOperation, Individual, Population, config as ea_config, set_seed

from config import (
    BLEND_ALPHA,
    CROSSOVER_TYPE,
    HIDDEN_SIZE,
    MUTATION_RATE,
    MUTATION_SIGMA,
    NUM_GENERATIONS,
    PLATEAU_MIN_IMPROVEMENT,
    PLATEAU_PATIENCE,
    POPULATION_SIZE,
    BASELINE_NUM_EVALUATIONS, BLEND_ALPHA, HIDDEN_SIZE, NUM_GENERATIONS, POPULATION_SIZE, SEEDS,
)
from environment.simulation import get_io_sizes
from evolution import evaluation
from evolution.baseline import random_search
from evolution.crossover import OPERATORS, crossover
from evolution.evaluation import evaluate
from evolution.genotype import make_individual
from evolution.mutation import mutate
from evolution.selection import parent_selection, survivor_selection
from evolution.statistics import record_stats

# ARIEL/EA write to "__data__" in the working directory - land in
# assignment_2/__data__ regardless of where this script is invoked from.
os.chdir(Path(__file__).resolve().parent.parent)

BASELINE = "baseline"
METHODS: list[str] = [BASELINE, *OPERATORS]  # run order: baseline first, then each crossover


def create_experiment_dir(root: Path, crossover_type: str) -> Path:
    """Create the next numbered folder for this crossover operator."""
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{crossover_type}_"
    iterations = [
        int(path.name[len(prefix):])
        for path in root.iterdir()
        if path.is_dir()
           and path.name.startswith(prefix)
           and path.name[len(prefix):].isdigit()
    ]
    iteration = max(iterations, default=0) + 1

    while True:
        output_dir = root / f"{prefix}{iteration}"
        try:
            output_dir.mkdir()
        except FileExistsError:
            iteration += 1
        else:
            return output_dir


def build_initial_population(input_size: int, output_size: int) -> Population:
    """Create the starting population of randomly initialised individuals."""
    return Population(
            [make_individual(input_size, output_size) for _ in range(POPULATION_SIZE)],
    )


def has_plateaued(log: list[dict]) -> bool:
    """True if the best fitness barely improved over the last PLATEAU_PATIENCE generations."""
    if len(log) <= PLATEAU_PATIENCE:
        return False
    improvement = log[-PLATEAU_PATIENCE - 1]["best"] - log[-1]["best"]  # lower is better
    return improvement < PLATEAU_MIN_IMPROVEMENT


def seed_everything(seed: int) -> None:
    """Seed every RNG the run draws from."""
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def run_baseline(seed: int, evaluations_path: Path) -> list[dict]:
    """Run random search for one seed and return pseudo-generation statistics.

    Random search has no generations, so its evaluations are cut into the
    same chunks the EA evaluates: POPULATION_SIZE for generation 0, then one
    generation's offspring each. `best` is the best found so far; the other
    columns describe the evaluations inside that chunk.

    Every evaluation (index, fitness, genotype) is saved to `evaluations_path`,
    the baseline's counterpart of the EA's per-seed database.
    """
    seed_everything(seed)
    input_size, output_size = get_io_sizes()

    individuals: list[Individual] = []
    random_search(input_size, output_size, BASELINE_NUM_EVALUATIONS, history=individuals)
    fitnesses = [ind.fitness for ind in individuals]

    with evaluations_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["eval_index", "fitness", *(f"gene_{i}" for i in range(len(individuals[0].genotype)))])
        writer.writerows([i, ind.fitness, *ind.genotype] for i, ind in enumerate(individuals))

    offspring = (BASELINE_NUM_EVALUATIONS - POPULATION_SIZE) // NUM_GENERATIONS
    log: list[dict] = []
    start = 0
    for generation in range(NUM_GENERATIONS + 1):
        end = start + (POPULATION_SIZE if generation == 0 else offspring)
        chunk = fitnesses[start:end]
        log.append(
                {
                    "generation"     : generation,
                    "best"           : min(fitnesses[:end]),
                    "mean"           : float(np.mean(chunk)),
                    "worst"          : max(chunk),
                    "fitness_std"    : float(np.std(chunk)),
                    "population_size": len(chunk),
                }
        )
        start = end
    return log


def run_evolution(seed: int, db_path: Path, crossover_type: str) -> list[dict]:
    """Run one seed and return its generation statistics."""
    seed_everything(seed)

    input_size, output_size = get_io_sizes()
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")

    ea_config.target_population_size = POPULATION_SIZE

    initial = evaluate(build_initial_population(input_size, output_size))
    log: list[dict] = []
    record_stats(initial, log=log)  # Generation 0

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover, crossover_type=crossover_type),
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
    # Step manually instead of ea.run() so the run can stop at a plateau;
    # NUM_GENERATIONS is only the upper limit.
    for _ in range(NUM_GENERATIONS):
        ea.step()  # record_stats appends this generation to `log`
        if has_plateaued(log):
            console.log(f"plateau reached at generation {len(log) - 1}")
            break
    return log


def run_method(method: str, seeds: list[int]) -> None:
    """Run one method for the requested seeds and save one row per generation and seed."""
    output_dir = create_experiment_dir(Path("__data__") / "a2_experiments", method)
    results_path = output_dir / "results.csv"
    console.log(f"experiment output: {output_dir}")

    settings = {
        "crossover_type"          : method,
        "seeds"                   : seeds,
        "num_generations"         : NUM_GENERATIONS,
        "plateau_patience"        : PLATEAU_PATIENCE,
        "plateau_min_improvement" : PLATEAU_MIN_IMPROVEMENT,
        "population_size"         : POPULATION_SIZE,
        "hidden_size"             : HIDDEN_SIZE,
        "blend_alpha"             : BLEND_ALPHA,
        "mutation_rate"           : MUTATION_RATE,
        "mutation_sigma"          : MUTATION_SIGMA,
        "baseline_num_evaluations": BASELINE_NUM_EVALUATIONS,
    }
    with (output_dir / "settings.json").open("x", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)
        file.write("\n")

    with results_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
                file,
                fieldnames=[
                    "seed", "generation", "best", "mean", "worst",
                    "fitness_std", "population_size",
                ],
        )
        writer.writeheader()

        for seed in seeds:
            console.log(f"{method}: running seed {seed}")
            if method == BASELINE:
                log = run_baseline(seed, output_dir / f"seed_{seed}_evaluations.csv")
            else:
                log = run_evolution(seed, output_dir / f"seed_{seed}.db", method)
            writer.writerows({"seed": seed, **entry} for entry in log)
            file.flush()
            console.log(f"{method} seed {seed}: final best fitness {log[-1]['best']:.4f}")

    console.log(f"saved {results_path}")


def main() -> None:
    """Run the baseline and each crossover variant in turn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
            "--workers", type=int, default=1,
            help="processes used for evaluation (default 1: no multiprocessing)"
    )
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Each seed must be listed only once")

    # Same results either way: simulations use no randomness, only run in parallel.
    with Pool(args.workers) if args.workers > 1 else nullcontext() as evaluation.pool:
        for method in args.methods:
            run_method(method, args.seeds)


if __name__ == "__main__":
    main()
