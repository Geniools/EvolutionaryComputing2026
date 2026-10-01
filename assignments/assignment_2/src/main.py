"""Entry point: assemble and run the neuroevolution EA.

Mirrors `A2_template_2026.py::main()` (report the problem size) plus the
EA-assembly pattern from `examples/new_EC_engine_example.py` (build an
initial Population, list the EAOperations, hand both to `ariel.ec.EA`).
"""

import os
from pathlib import Path
from typing import cast

from ariel import console
from ariel.ec import EA, EAOperation, Population, config as ea_config, set_seed

from config import MODE, NUM_GENERATIONS, POPULATION_SIZE, SEED
from environment.controller import Genotype
from evolution.crossover import crossover
from evolution.evaluation import evaluate
from evolution.genotype import make_individual
from evolution.mutation import mutate
from evolution.selection import parent_selection, survivor_selection
from environment.simulation import get_io_sizes, run_simulation

# ARIEL/EA write to "__data__" in the working directory - land in
# assignment_2/__data__ regardless of where this script is invoked from.
os.chdir(Path(__file__).resolve().parent.parent)

set_seed(SEED)


def build_initial_population(input_size: int, output_size: int) -> Population:
    """Create the starting population of randomly initialised individuals."""
    return Population(
            [make_individual(input_size, output_size) for _ in range(POPULATION_SIZE)],
    )


def run_evolution() -> None:
    """Run the EA and report the best evolved controller."""
    input_size, output_size = get_io_sizes()
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")

    ea_config.target_population_size = POPULATION_SIZE
    ea_config.is_maximisation = False  # fitness_function: lower is better

    initial = evaluate(build_initial_population(input_size, output_size))

    ops: list[EAOperation] = [
        # EAOperation(parent_selection),
        # EAOperation(crossover),
        # EAOperation(mutate),
        EAOperation(evaluate),
        # EAOperation(survivor_selection),
    ]

    ea = EA(initial, ops, num_steps=NUM_GENERATIONS)
    ea.run()

    best = ea.get_solution("best")
    console.log(f"best fitness: {best.fitness:.4f} (lower is better)")

    # Watch the evolved controller. Switch mode to "video" for report figures.
    run_simulation(cast("Genotype", best.genotype), mode=MODE)


if __name__ == "__main__":
    run_evolution()
