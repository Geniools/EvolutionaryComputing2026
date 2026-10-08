from typing import cast

from ariel.ec import Individual

from environment.controller import Genotype
from environment.simulation import run_simulation
from evolution.genotype import make_individual


def random_search(
        input_size: int,
        output_size: int,
        num_evaluations: int,
) -> Individual:
    """Evaluate `num_evaluations` random genotypes and return the best.

    Draws a fresh individual via `genotype.make_individual`, scores it with
    `simulation.run_simulation(..., mode="simple")`, and keeps the best
    (lowest fitness) seen across `num_evaluations` draws.
    """
    if num_evaluations < 1:
        msg = "num_evaluations must be >= 1"
        raise ValueError(msg)

    best: Individual | None = None
    for _ in range(num_evaluations):
        ind = make_individual(input_size, output_size)
        ind.fitness = run_simulation(cast("Genotype", ind.genotype), mode="simple")
        if best is None or ind.fitness < best.fitness:
            best = ind

    assert best is not None
    return best
