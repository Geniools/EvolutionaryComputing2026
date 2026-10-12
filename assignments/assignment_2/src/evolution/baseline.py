from typing import cast

from ariel.ec import Individual

from environment.controller import Genotype
from evolution.evaluation import evaluate_genotypes
from evolution.genotype import make_individual


def random_search(
        input_size: int,
        output_size: int,
        num_evaluations: int,
        history: list[Individual] | None = None,
) -> Individual:
    """Evaluate `num_evaluations` random genotypes and return the best.

    Draws fresh individuals via `genotype.make_individual`, scores them with
    `evaluation.evaluate_genotypes`, and keeps the best (lowest fitness)
    seen across `num_evaluations` draws. If `history` is given, every
    evaluated individual is appended to it in order.
    """
    if num_evaluations < 1:
        msg = "num_evaluations must be >= 1"
        raise ValueError(msg)

    individuals = [make_individual(input_size, output_size) for _ in range(num_evaluations)]
    fitnesses = evaluate_genotypes([cast("Genotype", ind.genotype) for ind in individuals])
    for ind, fitness in zip(individuals, fitnesses, strict=True):
        ind.fitness = fitness
    if history is not None:
        history.extend(individuals)

    return min(individuals, key=lambda ind: ind.fitness)
