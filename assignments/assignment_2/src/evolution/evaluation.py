from multiprocessing.pool import Pool
from typing import cast

from ariel.ec import Population

from environment.controller import Genotype
from environment.simulation import run_simulation

# Worker processes for evaluation. None (the default) evaluates one by one in
# this process; main.py sets it when started with --workers.
pool: Pool | None = None


def simulate(genotype: Genotype) -> float:
    """Run one genotype headlessly and return its fitness."""
    return run_simulation(genotype, mode="simple")


def evaluate_genotypes(genotypes: list[Genotype]) -> list[float]:
    """Fitness of every genotype, in order; spread over `pool` when it is set."""
    if pool is None:
        return [simulate(genotype) for genotype in genotypes]
    return pool.map(simulate, genotypes)


def evaluate(population: Population) -> Population:
    """Run every unevaluated individual headlessly and record its fitness. """
    individuals = list(population.unevaluated)
    fitnesses = evaluate_genotypes([cast("Genotype", ind.genotype) for ind in individuals])
    for ind, fitness in zip(individuals, fitnesses, strict=True):
        ind.fitness = fitness
    return population
