from typing import cast

from ariel.ec import Population

from src.environment.controller import Genotype
from src.environment.simulation import run_simulation


def evaluate(population: Population) -> Population:
    """ Run every unevaluated individual headlessly and record its fitness. """
    for ind in population.unevaluated:
        genotype = cast("Genotype", ind.genotype)
        ind.fitness = run_simulation(genotype, mode="simple")
    return population
