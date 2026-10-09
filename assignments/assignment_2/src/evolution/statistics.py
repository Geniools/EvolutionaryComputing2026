"""Per-generation statistics for Assignment 2 EA runs."""

from statistics import pstdev

from ariel.ec import Population


def record_stats(population: Population, *, log: list[dict]) -> Population:
    """Record fitness statistics among living individuals."""
    fitnesses = [ind.fitness for ind in population.alive if ind.fitness_ is not None]
    if not fitnesses:
        raise ValueError("No living, evaluated individuals to record")

    log.append({
        "generation": len(log),
        "best": min(fitnesses),
        "mean": sum(fitnesses) / len(fitnesses),
        "worst": max(fitnesses),
        "fitness_std": pstdev(fitnesses),
        "population_size": len(fitnesses),
    })
    return population
