"""Per-generation statistics for Assignment 2 EA runs."""

from ariel.ec import Population


def record_stats(population: Population, *, log: list[dict]) -> Population:
    """Record best, mean, and worst fitness among living individuals."""
    fitnesses = [ind.fitness for ind in population.alive if ind.fitness_ is not None]
    if not fitnesses:
        raise ValueError("No living, evaluated individuals to record")

    log.append({
        "generation": len(log),
        "best": min(fitnesses),
        "mean": sum(fitnesses) / len(fitnesses),
        "worst": max(fitnesses),
        "population_size": len(fitnesses),
    })
    return population
