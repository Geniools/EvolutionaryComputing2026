import random

from ariel.ec import Individual, Population


def parent_selection(population: Population) -> Population:
    """Pair evaluated individuals at random and select the fitter of each pair.

    Each individual enters at most one tournament. If the number of eligible
    individuals is odd, one sits out this generation. Lower fitness is better.
    """
    for ind in population:
        ind.tags = {"selected": False}  # Clear last generation's selection.

    candidates = [ind for ind in population.alive if ind.fitness_ is not None]
    random.shuffle(candidates)

    for index in range(0, len(candidates) - 1, 2):
        pair = candidates[index:index + 2]
        winner = min(pair, key=_fitness_of)
        winner.tags = {"selected": True}

    return population


def _fitness_of(individual: Individual) -> float:
    """Treat an unevaluated individual as worse than every scored one."""
    return float("inf") if individual.fitness_ is None else individual.fitness_


def survivor_selection(
        population: Population,
        *,
        population_size: int,
        num_elites: int = 1,
) -> Population:
    """Cull tournament losers until only `population_size` individuals live.

    Lower fitness is better. The best `num_elites` cannot be culled.
    """
    if population_size < 1 or not 0 <= num_elites <= population_size:
        raise ValueError(
            "Require population_size >= 1 and 0 <= num_elites <= population_size"
        )

    alive = population.alive.to_list()
    number_to_kill = max(0, len(alive) - population_size)
    ranked = sorted(alive, key=_fitness_of)
    pool = ranked[num_elites:]

    for _ in range(number_to_kill):
        candidates = random.sample(pool, min(2, len(pool)))
        loser = max(candidates, key=_fitness_of)
        loser.alive = False
        pool.remove(loser)

    return population
