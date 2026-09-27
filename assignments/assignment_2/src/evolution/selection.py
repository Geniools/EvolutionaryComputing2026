from ariel.ec import Population


def parent_selection(population: Population) -> Population:
    """Choose which individuals become parents this generation.

    TODO: implement (e.g. tournament, rank-based, ...). Mark chosen
    parents, e.g. via `ind.tags = {"selected": True}`.
    """
    raise NotImplementedError


def survivor_selection(population: Population) -> Population:
    """Decide which individuals survive into the next generation.

    TODO: implement (e.g. (mu + lambda), elitism, ...). Set
    `ind.alive = False` for individuals that do not survive, keeping the
    population near `ariel.ec.config.target_population_size`.
    """
    raise NotImplementedError
