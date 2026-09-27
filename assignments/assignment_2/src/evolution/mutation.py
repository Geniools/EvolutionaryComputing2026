from ariel.ec import Population


def mutate(population: Population) -> Population:
    """Perturb offspring genotypes.

    TODO: implement. Apply an `ariel.ec.FloatMutator` operator (mutation
    strength / probability are experiment design choices, see
    config.py) to each individual produced by `crossover.crossover`, and
    set `ind.requires_eval = True` so `evaluation.evaluate` re-scores it.
    """
    raise NotImplementedError
