from ariel.ec import Population


def crossover(population: Population) -> Population:
    """Recombine selected parents into new offspring individuals.

    TODO: implement. Read parents tagged by `selection.parent_selection`,
    call an `ariel.ec.Crossover` operator on their genotypes, and
    `population.extend([...])` with the resulting children (tagged so
    `mutation.mutate` and `evaluation.evaluate` know to process them).
    """
    raise NotImplementedError
