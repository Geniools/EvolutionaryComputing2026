from ariel.ec import Individual


def random_search(
        input_size: int,
        output_size: int,
        num_evaluations: int,
) -> Individual:
    """Evaluate `num_evaluations` random genotypes and return the best.

    TODO: implement. Draw a fresh individual via `genotype.make_individual`,
    score it with `simulation.run_simulation(..., mode="simple")`, and keep
    the best (lowest fitness) seen across `num_evaluations` draws.
    """
    raise NotImplementedError
