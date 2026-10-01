from ariel.ec import FloatsGenerator, Individual

from environment.controller import genotype_length

# Initial sampling scale, taken from the template's `make_random_weights`
# (RNG.normal(scale=0.5, ...)) - the same distribution, now used to seed
# the initial population rather than to fix the controller once.
INIT_SCALE: float = 0.5


def make_individual(input_size: int, output_size: int) -> Individual:
    """Create one individual with a randomly initialised genotype."""
    ind = Individual()
    ind.genotype = FloatsGenerator.normal(
            mean=0.0,
            std=INIT_SCALE,
            size=genotype_length(input_size, output_size),
    )
    return ind
