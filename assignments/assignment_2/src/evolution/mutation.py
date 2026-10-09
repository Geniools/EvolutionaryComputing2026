"""Gaussian mutation for real-valued genotypes (M3).

Two functions:
- `mutate(population, ...)`             -> the EA step used in main.py
  (`EAOperation(mutate)`): mutates every offspring in the population.
- `mutate_genotype(genotype, ...)`      -> mutates ONE genotype (pure function),
  used by `mutate` and handy for testing.
"""

import numpy as np
import numpy.typing as npt
from ariel.ec import FloatMutator, Population

# -- Default mutation parameters ---------------------------------------------
# Standard deviation of the Gaussian noise added to a mutated gene.
# The initial weights are drawn from N(0, 0.5) (evolution/genotype.py), so 0.25
# is half the typical weight size: big enough to move, small enough to stay
# near the parent. Chosen with a small sigma pilot: 0.02/0.1 improved slowly,
# 0.25/0.5 faster and consistently, 1.0 erratically.
MUTATION_SIGMA: float = 0.25

# Per-gene probability of being mutated. With a genotype of 138 weights
# (gecko: 15 inputs x 6 hidden + 6 hidden x 8 outputs), 0.1 changes ~14 genes
# per child on average.
MUTATION_RATE: float = 0.1


def mutate(
        population: Population,
        sigma: float = MUTATION_SIGMA,
        rate: float = MUTATION_RATE,
) -> Population:
    """Mutate every offspring (= not yet evaluated individual) in place.

    Offspring are recognised by `requires_eval`, which is True for freshly
    created individuals (e.g. children from `crossover.crossover`), so no
    extra tag is needed. Uses ariel's RNG, so `set_seed(seed)` makes the
    whole EA reproducible.

    Note: if there is no crossover (mutation-only), the crossover step must
    still create copies of the parents as new individuals, otherwise there
    are no offspring for this function to mutate.
    """
    for ind in population.unevaluated:
        child = mutate_genotype(ind.genotype, sigma=sigma, rate=rate)
        ind.genotype = child.tolist()  # the DB stores genotypes as JSON lists
        ind.requires_eval = True  # re-score after mutation
    return population


def mutate_genotype(
        genotype: npt.ArrayLike,
        sigma: float = MUTATION_SIGMA,
        rate: float = MUTATION_RATE,
        rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Return a mutated copy of `genotype` (the input is never modified).

    Each gene independently, with probability `rate`, gets noise from
    N(0, sigma) added:  x_i' = x_i + N(0, sigma^2)  if u_i < rate, else x_i.

    The two knobs:
    - sigma (step size): too large and a child no longer resembles its parent,
      so the EA degrades into random search; too small and children are
      near-copies, so the population stagnates.
    - rate (how many genes change): rate * len(genotype) genes change on
      average; rate=0 copies the parent, rate=1 perturbs every weight.

    No clipping: the controller squashes everything through tanh, so any
    real-valued weight is valid.

    rng:
    - None (default) -> `ariel.ec.FloatMutator.gaussian`, which draws from
      ariel's shared RNG (seeded with `ariel.ec.set_seed`);
    - a `numpy.random.Generator` -> the same formula in plain numpy, because
      FloatMutator has no `rng` argument (useful for tests).
    """
    parent = np.asarray(genotype, dtype=np.float64)

    if rng is None:
        child = FloatMutator.gaussian(parent, std=sigma, mutation_probability=rate)
        return np.asarray(child, dtype=np.float64)

    # Same steps as FloatMutator.gaussian, but with our own generator.
    noise = rng.normal(loc=0.0, scale=sigma, size=parent.shape)
    mask = rng.random(parent.shape) < rate  # True = this gene mutates
    return np.where(mask, parent + noise, parent)  # np.where builds a new array
