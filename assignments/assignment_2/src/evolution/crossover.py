"""Crossover operators: every pair of selected parents produces ONE child.

Variants (pick with `config.CROSSOVER_TYPE`):
    none     control: the child is a copy of parent A (mutation-only EA)
    uniform  per weight: each gene comes from parent A or B (p = 0.5)
    blend    per weight: BLX-alpha, a new value sampled around both parents'
    neuron   per hidden neuron: its in-weights (column of w1) AND out-weights
             (row of w2) come from parent A or B together
"""

from collections.abc import Callable
from functools import cache

import numpy as np
import numpy.typing as npt
from ariel.ec import Individual, Population
from ariel.ec.generators import _rng as rng  # pyright: ignore[reportPrivateUsage]  # ariel's shared RNG, seeded by set_seed()

from config import BLEND_ALPHA, CROSSOVER_TYPE, HIDDEN_SIZE
from environment.controller import decode_genotype
from environment.simulation import get_io_sizes

type Genes = npt.NDArray[np.float64]


# -- 0. No crossover (control) -------------------------------------------------

def no_crossover(a: Genes, b: Genes) -> Genes:
    """Copy parent A; any change to the child comes from mutation alone."""
    return a.copy()


# -- 1. Uniform (per weight) -------------------------------------------------

def uniform_crossover(a: Genes, b: Genes) -> Genes:
    """Flip a coin per weight: take it from B where `from_b`, else from A."""
    from_b = rng.random(a.size) < 0.5
    return np.where(from_b, b, a)


# -- 4. Blend / BLX-alpha (per weight) -----------------------------------------

def blend_crossover(a: Genes, b: Genes) -> Genes:
    """Sample each weight uniformly from [min - alpha*d, max + alpha*d], d = |a - b|."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    margin = BLEND_ALPHA * (hi - lo)
    return rng.uniform(lo - margin, hi + margin)


# -- 6. Neuron-level (per hidden neuron) -------------------------------------

io_sizes = cache(get_io_sizes)  # compiling the model is slow; sizes never change


def neuron_crossover(a: Genes, b: Genes) -> Genes:
    """Flip a coin per hidden neuron and take ALL its weights from that parent."""
    n_in, n_out = io_sizes()
    a_w1, a_w2 = decode_genotype(a, n_in, n_out)
    b_w1, b_w2 = decode_genotype(b, n_in, n_out)

    from_b = rng.random(HIDDEN_SIZE) < 0.5
    w1 = np.where(from_b[np.newaxis, :], b_w1, a_w1)  # neuron j's in-weights  = column j of w1
    w2 = np.where(from_b[:, np.newaxis], b_w2, a_w2)  # neuron j's out-weights = row j of w2
    return np.concatenate([w1.ravel(), w2.ravel()])  # same layout decode_genotype expects


# -- Dispatch ------------------------------------------------------------------

OPERATORS: dict[str, Callable[[Genes, Genes], Genes]] = {
    "none": no_crossover,
    "uniform": uniform_crossover,
    "blend": blend_crossover,
    "neuron": neuron_crossover,
}
if CROSSOVER_TYPE not in OPERATORS:
    msg = f"unknown CROSSOVER_TYPE {CROSSOVER_TYPE!r}, choose from {list(OPERATORS)}"
    raise ValueError(msg)


def crossover(population: Population) -> Population:
    """Shuffle the selected parents, pair them up, add one child per pair.

    Parents are those tagged `selected` by `selection.parent_selection`;
    k parents give k // 2 children (an odd one out is skipped). Children
    are tagged `mutate` for `mutation.mutate`; parents are untagged so
    they are not re-used next generation.
    """
    operator = OPERATORS[CROSSOVER_TYPE]
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False))).to_list()
    rng.shuffle(parents)

    children: list[Individual] = []
    for parent_a, parent_b in zip(parents[::2], parents[1::2], strict=False):
        child = Individual()
        child.genotype = operator(genes(parent_a), genes(parent_b)).tolist()
        child.tags = {"mutate": True}
        children.append(child)

    for parent in parents:
        parent.tags = {"selected": False}
    population.extend(children)
    return population


def genes(ind: Individual) -> Genes:
    """An individual's genotype as a float array."""
    return np.asarray(ind.genotype, dtype=np.float64)
