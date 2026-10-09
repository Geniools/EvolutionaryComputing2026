"""Tests for evolution/mutation.py.

Run with either:
    python -m pytest tests/test_mutation.py      (from src/)
    python tests/test_mutation.py                (from src/, no pytest needed)
"""

import sys
from pathlib import Path

import numpy as np

# Make `evolution` importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ariel.ec import EAOperation, Individual, Population, set_seed

from evolution.mutation import mutate, mutate_genotype

GENOTYPE = np.linspace(-1.0, 1.0, 138)  # same length as the gecko controller


def test_same_shape() -> None:
    child = mutate_genotype(GENOTYPE, sigma=0.1, rate=0.5, rng=np.random.default_rng(0))
    assert child.shape == GENOTYPE.shape


def test_input_unchanged() -> None:
    parent = GENOTYPE.copy()
    mutate_genotype(parent, sigma=0.1, rate=1.0, rng=np.random.default_rng(0))
    assert np.array_equal(parent, GENOTYPE)


def test_same_seed_same_output() -> None:
    a = mutate_genotype(GENOTYPE, sigma=0.1, rate=0.5, rng=np.random.default_rng(7))
    b = mutate_genotype(GENOTYPE, sigma=0.1, rate=0.5, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_same_seed_same_output_ariel_rng() -> None:
    # rng=None goes through ariel's FloatMutator; set_seed must make it repeatable.
    set_seed(7)
    a = mutate_genotype(GENOTYPE, sigma=0.1, rate=0.5)
    set_seed(7)
    b = mutate_genotype(GENOTYPE, sigma=0.1, rate=0.5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, GENOTYPE)


def test_rate_zero_is_identity() -> None:
    child = mutate_genotype(GENOTYPE, sigma=0.1, rate=0.0, rng=np.random.default_rng(0))
    assert np.array_equal(child, GENOTYPE)
    set_seed(0)
    assert np.array_equal(mutate_genotype(GENOTYPE, sigma=0.1, rate=0.0), GENOTYPE)


def test_rate_one_std_matches_sigma() -> None:
    # Many genes -> the empirical std of (child - parent) should be close to sigma.
    sigma = 0.3
    parent = np.zeros(100_000)
    child = mutate_genotype(parent, sigma=sigma, rate=1.0, rng=np.random.default_rng(0))
    assert abs(np.std(child - parent) - sigma) < 0.01
    set_seed(0)
    child = mutate_genotype(parent, sigma=sigma, rate=1.0)
    assert abs(np.std(child - parent) - sigma) < 0.01


def test_mutate_population_only_touches_offspring() -> None:
    # One evaluated parent and one fresh child: only the child may change.
    parent, child = Individual(), Individual()
    parent.genotype = GENOTYPE.tolist()
    parent.fitness = 1.0
    child.genotype = GENOTYPE.tolist()
    set_seed(0)
    EAOperation(mutate)(Population([parent, child]))  # same call as in main.py
    assert parent.genotype == GENOTYPE.tolist()
    assert child.genotype != GENOTYPE.tolist()
    assert child.requires_eval


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"all {len(tests)} tests passed")
