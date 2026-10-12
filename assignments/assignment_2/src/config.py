import os
from typing import Literal, cast

type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]
type CrossoverTypes = Literal["none", "uniform", "blend", "neuron"]

# -- Reproducibility ----------------------------------------------------------
SEEDS: range = range(1, 16)  # seeds 1..15, as used for the final runs; override with --seeds

# -- Task -----------------------------------------------------------------
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]
SIM_DURATION: float = 15.0
REACH_RADIUS: float = 0.15  # metres: xy-distance within which the target counts as reached

# -- Fitness ------------------------------------------------------------------
DISTANCE_WEIGHT: float = 0.7  # main term: final distance to target (relative to start)
TIME_WEIGHT: float = 0.2  # time taken to reach the target
PATH_WEIGHT: float = 0.1  # wasted movement (non-straight path)

# -- Controller architecture ----------------------------------------------
HIDDEN_SIZE: int = 6

# -- EA hyperparameters -----------------------------------------------------
POPULATION_SIZE: int = 100
NUM_GENERATIONS: int = 100

# -- Crossover (see evolution/crossover.py) -----------------------------------
# Default variant for `crossover()` when called without `crossover_type`.
# main.py ignores it and runs every variant in turn (see its --methods).
CROSSOVER_TYPE: CrossoverTypes = cast("CrossoverTypes", os.environ.get("CROSSOVER", "uniform"))
BLEND_ALPHA: float = 0.5  # BLX-alpha: how far beyond the parents' range a child may land

# -- Analysis (see analysis/analyze.py) ------------------------------------------
NUM_SEED_PLOTS: int = 3  # single runs drawn in grey in the fitness plot: the first N seeds

# -- Baseline -----------------------------------------------------------------
# Random search gets the same number of evaluations as one EA run:
# the initial population, plus the offspring of every generation. Each
# generation, pairwise tournaments select half the population as parents
# and every pair of parents makes one child, so POPULATION_SIZE // 4 offspring.
OFFSPRING_PER_GENERATION: int = POPULATION_SIZE // 4
BASELINE_NUM_EVALUATIONS: int = POPULATION_SIZE + OFFSPRING_PER_GENERATION * NUM_GENERATIONS
