import os
from typing import Literal, cast

type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]
type CrossoverTypes = Literal["none", "uniform", "blend", "neuron"]

# -- Reproducibility ----------------------------------------------------------
SEEDS: range = range(1, 6)  # seeds 1..5; default for main.py, override with --seeds

# -- Task -----------------------------------------------------------------
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [5.0, 0.0, 0.1]
SIM_DURATION: float = 30.0
REACH_RADIUS: float = 0.15  # metres: xy-distance within which the target counts as reached
MODE: ViewerTypes = "launcher"

# -- Fitness ------------------------------------------------------------------
DISTANCE_WEIGHT: float = 0.7  # main term: final distance to target (relative to start)
TIME_WEIGHT: float = 0.2  # time taken to reach the target
PATH_WEIGHT: float = 0.1  # wasted movement (non-straight path)

# -- Controller architecture ----------------------------------------------
HIDDEN_SIZE: int = 6

# -- EA hyperparameters -----------------------------------------------------
POPULATION_SIZE: int = 100
# Upper limit: a run stops earlier once the best fitness plateaus (below).
NUM_GENERATIONS: int = 1000

# -- Early stopping (plateau detection, see main.py) --------------------------
# Stop when the best fitness has improved by less than PLATEAU_MIN_IMPROVEMENT
# over the last PLATEAU_PATIENCE generations. Elitism keeps the best fitness
# monotone, so this cannot trigger on a temporary dip.
PLATEAU_PATIENCE: int = 50
PLATEAU_MIN_IMPROVEMENT: float = 0.005

# -- Crossover (see evolution/crossover.py) -----------------------------------
# Default variant for `crossover()` when called without `crossover_type`.
# main.py ignores it and runs every variant in turn (see its --methods).
CROSSOVER_TYPE: CrossoverTypes = cast("CrossoverTypes", os.environ.get("CROSSOVER", "uniform"))
BLEND_ALPHA: float = 0.5  # BLX-alpha: how far beyond the parents' range a child may land

# -- Mutation (see evolution/mutation.py) -------------------------------------
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

# -- Baseline -----------------------------------------------------------------
# Upper bound: with early stopping the EA may use fewer evaluations
# populations holds duels to reproduce, only half gets to reproduce, and then there is only 1 offspring per 2 parents,
# thus /4
OFFSPRING_PER_GENERATION: int = (POPULATION_SIZE // 2) // 2
BASELINE_NUM_EVALUATIONS: int = POPULATION_SIZE + NUM_GENERATIONS * OFFSPRING_PER_GENERATION
