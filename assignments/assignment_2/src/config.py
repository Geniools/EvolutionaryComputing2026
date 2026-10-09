import os
from typing import Literal, cast

type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]
type CrossoverTypes = Literal["none", "uniform", "blend", "neuron"]

# -- Reproducibility ----------------------------------------------------------
SEED: int = 42

# -- Task -----------------------------------------------------------------
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]
SIM_DURATION: float = 15.0
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
NUM_GENERATIONS: int = 20

# -- Crossover (see evolution/crossover.py) -----------------------------------
# Variant under study; override per run without editing code:
#   PowerShell: $env:CROSSOVER = "blend"; python src/main.py
#   bash:       CROSSOVER=blend python src/main.py
CROSSOVER_TYPE: CrossoverTypes = cast("CrossoverTypes", os.environ.get("CROSSOVER", "uniform"))
BLEND_ALPHA: float = 0.5  # BLX-alpha: how far beyond the parents' range a child may land

# -- Baseline -----------------------------------------------------------------
# "Random search with the same evaluation budget" (template, YOUR JOB section).
# Should equal POPULATION_SIZE * NUM_GENERATIONS once those are set.
BASELINE_NUM_EVALUATIONS: int = POPULATION_SIZE * NUM_GENERATIONS
