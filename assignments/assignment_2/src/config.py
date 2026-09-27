from typing import Literal

type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# -- Reproducibility ----------------------------------------------------------
SEED: int = 42

# -- Task -----------------------------------------------------------------
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]
SIM_DURATION: float = 15.0
MODE: ViewerTypes = "launcher"

# -- Controller architecture ----------------------------------------------
HIDDEN_SIZE: int = 6

# -- EA hyperparameters -----------------------------------------------------
# Not specified by the assignment. Choose these deliberately and keep them
# fixed across the runs/seeds you compare against each other.
POPULATION_SIZE: int | None = None  # TODO: decide
NUM_GENERATIONS: int | None = None  # TODO: decide

# -- Baseline -----------------------------------------------------------------
# "Random search with the same evaluation budget" (template, YOUR JOB section).
# Should equal POPULATION_SIZE * NUM_GENERATIONS once those are set.
BASELINE_NUM_EVALUATIONS: int | None = None  # TODO: decide
