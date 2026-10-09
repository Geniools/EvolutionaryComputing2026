import mujoco as mj
import numpy as np
import numpy.typing as npt

from config import (
    DISTANCE_WEIGHT,
    PATH_WEIGHT,
    SIM_DURATION,
    TARGET_POSITION,
    TIME_WEIGHT,
)

from ariel.simulation.tasks.targeted_locomotion import (
    distance_to_target,
    fitness_delta_distance,
    fitness_direct_path,
    fitness_speed_to_target,
)


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
        initial_position: npt.NDArray[np.float64],
        final_position: npt.NDArray[np.float64],
        time_to_target: float | None = None,
        min_distance_to_target: float = 0.0,
        path_length: float = 0.0,
        duration: float = SIM_DURATION,
) -> float:
    """Score one evaluation. LOWER IS BETTER (0 = reached instantly, straight).

    fitness = DISTANCE_WEIGHT * d_final / d_initial      (main term)
            + TIME_WEIGHT     * speed_to_target          (arrival time)
            + PATH_WEIGHT     * wasted_path / d_initial  (straightness)

    - distance: remaining distance relative to the start distance
      (0 = on target, 1 = no progress, >1 = moved away).
    - speed (ariel `fitness_speed_to_target`): in [0, 1] if the target was
      reached (arrival time / duration), otherwise 1 + closest distance.
    - path (ariel `fitness_direct_path`): its penalty is the wasted movement
      (path length - straight-line displacement), isolated by subtracting
      the delta-distance part that `fitness_direct_path` also contains.
    """
    target = np.asarray(TARGET_POSITION)
    d0 = max(distance_to_target(initial_position, target), 1e-6)
    d_final = distance_to_target(final_position, target)

    speed_fitness = fitness_speed_to_target(
            time_to_target, duration, min_distance_to_target,
    )
    path_penalty = (
            fitness_direct_path(initial_position, final_position, target, path_length)
            - fitness_delta_distance(initial_position, final_position, target)
    )

    return (
            DISTANCE_WEIGHT * d_final / d0
            + TIME_WEIGHT * speed_fitness
            + PATH_WEIGHT * path_penalty / d0
    )
