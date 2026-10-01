import mujoco as mj
import numpy as np
import numpy.typing as npt

from config import TARGET_POSITION


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
        initial_position: npt.NDArray[np.float64],
        final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    Baseline version from the template: distance from the final position
    to the target, ignoring z. `initial_position` is accepted but unused -
    see the template's "YOUR JOB" notes for reasoned alternatives
    (distance reduced, fall penalties, ...) before changing this.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final_position[:2] - target[:2]))
