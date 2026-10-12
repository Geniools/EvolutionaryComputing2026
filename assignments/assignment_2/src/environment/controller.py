"""Neural-network controller and the genotype <-> weights mapping.

MuJoCo calls `nn_controller` every physics step; it returns `model.nu`
values for `data.ctrl`, scaled to [-pi/2, pi/2].

The genotype is a flat list of all weights; `decode_genotype` reshapes it
into the two weight matrices [w1, w2] of the network.
"""

import mujoco as mj
import numpy as np
import numpy.typing as npt

from config import HIDDEN_SIZE

type Weights = list[npt.NDArray[np.float64]]
type Genotype = npt.NDArray[np.float64] | list[float]


def genotype_length(input_size: int, output_size: int) -> int:
    """Total number of floats needed to encode [w1, w2]."""
    return input_size * HIDDEN_SIZE + HIDDEN_SIZE * output_size


def decode_genotype(
        genotype: Genotype,
        input_size: int,
        output_size: int,
) -> Weights:
    """Reshape a flat genotype into the controller's [w1, w2] matrices."""
    flat = np.asarray(genotype, dtype=np.float64)
    expected = genotype_length(input_size, output_size)
    if flat.size != expected:
        msg = f"expected genotype of length {expected}, got {flat.size}"
        raise ValueError(msg)

    split = input_size * HIDDEN_SIZE
    w1 = flat[:split].reshape(input_size, HIDDEN_SIZE)
    w2 = flat[split:].reshape(HIDDEN_SIZE, output_size)
    return [w1, w2]


def nn_controller(
        model: mj.MjModel,
        data: mj.MjData,
        weights: Weights,
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: inputs -> hidden (tanh) -> outputs (tanh).

    One hidden layer of HIDDEN_SIZE neurons, no biases, the same for every
    experiment.
    """
    w1, w2 = weights

    # Inputs: data.qpos, i.e. the core's position and orientation plus every
    # hinge angle.
    inputs = data.qpos

    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]
