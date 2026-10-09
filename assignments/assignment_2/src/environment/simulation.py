import numpy as np
import mujoco as mj
from mujoco import viewer

from config import REACH_RADIUS, SIM_DURATION, SPAWN_POS, TARGET_POSITION, ViewerTypes

from environment.controller import Genotype, decode_genotype, nn_controller
from environment.fitness import fitness_function, get_core_position
from environment.world import build_robot, build_world

from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder


def _compile_model() -> tuple[mj.MjModel, mj.MjData]:
    """Build world + robot, spawn, and compile - shared setup step."""
    world = build_world()
    robot = build_robot()
    world.spawn(
            robot.spec,
            position=SPAWN_POS,
            correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


def get_io_sizes() -> tuple[int, int]:
    """Return (input_size, output_size) for the current world/robot."""
    mj.set_mjcb_control(None)
    model, data = _compile_model()
    return len(data.qpos), model.nu


def run_simulation(
        genotype: Genotype,
        mode: ViewerTypes = "simple",
        video_output_folder: str | None = None,
) -> float:
    """Set up the world, run one simulation, and return the fitness.

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    model, data = _compile_model()
    input_size = len(data.qpos)
    output_size = model.nu
    weights = decode_genotype(genotype, input_size, output_size)

    target = np.asarray(TARGET_POSITION)
    reach_time: float | None = None
    min_dist = float("inf")
    path_length = 0.0
    prev_position = get_core_position(data)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        # "nonlocal" allows the callback to modify variables in the enclosing scope (run_simulation function)
        nonlocal reach_time, min_dist, path_length, prev_position
        position = get_core_position(d)
        dist = distance_to_target(position, target)
        min_dist = min(min_dist, dist)
        # distance_to_target is a plain xy distance, so it also measures a step.
        path_length += distance_to_target(position, prev_position)
        prev_position = position
        if reach_time is None and dist <= REACH_RADIUS:
            reach_time = float(d.time)
        actions = nn_controller(m, d, weights)

        # Blown-up weights silently write NaN into d.ctrl (template warning).
        # assert np.all(np.isfinite(actions)), "controller produced NaN/inf"

        # DIRECT application (see the controller contract in controller.py).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    initial_position = get_core_position(data)

    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            viewer.launch(model=model, data=data)
        case "simple":
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            recorder = VideoRecorder(output_folder=video_output_folder or ".")
            video_renderer(model, data, duration=SIM_DURATION, video_recorder=recorder)
        case "frame":
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            viewer.launch(model=model, data=data)

    mj.set_mjcb_control(None)

    final_position = get_core_position(data)
    return fitness_function(
            initial_position,
            final_position,
            time_to_target=reach_time,
            min_distance_to_target=min_dist,
            path_length=path_length,
    )
