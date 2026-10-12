import mujoco as mj

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.simulation.environments import SimpleFlatWorld

from config import REACH_RADIUS, SPAWN_POS, TARGET_POSITION

# Extra floor (metres) drawn beyond the spawn/target so the target is on it.
FLOOR_MARGIN: float = 5.0
MARKER_RGBA: list[float] = [0.1, 0.9, 0.2, 0.8]  # green, slightly transparent
BEACON_HEIGHT: float = 1.0  # metres: pole so the target is visible from afar


def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    The default 10 x 10 m floor only reaches x = 5, so it is enlarged to
    cover both spawn and target.
    """
    half = max(abs(c) for c in (*SPAWN_POS[:2], *TARGET_POSITION[:2])) + FLOOR_MARGIN
    world = SimpleFlatWorld(floor_size=(2 * half, 2 * half, 1))
    add_target_marker(world)
    return world


def add_target_marker(world: SimpleFlatWorld) -> None:
    """Draw the target: a floor disc of radius REACH_RADIUS plus a thin pole.

    Visual only: contype = conaffinity = 0 means it never collides with the
    robot, and a body without joints adds no entries to data.qpos, so the
    controller's input size and the fitness are unaffected.
    """
    x, y, _ = TARGET_POSITION
    body = world.spec.worldbody.add_body(name="target_marker", pos=[x, y, 0.0])
    visual_only = {"contype": 0, "conaffinity": 0, "rgba": MARKER_RGBA}
    body.add_geom(  # disc on the floor = the "reached" zone
            name="target_zone",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[REACH_RADIUS, 0.001, 0],  # radius, half-height
            pos=[0, 0, 0.001],
            **visual_only,
    )
    body.add_geom(  # beacon pole
            name="target_beacon",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[0.01, BEACON_HEIGHT / 2, 0],
            pos=[0, 0, BEACON_HEIGHT / 2],
            **visual_only,
    )


def build_robot() -> CoreModule:
    """Create the robot body."""
    return gecko()
