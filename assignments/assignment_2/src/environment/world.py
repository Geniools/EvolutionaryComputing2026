from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld


def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in."""
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body."""
    return gecko()
