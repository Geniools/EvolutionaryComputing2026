"""EC A1 template code - evolving robot morphologies with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo* file for starting you out with assignment 1. 
It samples one body at random, decodes it, scores it
against a set of target bodies, and shows you the result. 

*Your Job* section at the bottom of this file summarises the programming task. Full assignment description can be found in the pdf file on Canvas.


THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve a robot BODY that is as structurally close as possible to a whole set
of given target bodies at once.

    fitness = mean tree edit distance to every body in TARGET_DIR,
              plus one standard deviation across those per-target distances
"""

# Standard library
import random
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
from mujoco import viewer

from ariel.ec import Individual, Population, EAOperation, EA
from ariel.ec.genotypes.tree import TreeGenome
# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec.genotypes.tree.operators import (
    random_tree, crossover_subtree, mutate_replace_node,
    mutate_subtree_replacement, mutate_shrink, mutate_hoist
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module for their randomness.
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]

POPULATION_SIZE: int = 100
MUTATION_PROBABILITY: float = 0.2
NUM_STEPS = 50


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def fitness_function(
        body: nx.DiGraph,
        targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)


def show_body(
        body: nx.DiGraph,
        mode: ViewerTypes = MODE,
        file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
            robot.spec,
            position=SPAWN_POS,
            correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


def make_individual() -> Individual:
    ind = Individual()
    tree = random_tree(NUM_OF_MODULES)
    ind.genotype = tree.to_dict()
    return ind


def evaluate(population: Population) -> Population:
    """Evaluate every *unevaluated* individual in the population.

    This is the only place where the genotype is decoded into a phenotype
    and scored against the target set. The EA does not know anything about
    the phenotype or the targets.
    """
    targets = load_targets()
    for ind in population.unevaluated:
        tree = TreeGenome.from_dict(ind.genotype)
        body = tree.to_networkx()
        ind.fitness = fitness_function(body, targets)
    return population


def parent_selection(population: Population) -> Population:

    shuffled = population.shuffle()
    n = len(shuffled)
    for idx in range(0, n - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ <= ind_b.fitness_:
                ind_a.tags = {"selected": True}
                ind_b.tags = {"selected": False}
            else:
                ind_a.tags = {"selected": False}
                ind_b.tags = {"selected": True}

    # Odd population size: the last individual has no opponent to compare against
    if n % 2 == 1:
        shuffled[-1].tags = {"selected": True}

    return shuffled


def crossover(population: Population) -> Population:
    selected = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    shuffled = selected.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        parent_a = shuffled[idx]
        parent_b = shuffled[idx + 1]
        tree_a = TreeGenome.from_dict(parent_a.genotype)
        tree_b = TreeGenome.from_dict(parent_b.genotype)

        g_a, g_b = crossover_subtree(tree_a, tree_b)

        child_a = Individual()
        child_a.genotype = g_a.to_dict()
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = g_b.to_dict()
        child_b.tags = {"mutate": True}
        population.extend([child_a, child_b])
    return population


def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        # Not every eligible individual actually mutates - roll the dice first.
        if RNG.random() >= MUTATION_PROBABILITY:
            continue

        new = TreeGenome.from_dict(ind.genotype)
        mutation_type = RNG.choice(
                ["point", "subtree", "shrink", "hoist"], p=[0.4, 0.4, 0.1, 0.1],
        )

        if mutation_type == "point":
            # Point mutation: change node type/rotation
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            # Subtree mutation: replace subtree with new random tree
            mutate_subtree_replacement(new, max_modules=NUM_OF_MODULES)
        elif mutation_type == "shrink":
            # Shrink mutation: replace node+subtree with single leaf
            mutate_shrink(new)
        elif mutation_type == "hoist":
            # Hoist mutation: promote child to replace parent
            mutate_hoist(new)

        ind.genotype = new.to_dict()
        ind.requires_eval = True
    return population


def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    n = len(shuffled)
    alive_count = n
    for idx in range(0, n - 1, 2):
        if alive_count <= POPULATION_SIZE:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        fitness_a = ind_a.fitness_ if ind_a.fitness_ is not None else float("inf")
        fitness_b = ind_b.fitness_ if ind_b.fitness_ is not None else float("inf")
        # Kill whichever has the HIGHER (worse) fitness - minimisation.
        if fitness_a >= fitness_b:
            ind_a.alive = False
        else:
            ind_b.alive = False
        alive_count -= 1

    # Odd number of alive individuals: kill the last one
    # if n % 2 == 1 and alive_count > POPULATION_SIZE:
    #     shuffled[-1].alive = False

    return population


def main() -> None:
    # Load targets
    targets = load_targets()

    console.log(f"module budget : {NUM_OF_MODULES}")
    console.log(f"targets       : {len(targets)} bodies from {TARGET_DIR.name}")
    console.log(
            "target sizes  : "
            + ", ".join(str(t.number_of_nodes()) for t in targets),
    )

    # How far apart are the targets from each other? Your fitness cannot go
    # below the best possible compromise, and this is the clue to where that is.
    spread = [
        tree_edit_distance(a, b)
        for i, a in enumerate(targets)
        for b in targets[i + 1:]
    ]
    console.log(f"target spread : mean pairwise distance {np.mean(spread):.2f}")

    # Initialize population
    initial_population = Population(make_individual() for _ in range(POPULATION_SIZE))
    initial_population = evaluate(initial_population)

    ea_operations: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection)
    ]

    ea = EA(
            initial_population,
            ea_operations,
            num_steps=NUM_STEPS,
            is_maximisation=False
    )
    ea.run()

    best = ea.get_solution("best", only_alive=False)
    best_tree = TreeGenome.from_dict(best.genotype)
    best_body = best_tree.to_networkx()

    console.log("--- Results ---")
    console.log(f"best = {ea.get_solution('best', only_alive=False)}")
    console.log(f"median = {ea.get_solution('median', only_alive=False)}")
    console.log(f"worst = {ea.get_solution('worst', only_alive=False)}")

    show_body(best_body, mode=MODE, file_name="best_body")


if __name__ == "__main__":
    main()

# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above samples ONE body at random and scores it. Your task is to
# replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# For morphology-specific evolution with the tree encoding, read:
#
#     examples/c_genotypes/1_body_evolution_tree.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- GENOTYPE - DEPENDENT "GOTCHA"S -------------------------
#
#   TREE: VARIABLE LENGTH - Tree genotypes grow; without pressure against it they
#     will grow forever, and every extra module costs an edit.
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing - repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline, a good standard is at least a random search.
#   Keep the encoding, module budget and target set identical across
#     everything you compare, change one thing at a time.
#
# ============================================================================ #
