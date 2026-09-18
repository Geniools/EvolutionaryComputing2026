"""EC A1 - evolving robot bodies with ARIEL.

Research question: how strict should parent selection be?
For example: Variant A lets the top 50% reproduce, variant B the top 10%. 
The rest stays constant

fitness = mean tree edit distance to every target body, plus one standard deviation across those distances

Run `python A1_implementation_2026.py --> for one run
Run `python run_experiments.py` --> for sequential testing (full experiment).
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
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree import TreeGenome
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "frame", "none"]
type CullModes = Literal["random", "tournament"]

# --- RANDOM GENERATOR SETUP --- #
# One generator for everything: ARIEL's tree operators use `random`, so
# seeding that seeds the whole run. ("nde" would also need numpy and torch.)


def set_seed(seed: int) -> None:
    """Seed every random source this file uses."""
    random.seed(seed)


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

# --- THE RESEARCH VARIABLE --- #
TRUNCATION_FRACTION: float = 0.1  # 0.5 = variant A, 0.1 = variant B

# --- OTHER VARIABLES --- #
POPULATION_SIZE: int = 100
NUM_OFFSPRING: int = 50  # children per generation
NUM_STEPS: int = 200  # generations
MUTATION_PROBABILITY: float = 0.2
CULL_MODE: CullModes = "tournament"  # or "random"
NUM_ELITES: int = 1  # best N are never killed

# Crossover returns 2 children per call
assert NUM_OFFSPRING % 2 == 0, "NUM_OFFSPRING must be even"

# The random-search baseline gets the same budget
EVALUATION_BUDGET: int = POPULATION_SIZE + NUM_OFFSPRING * NUM_STEPS


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


def _fitness_of(individual: Individual) -> float:
    """Fitness of an individual, where "not scored yet" is the worst possible."""
    if individual.fitness_ is None:
        return float("inf")
    return individual.fitness_


def _num_modules(individual: Individual) -> int:
    """Number of modules in an individual's genotype."""
    return len(individual.genotype["nodes"])


# ============================================================================ #
#  THE EA STEPS
# ============================================================================ #
# Steps pass information to each other through `individual.tags`. The `tags`
# setter MERGES and tags are saved to the database, so any tag a step sets
# must also be cleared by a step, or it lingers for the individual's whole life.
# ============================================================================ #


def make_individual() -> Individual:
    ind = Individual()
    tree = random_tree(NUM_OF_MODULES)
    ind.genotype = tree.to_dict()
    return ind


def evaluate(
        population: Population,
        *,
        targets: list[nx.DiGraph],
) -> Population:
    """Evaluate every *unevaluated* individual in the population.

    This is the only place where the genotype is decoded into a phenotype
    and scored against the target set. The EA does not know anything about
    the phenotype or the targets.
    """
    for ind in population.unevaluated:
        tree = TreeGenome.from_dict(ind.genotype)
        body = tree.to_networkx()
        ind.fitness = fitness_function(body, targets)
    return population


def parent_selection(
        population: Population,
        *,
        truncation_fraction: float,
        num_offspring: int,
) -> Population:
    """Truncation selection: only the best `truncation_fraction` may reproduce. (RESEARCH VARIABLE) 
    
    We then draw a fixed number of pairs from that group (with replacement) so the number of children does not depend on
    how strict we were. Each parent is tagged with the pair numbers it is in.
    """
    # Clear last generation's pairings, or they come back (tags merge)
    for ind in population:
        ind.tags = {"pairs": []}

    # Not Population.best(sort="min"): ARIEL's sort key maps an unscored
    # individual to -inf, so it would rank it BEST when minimising.
    scored = [ind for ind in population if ind.fitness_ is not None]
    scored.sort(key=_fitness_of)

    pool = scored[: max(2, round(truncation_fraction * len(scored)))]
    if len(pool) < 2:
        return population

    for pair_number in range(num_offspring // 2):
        for parent in random.sample(pool, 2):  # two different parents
            pairs = list(parent.tags.get("pairs", []))
            pairs.append(pair_number)
            parent.tags = {"pairs": pairs}

    return population


def crossover(population: Population) -> Population:
    """Make two children per pair tagged by `parent_selection`."""
    pairs: dict[int, list[Individual]] = {}
    for ind in population:
        for pair_number in ind.tags.get("pairs", []):
            pairs.setdefault(int(pair_number), []).append(ind)

    children: list[Individual] = []
    for members in pairs.values():
        if len(members) != 2:  # should not happen
            continue
        parent_a, parent_b = members
        tree_a = TreeGenome.from_dict(parent_a.genotype)
        tree_b = TreeGenome.from_dict(parent_b.genotype)

        # ARIEL returns unchanged copies of the parents if the swap would
        # make an invalid body --> some children are clones.
        g_a, g_b = crossover_subtree(tree_a, tree_b)

        for genome in (g_a, g_b):
            child = Individual()
            child.genotype = genome.to_dict()
            child.tags = {"mutate": True, "pairs": []}
            children.append(child)

    population.extend(children)
    return population


def mutate(
        population: Population,
        *,
        mutation_probability: float,
        num_modules: int,
) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        # Clear before the dice roll, so each child gets exactly one chance.
        ind.tags = {"mutate": False}

        # Not every eligible individual actually mutates - roll the dice first.
        if random.random() >= mutation_probability:
            continue

        new = TreeGenome.from_dict(ind.genotype)
        mutation_type = random.choices(
                ["point", "subtree", "shrink", "hoist"],
                weights=[0.4, 0.4, 0.1, 0.1],
        )[0]

        if mutation_type == "point":
            # Point mutation: change node type/rotation
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            # Subtree mutation: replace subtree with new random tree
            mutate_subtree_replacement(new, max_modules=num_modules)
        elif mutation_type == "shrink":
            # Shrink mutation: replace node+subtree with single leaf
            mutate_shrink(new)
        elif mutation_type == "hoist":
            # Hoist mutation: promote child to replace parent
            mutate_hoist(new)

        ind.genotype = new.to_dict()
        ind.requires_eval = True
    return population


def survivor_selection(
        population: Population,
        *,
        population_size: int,
        cull_mode: CullModes,
        num_elites: int,
) -> Population:
    """Kill individuals one at a time until `population_size` is left.

    "tournament"  --> kills the worse of two random individuals 
    "random" --> kills one random individual, which applies no pressure at all. 
    
    The best `num_elites` are kept out of the draw (elitism).
    """
    alive = population.alive.to_list()
    number_to_kill = max(0, len(alive) - population_size)
    if number_to_kill == 0:
        return population

    pool = sorted(alive, key=_fitness_of)[num_elites:]

    for _ in range(number_to_kill):
        if cull_mode == "random":
            loser = random.choice(pool)
        else:
            loser = max(random.sample(pool, min(2, len(pool))), key=_fitness_of)
        loser.alive = False
        pool.remove(loser)

    return population


def record_stats(population: Population, *, log: list[dict]) -> Population:
    """Log this generation. Inluding the two std columns (diversity measures)"""
    alive = [ind for ind in population.alive if ind.fitness_ is not None]
    if not alive:
        return population

    fitnesses = np.array([ind.fitness_ for ind in alive], dtype=float)
    modules = np.array([_num_modules(ind) for ind in alive], dtype=float)

    log.append({
        "generation": len(log),
        "best": float(fitnesses.min()),
        "mean": float(fitnesses.mean()),
        "worst": float(fitnesses.max()),
        "fitness_std": float(fitnesses.std()),
        "mean_modules": float(modules.mean()),
        "modules_std": float(modules.std()),
        "population_size": len(alive),
    })
    return population


# ============================================================================ #
#  RUNNING ONE EA
# ============================================================================ #


def run_ea(
        targets: list[nx.DiGraph],
        *,
        seed: int,
        truncation_fraction: float = TRUNCATION_FRACTION,
        population_size: int = POPULATION_SIZE,
        num_offspring: int = NUM_OFFSPRING,
        num_steps: int = NUM_STEPS,
        mutation_probability: float = MUTATION_PROBABILITY,
        cull_mode: CullModes = CULL_MODE,
        num_elites: int = NUM_ELITES,
        db_file_path: Path | None = None,
        quiet: bool = False,
) -> tuple[list[dict], nx.DiGraph]:
    """Run one EA and return (per-generation log, best body found).

    Each run needs its OWN `db_file_path`: ARIEL queries the whole
    database, so runs sharing a file would mix.
    """
    set_seed(seed)

    log: list[dict] = []

    initial_population = Population([
        make_individual() for _ in range(population_size)
    ])
    evaluate(initial_population, targets=targets)
    record_stats(initial_population, log=log)  # generation 0

    ea_operations: list[EAOperation] = [
        EAOperation(
                parent_selection,
                truncation_fraction=truncation_fraction,
                num_offspring=num_offspring,
        ),
        EAOperation(crossover),
        EAOperation(
                mutate,
                mutation_probability=mutation_probability,
                num_modules=NUM_OF_MODULES,
        ),
        EAOperation(evaluate, targets=targets),
        EAOperation(
                survivor_selection,
                population_size=population_size,
                cull_mode=cull_mode,
                num_elites=num_elites,
        ),
        EAOperation(record_stats, log=log),
    ]

    ea = EA(
            initial_population,
            ea_operations,
            num_steps=num_steps,
            is_maximisation=False,
            db_file_path=db_file_path,
            quiet=quiet,
    )
    ea.run()

    best = ea.get_solution("best", only_alive=False)
    best_body = TreeGenome.from_dict(best.genotype).to_networkx()
    return log, best_body


def run_random_search(
        targets: list[nx.DiGraph],
        *,
        seed: int,
        population_size: int = POPULATION_SIZE,
        num_offspring: int = NUM_OFFSPRING,
        num_steps: int = NUM_STEPS,
) -> tuple[list[dict], nx.DiGraph]:
    """Baseline: sample random bodies on the same budget as the EA.
    
    "best" is the best found so far
    """
    set_seed(seed)

    log: list[dict] = []
    best_fitness = float("inf")
    best_body: nx.DiGraph | None = None

    def sample_batch(size: int, generation: int) -> None:
        nonlocal best_fitness, best_body
        fitnesses = []
        module_counts = []
        for _ in range(size):
            genome = random_tree(NUM_OF_MODULES)
            body = genome.to_networkx()
            fitness = fitness_function(body, targets)
            fitnesses.append(fitness)
            module_counts.append(len(genome.nodes))
            if fitness < best_fitness:
                best_fitness, best_body = fitness, body

        batch = np.array(fitnesses, dtype=float)
        modules = np.array(module_counts, dtype=float)
        log.append({
            "generation": generation,
            "best": best_fitness,
            "mean": float(batch.mean()),
            "worst": float(batch.max()),
            "fitness_std": float(batch.std()),
            "mean_modules": float(modules.mean()),
            "modules_std": float(modules.std()),
            "population_size": size,
        })

    sample_batch(population_size, 0)
    for generation in range(1, num_steps + 1):
        sample_batch(num_offspring, generation)

    assert best_body is not None
    return log, best_body


def main() -> None:
    """One run with the settings at the top of this file."""
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
    console.log(f"truncation    : top {TRUNCATION_FRACTION:.0%} reproduce")
    console.log(f"budget        : {EVALUATION_BUDGET} evaluations")

    log, best_body = run_ea(
            targets,
            seed=42,
            db_file_path=DATA / "single_run" / "database.db",
    )

    console.log("--- Results ---")
    console.log(f"generation 0   : best {log[0]['best']:.4f}")
    console.log(f"generation {log[-1]['generation']:<3} : best {log[-1]['best']:.4f}")
    console.log(f"best body      : {best_body.number_of_nodes()} modules")
    console.log(
            "per-target     : "
            + ", ".join(f"{d:.1f}" for d in distances_to_targets(best_body, targets)),
    )

    show_body(best_body, mode=MODE, file_name="best_body")


if __name__ == "__main__":
    main()
