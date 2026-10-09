"""Tiny helpers so every run writes its output where analysis/analyze.py looks.

Folder layout (proposal, to agree on as a team):
    results/<config_name>/seed_<n>/database.db         ariel.ec SQLite log
    results/<config_name>/seed_<n>/best_genotype.npy   best controller weights
    results/<config_name>/seed_<n>/evaluations.csv     (random search only)
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt


def run_dir(results_root: str | Path, config_name: str, seed: int) -> Path:
    """Return (and create) results/<config_name>/seed_<n>/."""
    path = Path(results_root) / config_name / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_best(genotype: npt.ArrayLike, run_dir: str | Path) -> Path:
    """Save the best genotype as best_genotype.npy, so a video can be made later.

    Reload with `np.load(path)` and pass it to
    `environment.simulation.run_simulation(genotype, mode="video")`.
    """
    path = Path(run_dir) / "best_genotype.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(genotype, dtype=np.float64))
    return path


def save_evaluations(fitnesses: list[float], run_dir: str | Path) -> Path:
    """Save one row per evaluation (eval_index, fitness) for random search.

    Random search has no generations and no ariel DB; analyze.py turns this
    file into best-so-far "pseudo-generations".
    """
    path = Path(run_dir) / "evaluations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("eval_index,fitness\n")
        for i, fitness in enumerate(fitnesses):
            f.write(f"{i},{fitness}\n")
    return path
