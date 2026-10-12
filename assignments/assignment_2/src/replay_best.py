"""Replay the best individual stored in a run database.

Usage (from assignment_2/):
    python src/replay_best.py                       # best over all seeds, MuJoCo viewer
    python src/replay_best.py --seed 3              # best of seed 3
    python src/replay_best.py --mode video          # record a video instead
    python src/replay_best.py --mode simple         # headless, only print fitness

"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from environment.simulation import run_simulation

# Same working directory as main.py, so relative paths match.
os.chdir(Path(__file__).resolve().parent.parent)


def load_best(db_path: Path) -> tuple[int, float, np.ndarray]:
    """Return (id, fitness, genotype) of the lowest-fitness individual."""
    with sqlite3.connect(db_path) as con:
        row = con.execute(
                "SELECT id, fitness_, genotype_ FROM individual "
                "WHERE requires_eval = 0 AND fitness_ IS NOT NULL "
                "ORDER BY fitness_ ASC LIMIT 1",
        ).fetchone()
    if row is None:
        msg = f"no evaluated individuals in {db_path}"
        raise ValueError(msg)
    return row[0], row[1], np.asarray(json.loads(row[2]), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="__data__/a2_experiments", help="folder with seed_<n>.db files")
    parser.add_argument("--seed", type=int, help="only use this seed (default: best over all)")
    parser.add_argument(
        "--mode", default="launcher",
        choices=["launcher", "video", "simple", "frame"]
        )
    parser.add_argument("--video-folder", default="__data__/a2_experiments/videos")
    args = parser.parse_args()

    folder = Path(args.dir)
    dbs = [folder / f"seed_{args.seed}.db"] if args.seed else sorted(folder.glob("seed_*.db"))
    if not dbs or not all(db.exists() for db in dbs):
        parser.error(f"database(s) not found in {folder}")

    best = min(((*load_best(db), db) for db in dbs), key=lambda r: r[1])
    ind_id, stored_fitness, genotype, db = best
    print(f"best individual: id={ind_id} from {db.name}, stored fitness={stored_fitness:.4f}")

    if args.mode == "video":
        Path(args.video_folder).mkdir(parents=True, exist_ok=True)
    fitness = run_simulation(genotype, mode=args.mode, video_output_folder=args.video_folder)
    print(f"replayed fitness: {fitness:.4f}")


if __name__ == "__main__":
    main()
