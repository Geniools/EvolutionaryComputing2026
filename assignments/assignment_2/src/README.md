# Assignment Structure

- **`environment/`** - the task itself: body, world, controller, fitness.
- **`evolution/`** - the EA: representation, operators and the baseline.
- **`analysis/`** - plots and statistics over finished runs.

```
src/
├── config.py             # all shared constants (task + EA hyperparameters)
├── main.py               # entry point: runs the baseline and every crossover variant
├── replay_best.py        # watch or record the best controller of a run
│
├── environment/          # "the task"
│   ├── world.py          #   robot body (John Set gecko) + flat world with target marker
│   ├── controller.py     #   NN controller, genotype <-> weights
│   ├── fitness.py        #   fitness score (lower is better)
│   └── simulation.py     #   run one full simulation -> fitness
│
├── evolution/            # "the EA"
│   ├── genotype.py       #   individual factory (representation)
│   ├── evaluation.py     #   Population -> scored Population
│   ├── selection.py      #   parent_selection, survivor_selection
│   ├── crossover.py      #   recombination operators (none, uniform, blend, neuron)
│   ├── mutation.py       #   Gaussian mutation
│   ├── statistics.py     #   best/mean/worst fitness per generation
│   └── baseline.py       #   random-search comparison
│
├── analysis/
│   ├── analyze.py        #   fitness plot, summary table, Mann-Whitney U tests
│   └── paired_tests.py   #   Wilcoxon tests, each method against the "none" control
│
└── tests/
    └── test_mutation.py  #   checks for the mutation operator
```

Run everything via `main.py` (see its docstring for `--seeds`, `--methods`, `--workers`).

> Note: Make sure to set the root folder to `assignments/assignment_2/src` in your IDE.

## Execution flow

```
main.py  (for each method: baseline, none, uniform, blend, neuron; for each seed)
  │
  ├─ baseline: random_search() with BASELINE_NUM_EVALUATIONS     (evolution/baseline.py)
  │
  └─ EA:
       make_individual() × POPULATION_SIZE                       (evolution/genotype.py)
         │
         ▼
       evaluate()  ──────────►  run_simulation()  ─►  fitness_function()
       (evolution/evaluation.py) (environment/simulation.py)  (environment/fitness.py)
         │
         ▼
       ┌──────────────────── EA.run() loops NUM_GENERATIONS times ────────────────────┐
       │  parent_selection → crossover → mutate → evaluate → survivor_selection       │
       │  → record_stats                                                              │
       └──────────────────────────────────────────────────────────────────────────────┘

then: analysis/analyze.py + analysis/paired_tests.py  → __data__/a2_experiments/figures/
      replay_best.py --mode video                     → a video per EA method
```
