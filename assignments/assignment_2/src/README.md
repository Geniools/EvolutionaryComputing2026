# Assignment Structure

Two concerns, two folders:

- **`environment/`** - the task itself: body, world, controller, fitness.
  Mostly given by the assignment template; keep it fixed.
- **`evolution/`** - the EA: representation and operators. This is what
  you need to design and implement (currently `NotImplementedError` stubs).

```
src/
├── config.py            # all shared constants (task + EA hyperparameters)
├── main.py               # entry point - wires everything together
│
├── environment/          # "the task" (given)
│   ├── world.py          #   robot body + world
│   ├── controller.py     #   NN controller, genotype <-> weights
│   ├── fitness.py        #   position tracking + fitness score
│   └── simulation.py     #   run one full simulation -> fitness
│
└── evolution/            # "the assignment" (TODO)
    ├── genotype.py       #   individual factory (representation)
    ├── evaluation.py     #   Population -> scored Population
    ├── selection.py      #   parent_selection, survivor_selection
    ├── crossover.py       #   recombination operator
    ├── mutation.py        #   mutation operator
    └── baseline.py        #   random-search comparison
```

Run everything via `main.py`.

> Note: Make sure to set the root folder to `assignments/assignment_2/` in your IDE, so that `src/` is on the Python
> path.

## Execution flow

```
main.py
  │
  ├─ get_io_sizes()                      (environment/simulation.py)
  │
  ├─ make_individual() × N               (evolution/genotype.py)
  │        │
  │        ▼
  │   Initial Population
  │        │
  │        ▼
  │    evaluate()  ───────────────────►  run_simulation()  ─►  fitness_function()
  │   (evolution/evaluation.py)          (environment/simulation.py)   (environment/fitness.py)
  │        │
  │        ▼
  │   ┌─────────────────────────── EA.run() loops NUM_GENERATIONS times ───────────────────────────┐
  │   │  parent_selection → crossover → mutation → evaluate → survivor_selection                   │
  │   └────────────────────────────────────────────────────────────────────────────────────────────┘
  │        │
  │        ▼
  └─ best individual → run_simulation(mode=MODE)   # watch/record the evolved controller
```

