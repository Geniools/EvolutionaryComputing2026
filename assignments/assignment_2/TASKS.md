# Assignment 2 — Task List per Member

## Shared Kickoff (all members)

- Pick the fixed body from the John Set and the fixed world (anything but `SimpleTiltedWorld`).
- Agree on the main research question (one EA aspect to study).

## Dependency Overview (who blocks whom)

- **M2 and M4 are the critical path.** Nothing runs end-to-end until M2's pipeline is wired,
  and M4's crossover study is the single largest deliverable.
- **M3's plotting/aggregation helper is a shared blocker.** Build it early (even against a
  dummy run) so M2 and M4 aren't stuck waiting for it right before the report is due.
- M2's pipeline wiring needs a first working draft from **M3 (`mutate`) and M4 (`crossover`)**

---

## M1 — Task Definition & Baseline

**Development**

- Choose and justify the fitness definition: plain remaining distance (template default) or
  one of the `ariel.simulation.tasks.targeted_locomotion` variants (distance-reduced,
  effort-penalized, fall-penalized, path-wandering, speed-rewarding); implement it if it's
  not the default.
- Decide the controller inputs (bare `qpos` vs. `qpos` + vector-to-target vs. `qpos` + `qvel`)
  and implement the chosen version in `environment/controller.py`.
- Implement the `random_search` baseline in `evolution/baseline.py`, with its evaluation
  budget set to `POPULATION_SIZE × NUM_GENERATIONS` so it's comparable to the EA.

**Report**

- Write the **Introduction** (research question + crossover sub-question).
- Write the **Methods** subsection on task definition: body, world, fitness function (with
  justification), controller inputs/outputs.

**Extra**

- Prepare the cover page and own the final `groupnumber.zip` packaging and submission.

---

## M2 — EA Core Engine (Selection & Integration)

*Critical path: the full loop cannot run for anyone until this track is done.*

**Development**

- Implement `parent_selection` (e.g. tournament selection) and `survivor_selection`
  (e.g. (μ+λ) with elitism) in `evolution/selection.py`.
- Parametrize the random seed (CLI arg or env var) in `main.py` so every ≥5-seed repeat is a
  single script invocation. *Blocks: M4 experiment runs.*
- Run the main EA configuration × ≥5 seeds, logging best/mean/worst fitness per generation.

**Report**

- Write the **Methods** subsection describing the EA algorithm itself: population model,
  parent/survivor selection mechanisms, and why they were chosen.
- Write the **Results & Discussion** for the main research question (EA-vs-baseline plot
  and its interpretation). *Depends on: M3's plotting helper.*

---

## M3 — Mutation, Parameters & Analysis

**Development**

- Implement `mutate` (Gaussian mutation via `ariel.ec.FloatMutator`) in
  `evolution/mutation.py`, with mutation strength/rate as an explicit, documented parameter. *Blocks: M4's
  crossover-variant runs.*
- Aggregate the per-generation best/mean/worst fitness logged in `ariel.ec`'s SQLite DB
  across seeds, and build the shared plotting helper + summary-statistics table (final best
  fitness mean/std/min/max per configuration). *Blocks: M2's and M4's Results & Discussion.*

**Report**

- Write the **Methods** subsection on parameter settings, evaluation budget, and
  reproducibility (seed handling).
- Write the shared **Results & Discussion** intro: the summary-statistics table and a short
  framing paragraph that ties the two sub-studies (main question + crossover) together.

**Extra**

- Record a video/screenshot of the best evolved controller (`mode="video"`) for report
  figures, on behalf of whichever track produces the best run.

---

## M4 — Crossover Study

**Development**

- Implement at least three crossover operators in `evolution/crossover.py`: e.g. uniform
  crossover, one-point/arithmetic (blend-α) crossover, and SBX or whole-arithmetic
  crossover, plus a "no crossover" (mutation-only) control condition. *Blocks: M2, which needs a working `crossover()`
  to run the EA.*
- Make the active crossover variant selectable via `config.py` / a CLI flag, so switching
  variants doesn't require editing code.
- Run the EA with each crossover variant × ≥5 seeds. *Depends on: M2's seed parametrization, M3's mutation operator.*
- Produce the crossover-comparison plot (avg ± std fitness over generations per variant). *Depends on: M3's plotting
  helper.*

**Report**

- Write the **Methods** subsection describing each crossover operator.
- Write the **Results & Discussion** for the crossover sub-question (comparison plot +
  interpretation of which variant helped/hurt convergence and why).
- Write the **Conclusions**.

---

## Final Checklist

- [ ] Code cleanup done
- [ ] Page limit respected (≤6 pages, excl. cover + bibliography)
- [ ] Full-team report read-through completed
- [ ] `groupnumber.zip` packaged and submitted

