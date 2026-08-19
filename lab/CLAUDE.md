# lab/ working notes (for Claude)

This is the exploration sandbox. Rules and habits when working here:

- **After any substantial experiment, append an entry to `EXPERIMENT_LOG.md`**
  (newest first): the question, method in a sentence, the result with numbers,
  and what it opened up. Do this before moving on — the log is the sandbox's
  memory, and future sessions rely on it.
- **Update the log table in `lab/README.md`** with the one-line status of each
  experiment file.
- **Keep `lab/TODOS.md` current.** Strike threads as they're run (moving the
  result to the log), and append new "opened up" threads there so the backlog
  never lives only in a session's memory. It's grouped by compute cost so a
  session can pick work that fits its time budget.
- **Contract (see `lab/README.md`):** import `spiritbench`, never fork it;
  write results under `lab/results/` (gitignored); one file per experiment,
  `exp_<slug>.py`, with a docstring stating the question and verdict criterion
  up front. Changing the engine (`src/spiritbench/`) happens on a `feat/*`
  branch with tests, not here.
- **Nothing in the published narrative (README, report, `scripts/00–06`) may
  depend on `lab/`.** Graduate a finding out of the sandbox only once it is
  validated *and* chosen for the story; then it moves to a numbered `scripts/`
  file, gains a report section, and the release tag bumps.
- The frozen public artifact is tag **`v1.0`**.
