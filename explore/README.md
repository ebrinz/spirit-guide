# explore/ — open-ended looking

The third tier beside `scripts/` (the published narrative) and `lab/` (hypothesis-driven
experiments). This is where you play with the poetry generator, with or without the listener,
to see what it does and get ideas. No hypothesis, no verdict criterion, no obligation to write in
`lab/EXPERIMENT_LOG.md`.

## Contract

- **Import the engine, never fork it.** Use `spiritbench` and the vendored constructors under
  `vendor/`, same as `lab/`. If something needs the engine to change, that happens on a
  `feat/<name>` branch with tests.
- **One folder per exploration.** `explore/<slug>/run.py` with a one-paragraph docstring saying
  what you are looking at. Add `NOTES.md` for what you noticed, if anything.
- **Small text is committed next to the script.** Poems, generations, short tables: `.txt`,
  `.md`, `.csv`. They are the point, so they live in the repo where they can be read later.
- **Anything large goes in `explore/scratch/`,** which is gitignored: hidden states, arrays,
  probe pickles, checkpoints.
- **Nothing in the narrative depends on `explore/`,** and `explore/` does not feed `scripts/`
  directly. If an exploration sharpens into a question with a verdict criterion, it moves to
  `lab/exp_<slug>.py` and follows the lab contract from there.

## Layout

```
explore/
  README.md
  scratch/            # gitignored
  <slug>/
    run.py
    NOTES.md          # optional
    *.txt *.md *.csv  # committed
```
