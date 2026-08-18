# lab/ — exploration sandbox

New techniques live here **before** they earn a place in the narrative. The contract:

- **Imports the engine, never forks it.** Use `spiritbench` (the stable library in `src/`) and the
  vendored constructors. If a technique needs to *change* the engine itself, do that on a branch
  (`feat/<name>`) with tests, not here.
- **Nothing in the narrative depends on `lab/`.** The README, the report, and `scripts/00–06` (the
  canonical pipeline) never reference this directory, so anything here can be broken, wrong, or
  abandoned without derailing the published artifact. Explore freely.
- **Results stay local.** `lab/results/` is gitignored (like `data/`); commit code and findings notes,
  not large output.
- **One file per experiment**, named `exp_<slug>.py`, with a docstring stating the question and the
  verdict criterion up front.

## Promotion path

An experiment graduates out of `lab/` only when it is (1) **validated** — replicated, controlled, effect
size honest — and (2) **you decide it belongs in the story**. Graduation means: move it to a numbered
`scripts/NN_*.py`, add a report section, log it in `docs/experiments-journal.md`, and bump the release
tag (`v1.1`, …). Until then it stays here, labelled exploratory and not load-bearing.

The frozen, coherent version of the project is always retrievable at tag **`v1.0`**.

## Log

| exp | question | status |
|---|---|---|
| _(none yet)_ | | |
