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

**Relocated from the main pipeline (the "reach of language" horizon — summarized in the report's §16
but kept off the README, per the exploratory contract):**

| script | question | status |
|---|---|---|
| `20_e19_replication.py` | can feedback search reach directions language can't name? | replicated (drift + transfer controls) |
| `21_void_probe.py` | are there states no prompt can approach? | yes — voids exist, repulsive |
| `22_void_stress.py` | does the void survive a strong (exhaustive) optimizer? | yes — single-token floor > 1.0 |
| `23_soft_prompt_void.py` | geometry or dictionary limit? | dictionary — soft prompts close ~92% |
| `24_two_boundary_gemma.py` | does the two-boundary structure replicate across models? | floor: 3 archs; soft: 2 archs |
| `25b_shadow_demo.py` | what does a model say held at a void's edge? | coherence frays (the "shadow") |
| `26_void_cartography.py` | can a void be located/estimated from its rim? | triangulation exact; field-extension flags discontinuity |

These are complete and journaled (`docs/experiments-journal.md`, E18–E26); they live here rather than in
`scripts/` because the README's core narrative does not depend on them. New work goes below.

| exp | question | status |
|---|---|---|
|  | do Llama-1B and Gemma-2B share un-reachable pockets of the VAD map? | **yes: per-cell residual ρ=0.917; shared pocket = the high-arousal ceiling (A≥0.7, all valences). A joint hole of valley-constructor + contemplative corpus, inherited by both models.** |
