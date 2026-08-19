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
| exp_void_pockets.py | do Llama-1B and Gemma-2B share un-reachable pockets of the VAD map? | **yes: per-cell residual ρ=0.917; shared pocket = the high-arousal ceiling (A≥0.7, all valences). A joint hole of valley-constructor + contemplative corpus, inherited by both models.** |
| exp_close_arousal_pocket.py | can any constructor reach the high-arousal pocket valley can't? | no — 5 constructors all stall ~0.47 placed-A vs targets 0.75–0.80. Not a valley flaw; a corpus/model wall. |
| exp_sae_pocket_rims.py | what concepts distinguish the reachable rim from the pocket? | pocket is distress-saturated (anxiety/stress/overwhelm); rim carries mindfulness/calm. The model reads high arousal AS distress; "peaceful intensity" barely exists. |
| exp_awe_close_pocket.py | can awe/ecstasy content close the pocket or lower its distress? | decisive no — awe didn't raise arousal and RAISED distress (18.3 vs 11.1). Pocket is structural: arousal & positive valence inversely coupled. |
| exp_passage_probe_pocket.py | is the high-arousal pocket an instrument artifact or a model attractor? | **model attractor. Same passage-calibrated ruler reaches high VALENCE (V≥0.75 → 0.71) but not high AROUSAL (A≥0.75 → 0.50). The old valence ceiling was the ruler; the arousal cap is the model.** |
| exp_dominance_axis.py | does the dominance axis reach states the V/A plane can't? | **yes — D is ~40% orthogonal to V/A (R²=0.90). At matched valence, high-D content lifts arousal ΔA +0.37 to A=0.70, past the ~0.47 ceiling. The pocket is escapable via dominance (Llama-1B).** |
| exp_dominance_crossmodel.py | does the escape replicate on Gemma-2b (word probe)? | SUPERSEDED — word probe has an arousal ceiling that hides the escape; see exp_gemma_passage_probe.py. Kept to document the confound. |
| exp_gemma_passage_probe.py | fair cross-model test with a calibrated Gemma ruler | **the "structural pocket" was largely a WORD-PROBE artifact. With a calibrated probe: dominance escape replicates (A 0.24→0.64, layer-robust to L24), and the awe null FALLS (awe reaches A≈0.55, not ~0.45). High arousal is reachable on both models; only calm/valley content genuinely stays low.** |
| exp_dominance_vs_awe_distress.py | is dominance a CLEANER high-arousal route than awe? | **no — inverted. At matched A,V, awe distress 7.4 vs dominance 27.7 (t −3.36). AWE is the clean high-arousal-positive placement; dominance reaches intensity through distress. The "structural pocket" was a compound artifact (word-probe ceiling + distress features contaminated with dominance/order). Peaceful intensity is reachable.** |
| exp_llama_behavioral_distress.py | does the awe/dominance distress split replicate behaviorally on Llama (no SAE)? | **directionally yes, once persona is removed. Canonical meditation-preamble pathway: null — output homogenized by the assistant persona. Open first-person anchor: dominance-placed generates distressed text (crumbling/died/overwhelmed) vs awe's peaceful/excited (gen-V 0.598 vs 0.633, t −1.43). Cross-instrument support; internal state ≠ expressed state under alignment.** |
| exp_crossmodel_align.py | does the shared affect structure survive in full residual space? | **yes — architecture-invariant up to a linear map. 1200 paired anchors (same passages): affine Gemma→Llama map R²~0.5 vs shuffled −0.14; affect subspace ~100% shared (V/A read at 98% of within-Llama ceiling after mapping). Only ~50% of the full residual is shared, but the emotion axes are the same axes up to a change of basis.** |
| exp_dominance_align.py | is the dominance axis as architecture-invariant as V/A? | **yes — D transfers at 96% of ceiling (within 0.897 → cross 0.863), tied with V/A (98%). The dominance lever, incl. its orthogonal-to-VA part, is shared cross-architecture — which is why the arousal escape replicated on Gemma. Full V/A/D geometry is architecture-invariant up to a linear map.** |
| exp_dominance_corpus_slice.py | is dominance-distress the axis or the corpus? | **not separable here. High dominance in this corpus = martial/hierarchical power (battle/king/command/war/supreme); no "calm mastery" content exists. Gore is a minority (21%, r 0.16) but even non-gore high-D is conquest/command themed. Dominance-distress is a content property; a true axis test needs out-of-corpus calm-power content.** |
| exp_residual_char.py | what is the ~50% of the residual the cross-model map misses? | **model-private, not hidden nonlinear structure. Affect is a tiny (~4% of variance) but precisely-shared (R² 0.87) subspace; the non-affect ~96% aligns at only 0.36, CKA barely rises linear 0.39→RBF 0.42, uniform across coherent/incoherent. Emotion is a near-universal low-dim island in otherwise-divergent representations.** |
