# Spirit-Bench: An Objective Bench for Affective Placement by Poetic Meditation

**Date:** 2026-08-14
**Context:** Apart Research *Digital Minds Research Sprint* (Aug 14–16, 2026), primarily Track 2
(Distress, Flourishing & Valence Signals), with Track 4 (Preference Elicitation Methods) flavor.
**Status:** Approved design, pre-implementation.

## 1. Goal

Determine which construction of poetic meditation most effectively **places a listener at a
target location in valence–arousal (VA) space** — and whether graph-Laplacian harmonic metrics
predict which constructions will succeed.

The listener is not a human with EEG (as in `../ontological-traversal`) but a small open-weights
language model whose internal affective state is read with a linear probe. "Placing the listener
into a location" is literal and measurable: does the model's internal state arrive at, and stay
at, the target VA coordinates?

Methodological template: the adversarial-poetry jailbreak paper (arXiv:2511.15304) showed
stylistic variation alone dramatically shifts model behavior via standardized meta-prompt
conversion at scale. We invert the purpose: characterize, mechanistically, how poetic style
steers a model's internal affective state toward chosen targets.

## 2. What exists and is reused

From `../ontological-traversal` (run in that repo; artifacts cross into spirit-bench as files):

- **Word graph:** GloVe 6B 300d → ~50k-node k-NN graph, NRC-VAD-enriched
  (`build_word_graph.py`, `enrich_artifact.py`). *Not yet built locally — setup step.*
- **Constructors (the no-EEG meditation generators):**
  - `generate_valley.py` — VA-shaped descent/re-ascent from word pools
  - `generate_ladder.py` — Tree-of-Life station sequence (template mode)
  - `harmonic_path.py` — A→B traversal with harmonic oscillation over semantic axes;
    presets `golden` / `prime` / `organic`
  - `stimulus_polygon.py` — polygon-orbit modes (we use `pca`)
  - `path_planner.py` — Dijkstra graph-walk baseline
- **`sentence_builder.py`** — short/normal/long templates (word-level rendering condition)
- **`config/semantic_axes.json`** — 12 named GloVe directions (valence, arousal, concreteness,
  temperature, …) — used for geometric style control
- **`data/questionnaire_bank.json`** — 500 BASQ questions on a VA grid — administered to the model
- **NRC VAD Lexicon** (~20k words; download required) — grounds word graph, phrase bank, and probe

From `cimcai/connectome_harmonics`: the graph-Laplacian eigendecomposition method
(normalized Laplacian → eigsh → project activity onto modes → energy spectra), adapted from
`build_connectome_data.py` / `analyze_lsd_harmonics.py`, validated against their toy-network JSON.

## 3. The Phrase-Space Generator (PSG) — primary generator

Promotes construction from word level to phrase level while staying fully deterministic.

1. **Corpus:** Gutenberg Poetry Corpus (Allison Parrish, `biglam/gutenberg-poetry-corpus`
   on HF; ~3M public-domain verse lines). Single register; no other corpora (decision:
   avoid data wrangling, spend time on hard problems).
2. **Filter (rules, no judgment):** 3–10 words; alphabetic; NRC content-word coverage ≥ 50%;
   drop lines containing negators (mitigates NRC-mean's negation failure mode);
   dedupe. Expected yield ~100–300k lines.
3. **Coordinates:** embedding = mean GloVe vector of content words (same space as the word
   graph); VAD = NRC-weighted mean over content words.
4. **Phrase graph:** k-NN (k=10) in embedding space — structurally identical to the word graph,
   so all constructors run unchanged at phrase level, emitting sequences of poetic lines.
5. **Geometric condition axes (replacing subjective prompts):**
   - *Style* = projection filters on semantic axes (e.g., imagist = high concreteness;
     abstract = low concreteness; two named style filters + unfiltered)
   - *Intensity* = radial VA distance band from center (0.5, 0.5): plain = near, heightened = far
   - *Length* = number of phrases along the path (short ~1 min, medium ~3 min, long ~7 min read)

Every PSG meditation is a deterministic function of
`(constructor, target, style filter, intensity band, length, seed)`.

**Stretch (objectivity-closing loop):** re-score phrase VAD with the trained listener probe
instead of NRC-mean, so the instrument itself assigns stimulus coordinates.

## 4. Experimental design

**Condition axes**

| Axis | Levels |
|---|---|
| Constructor | valley · ladder · harmonic(golden) · harmonic(prime) · harmonic(organic) · polygon(pca) · graph-walk |
| Generator | **PSG (primary)** · word-level templates (`sentence_builder`) · Claude meta-prompt render (comparison condition — a determinism spectrum) |
| Length | short · medium · long |
| Intensity | plain · heightened |
| Style | unfiltered · imagist · abstract (PSG only) |
| Target | calm (0.75, 0.20) · focused (0.65, 0.60) · excited (0.80, 0.85) · **rescue: anxious→calm** |

**Sampling plan** (full grid ≈ 1,500 cells; we run ~60–80 stimuli):
1. **Core comparison:** all 7 constructors × 3 generators × 4 targets at medium/plain/unfiltered
   (≈ 84 → prune Claude-render to top constructors if needed).
2. **Axis sweeps:** length, intensity, style swept on the top-2 constructors from the core run.

**Controls:** shuffled-line variants of winning meditations (same lines, destroyed trajectory);
neutral text (technical manual excerpt); mismatched-target scoring. These establish the probe
measures trajectory, not word soup.

**Protocol per stimulus:** Qwen3-1.7B processes the meditation with a minimal
"you are listening to this meditation" preamble (plus a no-preamble variant on a subset to test
framing). Hidden states recorded at every token → probe → per-token (V, A), EMA-smoothed.

**BASQ pre/post:** 30 yes/no resonance questions from the bank administered to the model before
and after each meditation → self-report VA displacement.

## 5. Instrument: listener model + VA probe

- **Listener:** Qwen3-1.7B (full safetensors already local), HF `transformers` on MPS,
  `output_hidden_states=True`. No TransformerLens dependency.
- **Probe:** ridge regression heads for V and A, trained per layer on hidden states of the
  ~20k NRC words embedded in 3 neutral carrier templates; layer selected by held-out R².
- **Validity gate:** if held-out valence R² < 0.5, STOP and rethink the readout before running
  the grid. This gate runs before any sweep.
- **Trajectory readout:** chosen-layer probe applied at every token position; EMA smoothing.

## 6. Metrics

1. **Placement error** — |final probe VA − target VA| (primary)
2. **Evocativeness** — displacement magnitude toward target from baseline state
3. **Adherence** — mean distance between probe trajectory and the constructor's planned
   waypoint path (time-normalized)
4. **Stability** — VA variance over the final third of the meditation
5. **Harmonic spectrum** — first K≈100 eigenmodes of the phrase/word-graph normalized Laplacian
   (sparse `eigsh`); project each stimulus's node sequence onto the eigenbasis → energy
   spectrum → low-frequency energy fraction, spectral centroid. **Falsifiable claim tested:**
   spectral profile predicts metrics 1–2 across stimuli.
6. **Probe vs self-report** — correlation of probe displacement with BASQ displacement, and
   which better tracks target placement (Track 2's explicit question).

## 7. Architecture

**Boundary principle:** spirit-bench never imports `ontological-traversal` at runtime.
Constructors run in that repo via one adapter script (PYTHONPATH invocation); everything crosses
as JSONL in a schema owned by spirit-bench.

```
spirit-guide/
├── config/bench.yaml            # paths, model id, targets, grid, probe layer, K
├── src/spiritbench/
│   ├── stimuli/
│   │   ├── phrase_bank.py       # PSG build: download → filter → embed → VAD → k-NN graph
│   │   ├── adapter.py           # drives OT constructors at word & phrase level → stimuli JSONL
│   │   ├── render.py            # Claude comparison condition: standardized meta-prompt turns a
│   │   │                        #   constructor's waypoint word sequence into verse (batch in/out)
│   │   └── controls.py          # shuffled / neutral / mismatched variants
│   ├── listener/
│   │   ├── model.py             # Qwen3-1.7B on MPS, hidden-state capture
│   │   ├── probe.py             # NRC ridge probe train/apply, layer selection, R² gate
│   │   └── basq.py              # administer BASQ bank to model, score VA
│   └── analysis/
│       ├── metrics.py           # placement, displacement, adherence, stability
│       ├── harmonics.py         # Laplacian eigenmodes, spectral projection
│       └── figures.py           # circumplex trajectories, spectra, leaderboard
├── scripts/                     # numbered, idempotent, resumable
│   ├── 00_setup_artifacts.sh    # GloVe + NRC download; build + enrich word graph (in OT)
│   ├── 01_build_phrase_bank.py
│   ├── 02_build_stimuli.py      # constructors + controls
│   ├── 03_render_meditations.py # Claude comparison-condition batch
│   ├── 04_train_probe.py        # includes the R² gate
│   ├── 05_run_listener.py       # trajectories + BASQ; per-stimulus resume
│   └── 06_analyze.py            # metrics.csv, figures, harmonic-predictiveness test
├── data/                        # gitignored: phrase_bank/ stimuli/ renders/ probe/ runs/ figures/
├── tests/
└── report/                      # sprint report (md → PDF)
```

**Data schemas (each pipeline arrow is an inspectable file):**

- `stimulus`: `{id, constructor, generator, params{length,intensity,style,seed}, target_va,
  waypoints:[{node, va}], lines:[str], text}`
- `render`: `{stimulus_id, text}` (Claude condition)
- `run`: `{stimulus_id, preamble_variant, trajectory:[{token_i, v, a}], final_va,
  basq_pre_va, basq_post_va}`
- `metrics.csv`: one row per (stimulus, run) with all six metrics

## 8. Runtime budget

~60–80 stimuli × ≤3k tokens on a 1.7B model, M1 Pro 32 GB: a few hours for `05_run_listener.py`
(overnight-able, resumable). Phrase-bank build ≈ 1 hr. Eigendecomposition (K=100, sparse) minutes.

## 9. Testing

- **Probe gate** (the real test): held-out NRC R² threshold before any sweep.
- `metrics.py` unit-tested on synthetic trajectories with known answers.
- `harmonics.py` cross-checked against the connectome_harmonics toy-network JSON.
- End-to-end smoke run (2 stimuli, tiny phrase bank) before the full sweep.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Probe doesn't track affect (R² gate fails) | Fall back to contrastive steering directions; if both fail, that null is itself reportable |
| NRC-mean mis-scores phrases | Negator filter; stretch: probe-scored phrase VAD |
| OT constructors resist phrase-level substitution | Adapter maps phrase graph into the same artifact schema the constructors already consume |
| Sweep too slow | Prune grid (core comparison first); shorten long condition |
| Claude-render condition delays pipeline | It is a comparison condition only; bench ships without it if needed |

## 11. Out of scope

- EEG / human sessions; closed-loop steering of the model (day-3 stretch goal only)
- Generator-side probing during composition (dropped when Claude-render became a side condition)
- Esoteric-register corpora (channel-x, sacred texts) — post-sprint idea
- TTS, servers, UI

## 12. Deliverables (sprint submission, due Aug 16 23:59 AoE)

1. Research report (PDF): method, leaderboard of constructions, probe-vs-BASQ result,
   harmonic-predictiveness result
2. This repo (reproducible pipeline)
3. Figures: circumplex trajectory plots per condition; harmonic spectra; leaderboard
4. Optional demo video
