# Spirit-Bench: Measuring Affective Placement of a Language Model by Poetic Meditation

**Digital Minds Research Sprint, Aug 14–16 2026 — Track 2 (Valence Signals)**

## Abstract
(150 words: bench, probe, findings.)

## 1. Motivation
Adversarial poetry shows style alone steers models (arXiv:2511.15304); we measure the
benevolent inverse: can constructed poetry place a model at chosen affective coordinates?

## 2. Method
2.1 Constructors (valley, harmonic x3, polygon-pca, graph-walk) on VAD-enriched graphs
2.2 Phrase-Space Generator (Gutenberg Poetry Corpus, NRC scoring, deterministic)
2.3 Listener instrument: Qwen3-1.7B + NRC ridge probe (layer L, held-out R² = ...)
2.4 BASQ self-report for models
2.5 Graph-Laplacian harmonic metrics (after Atasoy; cimcai/connectome_harmonics)

## 3. Results
3.1 Leaderboard (Table 1: data/figures/leaderboard.csv)
3.2 Trajectories (Fig: trajectories_*.png)
3.3 Controls: shuffled / neutral / mismatch
3.4 Probe vs self-report (Fig: probe_vs_basq.png)
3.5 Does harmonic smoothness predict placement? (harmonic_predictiveness.txt)

## 4. Limitations & digital-minds relevance

- Ladder constructor descoped: Tree-of-Life station pools don't transfer to phrase space.
- Harmonic constructor VA/semantic axes: on the phrase artifact, single-word axis anchors
  ("good", "joy", "calm", ...) don't exist in a vocabulary of multi-word lines. Fixed in
  `adapter.harmonic()` by building axes from the word artifact (same 300-d GloVe space)
  while traversing the phrase artifact's own vectors/index; see
  `.superpowers/sdd/2026-08-14-spirit-bench/task-15-report.md` for detail.
- The harmonic spectrum metric (`low_freq_fraction`, `spectral_centroid`) is order-invariant:
  it measures the node set's locality on the graph Laplacian, not the order waypoints are
  visited in, so it cannot distinguish a stimulus from a shuffled-order version of itself.
- The `valley` constructor ignores `start_va`, so its `rescue`-target output is
  byte-identical to its `calm`-target output; `valley` is excluded from the rescue target
  everywhere (core grid and sweeps).
- `graph-walk` paths from Dijkstra can be far shorter than the requested line count
  (typically 3-7 hops); shorter paths are stretched to the target length by proportional
  repetition of waypoints, not by inserting new semantic content.
- "plain" intensity means the VA-magnitude mask is unfiltered (no intensity constraint),
  not that waypoints are drawn near the VA-space center.

## 5. Reproducing
`scripts/00–06` in order.
