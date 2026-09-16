# layer_check — the 9B null was largely a shallow-probe artifact

`reversal_test` confirmed on Llama that reversing a poem costs the pipeline's recency-weighted
metric far more than it costs a calibrated whole-context read, and at 9B found no such difference.
The notes flagged a suspect: the 9B passage probe selects **layer 6 of 43**, while Llama's selects
layer 15 of 17. This re-reads at every depth, the way the lab settled the same worry for the
Gemma-2B probe (`lab/EXPERIMENT_LOG.md`, 2026-08-18).

Method: rebuild the same 72 poems ordered and reversed, read each once at the anchor keeping all 43
layers, then train a passage probe at every layer from the already-collected
`data_gemma9b/passage_probe` states and rescore. One 9B pass (1.6 min), then offline.

## The ruler is good at every depth

Held-out valence R² ranges **0.886 to 0.932** across all 42 usable layers, arousal 0.897 to 0.932.
No layer is a bad ruler, so the layer-6 selection was not a quality problem — `train_probe` simply
picks the argmax of a nearly flat curve. Same depth-robustness the lab found for Gemma-2B.

## But reversal cost collapses with depth

| depth band | mean R²_v | reversal cost, all constructors | reversal cost, valley | ratio, EMA / calibrated |
|---|--:|--:|--:|--:|
| shallow, layers 1–6 | 0.927 | +0.075 | +0.165 | 0.9× |
| early-mid, 7–14 | 0.918 | +0.062 | +0.126 | 1.2× |
| mid, 15–28 | 0.893 | +0.022 | +0.053 | 2.8× |
| **deep, 29–42** | 0.919 | **−0.009** | **+0.034** | **4.4×** |
| *the 9B EMA read, for reference* | | *+0.063* | *+0.150* | |

Spearman correlation between layer and reversal cost: **−0.89** across all constructors, **−0.82**
for valley. The deeper the read, the less reversing a poem matters — and past layer 29 it costs
essentially nothing (−0.009 overall).

## What this resolves

**The mechanism's premise holds.** A genuinely integrated whole-context read *is* order-invariant, as
the account in `polygon_sweep` §3b requires. The 9B null came from reading at layer 6, which is
shallow enough to carry recency character of its own — at that depth the calibrated read penalises
reversal as much as the recency-weighted metric does (ratio 0.9×), which is exactly why the contrast
vanished.

Read deep instead and the Llama-style differential returns, larger than on Llama: valley's reversal
costs the pipeline's metric 0.150 against 0.034 for a deep calibrated read, a ratio of 4.4×, where
Llama's was 2.4×.

**So the earlier conclusion needs revising.** `reversal_test` concluded the mechanism was
"model-specific, not general". On this evidence it is better described as **instrument-specific**:
it shows up whenever the comparison read is deep enough to be order-invariant, which Llama's
layer-15 probe is and 9B's layer-6 probe is not.

## What it does not resolve

The rank correlation ρ(tail rise, reversal cost) does *not* cleanly go to zero at depth — it wanders
between −0.03 and +0.66 in the deep band. That is expected once the cost itself is near zero, since
the correlation is then computed on noise, but it means the *structured* half of the prediction
(cost should scale with tail rise on the metric and not on the calibrated read) is only cleanly
demonstrated on Llama. At 9B what is demonstrated is the simpler and more basic claim: deep reads
are order-invariant, shallow ones are not.

Also unresolved: whether `train_probe`'s argmax-R² layer selection should be trusted at all when the
R² curve is this flat. It picked layer 6 of 43 on a curve spanning 0.886–0.932, and that choice
changed a headline conclusion. A selection rule that preferred depth among statistically tied layers
would have avoided this, and the same concern applies to the Gemma-2B probe at layer 1.

## Opened up

- **Add a depth tie-break to probe selection.** When held-out R² is flat within noise, prefer the
  deeper layer. This is an engine change (`src/spiritbench/listener/probe.py`), so a `feat/*` branch
  with tests, and it would affect every calibrated result in the project.
- **Re-run `calibrated_leaderboard` and `gemma9b_check` at a deep layer.** Both used each probe's
  own argmax pick. The constructor rankings there may move, and the 9B ones most of all.
- **Check Llama's layer-15 pick the same way.** It is 15 of 17, so already deep, but the sweep is
  cheap now that the pattern is known.
