# order_matters — order matters for the metric more than for the model (2026-09-15, Llama-1B)

`line_ablation` found line order irrelevant to the self-report battery. This asks the same of
**placement**, the metric the published leaderboard is built on. Canonical instrument:
Llama-3.2-1B-Instruct with a freshly retrained `data/probe/probe.pkl` (layer 10, R²_v 0.718,
R²_a 0.584 — matching the value recorded in `docs/experiments-journal.md` for E14's gate, so the
ruler is the published one).

5 constructors × 3 targets × (as constructed + 8 reorderings + its exact reverse), 150 reads.

## Answer: yes for the pipeline's metric, much less for the model's state

Two readouts of the same 150 contexts. `ema_error` is the canonical measure from `scripts/05`:
probe every token, smooth with an EMA at alpha 0.1, score the final value against target.
`anchor_error` is the whole-context read used elsewhere in `explore/`: one probe read at a fixed
anchor after the poem.

| readout | mean cost of shuffling | cells hurt | Wilcoxon |
|---|--:|--:|--:|
| EMA (the pipeline's) | **+0.034** | 13/15 | p = 0.005 |
| anchor (whole context) | **+0.009** | 11/15 | p = 0.030 |

The EMA cost is reliably the larger of the two (paired Wilcoxon p = 0.015, larger in 12 of 15
cells). Both are detectable; the pipeline's metric is about four times more order-sensitive than
the whole-context read of the same texts.

**Why, and the prediction it confirms.** The EMA at alpha 0.1 puts 96% of its weight on the last
~30 tokens, roughly the final four lines of a 24-line poem; the first half of the poem contributes
0.01%. So the published placement number is close to "where do the closing lines leave the model".
The sharpest confirmation is `valley`, which is built to ground low and ascend to the target:
reversing it costs **+0.116**, nearly three times its average shuffle cost (+0.043) and the largest
reversal cost of any constructor. Putting valley's grounding band at the end is exactly what defeats
a recency-weighted read.

## The more uncomfortable observation

Under the whole-context read the constructors nearly stop differing:

| constructor | EMA, as constructed | anchor, as constructed |
|---|--:|--:|
| valley | **0.313** | 0.302 |
| harmonic-golden | 0.341 | 0.294 |
| harmonic-prime | 0.381 | 0.304 |
| polygon-pca | 0.384 | 0.295 |
| graph-walk | 0.393 | **0.285** |
| **spread** | **0.079** | **0.019** |

The spread across constructors shrinks fourfold, and the ranking does not survive (ρ = −0.50):
`valley` is first on the pipeline's metric and last-but-two on the anchor read, while `graph-walk`
is last on one and first on the other.

**Two readings, and this experiment cannot choose between them.** Either the constructors genuinely
converge on similar final states and the leaderboard ordering is substantially a property of the
recency-weighted metric; or the single-token anchor read through a *word*-trained probe is simply
less sensitive, compressing real differences. The second is a live possibility — this project has
already caught the word probe compressing arousal to a ~0.45 ceiling and manufacturing a "structural
pocket" (see `lab/EXPERIMENT_LOG.md`, 2026-08-18). Settling it needs a passage-calibrated Llama
probe, which does not exist yet; Llama has only the word probe, and the calibrated ones built so far
are Gemma's.

## Calibration: how big is the shuffle effect?

| quantity | value |
|---|--:|
| mean cost of shuffling (EMA) | +0.034 |
| SD across reorderings of one poem | 0.039 |
| published gap, valley → harmonic on Llama-1B | 0.040 |

Permuting the lines of a poem moves its placement about as much as the published difference between
the best and second-best constructor. Any leaderboard gap of that size is not distinguishable from
which permutation of the same lines you happened to score.

## The committed shuffled control was too thin to see this

`scripts/02` already builds one shuffled variant per constructor (calm, medium, seed 99), and those
rows are in `results/leaderboard*.csv` though not in the README table. Matched to their exact parents
they give a different picture than this run: valley +0.004 there against +0.043 here, polygon-pca
+0.044 there against −0.003 here. With one permutation per cell, the estimates move by more than the
effect. This is the `zone_zoom` lesson landing on the published pipeline's own control: single-seed
controls in this project are not reliable, and the fix is cheap, since all 150 reads here took
30 seconds.

## Caveats

One model, one construction seed per constructor, 8 permutations, 3 targets, medium length only.
The published leaderboard averages over more stimuli and three models, so none of these numbers
replace its numbers. Nothing here says the EMA read is wrong — it is a defensible way to measure a
trajectory, and the README does describe constructors as paths. What the run shows is that the
metric's recency weighting carries a large share of both the order effect and the constructor
ranking, which is worth stating explicitly wherever that ranking is used.

## Opened up

- **Build Llama a passage-calibrated probe.** It is the missing piece for deciding between the two
  readings above, it is the same recipe already used twice for Gemma, and it costs about 20 minutes.
- **Re-run the leaderboard under both readouts.** If the ranking is metric-dependent on the full
  stimulus set, the report should say so next to the table.
- **Multi-permutation controls in the pipeline.** `scripts/02` builds one shuffle; building eight
  and reporting the spread would give every leaderboard row an error bar for free.
- **Does the EMA window explain the constructor ranking?** valley puts target-band lines last by
  construction, which is precisely what this metric rewards. Scoring each constructor by the NRC
  coordinate of its final four lines alone would show how much of the leaderboard that predicts.
