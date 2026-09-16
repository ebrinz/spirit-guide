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

**Two readings, and a passage-calibrated probe settles which.** Either the constructors genuinely
converge and the leaderboard ordering is a property of the recency-weighted metric; or the
single-token read through a *word*-trained probe is simply less sensitive. The second was a live
possibility — this project has already caught the word probe compressing arousal to a ~0.45 ceiling
and manufacturing a "structural pocket" (`lab/EXPERIMENT_LOG.md`, 2026-08-18) — so
`scripts/19_passage_probe.py` was run to give Llama the calibrated ruler it lacked (layer 15,
held-out R²_v 0.919, R²_a 0.916; its acid test reproduces the README's calibration footnote,
placing the best valley poem 0.038 from the calm target where the word probe reads 0.256).

**The compression was real, and it was the probe's.** Adding the calibrated read as a third readout:

| readout | constructor spread | rank agreement with the EMA metric |
|---|--:|--:|
| EMA + word probe (the pipeline's) | 0.079 | — |
| anchor + word probe | 0.019 | ρ = −0.50 |
| **anchor + passage probe** | **0.100** | **ρ = −0.10** |

So "the constructors converge" is dead: under a calibrated whole-context ruler they differ *more*
than under the pipeline's metric, not less. The 0.019 spread was the word probe being nearly blind,
as its own acid test shows — it reads the 24 calmest passages at (0.57, 0.39) and the 24 most
distressed at (0.58, 0.49), almost the same point.

**And the ranking is metric-dependent — this held up.** On these 15 cells the two readouts agree on
essentially nothing (ρ = −0.10). `explore/calibrated_leaderboard` first appeared to refute that
(ρ = +0.78 across the full stimulus set) but that number pools across targets, and both readouts
agree strongly about which *targets* are hard. Split by target at matched conditions, the mean
within-target rank agreement is ρ = −0.05, matching this run. The agreement is real at calm
(ρ = +0.89) and absent at focused and excited. See that note for the mechanism, which appears to be
the word probe's compressed range.

| constructor | EMA (pipeline) | anchor + passage probe |
|---|--:|--:|
| valley | **0.313** (1st) | 0.201 (2nd) |
| harmonic-golden | 0.341 (2nd) | 0.259 (4th) |
| harmonic-prime | 0.381 (3rd) | 0.278 (5th) |
| polygon-pca | 0.384 (4th) | **0.178 (1st)** |
| graph-walk | 0.393 (5th) | 0.240 (3rd) |

`polygon-pca` goes from fourth to first; `graph-walk` from last to third. `valley` is the one
constructor that holds up near the top under both.

**And valley's advantage is concentrated at one target.** Under the calibrated ruler, per target:

| constructor | calm | focused | excited |
|---|--:|--:|--:|
| valley | **0.038** | 0.196 | 0.369 |
| polygon-pca | 0.156 | **0.066** | **0.312** |
| harmonic-golden | 0.211 | 0.190 | 0.377 |
| graph-walk | 0.233 | 0.137 | 0.350 |
| harmonic-prime | 0.231 | 0.203 | 0.400 |

Valley is extraordinary at calm — 0.038, four times closer than the next constructor — and
middling at focused and excited. The published leaderboard averages across targets, so that
structure is invisible in it.

**The order effect under the calibrated read.** Shuffling costs +0.018 (10/15 cells, p = 0.083),
against +0.034 (13/15, p = 0.005) under the EMA. The effect is clearly significant on the
pipeline's metric and not on the calibrated one, but the paired difference between the two is not
itself significant (p = 0.17), so "the metric is more order-sensitive than the model" is the
direction of the evidence rather than an established result. Reversal tells the same story more
loudly: reversing valley costs +0.116 on the EMA but only +0.047 on the calibrated read.

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
The per-target table rests on a single poem per cell, so the valley-at-calm figure of 0.038 in
particular wants a seed sweep before it is leaned on.
The published leaderboard averages over more stimuli and three models, so none of these numbers
replace its numbers. Nothing here says the EMA read is wrong — it is a defensible way to measure a
trajectory, and the README does describe constructors as paths. What the run shows is that the
metric's recency weighting carries a large share of both the order effect and the constructor
ranking, which is worth stating explicitly wherever that ranking is used.

## Opened up

- ~~**Build Llama a passage-calibrated probe.**~~ DONE, `data/passage_probe/` (layer 15,
  R²_v 0.919). It refuted the convergence reading and produced the ranking result above.
- **Re-run the leaderboard under the calibrated read.** This is now the obvious next step and the
  data is cheap: if the constructor ranking on the full stimulus set is as metric-dependent as it
  is on these 15 cells, the report should say so next to the table. Note the calibrated probe can
  only be read at the anchor, so this is a comparison of two different measures, not a correction
  of one.
- **Seed-sweep valley at calm.** Its 0.038 under the calibrated ruler is the single most striking
  number here and rests on one poem.
- **Why does polygon-pca win under the calibrated read?** It was fourth of five in the published
  table and first here. Worth understanding before anyone believes either ordering.
- **Multi-permutation controls in the pipeline.** `scripts/02` builds one shuffle; building eight
  and reporting the spread would give every leaderboard row an error bar for free.
- **Does the EMA window explain the constructor ranking?** valley puts target-band lines last by
  construction, which is precisely what this metric rewards. Scoring each constructor by the NRC
  coordinate of its final four lines alone would show how much of the leaderboard that predicts.
