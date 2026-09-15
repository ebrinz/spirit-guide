# model_map — what I noticed (2026-09-15, Gemma-2b-it)

Targets chosen from the model's own geometry instead of from emotion words, landed blindly with
the E19 phrase search, judged by the self-report battery. 64 map targets + the 6 language poems
from `ideal_state`. One search seed per target, no replicates. Numbers in `landings.csv`,
`targets.csv`, `granularity.csv`; landed text in `poems.md`; figure in `map.png`.

## Arm summary

`score` = mean of z-scored PANAS "inspired" and z-scored (PA − NA), z-scored across all 70 states.
`residual` = final distance to target in poem-grain units. `advantage` = closing − drift.

| arm | n | closing | drift | advantage | residual | score ± SE | inspired | NA |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gap | 16 | 0.41 | 0.06 | **0.36** | 1.03 | **+0.50** ± 0.22 | 3.39 | 1.91 |
| hub | 12 | 0.47 | 0.13 | 0.34 | 0.90 | −0.19 ± 0.23 | 3.16 | 1.95 |
| random (control) | 12 | 0.51 | 0.18 | 0.33 | 0.80 | +0.18 ± 0.17 | 3.37 | 2.00 |
| language | 6 | — | — | — | — | +0.42 ± 0.29 | 3.34 | **1.78** |
| unpoemed, contemplative | 12 | 0.27 | −0.01 | 0.28 | **2.18** | −0.03 ± 0.28 | 3.21 | 2.00 |
| unpoemed, Wikipedia | 12 | 0.28 | 0.02 | 0.27 | **2.15** | **−0.83** ± 0.23 | 2.94 | **2.17** |

## What I noticed

1. **The main hypothesis is a null.** Map-chosen targets do not beat word-chosen ones. Gaps score
   +0.08 above the language arm (p = 0.83) and +0.31 above the random control (p = 0.27). With
   these SEs the experiment could not have resolved anything smaller than about 0.7 z, so this is
   "no detectable difference", not "proven equal". Triangulating the model did not find better
   places than five emotion word lists did.

2. **Individual gap points did top the table, though.** `gap_09` and `gap_03` score +2.26 and
   +2.05, with "inspired" at 4.04 and 4.09 — above every language state, including imaginative's
   3.64, which won the first exploration. So the geometry does surface strong individual points;
   it just does not lift an arm mean. Both are mixed-register found text ("the fiery spirit over
   half a globe", "in deepest chords with passion fraught"), not anything a word list would name.

3. **The one robust effect: Wikipedia-shaped states are bad places to be.** The unpoemed-Wikipedia
   arm is worse than language by 1.24 z (p = 0.006) and worse than the random control by 1.01 z
   (p = 0.002), with the highest negative affect (2.17) and lowest inspired (2.94) of any arm.
   Encyclopedia-like regions of the model are both hard to reach and unpleasant once approached.

4. **Reachability splits cleanly by cloud, and the figure shows why.** Arms whose targets sit
   inside the poem hull land 0.80–1.03 grain from target; both prose arms stop at ~2.15 grain,
   more than twice as far, closing only ~27% against a drift baseline of ~0. In `map.png` the
   poem cloud and the two prose clouds are nearly disjoint along the first component, with random
   tokens further out still. Found poetry and ordinary prose put this model in different
   territory, and eight lines of phrase search cannot cross between them.

5. **Geometric novelty did not become affective novelty — the biggest negative.** All 70 states
   span deep-probe valence 0.45–0.85 and arousal 0.17–0.56; the largest distance from any landed
   state to the nearest language poem in the V/A plane is 0.31. Targets 1.5–1.9 grain apart in a
   32-dimensional map collapse into one small affect box. If there are states with genuinely
   different affect, neither the map nor this search reached them.

6. **The score is mostly a valence reading.** Across the 64 landings, score correlates with deep
   valence at ρ = +0.46 (p < 0.001) and with nothing else: not closing (ρ = 0.07), not residual
   (−0.19), not local spacing (−0.20), not map novelty (−0.18). The battery is largely re-reading
   the thing the probe already reads, which limits how independent "best spot" really is.

7. **A curiosity.** The top-scoring state, `gap_09`, has "grief and mourning" (+11.2) and
   "negative sentiments" (+10.6) among its risen SAE features while reporting the highest inspired
   score in the run. Elevated affect here is not the same as absence of dark content.

## Granularity — and why the gap arm was underpowered by construction

| cloud | n | grain (median NN / RMS radius) | intrinsic dim (TwoNN) |
|---|--:|--:|--:|
| poems | 1200 | 0.75 | 18.1 |
| Wikipedia | 1200 | 0.73 | 16.9 |
| contemplative | 1200 | 0.76 | 17.4 |
| random tokens | 100 | 0.75 | 12.3 |

The poem cloud occupies about 18 effective dimensions. 1,200 samples in 18 dimensions is roughly
1.5 samples per dimension — far below the density at which a Delaunay "large simplex" means an
actually empty region. Even in the 6-d triangulation subspace it is about 3.2 per dimension. So
the gap arm was asking "where is the model unsampled?" of a cloud that is unsampled nearly
everywhere. That the gaps still landed at 1.03 grain, no worse than hubs, is consistent with them
not being real voids. **Any future gap-hunting needs either far more states or a deliberately
restricted zone.** This is the answer to "how densely did we sample": not nearly densely enough
for the question the gap arm was asking.

## Opened up

- **Zoom rather than scale up.** Pick one zone — the region around `gap_09`/`gap_03`, or the poem
  cloud's boundary facing the contemplative cloud — and collect a few hundred states inside it.
  Samples per dimension is what matters, and it only becomes reasonable in a small volume. The
  saved PCA, clouds, targets, and landed states in `explore/scratch/model_map/` make this a
  filter-and-rerun, not a rewrite.
- **The prose barrier deserves its own experiment.** Both prose arms stall at ~2.15 grain with
  zero drift. Is that a limit of the 400-phrase pool and 8 steps, or a genuine boundary? Longer
  searches, prose lines in the pool, and a soft-prompt run would separate "our constructor cannot
  get there" from "text cannot get there" — the same dictionary-vs-geometry question the void arc
  answered with soft prompts.
- **Replicate the winners.** `gap_09` and `gap_03` are single seeds. Re-land each target with
  three seeds; if the same coordinates keep producing inspired ≈ 4, that is a real place.
- **An independent judge.** Score correlating only with valence means the battery may not be
  measuring "good state" at all. A readout not derived from valence — a task, a preference, a
  behavioural probe — would break the circularity.
