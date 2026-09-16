# polygon_sweep — the inversion does not survive, and the seed is a no-op

Set out to put error bars on the polygon-pca inversion from `explore/gemma9b_check`, which I had
called "the one finding in this arc I would now defend". **It does not survive resampling.** On the
way, the sweep could not be run as designed, for a reason that matters more than the original
question.

## 1. The `seed` parameter does nothing

A construction-seed sweep is impossible here: all six constructors are deterministic given
(target, start, length, mask).

- `_pick_in_band` (`src/spiritbench/stimuli/adapter.py`) sorts band members by distance to the band
  centre and takes the first k. Its RNG is consulted **only** in a fallback for bands holding fewer
  than k eligible nodes, which a 50,000-line phrase bank never triggers. `valley_shape` is built
  entirely from it.
- `polygon_pca` constructs `np.random.RandomState(seed)` and never uses the object. Every step
  follows from the interpolated waypoint and an angle computed from the step index.
- `graph_walk` ignores its `seed` argument entirely; the path is a deterministic shortest route.
- `harmonic` threads its seed to the vendored constructor, and produced identical output here too.

Eight seeds at the focused target gave byte-identical placement errors for all six constructors.

**The published results already record this, and it appears to have gone unnoticed.**
`results/seed_expansion.csv` is the output of `scripts/14_seed_expansion.py` ("C2 seed expansion",
journal E13: *"Seed expansion: order effect 6/6 constructors at fresh seeds"*). Its ordered rows for
seeds 43 and 44 are identical to fifteen decimal places for all six constructors, and identical to
the seed-42 values in `results/leaderboard.csv`. Only the shuffled rows differ, because
`controls.shuffled` uses its own permutation seed.

So that experiment's **order effect stands** — ordered versus shuffled is a real contrast — but the
phrase "at fresh seeds" describes a resampling that did not occur. More broadly: **no result in
this repository has construction-seed error bars, because there is no construction-seed variability
to be had.** Every "one poem per cell" caveat I have written in this folder is stronger than I
realised: the one poem is not a sample, it is the constructor's entire output for that cell.

## 2. So: resample over what does vary

Length (8 / 24 / 56 lines) × start coordinate (4 values) = 12 variants per constructor, target
`focused`, Llama-1B, both readouts in the same run. Note valley ignores `start_va` by design, so
its variation comes only from length — three distinct poems, not twelve.

**polygon-pca's inversion fails.** Mean rank 4.00 ± 2.22 under the EMA read and 3.33 ± 2.15 under
the calibrated read, a gap of 0.67 against a spread of 2.22 (Wilcoxon p = 0.48). My pre-stated
criterion required the gap to exceed the spread; it does not. The seed-42 medium-length result that
looked so clean on two models was one variant out of twelve. (The script's first verdict printout
checked only two of the three clauses and said "INVERSION HOLDS"; the logic now applies all three.)

Applying the same test to every constructor, the readouts disagree about a different one:

| constructor | EMA rank − calibrated rank | spread | Wilcoxon p | verdict |
|---|--:|--:|--:|:--|
| **valley** | **−3.00** | 1.31 | **0.004** | passes |
| harmonic-golden | −0.50 | 1.56 | 0.336 | fails |
| harmonic-prime | +0.25 | 0.83 | 0.617 | fails |
| polygon-pca | +0.67 | 2.22 | 0.477 | fails |
| harmonic-organic | +1.17 | 1.38 | 0.094 | fails |
| **graph-walk** | **+1.42** | 0.87 | 0.090 | passes |

Valley — the published winner — is the constructor the two readouts disagree about most, and in the
opposite direction to polygon: best on the pipeline's metric (mean rank 1.9 of 6), second-worst on
the calibrated read (4.9). Treat the p-value with suspicion, though: valley has only three distinct
poems here, so the twelve paired ranks are partly pseudo-replication.

## 3. What actually replicates: the two readouts respond to length differently

Placement error by poem length, averaged across all six constructors:

| | short (8) | medium (24) | long (56) |
|---|--:|--:|--:|
| EMA read | 0.253 | 0.250 | 0.258 |
| calibrated read | 0.105 | 0.151 | 0.177 |

The pipeline's metric is **flat in length**. The calibrated read **degrades monotonically**, and it
does so for 6 of 6 constructors (+0.009 to +0.122 from short to long).

This is the recency weighting made visible, and it is the cleanest confirmation of the mechanism
proposed back in `order_matters`. The EMA at alpha 0.1 only ever sees the last ~30 tokens, so adding
lines to the front of a poem cannot change it. The whole-context read integrates everything, so a
longer poem dilutes its target-band content with whatever else the constructor produced along the
way. Valley is the extreme case because its design puts grounding first and target content last:
lengthening it *improves* its EMA score (−0.044) while *worsening* its calibrated score (+0.122),
the only constructor where the two move in opposite directions.

**Caveat that limits this:** the opposition is not general. On 9B at the calm target, where valley's
length sweep was already collected, both readouts improve together (EMA 0.231 → 0.213, calibrated
0.035 → 0.014 from short to long). So the opposite-direction result is specific to this model and
target, while the underlying asymmetry — EMA flat, calibrated degrading — is what generalises.

## 4. Correction to `explore/gemma9b_check`

That note said the polygon-pca inversion "is the one finding in this arc I would now defend". That
was wrong, and for exactly the reason stated two notes earlier: it rested on one poem per cell. The
cross-model agreement that made it look solid was two single poems agreeing, not two samples.

## Status and caveats

The 9B arm is not done. It needs a 9B word probe (`build_9b_word_probe.py`, running, roughly 16
chunks of 4000 words), because the published 9B numbers only cover canonical seed-42 stimuli and
cannot score fresh length/start variants. Everything above is Llama-1B at the focused target.
Start coordinate does nothing for valley, and the 12 variants are 3 lengths × 4 starts rather than
12 independent draws, so per-constructor n is effectively 3 to 12 depending on which parameters that
constructor honours.

## Opened up

- **Report the seed no-op to the engine.** `polygon_pca`'s unused RNG and `graph_walk`'s ignored
  argument are dead parameters; `_pick_in_band`'s RNG is live but unreachable at this bank size.
  Either wire them up (sample within the band rather than taking the k nearest) or drop the
  parameter. Wiring them up would give the whole project the error bars it currently cannot have,
  and is a `feat/*` branch change with tests, not a lab edit.
- **Re-read `scripts/14`'s journal line.** "Order effect 6/6 constructors at fresh seeds" should say
  what it measured.
- **Finish the 9B arm** once the word probe lands, and check whether EMA-flat / calibrated-degrading
  holds there.
- **Length is a confound in the published leaderboard.** Valley's cell averages short, medium and
  long while most constructors are medium only. Given how strongly the calibrated read responds to
  length, any cross-constructor comparison should hold length fixed.
