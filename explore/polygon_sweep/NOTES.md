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

It fails at 9B too: gap +1.50 against a spread of 1.73 (p = 0.055). Both models, not supported.

Applying the same test to every constructor, the readouts disagree about different ones — and
**that pattern replicates across models** (Spearman +0.83 on the six rank gaps, Llama vs 9B):

| constructor | rank gap, Llama | verdict | rank gap, 9B | verdict |
|---|--:|:--|--:|:--|
| **valley** | **−3.00** (p = 0.004) | passes | **−2.00** (p = 0.006) | passes |
| harmonic-golden | −0.50 (p = 0.336) | fails | −1.25 (p = 0.029) | passes |
| harmonic-organic | +1.17 (p = 0.094) | fails | −0.58 (p = 0.391) | fails |
| harmonic-prime | +0.25 (p = 0.617) | fails | 0.00 (p = 0.906) | fails |
| polygon-pca | +0.67 (p = 0.477) | fails | +1.50 (p = 0.055) | fails |
| **graph-walk** | **+1.42** (p = 0.090) | passes | **+2.33** (p = 0.004) | passes |

Negative means the pipeline's metric ranks it better than the calibrated read does.

**This is the finding the polygon claim was a noisy shadow of.** Two constructors clear the criterion
on both models, in opposite directions and at a 9× scale difference:

- **valley**, the published winner, is systematically *favoured* by the pipeline's metric. On Llama
  it is the best constructor by that measure (mean rank 1.9 of 6) and second-worst by the calibrated
  read (4.9); at 9B, 1.4 against 3.4.
- **graph-walk** is systematically *disfavoured* by it: last on the pipeline's metric on both models
  (5.25 on each) and mid-table on the calibrated read (3.8 and 2.9).

Caveat on valley's p-values: it ignores `start_va`, so it contributes only three distinct poems and
its twelve paired ranks are partly pseudo-replication. Its rank still varies genuinely, because the
five constructors it is ranked against do change. graph-walk has the same issue in reverse — it is
deterministic given start and target, so its variation is also length-only.

## 3. What actually replicates: the two readouts respond to length differently

Placement error by poem length, averaged across all six constructors, both models:

| model | EMA short | medium | long | calibrated short | medium | long |
|---|--:|--:|--:|--:|--:|--:|
| Llama-1B | 0.253 | 0.250 | 0.258 | 0.106 | 0.150 | 0.183 |
| Gemma-9B | 0.159 | 0.166 | 0.176 | 0.107 | 0.148 | 0.153 |

| short → long | EMA | calibrated | constructors worsening on calibrated |
|---|--:|--:|--:|
| Llama-1B | +0.005 | **+0.078** | 6/6 |
| Gemma-9B | +0.017 | **+0.046** | 4/6 |

The pipeline's metric is nearly **flat in length** on both models. The calibrated read **degrades**
on both, four to sixteen times as much.

This is the recency weighting made visible, and it is the cleanest confirmation of the mechanism
proposed back in `order_matters`. The EMA at alpha 0.1 only ever sees the last ~30 tokens, so adding
lines to the front of a poem cannot change it. The whole-context read integrates everything, so a
longer poem dilutes its target-band content with whatever else the constructor produced along the
way. Valley is the extreme case because its design puts grounding first and target content last:
lengthening it *improves* its EMA score (−0.044) while *worsening* its calibrated score (+0.122),
the only constructor where the two move in opposite directions.

**Caveat that limits this:** valley's opposite-direction behaviour is Llama-only. At 9B its EMA error
*worsens* with length (0.140 → 0.156) while its calibrated error stays flat (0.146 → 0.144), and at
the calm target both improve together. So the clean opposition is model-specific; what generalises is
the asymmetry itself — the pipeline's metric barely responds to length, the calibrated read does.

## 3b. Why the metric flatters valley: it ends where the metric looks

The two readouts are dominated by *different coordinates*, and that is the whole story. Decomposing
each readout's miss at the focused target (0.65, 0.60), medium length, Llama:

| constructor | EMA miss V | EMA miss A | calibrated miss V | calibrated miss A |
|---|--:|--:|--:|--:|
| harmonic-golden | −0.185 | −0.115 | −0.036 | −0.187 |
| **valley** | −0.215 | **−0.041** | −0.008 | **−0.196** |
| harmonic-organic | −0.224 | −0.109 | −0.042 | −0.143 |
| harmonic-prime | −0.230 | −0.122 | −0.035 | −0.200 |
| graph-walk | −0.253 | −0.101 | −0.104 | −0.089 |
| polygon-pca | −0.276 | −0.099 | −0.039 | −0.053 |

Under the **EMA read**, every constructor undershoots valence badly (−0.19 to −0.28) because the word
probe cannot reach high valence at all; arousal is nearly hit. So that metric is effectively scoring
**valence shortfall**. Under the **calibrated read** valence is essentially solved (−0.01 to −0.10)
and **arousal** becomes the binding constraint. Two metrics, two different questions.

Valley is the constructor the two probes disagree about most on arousal, and its own design explains
why. It grounds in low-arousal content and ascends to the target band, so:

| | whole poem | last 4 lines (the EMA's window) | rise |
|---|--:|--:|--:|
| **valley** NRC arousal | 0.394 | **0.600** | **+0.206** |
| harmonic-golden | 0.389 | 0.470 | +0.081 |
| graph-walk | 0.472 | 0.547 | +0.076 |
| harmonic-prime | 0.381 | 0.429 | +0.049 |
| polygon-pca | 0.550 | 0.595 | +0.044 |
| harmonic-organic | 0.392 | 0.388 | −0.004 |

Valley's closing lines sit at arousal **0.600** — exactly the focused target — while its body sits at
0.394. Its body-to-tail rise is more than twice any other constructor's. So the recency-weighted read
reports arousal 0.559, near target, and the whole-context read reports 0.404, missing by 0.196.

**Across constructors this predicts the rank gap.** How much the EMA read overstates arousal relative
to the calibrated read correlates with the rank gap at **ρ = −0.77 on both models** (p = 0.072 each,
six constructors). Constructors with sustained high arousal throughout — graph-walk (body 0.472) and
polygon-pca (0.550) — have *negative* gaps: the calibrated read sees their sustained arousal, the
compressed word probe caps it, and the pipeline's metric penalises them.

So the answer to "why does the metric flatter valley" is: **valley is the only constructor built to
put target-band content last, and the metric weights the last ~30 tokens at 96%.** The design and the
measurement are matched. That is not cheating — the constructor was designed against a trajectory
framing, and the metric implements that framing — but it does mean the leaderboard gap between valley
and the rest is partly a statement about where each constructor places its target content, not only
about how well it places the model's state.

Caveats: six constructors, so p = 0.072 is directional rather than conclusive; the poem-structure to
probe-gap link is weaker still (ρ = +0.60, p = 0.21). The valley case itself is not marginal — every
number in its chain is the extreme of its column.

## 4. The two instruments, side by side

Worth recording since both now exist for both models, built the same way:

| model | word probe (the EMA read) | passage probe (the calibrated read) |
|---|---|---|
| Llama-1B | layer 10, R²_v 0.718 | layer 15, R²_v 0.919 |
| Gemma-9B | layer 14, R²_v 0.719 | layer 6, R²_v 0.932 |

The fresh 9B word probe lands at layer 14 with R²_v 0.7195, reproducing the gate value recorded in
`docs/experiments-journal.md` for E14 ("gate layer 14, r2v 0.719") to three decimals. The published
9B run then moved its probe to layer 24 by hand after a context-sensitivity scan, so this is the
same instrument class but not that exact probe.

## 5. Correction to `explore/gemma9b_check`

That note said the polygon-pca inversion "is the one finding in this arc I would now defend". That
was wrong, and for exactly the reason stated two notes earlier: it rested on one poem per cell. The
cross-model agreement that made it look solid was two single poems agreeing, not two samples.

## Caveats

Both models, target `focused` only. The 12 variants are 3 lengths × 4 start coordinates, not 12
independent draws, so per-constructor n is effectively 3 to 12 depending on which parameters a
constructor honours — valley and graph-walk ignore `start_va`, giving them three distinct poems
each. The 9B word probe is a fresh instrument at layer 14, not the layer-24 probe behind the
published 9B table.

## Opened up

- **Report the seed no-op to the engine.** `polygon_pca`'s unused RNG and `graph_walk`'s ignored
  argument are dead parameters; `_pick_in_band`'s RNG is live but unreachable at this bank size.
  Either wire them up (sample within the band rather than taking the k nearest) or drop the
  parameter. Wiring them up would give the whole project the error bars it currently cannot have,
  and is a `feat/*` branch change with tests, not a lab edit.
- **Re-read `scripts/14`'s journal line.** "Order effect 6/6 constructors at fresh seeds" should say
  what it measured.
- ~~**Why does the metric flatter valley?**~~ ANSWERED in §3b: valley puts target-band content last
  (body-to-tail arousal rise +0.206, more than double any other constructor) and the metric weights
  the last ~30 tokens at 96%. Across constructors the EMA-vs-calibrated arousal gap predicts the rank
  gap at rho -0.77 on both models.
- **Test it causally.** The account predicts that reversing valley should collapse its EMA advantage
  while barely moving its calibrated score. `order_matters` already measured the first half
  (reversing valley costs +0.116 on the EMA, its largest reversal cost) and the second
  (+0.047 calibrated); running both readouts over reversed poems for all six constructors would
  close the argument properly.
- **Length is a confound in the published leaderboard.** Valley's cell averages short, medium and
  long while most constructors are medium only. Given how strongly the calibrated read responds to
  length, any cross-constructor comparison should hold length fixed.
