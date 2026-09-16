# calibrated_leaderboard — the ranking mostly survives, and it corrects my last note

Runs the full canonical stimulus set (75 stimuli after `scripts/02` + `02b`) under two readouts on
the same texts and the same model:

- **EMA read** — the published measure: word-trained probe (`data/probe/probe.pkl`, layer 10) at
  every token, EMA at alpha 0.1, final value's distance to target.
- **calibrated read** — passage-trained probe (`data/passage_probe/probe_passage.pkl`, layer 15,
  held-out R²_v 0.919) read once at a fixed anchor after the poem.

These are two different measures, not a correction of one by the other. The passage probe was
trained on anchor-token states, so it cannot be run per-token inside an EMA without going
off-distribution. The only question asked here is whether the leaderboard's *ordering* depends on
which is used.

## First: I over-corrected `explore/order_matters`, and then corrected that too

**Read this section as the settled version; the paragraph that follows records a mistake I made in
this same file.** `order_matters` reported ρ = −0.10 between the two readouts and concluded the
constructor ranking is metric-dependent. My first pass at this file said the full stimulus set
refuted that (ρ = +0.78) and marked the earlier claim superseded. Then `why_polygon.py` showed the
+0.78 is an artifact of pooling.

**Why the pooled number is not a ranking agreement.** Both readouts agree that `excited` is a hard
target and `focused` an easy one. Pooling every stimulus together lets that shared target-difficulty
signal masquerade as agreement about constructors. Split by target, at matched conditions
(medium/plain/unfiltered, 6 constructors each):

| scope | n | rank agreement |
|---|--:|--:|
| pooled across targets | 40 | ρ = +0.79 |
| within target: calm | 6 | **ρ = +0.89** (the only one significant at n = 6) |
| within target: focused | 6 | ρ = −0.60 |
| within target: excited | 6 | ρ = −0.43 |
| **mean within-target** | | **ρ = −0.05** |

With six constructors, |ρ| must exceed 0.83 for p < 0.05, so only the calm agreement is established;
the two negative values are "indistinguishable from zero", not demonstrated disagreement. But the
mean of −0.05 is the honest summary, and it matches what `order_matters` found. **That note was
right and my correction of it was wrong.** The lesson is the mirror image of the one I drew there:
not "small samples mislead" but "pooling across a strong nuisance factor manufactures agreement".

## The headline result survives the calibrated ruler

Found-poetry rows, the set the README's table reports:

| constructor | EMA (published measure) | calibrated read | rank change |
|---|--:|--:|:--|
| **valley** | **0.308** (1st) | **0.128** (1st) | — |
| harmonic-golden | 0.348 (2nd) | 0.220 (3rd) | −1 |
| harmonic-organic | 0.369 (3rd) | 0.226 (4th) | −1 |
| graph-walk | 0.373 (4th) | 0.238 (5th) | −1 |
| polygon-pca | 0.374 (5th) | **0.203 (2nd)** | **+3** |
| harmonic-prime | 0.374 (5th) | 0.265 (6th) | −1 |
| via-negativa | 0.570 (7th) | 0.541 (7th) | — |

Valley first and via-negativa last under both readouts. The README's two central ordering claims —
that valley wins and that via-negativa is the worst control — hold under a calibrated ruler.

**The one real re-ordering is `polygon-pca`**, fifth on the published measure and second on the
calibrated one. This replicates the shift seen in `order_matters` (fourth → first there).

## Why polygon-pca rises

It does not rise everywhere. Its rank out of six, by target, at matched conditions:

| target | EMA read | calibrated read |
|---|--:|--:|
| calm | 0.338 (2nd) | 0.156 (2nd) |
| focused | 0.293 (**6th**) | 0.066 (**1st**) |
| excited | 0.521 (**5th**) | 0.312 (**1st**) |

The readouts agree about polygon at calm and invert completely at the other two targets. So the
"rise" is entirely a non-calm phenomenon, which is the same place the within-target rank agreement
falls apart.

**The reason looks like the word probe's range.** The two readouts do not use the same amount of the
plane:

| readout | valence range | arousal range | area |
|---|--:|--:|--:|
| EMA + word probe | 0.26–0.53 | 0.36–0.59 | 0.062 |
| anchor + passage probe | 0.36–0.73 | 0.19–0.66 | 0.173 |

And the EMA read barely responds to where the poem is aimed. Averaged over the six constructors, it
lands at almost the same coordinate whichever target the poem was built for:

| target | EMA lands | calibrated lands |
|---|---|---|
| calm (0.75, 0.20) | (0.46, 0.38) | (0.59, 0.26) |
| focused (0.65, 0.60) | (0.42, 0.50) | (0.61, 0.46) |
| excited (0.80, 0.85) | (0.45, 0.50) | (0.61, 0.55) |

The calibrated read's arousal tracks the target across the full sweep, 0.26 → 0.46 → 0.55 against
targets of 0.20 → 0.60 → 0.85. The EMA read's valence moves by 0.04 across targets that differ by
0.15, and its arousal saturates at 0.50 for both focused and excited. Its placement error is then
dominated by the fixed gap between its narrow landing zone and wherever the target sits, which is a
property of the target, not of the constructor. That is both why pooling across targets produces a
high correlation and why the within-target constructor ordering it reports is close to arbitrary
away from calm.

This is a hypothesis about the instrument, not a proof: the compression could come from the word
probe, from the EMA smoothing pulling values toward a running mean, or both. `order_matters` found
the anchor read through the same word probe also compressed (constructor spread 0.019), which points
at the probe as the larger contributor.

The calibrated read also gives uniformly lower errors (mean 0.258 against 0.398), as expected from a
probe with a working range: the word probe reads the 24 calmest and 24 most distressed passages at
almost the same point, the passage probe separates them by 0.5.

## Order, on the pipeline's own shuffled controls

The six built-in shuffled stimuli, each against its exact parent:

| constructor | EMA cost of shuffling | calibrated cost |
|---|--:|--:|
| harmonic-prime | +0.095 | +0.027 |
| graph-walk | +0.088 | −0.032 |
| valley | +0.040 | +0.008 |
| harmonic-golden | +0.035 | +0.021 |
| polygon-pca | +0.034 | −0.014 |
| harmonic-organic | −0.014 | +0.030 |
| **mean** | **+0.046** (hurt 5/6) | **+0.007** (hurt 4/6) |

Consistent with `order_matters`: shuffling costs the pipeline's metric several times what it costs a
calibrated whole-context read. Six pairs is not a statistical claim on its own, but combined with
that run's 15 cells the direction is steady. The cleanest single case is valley at calm, which moves
0.040 under the EMA and 0.008 under the calibrated read — 0.038 ordered against 0.046 shuffled.

## Caveats

Llama-1B only; the published table also reports Gemma-2B and Gemma-9B, which have calibrated probes
but were not re-run here. `data/renders/renders.jsonl` was absent (it needs `scripts/03`, which calls
an external model), so LLM-rendered verse is missing from this comparison and the README's
"raw found-poetry beats LLM-rendered verse" claim is untested here. Cells have unequal n — valley psg
averages 8 stimuli, each shuffled control is 1 — so the cell means are not equally precise.

## Opened up

- **Why does polygon-pca rise?** The only robust re-ordering across both runs. If it places well on
  a calibrated whole-context read but poorly on a recency-weighted one, that says something specific
  about how it distributes target-band lines.
- **Repeat on Gemma-2B.** Its calibrated probe already exists (`data_gemma2b/passage_probe/`), so
  this is a cheap check of whether the agreement is model-general.
- **Give leaderboard rows error bars.** Several cells here sit within 0.001 of each other
  (polygon-pca and harmonic-prime tie at 0.374 on the EMA read), which is far below the spread a
  different construction seed produces.
- **Render the LLM-verse arm** if that README claim is to be checked under the calibrated read.
