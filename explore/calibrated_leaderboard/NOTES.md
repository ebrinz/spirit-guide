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

## First: this corrects `explore/order_matters`

That note reported the two readouts agreeing on "essentially nothing" (ρ = −0.10) and concluded the
constructor ranking is metric-dependent. **That does not hold on the full stimulus set.**

| scope | n | rank agreement |
|---|--:|--:|
| `order_matters`, 5 constructors × 3 targets, 1 poem per cell | 5 | ρ = −0.10 |
| per stimulus, here | 75 | **ρ = +0.78** (Pearson +0.83) |
| all constructor × generator cells | 22 | **ρ = +0.78**, p < 0.0001 |
| found-poetry rows only, as the README table reports them | 7 | **ρ = +0.72**, p = 0.068 |

The earlier estimate rested on five constructors with one poem each. It was underpowered in exactly
the way `zone_zoom` warned about, and this time the single-seed problem bit my own analysis rather
than the pipeline's. The metric-dependence claim in `order_matters/NOTES.md` should be read as
superseded by this.

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
calibrated one. This replicates the same shift seen in `order_matters` (fourth → first there), so
it is not noise. It is worth understanding before either ordering is quoted with confidence.

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
