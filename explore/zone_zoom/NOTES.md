# zone_zoom — the score belongs to the text, not the place (2026-09-15, Gemma-2b-it)

Follow-up to `explore/model_map`, zooming into the region around its two best landings. The zoom
answered a prior question instead: **is the score a property of the coordinate at all?** It is
mostly not. That invalidates the premise of the two previous explorations' search framing, so it
is written up first and plainly.

## The decisive number

Eight targets in the zone, each re-landed with five fresh search seeds, same pool and same eight
steps. One-way decomposition of the score:

| quantity | value |
|---|--:|
| between-target SD | 0.28 |
| within-target (seed) SD | **0.85** |
| intraclass correlation | **0.10** |
| F(7, 32) | 1.55, p = 0.19 |
| original score vs replicated mean (n = 8) | ρ = +0.31, p = 0.46 |

Only about a tenth of the score variance is attributable to *where* the search was aimed. Which
text the search happened to write matters roughly three times as much. The original run's ranking
does not reproduce: the target that scored −1.23 came back at +0.18, and two of the top three
regressed toward the middle. `gap_09` does keep the highest replicated mean (+1.06), but its own
five seeds span −0.1 to +2.6, so even that is not a stable property of the location.

## The double dissociation

Two ways to perturb the winner, each moving a comparable distance in the map:

| manipulation | what is held | distance moved | PANAS inspired |
|---|---|--:|--:|
| re-land the same target, new seed | the place | lands 1.0 grain from target | 3.30 ± 0.38 |
| swap 1 line of 8, keep the rest | the text | 0.35–0.85 grain | **3.85 ± 0.23** |
| `gap_09` untouched | both | 0 | 4.04 |

Keeping the text and moving 0.35–0.85 grain preserves inspired near 4.0. Landing at the *same*
coordinate with different text drops it to 3.30 and scatters it twice as widely (Δ = +0.55,
t = 7.00, p < 0.0001). For scale, the two original winners' targets were 0.78 grain apart — inside
the range the swaps moved without losing the effect. Position is not what carries it.

Distance moved does not predict the swap's score either (ρ = −0.23 across 20 swaps), so it is not
that small moves are safe and large ones are not. The specific lines are doing the work.

## Zooming does not simplify the space

2,000 fresh phrase-graph passages, measured against the zone:

| cloud | n | grain | intrinsic dim |
|---|--:|--:|--:|
| fresh passages, all | 2000 | 0.72 | 17.8 |
| fresh passages within 1.5 grain of the midpoint | **1067** | 0.80 | **18.3** |
| original poem cloud, same zone | 627 | 0.84 | 18.6 |

Two things follow. First, **the zone is not a special place**: 53% of all freshly generated
passages land inside it, so it is simply where the constructor puts most poems. Second, **the
region is statistically self-similar** — intrinsic dimension inside the ball is 18.3 against 17.8
for the whole cloud, and relative grain barely moves. Restricting the volume did not buy
resolution. The samples-per-dimension problem from `model_map` cannot be fixed by zooming; it
would need orders of magnitude more states, or a different space to work in.

## What this means for the previous two explorations

- **`model_map`'s arm comparison is weaker than it read.** Its per-arm means rest on one seed per
  target, and the seed SD here (0.85) is larger than most of the arm differences it reported. Its
  one robust effect, the Wikipedia arm at −1.24 z, is about 1.5 seed-SDs and survives; the gap
  arm's +0.50 does not.
- **`ideal_state`'s ranking needs the same treatment.** Those were also one poem per state. The
  word-list centroids are at least fixed by the lexicon rather than by a stochastic search, so the
  poems themselves are deterministic, but nothing there establishes that a different seed at the
  same target would rank the same.
- **The score remains largely a valence reading**, replicating across runs: score vs deep valence
  ρ = +0.56, p < 0.001 here, +0.46 in `model_map`.

## Opened up

- **Report seed distributions, never single landings.** Any future search arm needs ≥3 seeds per
  target, and effects should be quoted against the seed SD. Cheap to do and it changes conclusions.
- **Ask what in the text carries it.** The swaps say specific lines matter more than location.
  A line-level ablation on `gap_09` — remove each of the eight in turn, and substitute matched
  lines — would identify whether one or two lines carry the whole effect.
- **The question may need a different space.** If score is text-determined and the residual
  coordinate does not predict it, then a 32-d PCA of layer 20 is the wrong index for "good state".
  Worth testing whether score is predictable from anything: SAE features of the landed state, or
  the lines' own lexical properties, versus the coordinate.
- **Not worth doing:** a finer geometric search in this zone. The density result says there is no
  resolution to be gained, and the ICC says there is nothing at the target end to resolve.
