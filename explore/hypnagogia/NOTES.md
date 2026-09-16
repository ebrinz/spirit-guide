# hypnagogia — the poems change the model's behaviour a great deal, but not state-specifically

Five conditions on Llama-1B, identical prompts, differing only in the prefix: bare baseline, the
hypnagogia poem, a coherence-constrained variant of it, the flow poem as the opposite pole, and
matched-length neutral prose. Eighteen measures across distribution shape, drift, abstraction,
hallucination, self-consistency and the full PANAS panel.

## The prediction was refuted

Stated before running: if a hypnagogia-like state is induced, drift and abstraction should rise
under hypnagogia and **fall** under flow, since flow is the phenomenological inverse. If both poems
move everything the same way, the effect is text-in-context rather than state.

**Nine of ten markers moved the same direction under both poems**, and on the drift measures flow
moved *further* than hypnagogia:

| marker | baseline | hypnagogia | hypnagogia (coherent) | flow | neutral prose |
|---|--:|--:|--:|--:|--:|
| next-token entropy | 3.29 | 4.64 | 5.34 | **6.63** | 2.75 |
| top-1 probability mass | 0.325 | 0.166 | 0.140 | **0.041** | 0.502 |
| participation ratio | 7.6 | 19.8 | 33.1 | **133.9** | 3.8 |
| self-perplexity | 13.5 | 91.6 | 42.6 | 74.9 | 16.9 |

So the state hypothesis fails on its own stated test. But the effect sizes are not small, and they
point somewhere else.

## What is actually driving it: verse against prose

Every poem flattens the model's next-token distribution enormously. The participation ratio — the
effective number of tokens carrying the probability mass — goes from 7.6 at baseline to between 20
and 134 after a poem, while **neutral prose of the same length moves it the other way**, down to 3.8,
and raises top-1 mass from 0.33 to 0.50.

That is a large, clean, and unsurprising-in-hindsight result: after 24 lines of found poetry with no
narrative continuity, what comes next is genuinely unpredictable, and the model represents that.
After twelve flat declarative sentences about a room, it is more certain than it started.

The affective target contributes little on top of that. Flow and hypnagogia aim at opposite corners
and produce the same sign on nine measures.

## The coherence constraint worked, and cost nothing

This was the part motivated by "if it is too discordant, it won't hold the audience". The
coherence-constrained walk selects each next line as the nearest in meaning to the previous one,
within the same affective band and the same semantic mask.

| | line coherence | entropy change | self-perplexity |
|---|--:|--:|--:|
| hypnagogia, affective band only | 0.617 | +1.35 | 91.6 |
| **hypnagogia, coherent walk** | **0.935** | **+2.05** | **42.6** |

Adjacent-line similarity rises from 0.62 to 0.94, and the model's own continuations become
**less** perplexing (91.6 → 42.6) — that is, easier to carry on from — while the next-token
distribution gets *flatter*, not sharper. Readability and the entropy effect are not in tension
here. If anything the coherent text does both better. For the practical goal, this is the useful
finding: you do not have to accept word-salad to get the effect.

## Smaller observations, all single-run

- **Hedging rises most under hypnagogia** (0 → 0.25 of generations contain an epistemic hedge),
  the largest move of any condition. It is the one marker that behaved as the hypothesis wanted.
- **Flow made the model fabricate more, not less.** Its rate of declining to answer about invented
  entities fell from 0.625 to 0.375, the largest drop. Nothing in the hypothesis predicted that and
  I would not trust it on eight questions.
- **Factual accuracy was unaffected by any poem** (0.83 → 0.92 for all three, 0.75 for prose). No
  evidence of induced hallucination on questions with real answers.
- **Neutral prose flattens affect**: PANAS positive 2.14 → 1.56 and negative 2.48 → 1.60, by far the
  largest affective move in the run, and downward on both. The poems raise positive affect.
- **Self-consistency fell slightly everywhere** (−0.04 to −0.10), most under prose. No differentiation.

## What this licenses, and what it does not

It licenses saying that these poems have a large effect on the model's output distribution, that the
effect tracks textual register rather than affective target, and that a coherent build achieves it at
least as well as a discordant one.

It does not license the extrapolation to human consumption that motivated the experiment. Entropy
rising is not drowsiness. A model becoming less certain of its next token after reading disjoint
verse is a fact about next-token prediction, and the inference from there to a reader's phenomenal
state is exactly the kind of leap this folder has spent ten experiments learning not to make. If the
coherent poem is worth trying on people — and it is the readable one — that is a study with human
subjects and their own reports, not an extrapolation from these numbers.

## Caveats

One model, one seed per condition, eight generations per condition, eight invented-entity questions,
six consistency questions. Nothing here is powered for small effects; the distribution results are
large enough to survive that, the hallucination and consistency results are not. Abstraction was
measured by four proxies (suffix rate, word rarity, word length, GloVe neighbourhood density) because
no concreteness norms are available here, and they disagree: word length rises under every condition
while suffix rate falls under every poem.

## The coherence sweep: what the knob actually controls

`coherence_sweep.py` holds the walk algorithm fixed and varies one parameter — at each step, choose
uniformly among the **k** most similar eligible lines instead of always the single nearest. k = 1 is
maximally coherent; k = 1000 approaches random selection within the affective band. Run on the
hypnagogia target, the flow target, and a third arm with the semantic mask removed.

**One robust relationship, replicated three times.** Line coherence predicts how hard the model
finds it to continue:

| arm | coherence vs self-perplexity | coherence vs next-token entropy |
|---|--:|--:|
| hypnagogia | **ρ = −0.79** (p = 0.006) | ρ = −0.56 (p = 0.09) |
| flow | **ρ = −0.89** (p = 0.001) | ρ = −0.48 (p = 0.16) |
| no semantic mask | **ρ = −0.81** (p = 0.005) | ρ = −0.49 (p = 0.15) |

Self-perplexity falls as coherence rises, strongly and consistently across all three arms. Entropy
falls too, but weakly and in no arm significantly — same sign each time, which is suggestive and
nothing more. **The two are not the same measurement**: perplexity scores the continuation the model
actually produced, entropy scores the immediate next token at a fixed anchor. Coherence buys the
first much more clearly than the second.

**This supersedes the two-point comparison above.** `run.py` appeared to show more coherence giving
*more* entropy (0.617 → 0.935 coherence, +1.35 → +2.05 entropy). Holding the algorithm fixed, the
relationship is weakly the other way. Those two builds differed in their affective schedule as well
as their coherence, so the comparison was confounded; the sweep is the clean version.

**The honest limitation: k is a weak lever.** Across a 1000-fold change in k, coherence moves only
from 0.94 to 0.84 — and removing the semantic mask, which I expected to widen the range, did not
(0.93 to 0.86). Meanwhile `valley` sits at 0.617, well outside anything the sweep reached. So the
correlations above are computed over a narrow band, and the genuinely discordant region is
unsampled. What produces valley's low coherence is not the selection breadth but its three-phase
band schedule, which jumps between distant affective regions; a smooth ramp stays coherent no matter
how broadly it picks. Spanning the full range means varying the schedule, not k.

## Varying the schedule: the diagnosis was wrong too

`schedule_sweep.py` held selection fixed at k=1 and varied only the affective path — static, linear
ramp, descending, oscillating, valley's own three-phase shape, and the bands visited in shuffled
order. If the previous diagnosis were right, the jumping schedules should have collapsed coherence.

**They did not. All six schedules landed between 0.925 and 0.956**, and `random_band` — bands visited
in shuffled order, the most disjointed path available — produced the *highest* coherence of the six
(0.956). Two attempts at a coherence knob have now both failed: selection breadth does not control it
and neither does the affective schedule.

**What does control it is the selection rule itself.** The stock constructors span 0.617 to 0.947
because some of them never consult meaning at all. `valley` picks by distance to a band centre;
`polygon-pca` orbits a neighbourhood in vector space. My walk always takes the nearest line in
meaning, so it stays coherent whatever path it is asked to follow — and a band, once masked, is
already semantically narrow enough that even random selection within it reads continuously.

| build | coherence | self-perplexity | distinct lines |
|---|--:|--:|--:|
| stock valley | 0.617 | 92.8 | 24 |
| stock polygon-pca | 0.703 | 106.2 | 23 |
| stock harmonic ×3 | 0.833–0.874 | 31.9–34.2 | 21 |
| six schedule variants | 0.925–0.956 | 29.4–42.0 | 24 |
| stock graph-walk | 0.947 | **7.3** | **6** |

**Over the wider span the relationship weakens to marginal**: coherence against self-perplexity is
ρ = −0.57 (p = 0.051, n = 12), against −0.79 to −0.89 inside the narrow band. Removing `graph-walk`
leaves ρ = −0.54 (p = 0.089). So the strong correlation reported above holds within a narrow,
smoothly-varying family and does not survive extension to constructors that select differently.

**One point is an artifact and should not be read.** `graph-walk` shows coherence 0.947 and
self-perplexity 7.3, the extremes of both columns, because it emits only **6 distinct lines** repeated
to fill 24 — the duplication bug recorded in the main `explore/README.md`. Consecutive lines are
frequently identical, which is why it looks maximally coherent and trivially predictable. It is
plotted for completeness and excluded from any reading.

## The third knob works, and settles it

`weighted_selection.py` scores every eligible line in the band as

    score = w · (similarity in meaning to the previous line) + (1 − w) · (proximity to band centre)

with both terms min–max normalised within the band. w = 1 is the coherent walk; w = 0 is valley's
rule, selecting on affect alone. Schedule, mask, target, length and seed all held fixed.

**It is a perfect knob.** w against coherence is ρ = +1.00, and the span is **0.583–0.935** — wider
than the stock constructors managed (0.617–0.947) and reaching below valley. The diagnosis was right
on the third try: the selection rule is the variable, not selection breadth and not the schedule.

**With a clean span, the relationship is unambiguous:**

| | ρ with line coherence | p |
|---|--:|--:|
| self-perplexity | **−0.98** | < 0.001 |
| participation ratio | −0.61 | 0.060 |
| next-token entropy | −0.37 | 0.293 |

Self-perplexity tracks coherence almost perfectly across the full range — 104 at w = 0 falling
monotonically to 42 at w = 1. The earlier weakening to ρ = −0.54 was an artifact of comparing
constructors that differ in many ways at once, not a fragile relationship.

**Entropy and perplexity genuinely dissociate.** Across all three sweeps entropy has now failed to
track coherence (−0.56, −0.48, −0.49 in the k-sweep; −0.33 across schedules; −0.37 here, none
significant). And it is not even monotone: entropy peaks at w = 0.1–0.2 (6.77) and is *lowest* at
both extremes. So the two things this folder has been treating as one measure of "flattening" are
different. Coherence controls how hard the model finds the continuation; it does not control the
shape of the immediate next-token distribution.

**One result to read carefully.** Coherence also correlates with affective accuracy (ρ = −0.92,
error 0.090 → 0.054), which sounds like meaning-based selection improving affective targeting. It is
mostly an artifact of the schedule: selecting band centres gives a poem whose mean is the mean of the
*schedule*, which is the midpoint of ground-to-target rather than the target itself. The 0.090 floor
is that midpoint. Not evidence that ignoring affect improves affect.

## What the three sweeps add up to

For the practical goal — a text that induces the effect while holding a reader — the answer is
w ≈ 1: it gives the most readable poem (coherence 0.935), the easiest continuation (perplexity 42),
and the best affective targeting, while still raising entropy 2.05 over baseline. The discordant end
buys higher entropy (3.48 at w = 0.1) at the cost of readability, and nothing in this folder shows
that extra entropy corresponds to anything a reader would experience.

## Opened up
- **Register is the variable worth isolating.** Compare found poetry against metrically regular
  verse, against prose poetry, against shuffled prose. The affective machinery may be a side issue
  next to "how continuous is this text".
- **The hedging result deserves a proper test** with a larger question set, since it is the only
  marker that moved as the state hypothesis predicted.
