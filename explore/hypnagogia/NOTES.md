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

## Opened up

- **Vary the affective schedule, not the selection breadth.** That is where coherence actually comes
  from, and it is the only way to reach the discordant end where valley sits.
- **Register is the variable worth isolating.** Compare found poetry against metrically regular
  verse, against prose poetry, against shuffled prose. The affective machinery may be a side issue
  next to "how continuous is this text".
- **The hedging result deserves a proper test** with a larger question set, since it is the only
  marker that moved as the state hypothesis predicted.
