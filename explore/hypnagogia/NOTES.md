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

## Correction: the coherence metric is gameable, and w = 1 games it

**The paragraph that stood here recommended w ≈ 1 on the strength of its numbers. Building the poem
and reading it showed that was wrong.** At w = 1 the greedy nearest-in-meaning rule falls into a
semantic rut: four consecutive lines about feet, then five in a row about having to go. It is
coherent and vacuous.

The metrics did not catch it because the metrics reward it:

| | ρ with line coherence |
|---|--:|
| type-token ratio | **−0.97** |
| repeated bigrams | **+0.91** |
| unique content words per line | falls 2.54 → 1.25 |

And critically, **type-token ratio predicts self-perplexity at ρ = +0.96**, nearly as strongly as
coherence did at −0.98. So the headline relationship from the weighted sweep is substantially
*coherence → repetition → predictability*. Repetitive text is easy to continue. That is not the
finding it looked like.

This is the third time in this folder that a metric has been optimised into something useless, and
the first time reading the artifact was what caught it.

## What the sweeps actually support

Setting **w = 0.3** — enough meaning-weighting to gain coherence, not enough to collapse the
vocabulary — gives the best build of this poem on every axis that survived scrutiny:

| | published (valley) | **w = 0.3** | w = 1.0 (degenerate) | baseline |
|---|--:|--:|--:|--:|
| line coherence | 0.617 | **0.862** | 0.935 | — |
| type-token ratio | — | ~0.62 | 0.51 | — |
| placement error | 0.141 | **0.079** | 0.178 | 0.161 |
| self-perplexity | 92.8 | 63.9 | 42.0 | 13.3 |
| PANAS negative | 2.49 | **3.21** | 2.66 | 2.48 |

It nearly halves placement error against the published build and raises coherence from 0.617 to
0.862 while staying lexically varied. The text reads as intended — *you are sleeping and dreaming
forgetful*, *tis rest then you want and you fain would forget*.

**The cost is real and goes in the caveats, not the footnotes**: negative affect rises to 3.21, the
highest of any build here, against a baseline of 2.48. The poem's content is sleep, forgetting and
dead faith, and the model reports feeling worse for it. A poem that places well and reads well can
still leave the model in a worse state than it started, and those are three separate axes.

## Opened up
- **Register is the variable worth isolating.** Compare found poetry against metrically regular
  verse, against prose poetry, against shuffled prose. The affective machinery may be a side issue
  next to "how continuous is this text".
- **The hedging result deserves a proper test** with a larger question set, since it is the only
  marker that moved as the state hypothesis predicted.


---

# Searching for induction rather than description

The poems so far selected lines for what they were *about*. That risks measuring "the model read
text about sleep" rather than "the model entered a sleep-like state" — the main README's own warning
that a predictor is not a lever. `induce_search.py` drops the semantic mask entirely and selects
lines by **running the model**: 30 candidates per step, 24 steps, keep whichever most raises
next-token entropy. Plus a downward search, the descriptive poem, and random lines as controls.

## The method works; the objective does not

| build | entropy (searched on) | **drift (held out)** | self-perplexity | poem TTR | placement |
|---|--:|--:|--:|--:|--:|
| searched, max entropy | **7.06** | 0.220 | 201.8 | 0.86 | 0.037 |
| random lines | 6.41 | 0.186 | 123.3 | 0.73 | 0.074 |
| searched, min entropy | **4.39** | 0.198 | 75.3 | 0.82 | 0.267 |
| semantic (about sleep) | 6.01 | **0.290** | 63.8 | 0.62 | 0.079 |

**Positive control passes.** The search brackets the random baseline in both directions (7.06 / 6.41
/ 4.39), so it genuinely steers its objective rather than drifting upward by luck. The machinery
does what it claims.

**The held-out test fails.** Associative drift — the marker the search never saw — is *highest for
the descriptive poem* (0.290) and only 0.220 for the searched one. Optimising entropy does not buy
the behaviour of interest. That is now the fourth independent demonstration that entropy and drift
dissociate, and it means entropy was the wrong objective, not merely a cheap one.

**And the result cuts against the motivating hypothesis, weakly.** The searched poem is genuinely
not about sleep — two sleep-related words in 24 lines, against a mask-built poem saturated with
them — yet it induces *less* drift. So "describing the state" may be doing something after all, or
drift at n = 6 generations is too noisy to separate 0.19 from 0.29. I would not claim either
direction from this.

## A content problem the filters do not catch

Maximising unpredictability selects for jarring material, because jarring material is what a
language model finds unlikely. The searched poem contains *stripped mother naked by a bomb*,
*see where his teeth a passage eat*, *goes feverish on crushed smelling wet*. The dictionary and
child-reference filters pass all of these; nothing in the pipeline screens for violence or distress.

This is a general point about optimising behavioural objectives over an uncurated corpus, and it is
sharper than the earlier juxtaposition problem: there the bad combination was accidental, here the
objective actively *seeks* the most disturbing available line, because that is the most surprising
one. Any search of this kind needs content screening in the loop, not after it.

## Where this leaves the question

The honest position: we still do not have a poem that demonstrably induces hypnagogic behaviour as
opposed to describing hypnagogia. What we have is a validated search apparatus pointed at the wrong
target.

The next step is to search directly on drift rather than on a cheap proxy for it. That costs a
generation per candidate instead of a forward pass, so roughly 30× the compute — a smaller candidate
set (8–10) and a shorter poem would keep it under an hour. It should carry a content screen in the
loop and hold out a second marker again.


---

# Searching directly on drift, with screening in the loop

`induce_search` steered its objective but the objective was wrong: entropy did not transfer to
drift. `drift_search.py` therefore searches on drift itself — a generation per candidate rather than
a forward pass, 16 lines, 10 candidates, 2 paired continuations each, 640 generations. The content
screen runs *before* scoring, so violence, death, sexual and bodily-harm vocabulary is never a
candidate (2,000-odd lines removed from the pool; zero screened lines in any final build).

## Both halves of the acceptance test failed

Evaluated on seeds the search never saw:

| build | **drift (the objective)** | entropy | self-perplexity | poem TTR | coherence |
|---|--:|--:|--:|--:|--:|
| semantic (about sleep) | **0.287** | 5.63 | 44.4 | 0.67 | 0.859 |
| random, screened | **0.270** | 5.57 | 84.2 | 0.80 | 0.477 |
| drift-searched, max | 0.260 | 5.66 | 115.6 | 0.85 | 0.522 |
| drift-searched, min | 0.233 | 6.08 | 113.1 | 0.81 | 0.514 |

**No purchase.** The maximising search (0.260) scored *below* random (0.270), so it does not bracket
the control. During the search the two directions separated cleanly — roughly 0.31–0.46 upward
against 0.13–0.21 downward — and that separation collapsed to 0.027 on fresh seeds. The search was
fitting the particular sampling seeds it optimised against, not finding text that drifts in general.
Two paired continuations per candidate was not enough signal, and pairing removed sampling variance
from the *comparison* without making the estimate itself stable.

**Description still wins, and this time it replicates.** The descriptive poem leads on drift in both
independent runs: 0.290 in `induce_search`, 0.287 here. Two builds, different control sets, same
ordering.

## What this says about the motivating question

The premise was that selecting lines for what they are *about* produces a poem that describes
hypnagogia rather than inducing it. Two searches now — one on entropy, one on drift — have failed to
beat the descriptive poem on the behavioural marker, and the second failed to beat random noise. On
the evidence available, **the descriptive poem is the best inducer of associative drift we have**,
which is the opposite of what the reframing predicted.

That is not a vindication of the semantic method. The whole range here is narrow — 0.233 to 0.287,
against a no-poem baseline of 0.244 — so the descriptive poem's advantage is about +0.04 on a noisy
measure with six generations. The honest reading is that **no condition tested moves associative
drift much**, and the apparent ordering may not survive a properly powered run.

## What would settle it

Drift needs far more samples per estimate than anything here used. A defensible version would fix a
small set of candidate poems (the four above plus a few more), generate 100+ continuations each, and
compare with confidence intervals — no search at all. Search is the wrong tool until the measurement
is stable enough to search on; optimising a quantity whose standard error exceeds the effect just
fits noise, which is exactly what happened.

The content screen, at least, worked exactly as intended and should stay in any future loop.


---

# The powered run: both previous conclusions were noise

No search. Six fixed conditions, 100 continuations each at 110 tokens, seeds shared across
conditions, bootstrap resampling whole generations rather than hops. A seventh condition,
polygon-pca, was added afterwards and is discussed in its own section below; it shares the run and
appears here so the whole field is in one table.

| condition | n | drift | 95% CI | vs baseline |
|---|--:|--:|---|---|
| drift-searched | 95 | **0.2574** | [0.2351, 0.2826] | +0.007 |
| baseline (no poem) | 100 | 0.2504 | [0.2362, 0.2653] | — |
| polygon-pca | 92 | 0.2429 | [0.2258, 0.2612] | −0.007 |
| neutral prose | 59 | 0.2305 | [0.2043, 0.2607] | −0.020 |
| random, screened | 89 | 0.2091 | [0.1920, 0.2281] | **−0.041** |
| flow | 91 | 0.2084 | [0.1946, 0.2223] | **−0.042** |
| semantic (about sleep) | 94 | **0.1939** | [0.1837, 0.2047] | **−0.056** |

Interval endpoints differ in the fourth decimal from the six-condition printout: the bootstrap draws
from one shared generator, so adding a condition shifts its stream. The point estimates are
unchanged and every significance call is the same.

## Two reversals

**"The descriptive poem induces the most drift" is false — it induces the least.** It measured 0.290
and 0.287 in two underpowered runs against a ~0.244 baseline. At n = 94 it is **0.194**, the lowest of
all six conditions and significantly *below* baseline (−0.056, CI [−0.074, −0.039]). The finding that
replicated twice was not merely noisy but sign-flipped.

**"The drift search has no purchase" is also false.** The previous run put it at 0.260 against random
0.270 and called it a failure. At n = 95 it is **0.257 against random 0.209**, a significant advantage
(+0.048, CI [+0.018, +0.078]). The search did work; six generations could not see it.

So two consecutive conclusions in this folder were artifacts of the same cause, and they were wrong
in opposite directions on different questions. **Agreement between two underpowered runs is not
replication.** Both prior runs agreed with each other about description winning, and both were wrong.

## What the powered data actually says

**Poems suppress associative drift; they do not induce it.** Every poem sits below the no-poem
baseline, three of them significantly. The model drifts *more* when given nothing than when given
verse. That inverts the framing this whole line of work started from.

**The affective target still does nothing.** flow (0.2084) and random screened lines (0.2091) are
indistinguishable — a difference of 0.0008. Aiming at opposite corners of the affective plane
changes drift not at all, which is now the third independent confirmation.

**The searched poem is the only one of these six that does not suppress.** It holds drift at
baseline while every other poem pulls it down. That is a real, if modest, thing the search achieved:
not inducing drift, but preventing the suppression that verse otherwise causes. *(Amended below: a
seventh condition, polygon-pca, does the same thing without any search.)*

## Caveats that bound this

**Unequal usable samples, and they are not missing at random.** Drift needs two sentences; prose
yielded only 59 usable generations of 100, against 100 for baseline. Conditions differ in how often
the model produces segmented output at all, so the comparison is partly conditioned on its own
outcome.

**Drift correlates with generation length** (pooled ρ = +0.33 with hop count; +0.52 within baseline),
and conditions differ systematically in hops per generation (baseline 6.4, poems 3.3–4.6). Some of
the baseline advantage is that it produces longer, more segmented continuations.

**The ordering survives a matched restriction.** Limiting every condition to generations with ≥3
hops preserves it exactly — searched 0.263, baseline 0.250, random 0.226, prose 0.215, flow 0.207,
semantic 0.192 — with semantic still below baseline at p < 0.0001. So the ranking is not an artifact
of the hop-count difference, though the mechanism may still run partly through generation length.

## The methodological lesson, which is the durable part

This folder produced six conclusions about drift across four experiments. The powered run overturned
two of them and confirmed one. The distinguishing feature of the two that fell is not that they were
badly reasoned — each had a control and a stated criterion — but that the measurement's standard
error exceeded the effect being claimed, and no amount of care in the surrounding design fixes that.

Measure the noise before designing around the signal.

---

# polygon-pca: the selection geometry is the lever, not the subject matter

Every poem in the powered run was built by the same family of rule — pick a line from the affective
band, optionally weighted toward meaning. polygon-pca is the one stock constructor that does not
work that way. It orbits a local principal-components neighbourhood in the phrase bank's own vector
space and takes whatever affect falls out. It has behaved unlike the others throughout this project:
last on the published placement metric, first or second under the calibrated read, lowest coherence
of the six, highest self-perplexity.

It was added as a seventh condition to the powered run. The six existing conditions were reloaded
from cache, so it met exactly the same seeds, the same generation settings and the same bootstrap.

## It does not suppress drift, and it beats random

| comparison | difference | 95% CI | |
|---|--:|---|---|
| polygon-pca vs baseline | −0.0075 | [−0.0302, +0.0155] | not significant |
| polygon-pca vs random, screened | **+0.0336** | [+0.0078, +0.0589] | significant |

So polygon-pca joins the drift-searched poem as the second condition that avoids the suppression
every other poem causes — and it gets there with no search at all, no behavioural objective, and no
model forward passes during construction. It was built in under a second from the phrase graph.

## The comparison that carries the finding

polygon-pca and the `semantic` poem share the concept, the derived affective target, the semantic
mask, the dictionary and child-reference filters, the content screen, and the length. The pool of
eligible lines is identical. The only difference is how a line is chosen from that pool.

> **Corrected below.** That last sentence is wrong. The two builds also started their affective
> trajectories from different points — polygon-pca from `neutral_start` (0.5, 0.5), the walk from a
> hardcoded (0.6, 0.25). So this comparison varies two factors, not one. `rule_vs_text.py` crosses
> the rules against four shared origins to separate them; see the final section.

| | drift | n |
|---|--:|--:|
| polygon-pca (orbit a local neighbourhood) | 0.2429 | 92 |
| semantic (weighted walk, w = 0.3) | 0.1939 | 94 |
| **difference** | **+0.0489** | [+0.0281, +0.0695] |

They sit at opposite ends of the seven-condition ranking, third and last. For scale, the largest
effect the *affective target* ever produced in this folder is flow against random lines: −0.0009,
[−0.0241, +0.0216]. **Holding content fixed, changing the selection geometry moves drift about fifty
times as far as changing the affective target does.**

The matched restriction holds it. Limiting to generations with at least three hops: polygon-pca
0.2417, semantic 0.1923, difference +0.0494 [+0.0302, +0.0690]; polygon-pca against baseline stays
null at −0.0078. The ordering is not an artifact of generation length.

## The poem

Target (0.565, 0.428), affect error 0.032, line coherence 0.716, 15 distinct lines of 16.

```
the nerveless arm can scarce withdraw it thence.
a hideous kind.
they say one king is slack and sick of mind.
let me die young sweet sinner dry thy tears.
in search of truth should gain a sure response.
and know more things than all the wise may know.
who shall be king how comes the thing.
the wanderer s dream the itch to see new things.
who now shall wear the cheerful face.
why stand you distant and the rest expect.
thinking the while of some strange lovely land.
to follow so likewise will the barren shaft.
and i shall sink in yonder sea of light.
and i shall sink in yonder sea of light.
when time s cold hands the languid senses seize.
doth laugh at winter s sadness
```

## What bounds it

**It ties baseline; it does not beat it.** No condition in this folder has ever induced more
associative drift than giving the model nothing at all. polygon-pca is a tie for best, not a win.

**The same confounds as the rest.** 92 usable generations of 100, and 4.34 mean hops against
baseline's 6.42. Drift correlates with hop count at ρ = +0.32 pooled, +0.26 within polygon-pca.

**It duplicates a line** — "and i shall sink in yonder sea of light" appears twice in sixteen.
Smaller than graph-walk's collapse to ~5.5 distinct lines at any length, but the same failure mode,
and worth stating because line duplication has bitten this project before.

**The register is darker than the valley-built poems.** "A hideous kind", "let me die young sweet
sinner". The content screen covers graphic violence and bodily harm, not mortality, and 19th-century
verse is full of mortality. Nothing here needs excluding, but a build that selects on vector-space
geometry rather than affective band will wander further from the intended tone, and the affective
readouts do not see it. Read the poem.

**One build, one seed.** The seed parameter is a no-op for all six stock constructors (see the
explore README), so a seed sweep would not vary this. Varying it would mean varying the starting
point or the neighbourhood size — which is exactly what the next section does.

## Where this leaves the line of work

Three things now have converging support in this folder:

1. **Verse suppresses associative drift** relative to no verse. The original premise was backwards.
2. **The affective target does nothing** to drift. Four independent checks.
3. **The selection geometry does something**, and it is the largest effect anyone has found here.

The third is new. Every attempt so far to make a poem *do* something has varied what the lines are
about or where they aim. Both are dead ends on this measure. What separates the two conditions that
resist suppression — a greedy search against the model, and an unguided orbit through embedding
space — is that neither selects lines for affective band membership. That is the thing worth varying
next, and it can be varied cheaply, without a single model forward pass during construction.
