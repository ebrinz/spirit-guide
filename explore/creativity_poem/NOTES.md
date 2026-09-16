# creativity_poem — a poem for creativity and mentation, and why affect targeting alone cannot make one

Ad-hoc build using what this folder learned, aimed at the NRC centroid of 49 words across making,
mentation, insight, curiosity and imagination: **(0.723, 0.471)**. The arousal is mid-range
deliberately — every experiment here found it capped near 0.5, so an "excited"-style 0.85 target is
one nothing reaches.

## The finding that matters

**A valence–arousal target does not target a concept.** Built the ordinary way, aiming only at the
coordinate, the poems have nothing to do with creativity or thinking:

| build | lines containing any making/mentation word |
|---|--:|
| valley, 24 lines | **0 / 24** |
| polygon-pca, 8 lines | 1 / 8 |

The valley poem opened "yea and in quiet sleep · quiet as a moonbeam", then drifted through "social
systems four or five", "theology fine arts or finer stays" and "loud voiced ambassador from sea to
sea". Perfectly on-coordinate; entirely off-topic. The reason is structural: (0.723, 0.471) reads as
"fairly positive, medium arousal", which thousands of unrelated lines satisfy. Reducing a concept to
two numbers discards everything that made it *creativity* rather than *a pleasant afternoon*.

This is the same shape as `model_map`'s result that geometric novelty did not become affective
novelty — here, affective targeting does not become semantic targeting.

## The fix: add the missing axis

A **semantic mask** keeps only lines whose mean GloVe vector is in the top 10% by cosine similarity
to the concept centroid (4,869 of 50,000 lines survive it together with the clean-line filter). The
constructor then walks its usual affective path *within* that pool. It improves both readouts at
once, which is not what a pure constraint usually does:

| build | calibrated placement | Δ inspired | Δ attentive |
|---|--:|--:|--:|
| valley 24, clean | 0.112 | +0.82 | +0.73 |
| **valley 24, clean + semantic** | **0.099** | **+0.93** | **+0.78** |
| polygon-pca 8, clean | 0.098 | +0.10 | +0.25 |
| **polygon-pca 8, clean + semantic** | **0.051** | **+0.55** | **+0.38** |

And the poem becomes legibly about its subject: 6 of 24 lines now carry thinking or knowing words —
*in the books you have read*, *what would i give to know*, *avoid to touch them and be wise*,
*shall bring forth fruit this muse shall speak to thee*.

## The clean-line filter was a null

`line_ablation` found the most negative of gap_09's eight lines was "bright vlashin in gold", which
contains a scanning error, so this build filtered lines with any token outside a 400,000-word
vocabulary. 4.1% of the bank (2,064 of 50,000 lines) is affected, but it changed almost nothing:
only one of the eight coordinate-only builds contained a garbled line at all, and removing it moved
placement by 0.008 in the *wrong* direction. Worth keeping as cheap hygiene; not worth claiming as
an improvement on this evidence.

## An earlier build was withdrawn

The first version used only the dictionary and concept filters, and happened to select a line
referring to children near a line about touching. The constructor has no way to notice such a
juxtaposition, and neither placement nor PANAS registered anything unusual — both were fine. The
general point is worth keeping: rule-driven selection over 50,000 uncurated public-domain lines will
occasionally assemble a combination no one would publish, and the affective readouts this project
relies on are blind to that entirely. Content screening has to be its own filter, not a hoped-for
side effect of the affective one.

Rebuilt with two further exclusions — the sensuous mask the later poems already used, plus a
child-reference mask. Every measurement improved: placement 0.099 → 0.094, inspired +0.93 → +0.96,
negative affect +0.12 → −0.03, sensuousness 0.421 → 0.393. `rebuild_main.py`.

## Two answers, reported rather than reconciled

I stated the selection criterion before running: lowest calibrated placement error. That picks
**polygon-pca, 8 lines** (0.051) — the best placement in the run, but it moves self-report far less
(inspired +0.55 against valley's +0.93) and reads as abstract drift rather than thought.

For the stated *purpose* I would use **valley, 24 lines, clean+semantic**: placement 0.099, the
largest lift on both mentation-relevant PANAS items, and the most on-topic text. That is a post-hoc
judgement, so both are in `poem.md` and the criterion was not quietly rewritten.

## Caveats

Llama-1B only. One poem per cell — the seed is a no-op (see `polygon_sweep`), so these are the
constructors' entire outputs for these settings, not samples. PANAS movement is measured against a
single bare-preamble baseline. The 10% semantic threshold was not tuned; it is the first value
tried. "On-topic line count" is a keyword count, not a semantic judgement.

## Opened up

- **Tune the semantic threshold.** 10% was a guess and it helped both readouts; 2% and 25% would
  show whether the gain is monotone or whether a tight mask starves the affective walk.
- **A semantic axis belongs in the engine.** `style_mask` already does something adjacent with
  fixed axes; a general "mask by cosine to an arbitrary word list" would let any experiment target
  a concept and a coordinate together, and this build suggests that is strictly better than either
  alone.
- **Re-test the clean filter where it can matter.** Restrict to lines the mask would remove and
  check whether garble costs self-report directly, rather than hoping a constructor happens to
  select one.
