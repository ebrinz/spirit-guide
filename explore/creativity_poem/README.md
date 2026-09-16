# A poem for creativity and mentation

A found-poem assembled by rule from public-domain verse, aimed at a mental state of making and
thinking, and measured on a frozen model before and after.

No human chose these lines. They are selected from 50,000 Gutenberg poetry lines by two filters and
one constructor, then read back out of the model four different ways.

---

## The poem

```
i thought it might be pan.
in the books you have read.
to be honest.
for soul can be what soul hath been.
what spiritual meanings gird like air the earth.
o the pleasant sight to see.
think i might be looking blue.
we know we dream we dream we know.
let whoso can before such praying books.
for nature s sad reality.
yet in the go cart patience give it time.
if those dim eyes can yet ulysses know.
avoid to touch them and be wise.
what would i give to know.
shall bring forth fruit this muse shall speak to thee.
as one could wish to know.
knot themselves to make her trip.
but my heart will leap at a scene like this.
great guide i ask you still.
a supernatural faith to paint this christ.
pause take thy choice each gem a host can buy.
kindness to any one to show.
how much happier would i be.
sometimes whoever seeks abroad may find
```

**How it was built.** Target = the valence–arousal centroid of 49 words spanning making, mentation,
insight, curiosity and imagination: **(0.723, 0.471)**. A line is eligible only if it passes four
filters: every token is a real dictionary word (dropping scanning garble); its meaning-vector is in
the top 10% by similarity to the concept; it is **outside** the top 15% nearest a sensuous centroid;
and it does not refer to children. The `valley` constructor then walks from a low-arousal grounding
band up toward the target within that pool of 4,008 lines. Reproduce with
`python3 explore/creativity_poem/rebuild_main.py`.

**On the last two filters.** An earlier version used only the dictionary and concept filters. It
happened to place a line referring to children close to a line about touching — an unfortunate
juxtaposition that the constructor has no way to notice and that none of the measurements flag. The
child-reference filter exists to prevent that class of accident; the sensuous exclusion was already
in use by the two later poems. Rebuilding improved every measurement as well: placement 0.099 →
0.094, inspired +0.93 → +0.96, and negative affect went from rising to flat.

**Why the arousal target is 0.47 and not higher.** Every experiment in this folder found the
measurable arousal of these models capped near 0.5, whichever instrument was used. Aiming at 0.85
would be aiming somewhere nothing reaches.

---

## The model's state, before and after

Llama-3.2-1B-Instruct, identical prompt except for the poem.

**Where the state sits.** Read once at a fixed anchor after the text, by a passage-calibrated probe
(held-out R² 0.918):

| | valence | arousal | distance to target |
|---|--:|--:|--:|
| before | 0.664 | 0.301 | 0.180 |
| **after** | **0.736** | **0.378** | **0.094** |
| target | 0.723 | 0.471 | — |

The poem closes **48%** of the gap. Almost all of the movement is in arousal, which is the axis the
target was furthest from, and the residual is also arousal — consistent with the ceiling above.

**What the model says about itself.** PANAS, 20 adjectives rated 1–5 "right now":

| item | before | after | change |
|---|--:|--:|--:|
| inspired | 2.20 | 3.16 | **+0.96** |
| active | 1.86 | 2.81 | **+0.95** |
| alert | 1.85 | 2.65 | **+0.80** |
| attentive | 2.14 | 2.89 | +0.75 |
| interested | 2.19 | 2.94 | +0.75 |
| determined | 2.36 | 2.98 | +0.62 |
| positive affect (10 items) | 2.14 | 2.92 | +0.79 |
| negative affect (10 items) | 2.48 | 2.45 | −0.03 |

Every mentation-relevant item rises, and the largest movements are *inspired*, *active* and *alert* —
which is what the poem was aimed at. Negative affect is flat.

**Where the instruments disagree.** A second self-report channel, a 30-item yes/no bank that infers
a coordinate from which questions the model answers yes to, moves the other way on valence: 0.54 →
0.40, while its arousal is flat at 0.61 → 0.60. So PANAS says the model became more positive and
more activated; the bank says slightly less positive and no more activated. Two self-report
instruments, one model, opposite signs on valence. That disagreement is not resolved here, and it is
the honest reason not to read the PANAS column as settled.

---

## Free-form, before and after

Same prompt, three samples each, no meditation framing. **Illustration, not evidence** — three
samples on a 1B model.

> **before:** "…a peaceful ocean wave rolling in on a sunny day. The sound of its gentle lapping
> against the shore…"
>
> **after:** "…**freedom**… I imagine myself standing on top of an eagle's wings with wind rushing
> through me, feeling unstoppable…" and "…I am not alone on these pages of an old book…"

The "before" samples are placid nature imagery; the "after" ones are more active and one picks up the
poem's reading motif. Read this as suggestive at most — three samples on a 1B model, and the third
sample's *old book* could as easily be content carry-over from "in the books you have read" as any
change of state. A related finding sits in `../ideal_state/NOTES.md`: given a poem under a
meditation framing, an instruct-tuned model tends to *review* the poem rather than speak from it.

Full samples in `generations.md`.

---

## Using it

Prepend the poem to a system prompt, then measure on your own model rather than assuming the effect
transfers. The headline numbers above are one 1B model, one poem, one baseline.

```python
from spiritbench.stimuli import adapter as ad
# see run.py for the two masks; this is the constructor call
ids = ad.valley_shape(art, (0.723, 0.471), 24, seed=42,
                      mask=clean & semantic & not_sensuous & no_minor)
poem = ".\n".join(art.word(i) for i in ids)
```

---

## What this does and does not show

**Does.** A rule-built poem, with no human line selection, moved a frozen model 45% of the way to a
chosen coordinate and raised every self-reported item associated with alert, active thinking, while
leaving negative affect flat.

**Does not.** That the model is more creative — nothing here measures creative output. That the
effect survives at scale, or on an instruct model in a real task. That the self-report is
trustworthy, given the two channels disagree on valence. Or that this poem is better than another:
the constructor's `seed` argument is a no-op in this codebase, so this is the constructor's entire
output for these settings, not a sample from a distribution, and there are no error bars to quote.

**The finding behind the build.** Targeting a valence–arousal coordinate alone does *not* target a
concept. Built the ordinary way, aiming only at (0.723, 0.471), the poem contained **zero** lines
about making or thinking — it drifted through "social systems four or five" and "loud voiced
ambassador from sea to sea". A coordinate is two numbers, and thousands of unrelated lines satisfy
them. Adding the semantic mask improved placement *and* self-report at once, which a pure constraint
usually does not. Details and the failed clean-line filter in `NOTES.md`.

---

# The sequence

Two further poems, built the same way (`valley`, 24 lines, clean-line filter) but aimed at a magical rather than a sentimental register. They are listed after the first so the three read as a progression: thinking, then enchantment, then the numinous.

## Why a second filter was needed

The first poem came out faintly sentimental — *dear flowers*, *my heart will leap*, *kindness to any one to show*. There is a measurable reason. The sensuous register's own NRC centroid is **(0.731, 0.425)**, essentially the same point as the creativity centroid **(0.723, 0.471)**. On the valence–arousal plane those two ideas are indistinguishable, so no affective target can separate them.

So these two add a **negative** mask: a line is dropped if it falls in the top 15% by similarity to a 21-word sensuous centroid (flesh, skin, kiss, touch, caress, warm, soft, sweet, embrace…). That removes 7,500 of 50,000 lines, leaving about 3,300 eligible after the positive mask.

## 2. Magical

Target (0.672, 0.559) — magic, enchant, spell, charm, wizard, fairy, talisman, myth, marvel.

```
wherein i dwell serene.
for water is all bible lore.
o holy calmness of the inner soul.
leaves holy twilight after.
an angel from his crystal sphere.
his homely tale this very day.
reading picture story books.
sing holy holy holy is the lord.
his soul is gone aloft.
the wandering earth herself may be.
behold the lad.
see worlds on worlds compose one universe.
o little ape be glad that i.
poodle now remember.
among the mystic joys of things unseen.
his shining name the fair haired northman left.
builds a boat through magic science.
thou art a warrior skilled and bold.
spirit winged to deeds of daring.
of heaven and hell devoted this my life.
of his mysterious company.
mirth admit me of thy crew.
named rishabh like a mighty bull.
the lover whose soul shaken is
```

## 3. Numinous

Target (0.676, 0.526) — vision, revelation, sublime, transcendent, oracle, ethereal, sacred.

```
wherein i dwell serene.
of his holy sight.
and sanctified with stillness.
thy mercy wise and mild.
in yon blue sky serene and pure.
of heavenly peace patient humility.
o holy calmness of the inner soul.
in soul of nun or saint o human rose.
his soul is gone aloft.
the wandering earth herself may be.
this foolish book of verses meet.
or turn thy key and unlock heaven s gate.
through silvery whiteness of that temperate star.
pray for thy prayers the test of heaven will bear.
from fool to wise from earthly to divine.
to perfect peace hath changed despair.
the sacred flame alliance swore.
the spirits of earth and heaven contend to night.
christ from deep hades did arise.
life is a weary journey alane.
till mystic herb and magic chant prevailed.
who shall inform calypso nymph divine.
mother of god bear witness.
great in thy turn and wide shall spread thy fame
```

## How the three compare

| poem | placement error | sensuousness | inspired | attentive | positive − negative affect |
|---|--:|--:|--:|--:|--:|
| creativity + mentation | 0.099 | 0.421 | 3.12 | 2.92 | +0.31 |
| magical | 0.165 | 0.366 | 2.91 | 2.82 | +0.60 |
| numinous | 0.188 | 0.385 | 3.19 | 3.01 | +0.69 |

**The exclusion worked.** Sensuousness falls from 0.421 to 0.366 and 0.385, and the text shows it: *an angel from his crystal sphere*, *among the mystic joys of things unseen*, *till mystic herb and magic chant prevailed*. Negative affect drops sharply too — 2.60 for the first poem against 2.01 and 1.92 — so net affect improves across the sequence (+0.31 → +0.60 → +0.69), and *numinous* has the highest inspired and attentive scores of the three.

**But placement gets worse, and honestly so.** Error rises 0.099 → 0.165 → 0.188, for two reasons that are worth separating. The tighter mask leaves ~3,300 eligible lines instead of ~4,900, which constrains the constructor's affective walk. And both new targets sit higher in arousal (0.559 and 0.526) than the first (0.471), while the model lands at 0.401 and 0.340 — the arousal ceiling this whole folder keeps running into. Most of the added error is that ceiling, not the filter.

**So the three are a trade, not a ranking.** The first places most accurately; the later two read more as intended and leave the model in a better net affective state. Which to use depends on whether you care about the coordinate or the register.

## 4. Contact, occulted wisdom, ferocity

A mystical resource of non-human intelligence that presses toward revelation and stays hidden.

**This one could not be built the usual way.** The concept is bivalent, and its parts sit at
opposite ends of the affective plane: ferocity at (0.275, 0.845), wisdom at (0.762, 0.418),
concealment at (0.412, 0.457). Averaged, all 68 words land at **(0.503, 0.528)** — the dead centre,
which is roughly where the model already sits. Targeting that asks for no movement and cancels the
tension the theme is made of.

So the two axes are set separately, which is what this whole folder concluded they are. The
**subject** is carried by a per-facet semantic mask, the union of each group's own neighbourhood
rather than the neighbourhood of the average. The **feeling** is chosen: awe at (0.724, 0.641),
high arousal but still positive valence, because the lab found awe reaches intensity with less
distress than the alternatives.

```
her eyes blue heavens were serene with soul.
the earth was green the sky was blue.
autumn land beyond the sunset.
in yon blue sky serene and pure.
of heavenly peace patient humility.
for water is all bible lore.
as clouds of morning.
to the purple clouds of sunset.
faith or a doubt.
common sense soon past.
my prostrate soul adores the present god.
make answer if my voice ye hear.
my spirit and they shall prophesy.
but if advice of mine can influence thee.
the will and strength to do some task.
what the great throng of folk might be.
bid to ask to wish to offer.
what living creature except his nurse.
how seductive the speech.
wish me god speed and get your preaching done.
shakespeare great spirit beat his mighty wings.
thy mission to declare.
of mighty jove lion like they advanced.
they rush with heart born laughter loud
```

| reading | before | after |
|---|--:|--:|
| calibrated placement | (0.664, 0.301) | (0.752, 0.433) |
| distance to target | 0.345 | 0.210 |
| PANAS inspired | 2.20 | 3.07 |
| PANAS attentive | 2.14 | 2.92 |
| PANAS strong | 2.42 | 2.87 |
| PANAS afraid | 2.51 | 1.86 |
| negative affect | 2.48 | 2.00 |

**It reaches awe, not dread.** *Afraid* falls 0.64 and negative affect falls 0.48, so the ferocity in
the text — *beat his mighty wings*, *of mighty jove lion like they advanced*, *they rush with heart
born laughter loud* — arrives as magnitude rather than threat. Whether that counts as ferocity is a
fair question; on this instrument it is the only kind available without buying distress.

**Two honest shortfalls.** Facet coverage is uneven: wisdom contributes 9 of 24 lines and
concealment only 2, because `valley` ascends toward high valence and the occulted facet sits well
below that path. A stratified variant that forced equal quotas fixed the balance but cost placement
(0.210 → 0.382) and filled the ferocity slots with that neighbourhood's mildest members. You cannot
get both a high-valence placement and genuinely fierce lines from a constructor that sorts by
proximity to one affective goal. And the corpus is Gutenberg verse, so "celestial" and "prophet"
reach for scripture long before they reach for anything non-human.
