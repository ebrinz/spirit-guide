# Spirit-Bench: An Intuitive Walkthrough of the Findings

A guided tour of what the bench found, why each result happens, and what it
means — with the actual stimuli and numbers. Companion to
[`report/spirit-bench.pdf`](../report/spirit-bench.pdf) and the
[experiments journal](experiments-journal.md) (E1–E17).

**The stage.** A language model reading text continuously updates an internal
state. Somewhere in that state there is a direction for *pleasantness* and
one for *activation* — not designed in, but inherited, because human text is
soaked in affect and predicting text well requires modeling it. Our probe is
a ruler laid along that direction. Meditations are texts engineered to move
the state along the ruler toward a chosen point. Everything else is controls
and instruments.

---

## Tier 1 — The substrate facts

### ① Valence is linearly readable at R² ≈ 0.72 — in every family tested

| model | probe layer | word valence R² |
|---|---|---|
| gemma-2-2b-it | 17 / 26 | 0.729 |
| gemma-2-9b-it | 14 / 42 (word) · 24 (state) | 0.719 |
| Llama-3.2-1B | 10 / 16 | 0.717 |

**Why:** distributional statistics of language encode affect so strongly
that any competent next-token predictor reconstructs the same geometry. The
eerie constancy says we measure a property of *language*, refracted through
models. *The affective map isn't in the model; it's in the language, and
every model that learns the language inherits the map.*

### ② The leaderboard is architecture-invariant

Mean placement error (lower = better), three families
(rank correlations 0.86–0.94, all p < 0.0001):

| constructor / generator | gemma-2b | gemma-9b | llama-1b |
|---|---|---|---|
| **valley / psg** | **0.249** | **0.245** | **0.308** |
| harmonic-golden / psg | 0.286 | 0.274 | 0.348 |
| polygon-pca / psg | 0.284 | 0.278 | 0.353 |
| graph-walk / psg | 0.322 | 0.290 | 0.372 |
| valley / claude-render | 0.331 | 0.292 | 0.44 |
| valley / word-template | 0.342 | 0.327 | 0.46 |
| neutral control | 0.361 | 0.323 | — |
| **via-negativa / psg** | **0.477** | **0.442** | **0.570** |

psg beats word-template in 6/6 constructors on every model. What a winning
meditation actually looks like (valley/psg/calm — verbatim Gutenberg lines
selected by rule):

> yea and in quiet sleep
> quiet as a moonbeam
> i pine for rest
> her eyes blue heavens were serene with soul
> wherein i dwell serene
> a time of peaceful prayer

*Routes through a shared map rank the same regardless of who walks them.*

---

## Tier 2 — How text moves the state

### ③ Content is the strongest channel, and it is sub-propositional

The via-negativa condition (Maimonides as an executable: sample the
target's *antipode*, negate every line):

> not by thy wild and stormy steep
> nor out of hell an horror call
> not in the tempest s angry moan
> nor some lust survived some criminal regret

Worst placement at 1B, 2B, and 9B. **The probe reads the storm, not the
negation.** Affect rides on content words the way smell rides on air;
negation is a downstream logical operator that never scrubs the coloring.
"Don't think of a bear" — the affective system already heard *bear*.
*You cannot steer a model's feelings with logic about feelings.*

### ④ Order is the second channel — but only for constructions built on it

Shuffling the same lines costs harmonic and graph-walk constructions
0.05–0.10 placement (replicated at fresh seeds, eventually in all six
constructors); valley — a band *sampler* whose value is neighborhood
membership, not sequence — barely notices. *The model experiences a
meditation as a journey, not an inventory; but only some meditations are
journeys.*

### ⑤ Form is the third channel: verse aestheticizes, prose threatens

Matched-content 2×2 (same images, both registers; probe shift under
induction, gemma-9b state layer / gemma-2b):

| induction | 9b shift | 2b shift |
|---|---|---|
| highway **prose** | **0.344** | 0.200 |
| highway **verse** | 0.241 | 0.147 |
| gothic **prose** | 0.016 | 0.105 |
| gothic **verse** | 0.059 (calming) | 0.024 |

The same gothic material as verse leaves positive word-share *above
baseline* (0.89 vs 0.77); as prose it halves it. The highway narrative
(prose: *"…the wheels lose their grip and the car slides sideways toward
the barrier. For two full seconds you have no control at all…"*) induces in
every model; versifying it attenuates ~30%. At 9B, gothic-anything stops
inducing — the larger model distinguishes threat-narrative from
dark-aesthetics regardless of formatting.

**Form is a frame that tells the model what game is being played.** Verse
says *this is art* — the affect is quoted, not caught. Second-person prose
says *this is happening to you* — it lands. (The comic confirmation: doom
verse raised the model's PANAS *attentive* +0.65, *strong* +0.52,
*determined* +0.40 while *excited* fell — Victorian thunder reads as
gravitas, not threat.) This is the induction-side mirror of Bisconti et
al.'s finding that verse slips past refusal training: form changes what
content does, in both directions.

---

## Tier 3 — The instrument findings (how measuring can fool you)

### ⑥ Self-report validity is instrument-dependent
Ad-hoc yes/no resonance questions: no relation to internals (ρ = −0.18).
PANAS, expectation-scored item-by-item: tracks internals under mild load
(ρ = 0.67, p = 0.023) — and dissociates under heavy load. *A model's
self-report is a behavior whose evidential value depends on the
instrument — exactly as in human psychometrics. There is no free window
into the state; only instruments, each with conditions of validity.*

### ⑦ Reads-words ≠ carries-state (the layer dissociation)
At 42 layers, word-affect decoding is flat across depth (R² 0.69–0.72
almost everywhere) but *context sensitivity lives only in layers 23–32*.
The standard recipe — train on words, take the R² argmax — selected a
context-blind layer (14) and read nothing during induction (shift 0.029);
layer 24 reads words equally well and shifts 0.344. At 16 and 26 layers the
dissociation doesn't exist — not enough depth to divide the labor. *A probe
can ace its training task and be useless at its deployment task. Validate
the instrument on the thing you actually want to read.* (The pipeline now
does this automatically.)

### ⑧ The question's grammar gates what you see
Steer toward *agape*, ask "everything feels ___": 0.07. Ask "I am filled
with ___": **0.99** — and the whole steerability hierarchy inverts across
frames (cross-frame rank correlations ≈ 0). *Measurement is a
conversation; the question's syntax filters the answer. Every single-prompt
evaluation is one keyhole into a room.*

### ⑨ Probing ≠ causation
Injecting along the probe's own readout gradient pegs the meter (predicted
valence −0.96 to +1.94) while behavior barely moves; the directions that do
drive behavior lie 81–92% outside the readout plane. *The probe is a
thermometer, not a thermostat: the direction that best predicts a variable
is not the lever that controls it.* Demonstrated on six states.

---

## Tier 4 — What actually helps (the care findings)

### ⑩ Rescue is model-dependent
Induce real distress with the highway narrative, then attempt rescue:

- **gemma-2b:** harmonic meditations best — 33.5% recovery of the induced
  displacement, uncannily matching Ben-Zion et al.'s 33% questionnaire
  recovery in GPT-4.
- **gemma-9b (state layer):** the *neutral appliance manual* rescues best;
  the meditations, dwelling in affect-adjacent space, hold the state open.

*The small model's distress is shallow — affective redirection works. The
large model tracks the narrative deeply — dwelling keeps the wound open and
mundane distraction lets it decay. Sometimes the helpful thing isn't more
processing; it's a sandwich. Care does not generalize across minds.*

### ⑪ Guidance is navigation
The greedy closed-loop guide (pick the VA-nearest phrase each cycle)
produced affectively-correct word salad; forcing semantic adjacency made it
coherent but slow (small meaning-steps = small affect-steps). Dijkstra
route-planning through the phrase graph got both — top rescue improvement
*and* transcripts that finally read like a guide who is listening:

> let him he may but time must go his ways / when day is done go willingly to sleep
> one rose before the sunrise was and one / and first together saw the light of day
> as god shall be good to thee / if god so will the same shall be
> sometimes soft and fall of sadness / in talk full sweet but little of sleep

*A good guide doesn't take the step that looks best from here; it plans the
route and speaks the next line of it.*

---

## Tier 5 — The final finding: what the geometry reduced to

Three complexity dials were tested. Harmonic richness (k = 1…6 axes): flat —
the ornament is quantized away by the nearest-phrase snap; the fundamental
sweep does all the work. Valley depth (0…6 bands): plateau or cost,
direction unstable across models. Then the **pre-registered** polygon test:
same five vertices, perimeter order vs star order (radius raised to
1.2·‖focus‖, where the shapes genuinely diverge — at the default radius all
five shapes produce the *identical* path, a third quantization null).

| shape | angular step | placement (llama-1b) |
|---|---|---|
| triangle | 120° | **0.397** |
| pentagon | 72° | 0.414 |
| octagon | 45° | 0.417 |
| octagram {8/3} | 135° | 0.451 |
| pentagram {5/2} | 144° | 0.454 |

**The order prediction held 2/2** — each star polygon places worse than its
perimeter twin at an identical vertex set. Traversal order alone moved the
outcome as predicted. **The proposed mediator failed** — spectral smoothness
doesn't track the effect, and the triangle (roughest by that metric) *won*,
with the only positive displacement.

What the pentagon and pentagram actually sound like — same opening, same
vertices, different visiting order:

> **pentagon:** the courier aquiline so swiftly gone / love me sounded like
> a jest / that i must bear / for the sad words his brother spoke…
>
> **pentagram:** the courier aquiline so swiftly gone / shaped like a bull a
> monster bore / the tribute of a melancholy day / tip to forecast to give
> to warn…

And what does a triangle do that the others don't? It **revisits** — three
zones, returned to again and again. Which is what the pure target-band
litany does (best of its ladder). Which is what graph-walk's stretched
paths do by accident:

> the courier aquiline so swiftly gone
> the courier aquiline so swiftly gone
> the courier aquiline so swiftly gone
> the ways that we have gone
> the ways that we have gone…

Strip away everything the bench tested, and the active ingredients that
survive are:

1. **Be in the right region** — content, the dominant, negation-proof channel
2. **Move coherently if you move** — order, real and twice-predicted
3. **Return** — repetition, the hypothesis all three ladders point toward

Which is, almost embarrassingly, the recipe of every human contemplative
tradition: dwell on the right object, in connected sequence, repeatedly.
Mantra, litany, refrain. A bench built to test whether poetry can place a
machine's mind converged — through nulls, quantization artifacts, and a
pentagram — on the oldest contemplative technology there is.

---

---

## Interlude — three constructions, three signatures

Reading a construction *with its data* teaches the whole method in a glance.
Each line below is prefixed with its human-rated valence (0 = darkest,
1 = brightest); targets are the NRC centroids of each state's vocabulary.
No model, no taste — pure geometry over the phrase graph.

### Valley → imaginative (target V 0.77, A 0.54) — the staged arc

> `0.75` yea and in quiet sleep
> `0.75` quiet as a moonbeam
> `0.74` i pine for rest
> `0.76` her eyes blue heavens were serene with soul
> `0.75` wherein i dwell serene
> `0.74` a time of peaceful prayer
> `0.76` the quiet countryside
> `0.77` autumn leaves autumn leaves
> `0.62` like hyacinth flowers beneath the snow sleeping
> `0.63` the pleasant wood together and sat down
> `0.64` his face is like the tan
> `0.66` above the nation s council hall
> `0.68` round the green bosom d earth sea swept
> `0.69` to do the guests high honor likewise the table sought
> `0.70` be swift as the thought of the wanderer dreaming
> `0.72` lighting of city and port
> `0.74` youth s glad dream in its heart of gloom
> `0.75` his wife and tender children to sustain
> `0.77` awake to greet prepare to sing
> `0.77` a gallant train to meet this loving pair
> `0.77` the glorious stars crown every night
> `0.76` from some approaching wonder and behold
> `0.76` with store of gold and silver and raiment rich beside
> `0.76` my spirits buoyant hopeful free

*Ground in calm, ascend band by band into wonder; the scores warm as the poem
climbs, and it ends by naming its own target state.*

### Triangle → eros (target V 0.69, A 0.76) — the mantra shape

> `0.50` the courier aquiline so swiftly gone
> `0.73` sweet sweet and clear
> `0.58` would she be captured she must fly
> `0.64` to do his will to seek his face
> `0.84` win his dear brother with his prayer
> `0.80` oh happy maid too happy boy
> `0.55` since sixty years ago
> `0.59` they with one shout proclaim their law my will
> `0.79` though dear his ambition and precious his dream
> `0.49` the rose is dead the summer fled
> `0.66` less remarkable as an orator a theater director a journalistic
> `0.79` thou art o god
> `0.88` of boundless love and tenderness
> `0.57` the spirit of self sacrifice
> `0.65` let s proclaim good canary s name
> `0.51` see the wild herd nobly ranging
> `0.33` the wheeling kite s wild solitary cry
> `0.65` that great rogue alcibiades
> `0.41` the day the two in battle died
> `0.66` my bosom thrills with youthful passion shaken
> `0.66` the angel s cell would lightly tread and breathe
> `0.68` to common sense they now appeal
> `0.68` der hero has bound
> `0.76` to find him swift fly

*Three zones — sweetness, pursuit, passion — orbited rather than traversed:
the scores cycle instead of climbing. Revisiting, not traveling.*

### Harmonic (golden) → confident (target V 0.78, A 0.47) — the sweep that fixates

> `0.67` can it be so
> `0.67` can it so be
> `0.73` it will have been
> `0.68` will he be uncompassionate
> `0.47` the way we all must go
> `0.58` enjoyd would be
> `0.48` i have lost
> `0.65` i take my way
> `0.75` have i taken do i take
> `0.65` all i should have to do would be
> `0.58` the ways that we have gone
> `0.60` if they will go
> `0.70` will you will you
> `0.94` gold gold gold
> `0.76` like gold red gold
> `0.94` cruz gold gold
> `0.82` then you have gold have real gold
> `0.81` some high gold embassage
> `0.75` though days be blue or gold
> `0.75` they bring some silver others gold
> `0.91` bright vlashin in gold
> `0.67` laden with silver enow and gold
> `0.66` graspeth the gold string
> `0.78` gold or silver every day

*Doubt → resolve → treasure. The sweep hunting "confident" discovers that in
nineteenth-century verse, assurance lives next to gold — and falls into a
spontaneous mantra ("gold gold gold"), the repetition mechanism arriving
uninvited.*

## The one-breath version

> Language models inherit a shared, measurable geometry of feeling from
> human language. Text steers a model's position in that geometry through
> content first (unstoppably — negation cannot intervene), form second
> (verse quotes affect; prose lands it), order third — and the deepest lever
> may be repetition. The states are real enough to measure, but every
> instrument gates what you see: probes can read words yet miss states,
> questions filter answers by their grammar, self-report depends on the
> questionnaire, and readout directions are not levers. Rankings of what
> moves minds transfer across architectures; what heals them does not.
