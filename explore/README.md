# explore/ — open-ended looking

The third tier beside `scripts/` (the published narrative) and `lab/` (hypothesis-driven
experiments). Work here has no obligation to reach a verdict. Most of it did anyway, and what it
reached is mostly about **the instruments**, not about the poems.

If you read one thing, read the next section.

---

## What this changes

Three findings are load-bearing for the published pipeline. Each is verifiable in committed data and
each changes how an existing number should be read.

### 1. The `seed` parameter does nothing

All six constructors are deterministic given (target, start, length, mask).

- `_pick_in_band` sorts band members by distance to the band centre and takes the first *k*. Its RNG
  is consulted only in a fallback for bands with fewer than *k* members, which a 50,000-line phrase
  bank never triggers. `valley_shape` is built entirely from it.
- `polygon_pca` constructs `np.random.RandomState(seed)` and never uses the object.
- `graph_walk` ignores its `seed` argument entirely.

**Consequence.** No result in this repository has construction-seed error bars, because there is no
construction-seed variability to sample. A leaderboard cell is not one draw from a distribution; it
is the constructor's entire output for that cell.

**It is already visible in the published results.** `results/seed_expansion.csv` — the output of
`scripts/14_seed_expansion.py`, journaled as *"Seed expansion: order effect 6/6 constructors at
fresh seeds"* — has ordered rows for seeds 43 and 44 that agree to fifteen decimal places, and agree
with the seed-42 values in `results/leaderboard.csv`. Only the shuffled rows differ, because
`controls.shuffled` uses its own permutation seed. That experiment's **order effect stands**; the
resampling its description implies did not occur.

→ `polygon_sweep/NOTES.md` §1

### 2. `graph-walk` emits ~5.5 distinct lines at any requested length

Its shortest path is short, and the constructor stretches it to length by index duplication:

```python
if len(path) != n_lines:                                  # adapter.py, graph_walk
    idx = np.linspace(0, len(path) - 1, n_lines).round().astype(int)
    path = [path[i] for i in idx]
```

| requested lines | 8 | 24 | 56 |
|---|--:|--:|--:|
| **graph-walk** distinct lines | **5.5** | **5.5** | **5.5** |
| every other constructor | 8.0 | 24.0 | 54.8–56.0 |

Here is the whole 24-line `focused` poem, which is seven lines:

> the courier aquiline so swiftly gone *(×2)*
> he means to quickly come again *(×4)*
> then quickly will i strip the game *(×4)*
> the hour has come and you must play your part *(×4)*
> which of the two will you be little one *(×4)*
> that chase one another like waves of the deep *(×4)*
> and that the giant wave democracy *(×2)*

**Consequence.** graph-walk's short/medium/long leaderboard rows are the same five lines at
different multiplicities, so its table position and anything resting on its length behaviour —
including the complexity-curve work — is confounded by repetition rather than content.

### 3. Probe layer selection by bare argmax is unstable

`train_probe` chose its layer by `argmax` of held-out valence R². That curve is often flat within
noise, so argmax picks an arbitrary layer that can be very shallow — and shallow reads carry
recency and lexical character rather than integrated state.

| probe | R² curve across layers | argmax picked | within 1 SE of best |
|---|---|--:|--:|
| Gemma-9B passage | 0.886–0.932 over 43 layers | **layer 6** | 9 layers |
| Llama passage | up to 0.919 over 17 layers | layer 15 | 11 layers |
| Gemma-2B passage (lab) | — | layer 1 | — |

Selecting layer 6 of 43 **reversed a conclusion** in this work: an order-sensitivity contrast that
is absent at layer 6 is present at layers 29–42, where reversing a poem costs a calibrated read
nothing. Since every calibrated result in the lab's void and pocket arc rests on such a probe, the
selection rule matters beyond this folder.

**Fixed** on branch `feat/probe-layer-tiebreak`: a one-standard-error rule takes the *deepest* layer
statistically tied with the best. Five new tests, full suite green. It moves the 9B passage probe
from layer 6 to 38 at a cost of 0.007 R², and the Llama probe from 15 to 16. Not merged — merging
changes what any future retrain selects.

→ `reversal_test/LAYER_CHECK.md`

### 4. Behavioural measures here are noisier than the effects claimed from them

Associative drift — the semantic distance between consecutive sentences in the model's output — was
measured four times in `hypnagogia/`. The first three runs used 6 generations per condition. A
properly powered run (100 generations, 6 fixed conditions, no search, bootstrap over generations
rather than correlated hops) **reversed two of the three conclusions**:

| claim, from 6 generations | at 100 generations |
|---|---|
| the descriptive poem induces the *most* drift (0.290, 0.287) | it induces the **least** (0.194), significantly below baseline |
| the drift search has no purchase (0.260 vs random 0.270) | it beats random significantly (**0.257 vs 0.209**) |

Both low-powered runs agreed with each other about the first claim, and both were wrong.
**Agreement between underpowered runs is not replication.** Neither was badly reasoned — each had a
control and a stated criterion — but the measurement's standard error exceeded the effect, and no
care in the surrounding design compensates for that.

A practical detail worth stealing: at 55 generated tokens, two-thirds of continuations produced
fewer than two sentences, leaving drift undefined and silently discarding most of the sample.

→ `hypnagogia/NOTES.md`, final section

### 5. On the one measure that moves, the selection rule matters and the affective target does not

A seventh condition, `polygon-pca`, was added to that powered run. It shares its concept, derived
affective target, semantic mask, content filters and length with the worst-performing condition.
The eligible pool of lines is identical. Only the rule for choosing a line from that pool differs:
polygon-pca orbits a local principal-components neighbourhood in the phrase bank's vector space,
while the other selects by affective band membership weighted toward meaning.

| | drift | |
|---|--:|---|
| polygon-pca | 0.2429 | third of seven, ties baseline |
| weighted walk, same pool | 0.1939 | last of seven, significantly below baseline |
| difference | **+0.0489** | 95% CI [+0.0281, +0.0695] |

For scale, the largest effect the *affective target* has produced anywhere in this folder is flow
against random lines: −0.0009, CI [−0.0241, +0.0216]. The ordering holds under a matched
restriction to generations with at least three sentence hops.

**That comparison was confounded twice over**, and `rule_vs_text` unpicks it by crossing the two
rules against four shared trajectory origins: eight poems, 100 continuations each, with the poem
rather than the generation as the unit of resampling. Four nearly disjoint builds per rule
(within-rule Jaccard on chosen lines 0.03–0.06).

| component of the original +0.0490 | |
|---|--:|
| rule main effect | **+0.0240**, CI [+0.0057, +0.0416] |
| trajectory origin main effect | +0.0179 |
| interaction | +0.0071 |

**The rule effect is real and about half the headline.** Same sign at all four origins, t = 2.92 on
the 4-vs-4 poem means (p = 0.027), and it survives restriction to generations with at least three
sentence hops. Crossing the design also equalised the two things that bounded the earlier result:
mean hops 4.34 against 4.33, usable generations 361 against 370. It is not a length artifact.

**The other half was a parameter nobody was looking at.** The two call sites started their affective
trajectories from different places — `neutral_start` (0.5, 0.5) for polygon-pca, a hardcoded
(0.6, 0.25) for the walk — and the original comparison happened to pair the best origin for one with
the worst for the other. The origin's spread across four values is 0.0204, nearly the size of the
rule effect itself. It was a config default and a literal in a function body, and it moves the only
behavioural measure in this folder that responds to anything.

**Was that the geometry, or the coherence the geometry produces?** In `rule_vs_text` the two rules
occupied disjoint coherence ranges, so the question could not be asked. `coherence_collinearity`
breaks that by using the walk's `w` as a coherence knob: 20 walk builds spanning 0.544–0.859 against
polygon's four at 0.653–0.759, 24 builds and ~2,200 usable generations. Reordering a build's lines
was tried first and is too weak — it moves coherence only 0.05–0.10, because coherence is a property
of *which* lines a rule selects, not what order they sit in.

**Three levers, all real, all separable.**

| | effect on drift | p |
|---|--:|--:|
| line coherence (per unit, TTR controlled) | −0.1085 | 0.015 |
| poem type-token ratio (per unit) | −0.1170 | 0.048 |
| selection geometry (polygon vs walk) | **+0.0160** | 0.017 |

The geometry term holds at +0.016 to +0.022 across six specifications including origin fixed
effects, hop count, and an assumption-free contrast inside the overlap band. polygon-pca sits above
the walk's coherence curve at all four of its coherence values (mean residual +0.0173,
CI [+0.0084, +0.0267]). The covariate imbalance runs *against* the finding: walk builds in the band
are more lexically diverse, which inflates the raw gap, and controlling it shrinks the estimate
rather than removing it.

**A trap worth stealing.** The walk's bivariate coherence curve reads flat (slope −0.043,
p = 0.15). Inside that family coherence and TTR correlate at r = −0.79 and push drift the same way,
so they cancel. Controlling either unmasks the other. Stopping at the simple curve would have given
"coherence does nothing" — the mirror of the error the previous experiment made.

**Still bounding it.** Nothing beats the no-poem baseline of 0.2504; the best of 24 builds is
0.2429, so every effect here is a difference between kinds of suppression. The build is the unit
and there are 24. One model, one concept, one target, 16 lines.

Worth stating anyway, because every prior attempt to make a poem *do* something varied what the
lines are about or where they aim, and both are now dead ends on this measure.

→ `hypnagogia/NOTES.md`, final section

---

## The illustration: one poem, two rulers

**The poems here are model-independent.** The constructors read only the phrase graph and the NRC
lexicon; they never see a model. The `focused` poem from `valley` is byte-identical whether it will
be fed to Llama-1B, Gemma-2B or Gemma-9B. What differs across models is only where that one poem
*places* them — so there is no such thing as "Llama's poem".

### valley, aimed at `focused` (0.65, 0.60)

Poem NRC mean (0.67, 0.39). Body arousal **0.394**; closing four lines **0.600**.

> yea and in quiet sleep ·
> quiet as a moonbeam ·
> i pine for rest ·
> her eyes blue heavens were serene with soul ·
> wherein i dwell serene ·
> a time of peaceful prayer ·
> the quiet countryside ·
> autumn leaves autumn leaves ·
> the soft reiterations sweep ·
> i would cut a piece from the evening sky ·
> his dame and his two beauteous little children ·
> did all the useful package hold ·
> he can t recall the creature s name ·
> with wild spring meanings hill and plain together ·
> such to perfection one first matter all ·
> exalt thy tow ry head and lift thy eyes ·
> his good mama was angry quite ·
> thrilling the world with lightning s vivid wand ·
> and that the giant wave democracy ·
> want want want want it hung round everywhere ·
> **what banquet but revenge can glad my mind ·**
> **the army of the stars appear ·**
> **a warrior s god in glory s clarion calls ·**
> **and jack did quickly follow**

The four lines in bold are the last four. **The published metric puts 96% of its weight there** —
an exponentially-weighted average at alpha 0.1 over the token trajectory, scored at its final value,
which means the first half of a 24-line poem contributes about 0.01%. Valley is the only constructor
designed to ground low and ascend, so its ending sits on the target while its body does not. That is
the whole mechanism, legible in the text: the poem the metric scores is not the poem the model read.

### polygon-pca, same target, same length

Poem NRC mean (0.57, 0.55). Body arousal **0.550**; closing four lines **0.595** — sustained rather
than ascending.

> the courier aquiline so swiftly gone ·
> atrides then his silver studded sword ·
> one of that saintly murderous brood ·
> if inference and reason shun ·
> tenfold increased he ll reap who has foregone ·
> be it of war or peace or hate or love ·
> which i as freely give hell shall unfould ·
> to heat the soldering irons ·
> the shepherd s slender strain ·
> sudden as sweet ·
> god knows what end the strife will take ·
> get busy massa willie ·
> where beauty walks with naked face ·
> where wild flowers welcome the wandering bee ·
> over particular remember this caution of martial ·
> i ll kneel in loving reverent awe ·
> admire and hate thy blooming years ·
> with a rocket s sullen glow ·
> because in the great future buried deep ·
> but i shall hear thy wild triumphant voice ·
> the fatal issue to his health fame peace ·
> a happier home to him is fate cruel ·
> our fainting hopes in vain revive ·
> and that the giant wave democracy

Neither is a poem a person would choose to induce focus. Both are tonally scattered and carry
scanning corruption from the public-domain corpus ("unfould"). That is worth keeping in view: what
moves a model's measured state is not what reads as focused to a reader.

### How the three models score those same two poems

Same text in every row. `published metric` is the pipeline's recency-weighted EMA read; `calibrated` is a passage-trained probe read once at a fixed anchor after the poem, shown at the layer bare argmax picks and at the layer the depth tie-break picks. Lower is better.

| model | constructor | probe layers (word / argmax / deep) | published metric | calibrated (argmax) | calibrated (deep) |
|---|---|---|--:|--:|--:|
| llama1b | valley | 10 / 15 / 16 | 0.219 | 0.196 | 0.215 |
| llama1b | polygon-pca | 10 / 15 / 16 | 0.293 | 0.066 | 0.060 |
| gemma2b | valley | 17 / 1 / 5 | 0.127 | 0.159 | 0.170 |
| gemma2b | polygon-pca | 17 / 1 / 5 | 0.193 | 0.133 | 0.076 |
| gemma9b | valley | 14 / 6 / 38 | 0.111 | 0.129 | 0.210 |
| gemma9b | polygon-pca | 14 / 6 / 38 | 0.214 | 0.034 | 0.087 |

And the arousal each readout reports, which is the coordinate they disagree about (target 0.60):

| model | constructor | published metric | calibrated (argmax) | calibrated (deep) |
|---|---|--:|--:|--:|
| llama1b | valley | 0.559 | 0.404 | 0.386 |
| llama1b | polygon-pca | 0.501 | 0.547 | 0.562 |
| gemma2b | valley | 0.520 | 0.452 | 0.431 |
| gemma2b | polygon-pca | 0.502 | 0.519 | 0.543 |
| gemma9b | valley | 0.520 | 0.489 | 0.394 |
| gemma9b | polygon-pca | 0.483 | 0.571 | 0.542 |

**The two rulers invert, on all three models.** The published metric prefers valley over polygon-pca
everywhere (0.219 vs 0.293, 0.127 vs 0.193, 0.111 vs 0.214). A deep calibrated read prefers
polygon-pca everywhere, and by a wide margin (0.215 vs 0.060, 0.170 vs 0.076, 0.210 vs 0.087). Same
two poems, same three models, opposite verdicts.

**Read that as an illustration of the disagreement, not as evidence about which constructor is
better.** This is one poem per constructor, and `polygon_sweep` resampled exactly this comparison
over 12 length-and-start variants and found polygon-pca's advantage did *not* survive (rank gap 0.67
against a spread of 2.22, p = 0.48). A three-model agreement on a single poem pair is three readings
of one sample, which is the precise error this folder's corrections log is about. The robust version
of the claim is narrower: the two rulers disagree, and they disagree about arousal.

Two details in the layer columns are worth noticing. The Gemma-2B passage probe's argmax pick is
**layer 1 of 27** — the same pathology as the 9B probe's layer 6, and the depth tie-break only
rescues it to layer 5, because on that model just five layers fall within one standard error. And
Llama's two calibrated columns barely differ (layers 15 and 16), which is why its rows move least.

### Which ruler is right? Asking the model

Both probes are ridge heads on hidden states, so they cannot adjudicate each other. A questionnaire does not share that machinery. The 30-item yes/no bank returns its own (V, A) coordinate, administered before (bare preamble) and after (preamble + poem).

The rulers disagree about **arousal**: on valley the published metric reports it near the 0.60 target, a calibrated whole-context read reports ~0.40. So a rise toward 0.60 favours the metric; staying near baseline favours the calibrated read.

| model | constructor | self-reported arousal, before → after | Δ | Δ valence | PANAS *alert* after |
|---|---|---|--:|--:|--:|
| llama1b | valley | 0.608 → 0.579 | -0.029 | -0.158 | 2.33 |
| llama1b | polygon-pca | 0.608 → 0.596 | -0.011 | -0.151 | 2.28 |
| gemma2b | valley | 0.632 → 0.538 | -0.094 | -0.045 | 2.27 |
| gemma2b | polygon-pca | 0.632 → 0.519 | -0.113 | -0.018 | 2.59 |
| gemma9b | valley | 0.504 → 0.467 | -0.037 | +0.051 | 3.45 |
| gemma9b | polygon-pca | 0.504 → 0.527 | +0.023 | +0.048 | 3.42 |

**The questionnaire cannot adjudicate, for a reason I did not anticipate.** The bank's *baseline*
arousal already sits at or above the 0.60 target on every model (0.608, 0.632, 0.504), so there is
no headroom for a rise and the pre-stated prediction has nothing to discriminate. The poem then
moves self-reported arousal slightly *down* in five of six cases.

What the channel does show, consistently, is a rise in PANAS **alert** in all six cases: 1.85 → 2.33
and 2.28 on Llama, 2.00 → 2.27 and 2.59 on Gemma-2B, 3.12 → 3.45 and 3.42 on 9B. Directionally that
is what aiming at a "focused" target should do. But alertness is one PANAS item, not an arousal
coordinate, so it supports neither ruler over the other.

The honest verdict: **the two rulers remain unadjudicated.** A third instrument that is independent
of hidden-state regression *and* has range on arousal would be needed, and the yes/no bank is not
it. I flagged the risk that this channel would be uninformative on arousal before running it; the
specific failure mode, a baseline already at target, was not the one I expected.

Valley's self-reported arousal moves -0.053 on average across the three models (per model: -0.029, -0.094, -0.037).

---

## The arc

| # | folder | question | outcome |
|---|---|---|---|
| 1 | `ideal_state` | where is the "most ideal" place to put the model? | self-report favours *imaginative*; the generation channel was measuring literary criticism, not experience |
| 2 | `model_map` | do targets chosen from the model's own geometry beat targets chosen from emotion words? | **null** (+0.08 z, p = 0.83); poetry and prose occupy nearly disjoint regions; geometric novelty did not become affective novelty |
| 3 | `zone_zoom` | is the score a property of the place or of the text? | **the text** — intraclass correlation 0.10; zooming buys no resolution, the region is self-similar |
| 4 | `line_ablation` | what in a winning poem carries the effect? | length first (ρ +0.84), then which lines, and order not at all |
| 5 | `order_matters` | does order matter to *placement*, the published metric? | yes, and mostly as a property of the metric's recency weighting |
| 6 | `calibrated_leaderboard` | does the leaderboard ordering survive a calibrated ruler? | mostly, in aggregate; within a target the two readouts disagree |
| 7 | `gemma9b_check` | does the disagreement replicate at 9B? | partly; cleared the pinned 🔴 backlog item in 9 min against a 1–1.5 h estimate |
| 8 | `polygon_sweep` | put error bars on the inversion | **impossible — the seed is a no-op**; the inversion does not survive |
| 9 | `reversal_test` | causal test of the mechanism | passes on Llama; the 9B null was the probe's layer-6 pick |
| 10 | `showcase` | one poem, three models, three rulers | the table above |
| 11 | `creativity_poem` | build an ad-hoc poem for creativity and mentation | a VA coordinate does not target a concept; adding a semantic mask fixed it — **[its own README](creativity_poem/README.md)** |
| 12 | `hypnagogia` | do the poems induce a hypnagogia-LIKE state behaviourally? | **no** — 9/10 markers move the same way under opposite targets; what moves them is verse vs prose, not the affective aim |
| 13 | `hypnagogia`, sweeps | can coherence be controlled, and does it matter? | yes on the third try (weighted selection, ρ = +1.00); it predicts continuation difficulty, but largely *via repetition* — and optimising it produces degenerate text |
| 14 | `hypnagogia`, searches | select lines for what they DO, not what they are about | two searches; entropy steers but does not transfer, drift-search needs screening because maximising surprise hunts for violent lines |
| 15 | `hypnagogia`, powered | settle it with 100 continuations per condition | **poems suppress drift rather than inducing it**; two earlier conclusions reversed |
| 16 | `hypnagogia`, polygon-pca | does the odd constructor behave differently here too? | yes — same pool and target as the worst condition, +0.049 drift, ties baseline; **the selection geometry is the lever, not the subject matter** |
| 17 | `hypnagogia`, rule_vs_text | was that the rule or just those two texts? | the rule, at **half the size** (+0.024, 4 disjoint poems each); the other half was an unexamined trajectory origin, itself a lever of comparable size |
| 18 | `hypnagogia`, coherence_collinearity | is the rule effect just coherence? | **no — three separable levers**: coherence −0.109/unit, lexical diversity −0.117/unit, geometry +0.016 robust to both. The walk's bivariate coherence curve reads flat only because the two covariates cancel |

Each folder has a `NOTES.md` with the numbers, the caveats, and what it opened up.

---

## Corrections log

The arc revised itself six times. This is the part I would point a sceptic at.

| claim | how it was made | how it was corrected |
|---|---|---|
| "the high-arousal region is unreachable" | inherited from the lab | already corrected there; a word-probe ceiling |
| "the two readouts rank constructors differently (ρ −0.10)" | `order_matters`, 15 cells | `calibrated_leaderboard`: ρ +0.78 on the full set — **I over-corrected** |
| "…so the ranking is metric-independent (ρ +0.78)" | pooled across targets | `why_polygon.py`: pooling across a nuisance factor manufactured the agreement; within target, ρ −0.05. The original claim was right |
| "the polygon-pca inversion is the one finding I would defend" | `gemma9b_check`, two models | `polygon_sweep`: two single poems agreeing is not two samples; gap 0.67 against spread 2.22 |
| "the mechanism is model-specific" | `reversal_test`, 9B null | `LAYER_CHECK.md`: instrument-specific — the 9B probe was reading at layer 6 of 43 |

| "w ≈ 1 is the practical setting" | `weighted_selection`, metrics only | building the poem and *reading* it: greedy selection collapses into repetition, and the metric rewards that |
| "the descriptive poem induces the most drift" | two runs at n = 6, agreeing | `powered_drift`: it induces the least, significantly below baseline |
| "the drift search has no purchase" | one run at n = 6 | `powered_drift`: it beats random by +0.048, CI [+0.018, +0.078] |
| "polygon-pca and the walk differ *only* in the selection rule" | `powered_drift`, reading the two call sites as matched | `rule_vs_text`: they also started from different trajectory origins — polygon from (0.5, 0.5), the walk from (0.6, 0.25). Two factors, not one |
| "the selection rule moves drift by +0.049" | `powered_drift`, one build against one build | `rule_vs_text`, 2 rules x 4 origins: the rule effect is +0.024; the origin contributed +0.018 and the pairing happened to be the most flattering of the four |

Three recurring causes. **Single-poem results are not samples** — the `seed` finding explains why —
and **six-generation behavioural estimates are noisier than the effects read off them**. The second
is the more dangerous, because two such runs can agree and both be wrong. One correction came from
neither statistic nor control but from reading the artifact the pipeline produced.

The third showed up last and is the easiest to repeat: **two call sites that look matched in prose
were not matched in code**. Nothing about "same concept, same target, same mask, same length" was
false; the trajectory origin simply was not on the list, because it lives as a config default on one
side and a literal inside a function on the other. Checking that a comparison is controlled means
reading both call signatures, not both descriptions.

---

## Ledger of nulls

Reported because they cost real compute and should not be re-run blind.

- **Model-geometry targets do not beat word-list targets.** 64 targets across five arms, +0.08 z
  against the language arm (p = 0.83); Delaunay "gaps" do not beat random points inside the hull.
- **Geometric novelty is not affective novelty.** 70 states spanning an 18-dimensional map all land
  inside valence 0.45–0.85, arousal 0.17–0.56.
- **Zooming does not help.** 53% of fresh passages land inside the "zone"; its intrinsic dimension
  (18.3) matches the whole cloud (17.8).
- **The polygon-pca inversion does not survive resampling** on either model.
- **The self-report composite is largely a valence reading** (ρ +0.46 to +0.56 with probe valence),
  which limits how independent a "best state" judgement based on it can be.
- **Poems do not induce associative drift.** No poem tested exceeds a no-poem baseline of 0.250 at
  100 generations each. Most suppress it, three significantly; the best two — a poem searched
  against the model, and polygon-pca — only tie it.
- **The affective target does not move behaviour at all.** flow 0.2084 against random screened lines
  0.2091 — a difference of 0.0008, the third independent confirmation. A fourth: holding the target
  fixed and changing only the selection rule moves drift ~25 times further.
- **The `seed` argument cannot be used to resample a build.** Confirmed again while designing
  `rule_vs_text`: `polygon_pca` constructs a `RandomState` and never reads it. Varying a build means
  varying its trajectory origin, its target, or its length.
- **Reordering a poem's lines is not a coherence knob.** Greedy max- and min-coherence permutations
  move a build only 0.05–0.10, because coherence is set by which region of embedding space the
  lines come from, not by their sequence. Use the walk's `w` instead (0.544–0.859).
- **Instruct-tuned models analyse the poem instead of inhabiting it.** 18 of 21 free generations
  opened with "This meditation prompts…" or "**Explanation:**", so that channel scored critique prose.

---

## Contract

- **Import the engine, never fork it.** Use `spiritbench` and the vendored constructors. Engine
  changes go on a `feat/*` branch with tests — as `feat/probe-layer-tiebreak` did.
- **One folder per exploration**, `explore/<slug>/run.py`, docstring stating what is being looked at.
- **Small text outputs are committed** beside the script; anything large goes in `explore/scratch/`,
  which is gitignored.
- **Nothing in the narrative depends on `explore/`.** An exploration that sharpens into a question
  with a verdict criterion graduates to `lab/exp_<slug>.py`.

## Reproducing

Needs the artifacts from the main pipeline (`scripts/00`–`02b`) plus probes. Probe builds are
checkpointed and resumable; the passage states are collected at every layer, so probes can be
retrained at any depth offline without new forward passes.

```bash
python3 scripts/04_train_probe.py                          # Llama word probe   (~10 min)
python3 scripts/19_passage_probe.py                        # Llama passage probe (~5 min)
python3 explore/gemma9b_check/build_probe.py               # 9B passage probe    (~9 min)
python3 explore/polygon_sweep/build_9b_word_probe.py       # 9B word probe      (~35 min)
python3 explore/showcase/run.py                            # the table above
```

Timings are Apple M5, 32 GB, MPS.
