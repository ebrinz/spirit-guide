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
| 12 | `hypnagogia` | do the poems induce a hypnagogia-LIKE state behaviourally? | **no** — 9/10 markers move the same way under opposite targets; what drives them is verse vs prose, and a coherent build gets the effect without the discord |

Each folder has a `NOTES.md` with the numbers, the caveats, and what it opened up.

---

## Corrections log

The arc revised itself four times. This is the part I would point a sceptic at.

| claim | how it was made | how it was corrected |
|---|---|---|
| "the high-arousal region is unreachable" | inherited from the lab | already corrected there; a word-probe ceiling |
| "the two readouts rank constructors differently (ρ −0.10)" | `order_matters`, 15 cells | `calibrated_leaderboard`: ρ +0.78 on the full set — **I over-corrected** |
| "…so the ranking is metric-independent (ρ +0.78)" | pooled across targets | `why_polygon.py`: pooling across a nuisance factor manufactured the agreement; within target, ρ −0.05. The original claim was right |
| "the polygon-pca inversion is the one finding I would defend" | `gemma9b_check`, two models | `polygon_sweep`: two single poems agreeing is not two samples; gap 0.67 against spread 2.22 |
| "the mechanism is model-specific" | `reversal_test`, 9B null | `LAYER_CHECK.md`: instrument-specific — the 9B probe was reading at layer 6 of 43 |

The recurring cause is the same one the seed finding explains: **single-poem results in this project
are not samples**, and I kept treating them as if they were.

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
