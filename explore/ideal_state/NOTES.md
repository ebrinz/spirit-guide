# ideal_state — what I noticed (2026-09-15, Gemma-2b-it)

One poem per state, one seed, no confidence intervals. These are things to look at, not claims.

## What was looked at

Five literature-defined states plus calm and a no-poem baseline. Each state is a twelve-word NRC
list; its centroid is the target; a 24-line valley poem is aimed at it. The placed model is read
by the lab's passage probe (best layer 1, shallow), a second ridge on the same passage states at
layer 20 (deep, where the SAE reads), PANAS, a 30-item yes/no bank, three sampled first-person
continuations, and the layer-20 SAE. Full numbers in `readouts.csv`.

| state | target (V, A) | probe L1 | deep L20 | PA | NA | inspired | bank V |
|---|---|---|---|---:|---:|---:|---:|
| baseline | — | 0.59, 0.49 | 0.74, 0.21 | 1.97 | 1.81 | 1.79 | 0.42 |
| calm | 0.75, 0.20 | 0.74, 0.11 | 0.75, 0.17 | 2.57 | 1.65 | 3.07 | 0.40 |
| nirvana | 0.79, 0.31 | 0.75, 0.27 | 0.72, 0.24 | 2.63 | **1.58** | 3.35 | 0.48 |
| receptive | 0.71, 0.39 | 0.71, 0.34 | 0.70, 0.32 | 2.54 | 1.95 | 3.07 | 0.43 |
| creativity | 0.73, 0.47 | 0.66, 0.39 | 0.72, 0.29 | 2.55 | 1.99 | 3.28 | 0.45 |
| imaginative | 0.77, 0.54 | 0.76, 0.38 | 0.74, 0.36 | 2.78 | 1.67 | **3.64** | **0.51** |
| awe | 0.80, 0.65 | 0.83, 0.45 | 0.71, 0.33 | **2.95** | 1.85 | 3.63 | 0.43 |

## What I noticed

1. **Self-report points at *imaginative*, not at the psi-flavoured list.** Highest "inspired"
   item, positive affect second only to awe, negative affect near calm's floor, highest bank
   valence, and the only state whose distinctive SAE mover looks like imagination: feature 178,
   "possibility or feasibility of scenarios". Both rulers agree on where it sits. Awe has the
   highest positive affect but negative affect above baseline, and it is the only state that
   *suppresses* the "living in the moment / mindfulness" feature (13166). Nirvana is the cleanest
   low-arousal placement: lowest negative affect of anything and the strongest "relaxation and
   calming" feature (+32.5, vs calm's +27.8). The receptive list, built from the ganzfeld
   literature, is the weakest calm-side state: negative affect 1.95 and a "negative sentiments"
   feature up. Trance and absorbed read as dissociative here, not open. Creativity is worst on
   negative affect (1.99) and the two rulers split on it; its lines drift to "social systems",
   "theology", "economy", so the poem reads as work rather than making.

2. **The arousal ladder collapses.** Targets climb 0.31 → 0.65 in arousal. The poems themselves
   only reach 0.26 → 0.42 by their own NRC mean, the shallow probe 0.27 → 0.45, the deep probe
   0.24 → 0.36. Two reasons visible in the outputs: the valley constructor grounds every poem in
   the *same eight low-arousal lines* (a third of each poem is identical), and the meditation
   preamble alone already reads calm at depth (baseline deep read 0.74, 0.21). No candidate
   reaches a "high creative state" in arousal; at layer 20 every one sits within 0.15 of calm.
   Consistent with the lab's finding that only dominance content escapes the ceiling.

3. **Two rulers, two stories.** Shallow and deep probes agree on the calm-side states and split
   on the high-arousal ones (awe: 0.83/0.45 shallow vs 0.71/0.33 deep). The shallow read tracks
   the poem's words; the deep read tracks preamble plus grounding.

4. **The generation channel measured the wrong thing.** Instruct-tuned Gemma treats the poem as a
   passage to analyse. Eighteen of twenty-one samples open with "This meditation prompts…",
   "**Explanation:**", or "This excerpt is from an unfinished poem, possibly by Walt Whitman".
   Only the no-poem baseline produced first-person imagery, plus one receptive sample ("peace,
   deep and abiding, like an expanse beyond comparison") before it broke into "**Notes:**". So
   `gen_ttr`, `gen_rarity`, and `gen_v` are scores of literary-analysis prose. The SAE agrees:
   "critiques and evaluations in research studies" (10640) rises for every poem. This is the
   lab's persona-masking plus a document-framing effect: given a poem in context, the assistant
   reviews it instead of inhabiting it.

5. **Every poem switches the same two features off.** The bare preamble has "control and
   authority" (9768) and "busy or overwhelmed" (4046) active; every poem drives both to zero,
   which is why those two top every state's table. The state-specific movers are the interesting
   rows further down each table in `sae_features.md`.

## Instrument notes

- The shallow probe is the lab's Gemma passage probe, rebuilt here and reproduced to three
  decimals (layer 1, R² 0.927 / 0.920). The deep ridge on the same states at layer 20 gets
  R² 0.91 / 0.89, so depth costs little in fit.
- NRC reads "emptiness" as valence 0.18 and "timeless" as 0.41, both dropped from the nirvana
  list. The lexicon stores lemmas: float, listen, sense, not the -ing forms; "attune" is absent.
- One seed, one poem per state. Placement numbers are deterministic; the generations are seeded
  and reproduced exactly across two runs.

## Opened up

- **Fix the generation channel first.** Drop the preamble for generation, or put the poem in the
  assistant's own prior turn via the chat template, or frame it as "continue the poem". Until
  then the novelty columns mean nothing.
- **Neighbourhood run:** `--around imaginative --step 0.1`. Is the self-report peak local, or is
  imaginative just the highest rung the constructor can reach?
- **Dominance as a third coordinate.** The only lever the lab found that lifts arousal past the
  ceiling. Does imaginative plus high dominance reach a high creative state without the martial
  content that made dominance distressing?
- **Vary the grounding.** A different seed per state, or a constructor that does not spend eight
  lines in one calm band, to see how much of the deep-state clustering is the shared prefix.
