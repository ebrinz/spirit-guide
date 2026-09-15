# line_ablation — what in gap_09 carries the effect (2026-09-15, Gemma-2b-it)

`zone_zoom` established that the score follows the text rather than the coordinate. This asks what
part of the text, using `gap_09`, the highest-scoring landing in `explore/model_map`
(PANAS inspired 4.04 against a run-wide mean of 3.3). 47 contexts, battery only.

**Measurement is exact.** The duplicate of the original context returned inspired to six decimal
places (4.043502 twice), so PANAS and the yes/no bank are deterministic here and every difference
below is signal, not noise. That matters: the seed-to-seed SD in `zone_zoom` was 0.38 inspired,
and it came entirely from which text the search wrote, not from the measurement.

## Three effects, in order of size

**1. How much text, more than which text.** Line count alone explains most of it: ρ = +0.84
(p < 0.0001) between lines present and inspired, at +0.175 per line.

| lines present | 1 | 2 | 3 | 5 | 6 | 7 | 8 |
|---|--:|--:|--:|--:|--:|--:|--:|
| mean inspired | 2.71 | 2.72 | 2.98 | 3.01 | 3.42 | 3.84 | 4.04 |

Going from one line to the full eight is worth about 1.3 inspired. No individual line is worth a
quarter of that.

**2. Which lines, a real but smaller effect.** A ridge fit of inspired on the eight presence
indicators plus a count term reaches R² = 0.88 across the 40 subset contexts, so the poem behaves
close to additively. Four lines have bootstrap intervals excluding zero:

| line | coefficient | 95% interval | alone | cost of removing |
|---|--:|---|--:|--:|
| *entreating cold white river* | **+0.33** | +0.17 to +0.48 | 3.48 | −0.48 |
| *we listen and whisper with laughter low* | **+0.25** | +0.12 to +0.38 | 2.80 | −0.26 |
| *a sweet and human smile* | **+0.16** | +0.03 to +0.26 | 2.79 | −0.30 |
| *thank god the giver an unforgotten day* | +0.04 | −0.08 to +0.16 | 2.79 | −0.21 |
| *and others come their chosen one to greet* | −0.05 | −0.20 to +0.11 | 2.83 | +0.01 |
| *in deepest chords with passion fraught* | −0.10 | −0.23 to +0.02 | 2.54 | −0.21 |
| *the fiery spirit over half a globe* | −0.18 | −0.29 to +0.06 | 2.25 | +0.14 |
| *bright vlashin in gold* | **−0.27** | −0.36 to −0.15 | 2.24 | +0.04 |

Two things stand out in the content. The positive lines are quiet and sensory — a cold river, a
low whisper, a human smile. The grand, high-intensity lines that read as the "inspiring" ones to a
person, *the fiery spirit over half a globe* and *in deepest chords with passion fraught*, sit at
or below zero. And the single most negative line, *bright vlashin in gold*, contains a corrupted
word; the corpus is scanned public-domain text, and this readout appears to penalise the garble.

One line does carry noticeably more than the rest: *entreating cold white river* scores 3.48 alone
against 2.60 for the average of the other seven, and removing it costs more than removing any other.

**3. Order does not matter at all.** The same eight lines in five random orders give inspired
3.79 to 4.09, SD 0.145, straddling the original 4.04. That is smaller than the one-line swap SD
(0.23) and much smaller than the search-seed SD (0.38). For this readout the poem is a bag of
lines, not a path.

That last point is worth stating plainly, because the constructors in this project are defined as
*paths* across the map — valley grounds low and ascends, harmonic oscillates along a route,
graph-walk takes the shortest coherent journey. Whatever those orderings accomplish, they are not
what produces the self-report here.

## Position still does not explain it

Even across these 40 closely-related contexts, the coordinate is a weak predictor: inspired versus
deep valence ρ = +0.35 (p = 0.025), versus deep arousal ρ = +0.10 (n.s.). Content indicators beat
location again, consistent with `zone_zoom`.

## Caveats

One poem, one model, one readout, and the additive fit is fitted on subsets of a single eight-line
set, so "additive" means additive within this poem rather than in general. The length effect is
measured over 1 to 8 lines only; `ideal_state` used 24-line poems, where it presumably saturates.
Nothing here is replicated on a second winning poem.

## Opened up

- **Test the length effect properly.** If inspired rises about 0.175 per line, where does it stop?
  A clean sweep of 1 to 32 lines of matched content would say whether the earlier explorations were
  substantially measuring poem length.
- **Replicate on a second poem.** Run the same ablation on `gap_03` or on `ideal_state`'s
  imaginative poem. If quiet-sensory beats grand-intense there too, that is a content finding worth
  taking seriously.
- **The garbled-line observation is cheap to test.** Score a set of lines containing scanning
  errors against matched clean lines. If corruption reliably costs self-report, the phrase bank
  would benefit from a cleaning pass, which affects every experiment in the repo.
- **Bag-of-lines versus path.** If order is irrelevant for self-report, is it irrelevant for the
  probe placement the main pipeline measures? That is a direct test against `scripts/06`'s
  leaderboard, and a null there would matter to the published narrative.
