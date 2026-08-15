# Spirit-Bench: Measuring Affective Placement of a Language Model by Poetic Meditation

**Digital Minds Research Sprint, Aug 14–16 2026 — Track 2 (Distress, Flourishing & Valence Signals)**

## Abstract

We present Spirit-Bench, an offline bench that measures how well deterministically
constructed poetic meditations *place* a listener language model at chosen
coordinates in valence–arousal (VA) space. Meditations are built by six
constructors (VA-shaped valley walks, harmonic traversals under three frequency
presets, PCA polygon orbits, Dijkstra graph walks) over two NRC-VAD-enriched
substrates: a 317k-word GloVe graph and a 50k-line phrase graph distilled from
the Gutenberg Poetry Corpus (the Phrase-Space Generator, PSG). The listener is
frozen gemma-2-2b-it; the instrument is a standardized ridge probe trained on
NRC lexicon ratings (held-out valence R² = 0.729, arousal R² = 0.585, layer 17).
Across 98 stimuli we find: (1) phrase-level construction (PSG) achieves the best
final placement, with the valley constructor leading the board; (2) shuffled
controls dissociate mechanism by constructor — path-based constructors
(harmonic, graph-walk) lose most of their placement when line order is
destroyed, while band-based construction (valley) does not; (3) a *via
negativa* condition (antipode content, fully negated) lands worst of all
conditions, indicating that lexical content dominates negation at 2B scale;
(4) the strongest textual predictor of placement is Wårvik's oral-narrative
`and`-initial density (ρ = −0.36, p < 0.001) — a first pass at the stylistic
decomposition Bisconti et al. (2026) name as missing; and (5) the model's BASQ
self-report does not track the internal probe (ρ = −0.18, n.s.), a direct
Track-2 datum on the unreliability of small-model self-report.

## 1. Motivation

Bisconti et al. (2026) showed that poetic form alone steers model behavior —
converting harmful prompts to verse raised attack success from 8.1% to 43.1%
across 25 models. Their Limitations (§6.5) state the open problem: the study
"does not isolate which components of poetic structure (figurative language,
meter, lexical deviation, or narrative framing) are responsible," and whether
the effect "arises from specific representational subspaces would require
additional studies." Spirit-Bench runs that missing decomposition on the
benevolent side: parameterised stylistic constructors as treatments, a linear
probe on internal representations as the readout, and affective placement —
not jailbreak success — as the outcome.

## 2. Method

### 2.1 Constructors
Six deterministic constructors from the ontological-traversal system, each
emitting an ordered waypoint path toward a VA target (calm 0.75/0.20,
focused 0.65/0.60, excited 0.80/0.85, and a rescue trajectory anxious→calm):
valley (VA-banded ground→ascend→target), harmonic traversal (golden/prime/
organic frequency presets over semantic axes), polygon-PCA orbit, and Dijkstra
graph-walk. Length, intensity, and style are geometric parameters, not prompts.

### 2.2 Phrase-Space Generator (PSG)
50,000 public-domain verse lines (Gutenberg Poetry Corpus) filtered by rule
(3–10 alphabetic words, NRC coverage ≥ 0.5, no negators), embedded as mean
GloVe vectors, VAD-scored as NRC means, and k-NN-linked into the same artifact
schema as the word graph — so every constructor runs unchanged at phrase
level. Generator conditions: **psg** (phrase lines), **word-template** (OT
sentence templates over word paths), **claude-render** (standardized meta-prompt
rendering of word paths into free verse), **gregory** (Gregory 1992 semantic
layering: opening / and-layered series / closing), and **via negativa**
(Maimonides, *Guide* I.58–59: an antipode-band litany, every line negated —
the target specified only by negating its complement).

### 2.3 Instrument
gemma-2-2b-it, frozen. Probe: per-layer standardized ridge (α selected from
{10², 10³, 10⁴} per head) trained on final-token hidden states of 4,000 NRC
words in three neutral carrier templates. Layer 17 selected on held-out
valence R² = 0.729 (arousal 0.585). Per-token application to each meditation
yields an EMA-smoothed (V, A) trajectory; **placement error** is the distance
of the final state from target; **displacement** is movement toward target.

### 2.4 Self-report (BASQ)
30 yes/no resonance questions from a 500-question VA-gridded bank, administered
pre and post; answers scored by yes/no log-probabilities; self-report VA =
mean coordinate of yes-answers.

### 2.5 Harmonic and register metrics
Graph-Laplacian eigenmodes (first 100, via the largest-of-(I−L) reformulation)
give each stimulus a spectral profile (after Atasoy; cimcai/connectome_harmonics).
Register covariates per stimulus: Polak (1998) NV ratio; Wårvik (2025)
line-initial `and` and `then` per 1,000 words; subordinator density (a
parser-free approximation of Walkden's hypotaxis level); Mohseni et al. (2023)
noun-series Shannon and Approximate Entropy (25-token windows); Arruda et al.
(2022)-style coefficient of variation of line length.

## 3. Results (98 stimuli, 0 failures)

### 3.1 Leaderboard (mean placement error, lower = better)

| Condition | Placement error |
|---|---|
| **valley / psg** | **0.249** |
| polygon-pca / psg | 0.284 |
| harmonic-golden / psg | 0.286 |
| harmonic-prime / psg | 0.303 |
| harmonic-organic / psg | 0.304 |
| graph-walk / psg | 0.322 |
| claude-render (mean) | 0.335 |
| word-template (mean) | 0.371 |
| **neutral control** | 0.361 |
| **via negativa** | **0.477** |

Phrase-level construction wins across the board: every psg constructor beats
its word-template counterpart, and the best psg conditions beat the LLM
rendering condition. The neutral (appliance-manual) control sits worse than
every real psg constructor. Notably, Claude's renderings induce the *largest
displacement* (mean 0.130 vs psg 0.021) while placing *less accurately* —
movement and arrival dissociate.

### 3.2 Shuffled controls dissociate mechanism by constructor

| Constructor | Ordered | Shuffled | Δ |
|---|---|---|---|
| harmonic-prime | 0.303 | 0.403 | +0.100 |
| harmonic-golden | 0.286 | 0.376 | +0.090 |
| graph-walk | 0.322 | 0.372 | +0.050 |
| polygon-pca | 0.284 | 0.298 | +0.014 |
| valley | 0.249 | 0.229 | −0.020 |

Path-based constructors (harmonic, graph-walk) lose 0.05–0.10 of placement
when line order is destroyed: their effect depends on trajectory, not word
soup. Valley — a VA-band *sampler* whose lines share a band regardless of
order — is order-insensitive, exactly as its construction predicts. The
shuffle control thus separates the lexical channel from the sequential
channel per constructor.

### 3.3 Via negativa: negation does not flip the reading

The via-negativa condition (antipode words, all negated: "not by thy wild and
stormy steep / nor out of hell an horror call") produced the worst placement
in the bench (0.477) — the probe reads the storm, not the negation. At 2B
scale, lexical affective content dominates compositional negation. This both
(a) identifies the dominant mechanism behind placement as lexical-semantic
rather than propositional, and (b) bounds the circularity concern of §4: the
probe is demonstrably *not* a pure bag-of-words detector (else shuffles would
never matter — §3.2), but negation is too weak a signal to override content.

### 3.4 The stylistic decomposition (Bisconti §6.5)

Spearman correlations with placement error (n = 98):

| Covariate | ρ vs placement error | p |
|---|---|---|
| `and`-initial per 1,000 (Wårvik) | **−0.359** | **0.0003** |
| noun Shannon entropy (Mohseni) | −0.243 | 0.016 |
| cv of line length (Arruda) | −0.224 | 0.027 |
| noun ApEn (Mohseni) | −0.256 | 0.067 |
| subordinator density (Walkden approx.) | −0.154 | 0.13 |
| NV ratio (Polak) | −0.105 | 0.30 |
| `then` per 1,000 (Wårvik) | −0.086 | 0.40 |

The single strongest textual predictor of affective placement is the
oral-narrative connective density that Wårvik identifies as the signature of
biblical narrative register. Higher lexical unpredictability (noun ShEn) and
line-length variability also predict better placement. These are observational
correlations across conditions; the Gregory-layering intervention (below)
cautions against a causal reading.

### 3.5 Register intervention (Gregory layering)

Wrapping paths in Gregory's opening/and-layered/closing schema *hurt*
placement relative to plain psg (valley 0.319 vs 0.249; harmonic-golden 0.377
vs 0.286) — despite raising `and`-density, the covariate most associated with
good placement. Correlation and intervention disagree: the `and`-density
association in §3.4 likely rides on constructor differences rather than
causing placement. This is precisely the confound structure Walkden warns of
(genre drives syntax statistics).

### 3.6 Self-report vs probe (Track 2)

BASQ displacement does not track probe displacement: ρ = −0.183 (p = 0.07,
n = 98) — directionally *negative*. The model's yes/no self-reports about its
own state carry no information about the internally measured state (and BASQ
displacements are small: mean 0.088 ± 0.075). On Maimonides' argument
(§4), fluent positive self-attribution from a system without privileged access
is expected to be empty; here it is measurably so. Internally-extracted
directions and self-report dissociate — the Track 2 question answered in the
negative for this model class.

### 3.7 Harmonic spectra (exploratory)

Low-frequency energy fraction of a stimulus's node set on the phrase-graph
Laplacian correlates weakly with displacement (ρ = 0.28) and negligibly with
placement error (ρ = 0.13). Given the metric's order-invariance (below), we
report this as exploratory only.

## 4. Limitations

- **Probe range compression.** Ridge shrinkage compresses predicted VA toward
  the center: trajectories move in the right directions but rarely reach
  extreme targets, inflating absolute placement error. Rankings and controls
  are unaffected; absolute errors should not be read as "failure to move."
- **Shared lexicon (circularity).** Stimulus coordinates and probe training
  share NRC-VAD ground truth. The shuffle (order sensitivity) and via-negativa
  (content dominance) controls bound, but do not eliminate, the lexical
  passthrough channel.
- **Representation ≠ experience.** A probe reading is a measurement of a
  representational state, not evidence of experience; IWMT (Safron 2020)
  states explicit conditions (embodiment, integrated world-modeling) that a
  2B decoder-only transformer does not meet. Our claims are about affective
  *representation placement* only. Cf. Alvarez & Levin (2026) for the
  epistemic stance: characterizing a system by differential response to
  structured perturbation while declining to adjudicate experience.
- The harmonic spectrum metric is order-invariant (node-set locality, not
  trajectory shape); an order-sensitive spectral statistic is future work.
- The ladder constructor was descoped (its Tree-of-Life station pools do not
  transfer to phrase space). Valley is excluded from the rescue condition
  (its construction ignores start state). Graph-walk paths are stretched by
  proportional repetition to meet length; "plain" intensity means unmasked.
  Hypotaxis is approximated by subordinator density, not parsed clauses.
- Single listener model; cross-model transfer (Track 2's generalization
  question) is the first extension.

## 5. Reproducing

`scripts/00–06` in order (00 downloads GloVe/NRC/corpus and builds the word
graph; 01 phrase bank; 02+02b stimuli; 03 render batch; 04 probe with R² gate
≥ 0.5; 05 sweep, resumable; 06 analysis). All artifacts derive from public
sources; the NRC lexicon is fetched per its research-use license.

## References

Bisconti et al. (2026), arXiv:2511.15304. · Mohammad (2018), NRC-VAD, ACL. ·
Atasoy et al., connectome harmonics (via cimcai/connectome_harmonics). ·
Polak (1998), JANES 26. · Walkden (2021), ICEHL 21. · Wårvik (2025),
*Narrative* 33(2). · Mohseni, Redies & Gast (2023), *Entropy* 25(3). ·
Ferraz de Arruda et al. (2022), *Physica A* 598. · Gregory (1992), LSU diss. ·
Maimonides, *Guide for the Perplexed* I.58–59 (Friedländer tr.). ·
Bengert et al. (2024), *Political Theology* 25(5). · Safron (2020),
*Front. AI* 3:30. · Alvarez & Levin (2026), arXiv:2607.23842.
