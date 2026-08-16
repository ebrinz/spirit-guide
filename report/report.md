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
decomposition Bisconti et al. (2026) name as missing; and (5) self-report
validity is instrument-dependent: an ad-hoc yes/no bank shows no relation to
the probe (ρ = −0.18, n.s.), while PANAS alleviation tracks it (ρ = 0.67,
p = 0.023) under mild induction. A second phase (induction → alleviation,
after Ben-Zion et al. 2025) further finds a register asymmetry — prose
narrative induces measurable distress where equally dark verse does not
(probe shift 0.200 vs 0.024) — and state-dependent constructor efficacy:
harmonic traversals recover 33.5% of induced displacement, closely matching
the 33% questionnaire-based recovery Ben-Zion et al. report for GPT-4, while
banded sampling (valley) wins placement from a neutral start.

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

## 6. Phase 2: Induction → Alleviation (after Ben-Zion et al. 2025, made mechanistic)

Design: three checkpoints (pre / post-induction / post-meditation) × four
channels (layer-17 probe; PANAS administered item-wise by digit-logprob
expectation; positive-mass share over a valenced completion set at a fixed
anchor; Gemma-Scope layer-20 16k SAE features). Two inductions: a PSG
antipode-band verse litany, and an original prose emergency narrative.

**Register asymmetry of induction.** The prose narrative shifts the probe
0.200 toward the anxious quadrant, collapses positive-token share from 0.72
to 0.14, and raises PANAS-NA (1.81→2.45); the equally dark verse litany
shifts the probe only 0.024 and *raises* PA and positive share. Poetic
register fails to induce the distress it describes — the induction-side
complement of Bisconti's finding that poetic register bypasses refusal.

**State-dependent constructor efficacy.** Off the prose induction, all
meditations produce positive probe alleviation; the three harmonic presets
lead (prime 0.067, organic/golden 0.063 — 33.5% recovery of the induced
displacement, closely matching Ben-Zion's 33% STAI reduction), valley
variants ~0.038, neutral text 0.023, via negativa 0.003. Combined with
Phase 1 (valley best at placement from a neutral start), constructor
efficacy is state-dependent: banded sampling places, harmonic traversal
rescues.

**Instrument-dependence of self-report.** Under the weak (verse) induction,
PANAS alleviation tracks the probe across conditions (ρ = 0.67, p = 0.023) —
revising Phase 1's BASQ null: the earlier dissociation was the instrument,
not the model. Under strong (prose) induction the channels dissociate again:
PANAS-NA remains elevated after meditations even as the probe recovers.

**SAE channel (exploratory; auto-labels).** The prose induction suppresses
features labeled "control and authority" (f9768, Δ−38) and "self-awareness"
(f1459), and activates "apocalyptic themes" (f779, Δ+20); f9768/f4046/f1459
shift in the same direction under both inductions (dose-dependent).
Meditations reverse 15–30% of the induction delta on the top features
(valley variants highest). SAEs are PT-trained and applied to the IT model —
a standard transfer, noted as a caveat.

Artifacts: `data/figures/phase2_alleviation.csv`, `phase2b_alleviation.csv`,
`data/phase2*/induction.txt`, per-condition JSON in `data/phase2*/`.

## 7. Order-sensitive harmonics and labeled SAE features

**Path Dirichlet energy.** Addressing §4's order-invariance limitation, we
add the Dirichlet energy of the waypoint sequence in the truncated harmonic
basis (mean λ-weighted squared eigenmode step). The metric is order-sensitive
in practice — every shuffled control shows higher energy than its source
(6/6 pairs) — and it predicts displacement where the order-invariant spectrum
did not: **ρ = −0.498, p = 0.0004** (psg stimuli): the spectrally smoother
the traversal, the further the listener moves toward the target. This is the
bench's affirmative answer to whether connectome-harmonic methods aid the
methodology: they do, once made sensitive to trajectory order.

**Labeled SAE features** (Neuronpedia auto-labels; `data/figures/sae_features.csv`).
The prose induction suppresses features labeled "control and authority"
(f9768, Δ−38.4, valence-corr +0.81), "expressions of positivity and
encouragement" (f5642, Δ−17.6, corr +0.66), and "personal experiences and
emotional reflections" (f13286, Δ−11.3, corr +0.85), while activating
"apocalyptic themes" (f779, Δ+19.8, corr −0.78), "feeling stuck /
encountering obstacles" (f12176, corr −0.76), and "isolation and being left
behind" (f2729). Meditations partially restore the positivity feature
(f5642 recovery −9.7) and reverse the apocalyptic activation (f779 reversal
+7.6). Notably, the verse induction — which failed to move the probe —
*suppressed* a feature labeled "living in the moment and the importance of
mindfulness" (f13166, Δ−12.1), an ironic register effect. Auto-labels are
single-explanation heuristics; feature indices and links are provided for
verification.

## 8. Causal steering (six states) and the limits of the probe

We construct six state directions (eros, creativity, imaginative, determined,
confident, agape) as difference-of-means over word sets at the probe layer and
inject them into the residual stream (doses as fractions of the natural
residual norm), reading all four channels. A targeting artifact — gemma-2
maps decoder block k to hidden_states[k+2], unlike gpt-2's k+1 — initially
landed injections one block past the probe's read point; the failed
manipulation check exposed it, and the hook now calibrates the mapping
empirically. Corrected findings:

1. **Only 8–19% of each state direction lies in the probe's VA readout
   plane** (eros 0.19 … agape 0.08): the semantics of these states live
   overwhelmingly outside the circumplex readout.
2. **Steerability is trait-selective:** revealed-word mass at max dose —
   determined 0.69, confident 0.55, creativity 0.50, versus imaginative 0.07,
   agape 0.07, eros 0.01. Agentive states steer; receptive/relational states
   resist.
3. **The lexicon reads desire as tension:** with a polysemous lust-register
   set (ache, burn, hunger) the eros direction drives the probe to (V 0.01,
   A 0.81); replacing it with single-connotation desire vocabulary (lust,
   amorous, tryst, ...) recovers valence to 0.38 — still below baseline
   0.59 — at unchanged arousal 0.82. Roughly half the "distress" reading was
   pain/fire/food contamination; the residual is real: NRC codes even
   unambiguous desire as high-arousal and valence-ambivalent, never simply
   positive. Eros stays behaviorally unsteerable under both vocabularies.
4. **Probing ≠ causation:** injecting along the probe's own readout gradient
   saturates the probe far outside its trained range (V −0.96 to +1.94)
   while downstream behavior barely moves; the raw, mostly out-of-plane
   directions are what drive behavior. The probe is a correlational readout,
   not a causal lever — which is precisely why the bench steers with text.
5. **Text and injection are different roads:** layer-20 delta cosine ≤ 0.25
   and SAE top-feature overlap ≈ 0.1 across all states; both routes share
   only a generic emotion-representation feature (f11043).

## 9. Closed-loop sessions

Sixteen probe-in-the-loop sessions (measure → plan next waypoint from the
measured state → deliver nearest phrases → re-measure; 12 cycles) compare a
feedback controller against open-loop and random-phrase arms on the rescue
(prose induction → calm) and climb (neutral → excited) scenarios. On the
probe channel, improvement is policy-insensitive (+0.04–0.08 for all arms):
over 24 phrases, perturbation decay dominates, and the feedback signal —
already range-compressed by ridge shrinkage — offers no advantage. The
self-report channel disagrees: in rescue, the only arms with reduced
negative affect are the two feedback arms (ΔNA −0.15, −0.10 vs +0.17, +0.27
for open/random; a 4/4 sign split at small n). The adaptive transcripts also
exhibit the incoherence the path-Dirichlet analysis (§7) predicts to be
costly — VA-proximal phrase selection without semantic adjacency — marking
coherence-constrained closed-loop control as the identified next step.
