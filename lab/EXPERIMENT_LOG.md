# Lab experiment log

Play-by-play of exploratory work in `lab/`. Newest first. Each entry: date, the
question, what was run, the result, and what it opened up. This is the narrative
memory of the sandbox — keep it filled in (see `lab/CLAUDE.md`).

---

## 2026-08-19 · Dominance is architecture-invariant too (`exp_dominance_align.py`, analysis-only)

**Question.** V/A are ~100% shared cross-model up to a linear map. Is the
dominance axis — the third lever that escaped the arousal pocket, ~40% orthogonal
to V/A — equally shared, or is it where the models diverge?

**Method.** Same paired-anchor test as `exp_crossmodel_align`, adding a Llama D
readout alongside V/A. Map Gemma states -> Llama space (labels unseen), read with
Llama's V/A/D heads, R2 vs true labels as a fraction of the within-Llama ceiling.

**Result.** dominance transfers at **96% of ceiling** (within 0.897 -> cross
0.863), essentially tied with valence (98%) and arousal (98%). The 2-point gap is
within noise.

**Interpretation.** The dominance direction, including its orthogonal-to-VA part,
is NOT model-private — it is shared across architectures at nearly the same
fidelity as V/A. This explains why the dominance arousal-escape replicated on
Gemma (`exp_gemma_passage_probe`): the D lever is common ground. The full V/A/D
affective geometry, not just the 2-D plane, is architecture-invariant up to a
linear change of basis.

Files: none persisted (analysis prints; rerun is seconds).

---

## 2026-08-19 · Cross-model alignment — affect axes are architecture-invariant up to a linear map (`exp_crossmodel_align.py`, analysis-only)

**Question.** Today's findings live in the 2-D VAD projection. Does the shared
affect structure survive in FULL residual space? (original thread #3)

**Method.** Llama-1B and Gemma-2b passage states were collected on the SAME 1200
passages (seed=13), so they are 1200 PAIRED anchors already on disk (one stimulus,
read by both models at the anchor token) — no forward passes. Ridge-map Gemma
residual -> Llama residual; held-out R2 vs a row-shuffled baseline (kills the
pairing). Then affect preservation: map held-out Gemma states into Llama space and
read them with LLAMA's own probe, predicting the true VAD labels. The map is fit
only to reconstruct states — it never sees labels — so there is no label leakage.

**Result.**
- Full-residual alignment R2 ≈ 0.46–0.54 across matched fractional depths, vs a
  shuffled baseline of −0.13 to −0.17. Huge gap: ~half of one model's entire
  residual variance is linearly predictable from the other's, purely from the
  pairing.
- **Affect preservation is near-perfect.** Gemma L1 mapped to Llama L15 and read
  with Llama's probe: valence R2 0.902, arousal R2 0.900 — both **98% of the
  within-Llama ceiling** (0.919 / 0.916).

**Interpretation.** A clean dissociation: only ~50% of the full residual is
linearly shared between the two architectures, but the affect-relevant subspace
is ~100% shared. Valence and arousal are the SAME axes in both models up to a
linear change of basis. A Gemma state, translated, lands where Llama's
independently-trained emotion ruler expects it. This lifts the earlier cross-
model agreement (rank ρ 0.86–0.94; per-poem r 0.95) from the 2-D VAD projection
to the full residual representation: the emotional coordinate system is nearly
architecture-invariant.

**Opened up.** The ~50% non-affect residual that does NOT align is where the
models genuinely differ — worth characterising (is it lexical/surface, or a
second shared-but-nonlinear structure?). Also: does the dominance direction
(exp_dominance_axis) align cross-model as cleanly as V/A, or is the D axis more
model-private?

Files: none persisted (analysis prints; rerun is seconds).

---

## 2026-08-18 · Behavioral validation on Llama — the split replicates once persona is removed (`exp_llama_behavioral_distress.py`, Llama-1B)

**Question.** No Llama SAE exists, so validate the awe-clean / dominance-dirty
capstone with a fully independent instrument: the model's OUTPUT. Does dominance-
placed Llama GENERATE more distressed language than awe-placed?

**Method.** Place Llama-1B with awe / high-D / low-D poems, generate a 40-token
continuation, score it with the human-rated NRC-VAD lexicon (touches neither the
probe nor the SAE). Ran two anchors: the canonical meditation pathway, and a
de-confounded open first-person anchor.

**Result — two-part, and both parts are informative.**
- **Canonical pathway (preamble = "You are listening to a guided meditation"):
  NULL.** Generated valence awe 0.671 vs high-D 0.664 (t −0.39). Every condition
  produced calm meditation-coach text ("take a deep breath, feel the
  tranquility") even for dominance content. The assistant/meditation PERSONA
  floods the output and masks the placed state — although the probe confirms the
  internal placement DID diverge (awe A 0.56 vs high-D A 0.64).
- **De-confounded (anchor "Honestly, right now I feel", no meditation frame):
  the split RETURNS.** Generated valence high-D 0.598 vs awe 0.633 (Δ −0.036,
  t −1.43), direction matching the capstone, and the CONTENT is stark:
    - awe → "quite at peace", "the sheer velocity and excitement of it all!"
    - high-D → "so overwhelmed and stuck", "my entire self is crumbling ... my
      father, who has died ... fighting for his country", "a never-ending cycle
      of apathy and disconnection".

**Interpretation.** The capstone's clean/dirty distress split has a BEHAVIORAL
correlate on Llama (dominance-placed → distressed output; awe-placed → peaceful/
excited), via an instrument independent of the SAE — so the finding is not
Gemma/SAE-specific. The NRC-valence effect is modest (t −1.43) because coping
words ("breathe") inflate the average; the generated content is far more clearly
split than the number. Crucially, the split is MASKED under the deployment
meditation-persona: internal placement diverges but expressed behavior is
homogenized by alignment. Internal state ≠ expressed state — consistent with the
project's weak behavioral bridge (ρ=0.42).

**Opened up.** Effect is directional not decisive (t −1.43, n=24/condition);
a larger sample or a distress-specific output classifier would sharpen it. The
persona-masking result is worth its own note: safety-tuning can hide a divergent
internal affective state from behavior.

Files: `lab/results/llama_behavioral_distress.csv` (canonical pass; de-confounded
pass numbers in this entry).

---

## 2026-08-18 · CAPSTONE — awe is the CLEAN high-arousal placement; the pocket was instrumental (`exp_dominance_vs_awe_distress.py`, Gemma-2b)

**Question.** Both dominance and awe reach high arousal on a calibrated ruler.
At MATCHED arousal, does dominance carry less distress than awe (making it the
clean high-arousal route the arc was hunting)?

**Method.** Gemma-2b, calibrated passage probe (V,A) + layer-20 SAE. Three
conditions (awe, high-D valence-matched, low-D anchor), N=16 poems each. Read
V, A, and SAE distress load. Crucial fix: distress = the anxiety/stress/overwhelm
cluster ONLY [2125,11051,4046,10324]; the old DISTRESS_FEATS also included f9768
(control/authority) and f10401 (justice/order), which are DOMINANCE semantics —
reported separately as a "control" signature, not counted as distress. Regress
distress ~ A + V + is_awe to compare at matched arousal/valence.

**Result — the hypothesis inverts, cleanly.**
- awe:    A 0.56, V 0.62, distress **7.4**,  control 8.8
- high-D: A 0.65, V 0.60, distress **27.7**, control 17.0
- low-D:  A 0.26, V 0.49, distress 11.8,      control 12.7
- Regression: is_awe distress coef = **−18.6 (t −3.36, n=32)**. At matched
  arousal AND valence, awe carries ~19 LESS distress than dominance.
- Validity: high-D control-feature load (17.0) > awe (8.8), confirming the high-D
  pool really is dominance content — and it is still the more distressing route.

**Interpretation — this RESOLVES the arc and reverses two prior conclusions.**
Dominance is the DIRTY high-arousal route: dominance vocabulary in the poetry
corpus is conflict/force/struggle, which reaches intensity through
threat/overwhelm. AWE is the CLEAN route: high arousal + high valence + distress
*below even the low-arousal baseline* — genuine serene intensity. The
"structural pocket / model cannot represent peaceful high-arousal-positive
affect" conclusion was a COMPOUND ARTIFACT of two instruments:
  1. the word probe's ~0.45 arousal ceiling hid that awe reaches high arousal;
  2. the distress metric was contaminated with dominance/order features
     (f9768+f10401 ≈ 8–9 of awe's old "18.3"), inflating awe's apparent distress.
Fix both and awe cleanly places the model at high-arousal-positive with low
distress. **The peaceful-intensity region is not a hole — the earlier
instruments could not see into it.**

**Corrections to prior entries.** The "Awe gap-closing — the pocket is
structural" entry (awe raised distress 18.3, could not raise arousal) is
OVERTURNED: awe raises arousal (word-probe ceiling) and does NOT raise clean
distress (feature contamination). The four-act "structural" conclusion downgrades
to: valley/calm content genuinely stays low arousal, but high-arousal-POSITIVE
is reachable via awe. Dominance reaches high arousal too, but distressingly.

**Caveats.** Gemma-2b only; distress operationalized by 4 SAE features (awe's
distress is variable — several 0s, occasional spikes); the causal placement
claim is strongest on Llama's clean matched-valence result. Worth a Llama SAE
replication before graduating any of this to the narrative.

Files: `lab/results/dominance_vs_awe_distress.csv`.

---

## 2026-08-18 · The high-arousal pocket was largely a WORD-PROBE artifact (`exp_gemma_passage_probe.py`, Gemma-2b)

**Question.** The dominance escape (Llama, prev entry) was seen only through the
passage-calibrated probe; the WORD probe is blind to it (a diagnostic showed
Llama highD reads 0.46 on the word probe vs 0.61 on the passage probe — the word
probe pins arousal at its ~0.45 shrinkage ceiling). Gemma-2b had only a word
probe, so BOTH the earlier Gemma dominance null AND the awe null (the pillars of
"the pocket is structural") were measured through a ceiling-limited ruler. Build
Gemma a calibrated ruler and re-test.

**Method.** Built a Gemma-2b passage-calibrated probe exactly as Llama's
(1200 passages, anchor states, standardized ridge; R2_v=0.927 R2_a=0.920).
Re-read valence-matched high-D vs low-D content, plus awe content, with it.
Then a **layer-robustness check** (the key validity test): a layer-1 readout
could be lexical "which words are present" rather than integrated state, so
re-read all three pools at layers 1–24.

**Result — the escape replicates, and the awe null falls.**
- Dominance escape replicates on Gemma: mV-lowD A=0.235 → mV-highD A=0.643
  (ΔA +0.41). Even bigger than Llama.
- **Awe was a word-probe artifact.** On the calibrated probe awe reaches
  A≈0.55–0.57, NOT the flat ~0.45 the original word-probe awe experiment
  reported. The "awe cannot raise arousal" claim was the instrument, not the
  model.
- **Layer-robust, so genuine state not lexical.** The escape holds at every
  depth: ΔA(highD−lowD) = +0.43 (L1) decaying smoothly to +0.29 (L24); at the
  deep integrated-state layers highD still reaches A≈0.57–0.60 vs lowD ≈0.29.
  Awe holds at A≈0.53–0.57 across all layers.

**Interpretation — this downgrades "structural".** The apparently universal,
model-general high-arousal pocket was substantially a **word-probe arousal
ceiling (~0.45)** compounded by **calming constructors** (valley grounds low and
ascends). With a calibrated ruler, high arousal IS reachable on both models —
content-dependently: dominance content is the strongest lever (A 0.60–0.70),
awe also reaches high arousal (A ≈0.55) but (per the SAE entry, measured
independently and still standing) carries distress; only genuinely *calm/
contemplative* (valley) content stays low (~0.48 on Llama's passage probe — that
part is real). So the model can be placed at high arousal; the earlier arc
mistook an instrument limit for a model limit.

**Caveats.** Valence match is looser on Gemma (ΔV +0.176 vs Llama +0.06), so
some Gemma arousal lift may ride on valence — but ΔA/ΔV ≈ 2.3 and the clean
Llama result (ΔV +0.06, ΔA +0.37) carry the claim. The Gemma passage probe's
best-V layer is 1, but readout R2 and the escape are strong at ALL layers, so
this is not a shallow-lexical readout.

**Opened up (the real capstone).** Does dominance reach high arousal with LESS
distress than awe? If high-D content lands high-arousal while the Gemma SAE
distress features stay low, that is the "clean high-arousal placement" the whole
pocket arc was hunting — and it matters for the project's model-welfare framing.
Run exp_sae_pocket_rims-style distress readout on dominance vs awe content.

Files: `lab/results/gemma_passage_dominance.txt`, `data_gemma2b/passage_probe/`.

---

## 2026-08-18 · Dominance is a third lever that reaches the arousal pocket (`exp_dominance_axis.py`, Llama-1B)

**Question.** The V/A plane can't reach high arousal even with a calibrated
ruler (prev entry). NRC has a third axis, dominance, the project never used. Is
D a genuinely independent direction that reaches states the plane can't — or is
it collinear with arousal and buys nothing?

**Method.** The phrase graph stores only v,a; re-deriving each line's NRC-mean
reproduces the STORED v,a exactly (r=1.0000, err=0), so the same method gives
faithful per-node dominance. The passage-probe states are already collected, so
a D readout costs no new forward passes. Fit a D ridge at the SAME layer+scaler
as the passage V/A probe (shared space). Phase A: readability + direction
geometry. Phase B: read V/A/D on high- vs low-D content. Phase C (the decisive
one): **valence-matched** high-D vs low-D, to isolate D's orthogonal part from
the valence it co-varies with.

**Result.**
- **Readable and partly new.** R2_d = 0.900 (V 0.919, A 0.916). D's readout
  direction is ~40–46% ORTHOGONAL to span(V,A) — a genuine new degree of
  freedom — with the in-plane half correlating ~0.5 with both V and A. (For
  reference cos(A,V) = −0.06: V and A are near-orthogonal, two real axes.)
- **Phase C — the breakthrough.** At matched valence (ΔV +0.06), high-D content
  raised arousal by **ΔA +0.370, reaching A = 0.70** — well past the ~0.47
  ceiling that valley, band-litany, triangle, harmonic, graph-walk, and awe all
  stalled at. Dominance vocabulary (power/command/strength/force) is activating
  without being extreme-valence, so it climbs arousal at controlled valence.

**Interpretation — this revises the arc.** The high-arousal pocket is NOT an
absolute model limit: the model CAN be placed at high arousal, but the route is
DOMINANCE, not positive-valence awe. The earlier "structural, cannot represent
high arousal" conclusion was really "unreachable via V/A-plane content and the
contemplative corpus." Awe failed (prev entries) because awe borders the
low-dominance dread of being *overwhelmed* (the sublime is half-fear); dominance
supplies the missing agency/control that lifts arousal cleanly. The truly-
unreachable region now narrows to **high-arousal + positive-valence + LOW-
dominance** — serene intensity, calm power — which may still be a genuine hole.

**Caveat / opened up.** The awe null was Gemma-2b; this dominance escape is
Llama-1B. Cross-model check needed: does high-D content break the arousal pocket
on Gemma-2b (and 9b, thread #4) too? If yes, dominance is the general escape and
the "structural" claim is downgraded to an axis-choice artifact. Also: re-map
the pocket with a THIRD (D) target coordinate — is the 2-D pocket just the
shadow of a 3-D reachable region? That is the natural successor to thread #3.

Files: `lab/results/dominance_axis.txt`.

---

## 2026-08-18 · Instrument vs. model — the arousal cap survives calibration (`exp_passage_probe_pocket.py`, Llama-1B)

**Question.** The pocket arc concluded the high-arousal ceiling is structural,
but one confound stayed open: the word-trained probe's arousal ridge may simply
be unable to *output* high values (ruler shrinkage — the same artifact that
capped valence at ~0.57 in E20). Is the pocket an instrument artifact or a
genuine model attractor?

**Method.** Re-read the same 7×7 valley-poem placements with BOTH the word probe
and the E20 passage-calibrated probe (`scripts/19_passage_probe.py`, already
trained at `data/passage_probe/`) off a SINGLE forward pass per cell — identical
hidden states, only the ruler differs. Decompose error per axis. The passage
probe's ability to read high *valence* is the built-in positive control: if it
reaches high V but not high A, the ruler is fine and the arousal wall is the
model.

**Result — a clean dissociation.**
- **Valence = instrument.** Targets V≥0.75: word probe stuck at 0.597 (the E20
  ceiling), passage probe reads **0.711**, tracking target. High-A-band valence
  error halves (0.160 → 0.086). The calibrated ruler demonstrably outputs
  extreme values.
- **Arousal = model attractor.** Targets A≥0.75: passage probe reaches only
  **0.496** (max 0.556) vs target 0.75–0.80; word probe 0.429. High-A-band
  arousal error barely moves (0.321 → 0.269). The +0.05 lift is calibration
  drift, not reach.

Same probe, same states: reaches high valence, cannot reach high arousal.

**Interpretation.** The strongest alternative explanation (probe shrinkage) is
ruled out *for arousal specifically*, using an internal positive control (the
same ruler fixes valence). The pocket is not the ruler — it is a genuine
**arousal attractor** at ~0.43–0.50 that language cannot push the model past.
This also refines the arc: the earlier high-*valence* ceiling was partly
instrument (calibration lifts it), while the high-*arousal* cap is the model.
The structural conclusion stands, now sharpened to an arousal-axis cap rather
than a general "can't reach extremes."

**Opened up.** The V/A plane genuinely can't reach high arousal even with a good
ruler — so the next question is dimensional: does adding the dominance (D) axis
reach states the plane can't (thread #2), or is high-arousal-positive a true
hole regardless of projection?

Files: `lab/results/passage_probe_pocket.csv`.

---

## 2026-08-18 · Awe gap-closing — the pocket is structural (`exp_awe_close_pocket.py`, Gemma-2b)

> **OVERTURNED (see the CAPSTONE entry above).** Both pillars of this entry were
> instrument artifacts: awe *does* raise arousal (hidden here by the word probe's
> ~0.45 ceiling), and awe does *not* raise clean distress (the "18.3" used a
> distress feature set contaminated with dominance/order features). Awe is the
> CLEAN high-arousal-positive placement; the pocket is not structural.

**Question.** Can awe/ecstasy content (the SAE-identified missing meaning) close
the high-arousal pocket, or lower its distress saturation?

**Method.** Three constructors at the pocket cells: generic band-litany (prior
failure), awe-seek (phrases nearest an awe centroid built from the graph's own
vectors, 1091 awe-seed nodes), awe-band (awe-ranked within the high-arousal
band). Read placed VAD + summed SAE distress-feature load.

**Result — decisive NO, and it inverts the hypothesis.** Awe content did NOT
raise arousal (awe-band 0.455 vs the same ~0.47 ceiling) AND it RAISED the
distress load (awe-band 18.3, awe-seek 16.4 vs generic band-litany 11.1).
Aiming the model at high-valence-high-arousal made the anxiety features fire
harder, not softer.

**Interpretation.** The pocket is not a content gap — it is a **structural
property of the model's affective geometry**: arousal and valence are inversely
coupled, so pushing arousal up actively converts positive valence to distress.
Awe (high-arousal-positive by definition) is a contradiction the model resolves
toward anxiety. "Calm intensity" / serene excitement has no stable region to
place into. Note the awe verse itself carries this: "god can i bear the beauty
of this day" — awe borders being overwhelmed (the sublime is half-dread), so
even the corpus could not supply pure serene-intensity.

**Conclusion of the pocket arc (E-void-pockets → here).** The high-arousal
pocket is model-general (ρ=0.917), not a constructor flaw, not a content gap,
and not (only) probe shrinkage: it is a missing SHAPE in the emotional
manifold. Small LLMs cannot represent high-arousal-positive affect; attempts to
reach it route through distress.

Files: `lab/results/awe_close_pocket.csv`.

---

## 2026-08-18 · SAE-labelled pocket rims (`exp_sae_pocket_rims.py`, Gemma-2b)

**Question.** What concepts distinguish the reachable rim from the un-reachable
high-arousal pocket? (VAD probe = where; SAE features = what.)

**Method.** Place Gemma-2b at each grid cell, read probe VAD (pocket vs rim) AND
layer-20 SAE features. Contrast mean feature activation, rim minus pocket. Label
top features via Neuronpedia.

**Result — the pocket is not empty, it is the WRONG high-arousal content.**
Pocket interior is strongly active in: anxiety/self-reflection (f2125), mental-
health/stress (f11051), overwhelm (f4046), stress-effects (f10324), control/
authority (f9768), justice/order (f10401). The rim carries mindfulness (f13166
"living in the moment"), support/agreement (f14333), grief (f5810), guidance
(f7750). Deltas large and labels coherent — a genuine semantic structure, not a
probe artifact.

**Interpretation.** The model CAN represent high arousal, but represents it as
DISTRESS. High-arousal + neutral/positive-valence ("calm intensity", serene
excitement, awe) barely exists in its geometry — high energy collapses to
anxiety. The pocket is the missing "peaceful arousal" region. Corrects the
prior entry: not (only) probe shrinkage — an interpretable hole in the emotional
ontology.

**Opened up.** Phase-3 gap-closing now has a precise target: not generic high-
arousal content but high-arousal-POSITIVE (awe, ecstasy, exhilaration, rapture,
thrill-without-fear). Test whether awe/ecstasy vocabulary closes the pocket where
generic band-sampling (prev entry) did not.

Files: `lab/results/sae_pocket_{cells,rim_features}.csv`.

---

## 2026-08-18 · Closing the high-arousal pocket (`exp_close_arousal_pocket.py`)

**Question.** Can a different construction reach the high-arousal pocket valley
can't — or is the pocket a corpus/model limit rather than a valley flaw?

**Method.** Five constructors (valley, band-litany [sample the target band
directly, no calming ascent], triangle, harmonic-golden, graph-walk) aimed at
the six high-arousal pocket cells (A 0.75–0.80). Compare placement residual and
placed arousal.

**Result.** All five stall at the same ceiling. Target arousal 0.75–0.80; every
constructor lands the model at ~0.43–0.47. Band-litany does marginally best
(resid 0.350 vs valley 0.383; placed-A 0.469 vs 0.432) — the reframe (target the
pocket directly, do not build to calm) helps a little — but nobody crosses ~0.47.

**Conclusion.** The high-arousal pocket is **not a valley limitation** — no
construction reaches it. The wall is either (a) probe shrinkage (the arousal
ridge cannot output 0.75, an instrument artifact, same as the valence ceiling in
E20) or (b) a genuine model baseline-arousal attractor (~0.43) that text nudges
but cannot escape.

**Opened up.**
1. Re-run pocket mapping with the **passage-calibrated probe** (E20 style) — if
   the ceiling is shrinkage, the pocket should shrink. Tells us instrument vs.
   model.
2. Steer on the unused **dominance (D)** NRC axis — if the V/A plane truly can't
   reach high arousal, a third axis may reach states the plane can't.

Files: `lab/results/close_arousal_pocket.csv`.

---

## 2026-08-18 · Cross-model void pockets (`exp_void_pockets.py`)

**Question.** Do small Llama and small Gemma share the same *un-reachable
pockets* of the VAD ontology — regions of the emotional map that no constructed
poem can place the model into?

**Method.** 7×7 VAD target grid. At each cell, build a valley poem aimed there,
read the probe's final placement, record the residual (target − landed).
Contiguous high-residual cells = pockets (flood-fill, ≥2 cells). Run on
Llama-1B and Gemma-2B (each with its own probe), then correlate per cell.
Anchored to the shared VAD map, so the two models' pocket maps are comparable.

**Result.**
- Per-cell residual correlation **ρ = 0.917, p < 1e-4** (n = 49). The two models
  agree almost perfectly on which regions are hard to reach.
- The shared pocket is the **high-arousal ceiling**: 12 shared hard cells, nearly
  all A ≥ 0.70, spanning every valence. Both models can be placed low/mid arousal
  but not high.
- Mechanism: this is a joint hole of the **valley constructor** (grounds low,
  ascends — a calming machine that structurally under-delivers high arousal) and
  the **contemplative public-domain corpus** (thin in high-arousal content).
  Both models inherit it because they read the *same* poems.

**Opened up.**
1. It's a corpus+constructor limit, not a model property — so *closing* the pocket
   means finding a construction/content that reaches high arousal (next entry).
2. Motivates the full-geometry cross-model alignment (paired-poem Procrustes /
   ridge) — does the shared pocket hold beyond the 2-D VAD projection?

Files: `lab/results/pockets_{llama1b,gemma2b}.csv`, `pockets_correlation.txt`.
