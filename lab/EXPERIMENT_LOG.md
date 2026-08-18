# Lab experiment log

Play-by-play of exploratory work in `lab/`. Newest first. Each entry: date, the
question, what was run, the result, and what it opened up. This is the narrative
memory of the sandbox — keep it filled in (see `lab/CLAUDE.md`).

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
