# Lab TODOs — open threads

Living backlog of exploratory threads. Grouped by **compute cost** so a session
can pick work that fits its time budget. Each item: the question, what it would
establish, cost, and dependencies. When a thread is run, move its result to
`EXPERIMENT_LOG.md` (newest-first) and strike it here. See `lab/CLAUDE.md` for
the sandbox contract.

Legend — cost: 🟢 low (analysis on saved data, no model) · 🟡 moderate (a few
dozen–hundred forward passes on a small model) · 🔴 heavy (state collection or a
big model; tens of minutes to hours).

---

## 🟢 Low compute (no new forward passes — analysis on saved states)

- [x] ~~**Cross-model paired-anchor alignment (was thread #3).**~~ DONE
  (`exp_crossmodel_align.py`, 2026-08-19): affine map aligns the two residual
  spaces (R2 ~0.5 vs shuffled −0.14); affect subspace is ~100% shared (V/A read
  at 98% of ceiling after mapping Gemma->Llama). Emotional axes are architecture-
  invariant up to a linear change of basis. See EXPERIMENT_LOG.

- [ ] **Stabilise the D-in-plane / out-of-plane fraction.** The orthogonal
  fraction of the dominance direction came out 36% vs 46% depending on ridge
  alpha (`exp_dominance_axis`). Re-fit across the alpha grid + bootstrap the
  passage set to report a stable fraction with a CI. Saved states only.

- [x] ~~**Does the dominance direction align cross-model?**~~ DONE
  (`exp_dominance_align.py`, 2026-08-19): D transfers at 96% of ceiling, tied with
  V/A (98%). The dominance axis is architecture-invariant too — full V/A/D
  geometry is shared up to a linear map. See EXPERIMENT_LOG.

- [ ] **Characterise the ~50% non-aligning residual.** Only ~half the full
  residual is linearly shared cross-model; the rest is where the architectures
  genuinely differ. Is it lexical/surface, or a second shared-but-nonlinear
  structure? Probe with a nonlinear map (kernel ridge / small MLP) on the paired
  states, or correlate the residual-of-the-map with surface features. 🟢

- [x] ~~**Is dominance-distress the corpus or the axis?**~~ DONE
  (`exp_dominance_corpus_slice.py`, 2026-08-19): NOT separable in this corpus.
  High dominance here = martial/hierarchical power (battle/king/command/war/
  warrior/supreme); no "calm mastery" content exists. Gore is a minority (21%,
  r 0.16) but even non-gore high-D is conquest/command themed. Dominance-distress
  is a content property; a true axis test needs out-of-corpus calm-power content.
  See EXPERIMENT_LOG.

- [ ] **(🟡, spun off) Gore vs non-gore high-D distress.** Forward-pass SAE
  distress comparison of the violent (64) vs non-violent (236) high-D sub-pools
  at matched V/A/D — does explicit gore add distress on top of the command/
  conquest baseline, or is the whole martial-power theme uniformly distressing?
  Gemma-2b + SAE, ~30 forward passes.

## 🟡 Moderate compute (small-model forward passes)

- [ ] **3-D pocket re-map — is the 2-D pocket a shadow?** Re-run the pocket grid
  with dominance as a THIRD target coordinate (V,A,D). If cells unreachable in
  the V/A plane become reachable once D is recruited, the "pocket" is a projection
  artifact of ignoring D. Builds on the D readout from `exp_dominance_axis`.
  Llama-1B, ~50–150 forward passes.

- [ ] **Sharpen the Llama behavioral split.** `exp_llama_behavioral_distress`
  found the awe/dominance split directional (t −1.43) but modest, and masked by
  the meditation persona. Raise N, and score output with a distress-specific
  readout (not just NRC valence, which coping words inflate). Llama-1B generation.

- [ ] **Persona-masking as its own finding.** The deployment meditation-preamble
  homogenised generated behavior while internal placement diverged (internal ≠
  expressed). Sweep persona strength (no preamble → mild → strong) and measure at
  what point the internal awe/dominance split stops surfacing in output. Speaks
  to safety-tuning hiding a divergent internal affective state — arguably the most
  novel thread. Llama-1B generation.

## 🔴 Heavy (state collection or 9B) — PINNED

- [ ] **Gemma-9B scale test (was thread #4). PINNED for later.** Does the
  awe-clean / dominance-dirty split and the arousal escape hold at 9B? Fair
  version needs a 9B passage probe (~1200 forward passes, the long pole) + the
  gemma-scope-9B SAE for the distress capstone. **ETA ~1–1.5 h**, dominated by
  passage-state collection; OOM/swap risk (18 GB model on 34 GB). Checkpointed
  and backgroundable. Note: the existing 9B *word* probe is weak (R²_a 0.60),
  which is why the passage probe is required.

## Blocked / not feasible with current instruments

- **Llama SAE distress replication.** No Llama-3.2-1B SAE exists (Gemma-Scope is
  Gemma-only; Llama-Scope is 3.1-8B). The capstone's SAE-distress metric can't be
  reproduced on Llama directly — the behavioral test (above) is the substitute.

---

## Done this arc (see EXPERIMENT_LOG.md)

- ✅ #1 Passage-calibrated pocket re-map — arousal cap is a real model attractor.
- ✅ #2 Dominance axis — a third lever that reaches the arousal pocket.
- ✅ Fair cross-model dominance (Gemma passage probe) — escape replicates; awe
  null was a word-probe artifact.
- ✅ Capstone (dominance vs awe distress) — awe is the CLEAN high-arousal route;
  the "structural pocket" was instrumental.
- ✅ Llama behavioral validation — split replicates once persona removed.
- ✅ #3 Cross-model alignment — affect axes architecture-invariant up to a linear
  map (V/A read at 98% of ceiling after Gemma→Llama mapping).
- ✅ Dominance alignment — D transfers at 96% of ceiling; full V/A/D geometry is
  architecture-invariant.
- ✅ Dominance-distress corpus-vs-axis — not separable; high dominance in this
  corpus is martial/command content (no calm mastery exists).
