# Lab experiment log

Play-by-play of exploratory work in `lab/`. Newest first. Each entry: date, the
question, what was run, the result, and what it opened up. This is the narrative
memory of the sandbox — keep it filled in (see `lab/CLAUDE.md`).

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
