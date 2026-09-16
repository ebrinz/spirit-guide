# gemma9b_check — does the readout disagreement replicate at 9B? (2026-09-15)

`explore/calibrated_leaderboard` found on Llama-1B that the published placement metric and a
passage-calibrated anchor read rank constructors differently within a target. This asks whether
that is a small-model instrument quirk. It also clears the pinned 🔴 item in `lab/TODOS.md`.

**The 9B probe.** `build_probe.py`, same recipe and passage seed as the Llama and Gemma-2B builds:
layer 6, held-out R²_v 0.932, R²_a 0.909. Acid test separates the 24 calmest passages (0.76, 0.28)
from the 24 most distressed (0.21, 0.73) — the separation the Llama *word* probe cannot make, where
both read near valence 0.57. Written to `data_gemma9b/passage_probe/` (gitignored).

Cost note for the backlog: the lab estimated 1–1.5 h with OOM risk for an 18.5 GB model on 32 GB.
It took **9 minutes** for 1200 passages, no memory trouble. The 75-stimulus read took 22 minutes,
slower per item because poems are longer and all 43 layers are returned.

No 9B word probe was retrained. The stimulus build is deterministic — all 75 freshly built ids
match `results/leaderboard_9b.csv` exactly — so the published per-stimulus placement errors serve
as the EMA side and only the calibrated side was computed.

## The disagreement is weaker at 9B, but its core replicates

Rank agreement between the two readouts, 6 constructors, matched conditions:

| scope | Llama-1B | Gemma-9B |
|---|--:|--:|
| within target: calm | +0.89 | +0.49 |
| within target: focused | **−0.60** | **−0.43** |
| within target: excited | −0.43 | +0.37 |
| mean within-target | −0.05 | +0.14 |
| aggregated over all found-poetry rows | +0.79 | **+0.93** |
| same, excluding the via-negativa control | +0.66 | +0.89 |

At 9B the two measures agree much better in aggregate. The one piece that replicates at both scales
is **focused**, where the readouts disagree (−0.60 and −0.43). **excited** flips sign between models,
so the Llama disagreement there does not generalise. With six constructors per cell, |ρ| must exceed
0.83 for p < 0.05, so none of the within-target values is individually significant at either scale;
these are directions, not established effects.

## polygon-pca replicates exactly, and it is the sharpest result here

Its rank out of six, EMA → calibrated:

| target | Llama-1B | Gemma-9B |
|---|---|---|
| calm | 2 → 2 | 2 → 2 |
| focused | **6 → 1** | **6 → 1** |
| excited | 5 → 1 | 3 → 1 |

Both models rank polygon-pca *last* on the published metric at the focused target and *first* on the
calibrated read. That is the same inversion, on two architectures, at a scale difference of 9×. It
is the one finding in this arc I would now defend.

Consequently, on matched cells the calibrated read puts polygon-pca first at both scales:

| | Llama EMA | Llama calibrated | 9B EMA | 9B calibrated |
|---|--:|--:|--:|--:|
| valley | **0.313** | 0.201 | **0.234** | 0.146 |
| polygon-pca | 0.384 | **0.178** | 0.303 | **0.136** |

The published winner is second under a calibrated ruler on both models. The margins are small
(0.023 and 0.010) and each constructor here rests on three stimuli, so this wants a seed sweep
before it is quoted — but it points the same way twice.

## An independent confirmation: the arousal ceiling is not the word probe

Both calibrated probes saturate in arousal well below the excited target, averaged over the six
constructors:

| target arousal | Llama calibrated | 9B calibrated |
|---|--:|--:|
| 0.20 (calm) | 0.26 | 0.27 |
| 0.60 (focused) | 0.46 | 0.49 |
| 0.85 (excited) | **0.55** | **0.52** |

Calm and focused are tracked; excited is not reached by either model on either ruler. This is the
arousal cap the lab attributed to the model rather than the instrument
(`lab/EXPERIMENT_LOG.md`, 2026-08-18, which found the passage probe reaching only ≈0.50 for A ≥ 0.75
targets on Llama). Two fresh calibrated probes, one on a 9B model, reproduce it. That entry's
conclusion holds.

## Caveats

Six constructors × three targets per model, one construction seed each; nothing here is powered to
resolve small ranking differences, which is exactly why only the polygon-pca inversion is claimed.
The 9B EMA numbers come from the committed results rather than a rerun, so they carry whatever
probe and conditions that run used — the ids match, but the instrument on that side was not rebuilt.
The 9B passage probe selects layer 6 of 43; the Gemma-2B build selected layer 1 and the lab checked
at the time that its result held across depths, but that check was not repeated here.

## Opened up

- **Seed-sweep the polygon-pca inversion.** Three stimuli per constructor per model is thin for a
  claim this specific. Eight construction seeds at the focused target on both models would settle it.
- **Why focused?** The inversion is target-specific and replicates. Something about how polygon-pca
  distributes lines suits a mid-arousal target under a whole-context read and not under a
  recency-weighted one.
- **Rerun the 9B EMA side.** Joining to committed numbers is efficient but mixes instruments across
  the comparison; a fresh 9B word probe would make both sides same-run.
- **Gemma-2B is the missing middle.** Its calibrated probe already exists, so adding it would give
  three scales and show whether the aggregate agreement rises monotonically with size (+0.79 at 1B,
  +0.93 at 9B).
