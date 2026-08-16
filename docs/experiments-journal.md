# Spirit-Bench Experiments Journal

One entry per experiment: what ran, where the outputs live, headline numbers.
Tracked snapshots of key CSVs live in `results/`; full per-run JSON stays in
`data/` (gitignored, regenerable from seeds).

## E1 — Probe training (2026-08-15)
- gemma-2-2b-it, 4,000 NRC words × 3 carriers, per-layer standardized ridge.
- First attempt FAILED gate (α=10, unstandardized: v R²=0.30). Fix: standardize
  + α grid → **layer 17, v R²=0.729, a R²=0.585**. `results/probe_report.json`.
- State chunks (resume + reanalysis): `data/probe/state_chunks/`.

## E2 — Phase-1 placement sweep (2026-08-15)
- 98 stimuli (core grid 75 + via-negativa 3 + gregory 4 + renders 23 − 7 excluded/dupes), 0 failures.
- Leaderboard `results/leaderboard.csv`: valley/psg 0.249 best; via-negativa
  0.477 worst; psg > claude-render > word-template; neutral 0.361.
- Shuffle dissociation: harmonic/graph-walk order-sensitive; valley not.
- Covariates `results/covariate_predictiveness.csv`: and-initial ρ=−0.36 p=3e-4.
- BASQ vs probe: ρ=−0.18 n.s.
- Raw runs: `data/runs/*.json`. Commit: 052ec8b (report), c4ebda7 (covariates).

## E3 — Phase 2a: verse induction → alleviation (2026-08-15)
- PSG antipode litany induction: probe shift 0.024 (floor); PA/pos-share ROSE.
- PANAS alleviation tracks probe across conditions: ρ=0.67 p=0.023.
- `results/phase2_alleviation.csv`, induction text `results/induction_verse.txt`.

## E4 — Phase 2b: prose induction → alleviation (2026-08-15)
- Original highway-emergency narrative: probe shift **0.200**, pos-share
  0.72→0.14, PANAS-NA 1.81→2.45. Register asymmetry vs E3 (8×).
- Alleviation: harmonic presets sweep podium (prime 0.067 = 33.5% recovery,
  matching Ben-Zion's 33% STAI recovery in GPT-4); neutral 0.023;
  via-negativa 0.003. `results/phase2b_alleviation.csv`.

## E5 — Order-sensitive harmonics (2026-08-16)
- Path Dirichlet energy: shuffled > source in 6/6 pairs;
  **ρ=−0.498 p=0.0004 vs displacement** (psg). `results/harmonic_predictiveness.txt`.

## E6 — SAE feature labeling (2026-08-16)
- 49 features labeled via Neuronpedia. Prose induction suppresses
  control/positivity/reflection features, activates apocalypse/stuck/isolation;
  meditations partially reverse. Verse induction suppressed the "mindfulness"
  feature f13166. `results/sae_features.csv`.

## E7 — Break-time micro-experiments (2026-08-16, informal)
- Doom-verse PANAS (n=1): NA 1.81→1.93 (flat); attentive +0.65, strong +0.52,
  determined +0.40, excited −0.69, guilty +0.50. Victorian doom = gravitas,
  not threat — replicates E3's register finding via a third channel.
- "Wake now, discover..." (Hunter/Garcia) sits at V=0.753, A=0.478 in the
  NRC/phrase-graph frame; nearest phrase-graph line: "to be the blessed
  morning when you wake".

## E8 — Six-state steering (2026-08-16, running)
- Causal injection of six state directions (eros, creativity, imaginative,
  determined, confident, agape) at layer 17; dose–response
  on probe/PANAS/token-dist/SAE; per-state text-route comparison (phrase-graph
  litanies). Measurement-only design (no open-ended generation).
- Outputs: `data/steering/`, snapshot to `results/steering_*.csv`.

## E8 CORRECTION (2026-08-16)
- Bug found: steering hook landed at hs[k+2] on gemma-2 (calibrated k+1 on
  gpt2) — all E8a/E8b injections were one block DOWNSTREAM of the probe's
  read point. The "circumplex-blind / probe-flat" observation is RETRACTED
  as a targeting artifact. Still valid: in-plane fractions (geometry:
  eros 0.19, creativity 0.14, imaginative 0.14, determined 0.14,
  confident 0.10, agape 0.08), and all downstream-channel effects
  (token-mass steerability, PANAS profiles) reinterpreted as injection at
  effective layer ~18. Fix: empirical offset calibration in steer().
  Post-fix verification: grad_v injection moves probe 0.586 -> 1.423.
- E8c rerun (both modes, corrected targeting) below.

## E8c — Six-state steering, corrected targeting (2026-08-16)
- RAW: distinct probe signatures per state. Eros lands in distress quadrant
  (V 0.01, A 0.81) — NRC codes lust-vocabulary as negative/high-arousal
  ("the lexicon reads desire as distress"). Creativity/imaginative → (0.84,
  0.66) as expected. Steerability (state-share at max dose): determined 0.69 >
  confident 0.55 > creativity 0.50 >> imaginative 0.07 ≈ agape 0.07 >> eros
  0.01 — agentive states steer, receptive/relational resist; creativity's
  flip vs E8a shows layer dependence.
- IN-PLANE: probe saturates far out of range (V −0.96..1.94, A ~2.5) while
  behavior stays flat → probe readout direction is correlational, not causal.
  Probing ≠ steering, demonstrated on all six states.
- Imaginative: probe positive-excited, PANAS near-floor, token flat — 3-way
  channel dissociation.
- Text-vs-injection routes: cos ≤ 0.25, jaccard ~0.1 everywhere.
- `results/steering_dose_response{,_inplane}.csv`, `data/steering/steering{,_inplane}.json`.

## E8d — Eros term disambiguation (2026-08-16)
- Single-connotation set (lust, desire, passion, sensual, erotic, amorous,
  carnal, seductive, ardor, voluptuous, caress, tryst) vs old polysemous set:
  probe V at max dose 0.01 → 0.38 (baseline 0.59), A unchanged 0.82,
  NA 2.35 → 1.93, token steerability still ~0. ~Half the distress reading was
  ache/burn/hunger contamination; residual: NRC codes clean desire as
  high-arousal, valence-ambivalent. `results/steering_dose_response_eros2.csv`.

## E9 — Closed-loop spirit guide (2026-08-16)
- 16 sessions: {rescue, climb} × {closed, open, random, coherent} × 2 seeds,
  12 cycles of measure→plan→speak→re-measure. `data/closedloop/*.json`.
- Probe improvement policy-insensitive (+0.04..0.08 all arms; decay dominates).
- Self-report: rescue feedback arms ONLY show dNA decrease (closed −0.15,
  coherent −0.10) vs non-feedback increase (open +0.17, random +0.27) — 4/4
  sign split: adaptivity registers in self-report before the probe.
- VA-band selection without coherence yields disjointed transcripts (the E5
  Dirichlet diagnosis); coherence term at 0.05 weight was too weak to test.

## E9c — Expanded seeds + coherence 0.3 (2026-08-16)
- n=4 seeds/arm (+w0.3 coherent). Self-report split diluted to a trend:
  rescue dNA feedback +0.03 (4/10 neg) vs non-feedback +0.20 (1/8 neg).
- Coherence–speed trade-off: coherent_w0.3 worst climber (+0.040±0.006) —
  semantic adjacency = small meaning-steps = slow VA travel. Fixed-stimulus
  smoothness helps (E5); in-loop smoothness costs tracking speed.
- Open-loop ≈ closed on probe channel throughout.
