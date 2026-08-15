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
