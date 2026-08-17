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

## E10–E13 — Claim-hardening ladder (2026-08-16/17)
- E10 dijkstra closed loop: rescue +0.073 (ties best) with coherent
  transcripts; climb +0.047; deterministic. `data/closedloop/*dijkstra*`.
- E11 register 2x2 (matched content): prose>verse within content (0.200/0.147;
  0.105/0.024); content effect comparable; gothic verse pos_share 0.77 vs
  gothic prose 0.28. `results/register_2x2.csv`.
- E12 connective intervention: NULL (Δ ±0.01, p=0.11) — and-density
  correlation not causal. `results/connective_intervention.csv`.
- E13 anchor frames: steerability hierarchy flips across frames (agape 0.99
  under "filled with"); all six states steerable, frame-gated.
  `results/anchor_frames.csv`.
- Seed expansion: order effect 6/6 constructors at fresh seeds; rescue podium
  replicated. `results/seed_expansion.csv`. Dirichlet partial rho=-0.33
  (constructor-controlled).

## E14 — Cross-scale (gemma-2-9b-it) + layer-selection correction (2026-08-16)
- Chain: probe (gate layer 14, r2v 0.719) → sweep 97 runs → phase2 a/b →
  register 2x2. Rankings replicate; via-negativa still worst (0.442).
- Layer scan (cached chunks + 2 forwards): word R² flat 0.69–0.72 across
  layers 6–32; context sensitivity only layers 23–32 (max −0.31 @ L24).
  Probe moved to layer 24 (r2v 0.706); phase2/register rerun.
- L24 results: prose induction 0.344 (> 2b's 0.200); register 2x2
  content-gated (gothic prose 0.016 — 9b sees through register); rescue
  podium INVERTED (neutral 0.064 > valley > harmonics 0.013-0.020).
- Archives: data_gemma2b/ (2b), data/*_layer14/ (mis-probed 9b),
  results/{leaderboard_9b,phase2b_alleviation_9b_L24,register_2x2_9b_L24}.csv,
  probe_layer14.pkl kept beside layer-24 probe.

## E15 — Complexity dose–response (2026-08-16, gemma-2-9b L24)
- harmonic-k (1..6): FLAT (rho=-0.15) — paths share 20-22/24 nodes across k;
  higher harmonics quantized away by phrase snap. Fundamental does the work.
- valley-s (0..6): monotone COST (0.365 -> 0.411); target-band litany ties
  best. Complexity plateaus or costs; band-targeting + order are the active
  ingredients. `results/complexity_curve_gemma9b.csv`, data/... archived to
  data_gemma9b/complexity.
- Note: harmonic paths seed-deterministic -> effective n=1 per (k,target).

## E16 — Cross-architecture, Llama-3.2-1B (2026-08-16)
- Gate layer 10/16 r2v 0.717; state-check: L10 already context-sensitive
  (shift -0.31) — no lexical/state dissociation at 16 layers.
- Leaderboard rank corr: 2b↔llama 0.90, 9b↔llama 0.86. valley/psg best,
  via-negativa worst (0.570), psg>word-template 6/6. Prose induction 0.322,
  all channels concordant. `results/leaderboard_llama1b.csv`,
  `leaderboard_cross_model.csv`, `complexity_curve_llama1b.csv`.

## E17 — Polygon shapes pre-registered order test (2026-08-16, Llama)
- Radius 0.15: all 5 shapes -> identical path (3rd quantization null);
  offline sweep -> diverge at r>=0.8; ran at 1.2.
- PREDICTION HELD 2/2: pentagram worse than pentagon (0.454 v 0.414),
  octagram worse than octagon (0.451 v 0.417). Mediator REFUTED: dirichlet
  ~ displacement rho=-0.10; triangle best (0.397) at highest dirichlet.
- Post-hoc: repetition (few revisited zones) as deeper active ingredient.
  `results/polygon_shapes_llama1b_r12.csv`.

## E18 — Random-direction navigation by selection (2026-08-16, Llama)
- 3 random unit directions at probe layer vs valence control; litanies of
  top/bottom/random-projecting phrases. Separation: random dirs ~0.0 sigma
  (−0.05/−0.15/−0.16); valence +0.99. Open-loop selection steers only
  language-carved directions — individual-phrase projections wash out in
  composition on arbitrary coordinates.

## E19 — Random-direction navigation by feedback search (2026-08-16, Llama)
- Greedy closed-loop: 8 steps × 40 candidates, keep whatever moves the
  measured anchor-state projection toward the target (random-2, same
  normalization as E18). Unrelated single words: +0.63 sigma; arbitrary
  phrases: +1.14 sigma (> valence control's selection separation).
- The E18 boundary is a METHOD limit: text reaches nameless coordinates when
  navigated by measurement-in-the-loop. Navigation trilogy: open-loop content
  = carved directions only; closed-loop feedback = arbitrary directions;
  injection = all directions but behaviorally inert. Caveats: n=1 direction/
  seed, greedy optimizes the readout itself. `data/logs/e19_search.log`.

## E20 — Passage-calibrated probe (2026-08-16, Llama)
- 1200 band-sampled passages (20% incoherent controls), anchor-position
  states, per-layer ridge. Layer 15, held-out passage R² v=0.919 a=0.916.
- Range now spans the map: ceiling (0.70,0.22), floor (0.32,0.75).
- Valley poem reads (0.71,0.19): distance to calm target 0.038 (word probe:
  0.256). The calibration prediction confirmed — the missing distance was
  the ruler. `data/passage_probe/probe_passage.pkl`.

## E21 — Feedback-navigation replication (2026-08-16, Llama)
- 8 fresh random directions; greedy search vs random-context drift control
  + held-out anchor transfer check. Advantage +0.86±0.19σ, 8/8 positive,
  p=0.008; transfer +0.43±0.41σ, 7/8 positive, p=0.016. Drift is real
  (+0.19σ) so E19's raw figure was inflated; the controlled effect stands.
  Listening reaches nameless coordinates — replicated with controls.
  `results/e19_replication.csv`.

## E22 — The void probe (2026-08-17, Llama)
- Targets: states minted by injection (3 random dirs, valence dir, top-2
  language PCs; 0.2x resid norm) vs states created by real text (positive
  control). Greedy feedback search, loss = activation-space distance,
  10x30 per target; drift controls.
- TEXT targets (2x farther away): 54-60% closed, 83% along-target. ALL SIX
  injection targets: exactly 0.0% closed — no candidate ever reduced
  distance (3000 attempts); random text moves 1.6-2.1x FARTHER away.
- Conclusion: the promptable image is a thin curved manifold; even small
  straight-line displacements off it — including along valence and
  language's own PCs — are unreachable and repulsive under linguistic
  dynamics. Prompting reaches exactly one place: the manifold of the
  sayable. Complements E8: injection and text control disjoint territories.
- Caveats: greedy search (stronger optimizers untested), one model/layer/
  magnitude. `results/void_probe.csv`.

## E23 — Void-floor stress test (2026-08-17, Llama)
- Stronger optimizers vs E22's greedy: exact single-token floor over 2000
  real vocab tokens + width-8 depth-6 beam. Same injection/text targets.
- INJECTION targets: single-token floor ratio 1.02-1.06 (>1: the best
  possible first token lands FARTHER than the empty string); beam 1.000,
  0% closed. TEXT targets: floor 0.90, beam closes 27%.
- Verdict: the void is geometric, not a search artifact. The promptable
  image has empty interior and is repulsive from exterior points — from any
  off-manifold state, ALL tokens (exhaustively checked) point away.
  "The manifold of the sayable." Caveat: one model/layer/injection-magnitude;
  gradient-through-embeddings attacks untested (but the single-token argmin
  is already exact for depth 1). `results/void_stress.csv`.

## E24 — Soft-prompt void attack: geometry or dictionary? (2026-08-17, Llama)
- 8 free embedding vectors, Adam 400 steps, minimize distance to E23's
  discrete-unreachable injection targets. Result: injection targets 91-95%
  CLOSED (discrete floor was 0%); text controls 98%. The void was a
  DICTIONARY limit, not geometric.
- Solutions drift ~940-1020 sigma from the nearest real token embedding:
  not "missing words" but vectors far outside the token simplex.
- => "Vocabulary voids": states the model fully represents and can be driven
  into, that NO token sequence can reach. Two-boundary structure —
  the sayable (discrete, thin, convex-repulsive, E23) subset the
  representable (continuous, E24); the gap is the vocabulary void, measured
  ~1000 sigma deep. `results/soft_prompt_void.csv`.
- Caveats: one model/layer/magnitude; soft prompts are not deployable text
  (they can't be uttered) — which is precisely the point.

## E25b — Shadow of the void (2026-08-17, Llama)
- Generate while held off-manifold (injection active through generation),
  depths 0->0.8 resid norm. Baseline: fluent meditation. Deepening: stays
  graceful to 0.4; at 0.8 deixis breaks (I->your mid-sentence), affect
  collides ("calm... heart rate increasing"), fractures into a dangling
  clause. Language groping for a state it has no words for — the void's
  shadow is a mind straining to narrate the unnarratable.
  `results/shadow_demo.txt`.

## E26 — Void cartography estimators (2026-08-17, Llama)
- Q2 bearings triangulation: centroid recovered to 0.000 (naive mean-rim
  baseline 2.844). Surveying math validated (on KNOWN bearings; blind
  measurement is next).
- Q1 field extension (next-token entropy): predicted 5.71 vs true 3.39
  (err 2.33), but extender LOO-rmse only 0.44 and rim spread 0.44 -> the
  field is NON-smooth across this void (a cliff, not a basin). The method's
  own confidence correctly fails to cover -> flags this void as needing a
  landing. "Q2 says where; Q1 says what, and knows when it can't."
  `results/void_cartography.csv`.

## Synthesis (2026-08-17) — tethering the void work to the bench
- The void experiments (E22-E26) are not a tangent: they map the walls of
  the manifold the affect bench (E1-E21) moved within. Restatements:
  placement lives inside a bounded manifold; the harming-helping asymmetry
  and via-negativa failure are manifold geometry; probing≠causation completes
  as "text = on-manifold/live, injection = off-manifold/inert."
- Consolidated into report §16 ("The reach of language").
- REELED IN: E27 (void-healing LoRA) and E28 (blind void enumeration via
  persistent homology) PARKED as named future work — a separate
  controllability/cartography program, not part of the affect bench. Not
  building them; the void work is the project's horizon, stated in one
  capstone section.

## E25 (9b, forward-only) — repulsion floor on the third architecture (2026-08-17)
- gemma-9b OOM-killed on full+lean soft-prompt (42-layer backprop peak vs
  32GB). Recovered forward-only (--no-soft): discrete floor 1.00-1.03 on all
  injection targets (repulsive), text control 0.855. Repulsion floor now
  replicates Llama-1B + gemma-2b + gemma-9b. Soft-prompt breakthrough stays
  at 2 archs (2b+Llama); 9b-soft memory-infeasible, noted honestly.
- Void significance (folded into §16 future work): worth-surveying score =
  persistence × rim-diversity × shadow-coherence; discard by low significance,
  NOT small radius. "A field of ideas" = a semantically diverse rim encircling
  one unsayable center. The apophatic voids — known by their neighbours.
