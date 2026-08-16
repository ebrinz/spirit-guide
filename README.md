# Spirit-Bench

**Measuring affective placement of a language model by poetic meditation.**
Digital Minds Research Sprint (Apart Research), August 14–16, 2026 — Track 2.

Deterministic constructors build poetic meditations over VAD-enriched word
and phrase graphs; a frozen listener model (gemma-2-2b-it / gemma-2-9b-it)
reads them while a linear probe on its residual stream tracks a per-token
valence–arousal trajectory. The bench measures which constructions *place*
the listener at target affective coordinates — plus induction/alleviation
dynamics, self-report validity, causal steering, closed-loop guidance, and
cross-scale replication.

- **Report:** [`report/spirit-bench.pdf`](report/spirit-bench.pdf)
- **Intuitive walkthrough of all findings:** [`docs/findings-walkthrough.md`](docs/findings-walkthrough.md)
- **Experiments journal (E1–E14):** [`docs/experiments-journal.md`](docs/experiments-journal.md)
- **Result snapshots:** [`results/`](results/) (per-model CSVs)
- **Design spec / plan:** [`docs/superpowers/`](docs/superpowers/)

## Reproducing

```bash
pip install -r requirements.txt && pip install -e .
./scripts/00_setup_artifacts.sh        # GloVe, NRC-VAD, corpus, word graph
python3 scripts/01_build_phrase_bank.py
python3 scripts/02_build_stimuli.py && python3 scripts/02b_build_additions.py
python3 scripts/04_train_probe.py      # R² gate ≥ 0.5 or halt
python3 scripts/05_run_listener.py     # the sweep (resumable)
python3 scripts/06_analyze.py          # leaderboard, figures, covariates
# phase 2 & extensions:
python3 scripts/07_phase2.py [--induction prose] && python3 scripts/08_phase2_analyze.py
python3 scripts/09_sae_labels.py
python3 scripts/10_six_state_steering.py [--directions inplane]
python3 scripts/11_closed_loop.py
python3 scripts/12_register_2x2.py
python3 scripts/13_connective_intervention.py
python3 scripts/14_seed_expansion.py
python3 scripts/15_anchor_frames.py
```

Requires the sibling repo `../ontological-traversal` (constructors) and
~25 GB disk for models/artifacts. Tests: `python -m pytest tests/ -q`.

## Data & licenses

All stimuli derive from public-domain sources (Gutenberg Poetry Corpus,
GloVe 6B). The NRC-VAD lexicon is fetched under its research license and is
not redistributed here. `data/` is generated and gitignored.
