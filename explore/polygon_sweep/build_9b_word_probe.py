"""build_9b_word_probe — a word-trained probe for gemma-2-9b-it.

The seed sweep needs BOTH readouts on both models, same run. Llama has both
(`data/probe/probe.pkl`, layer 10, and `data/passage_probe/`). 9B had only the calibrated one:
`explore/gemma9b_check` borrowed the EMA side from the committed `results/leaderboard_9b.csv`,
which is fine for the canonical seed-42 stimuli but cannot score fresh seeds.

Same recipe as `scripts/04_train_probe.py` (4000 NRC words, the config's carrier templates,
standardized ridge per layer, layer by held-out valence R2), writing to `data_gemma9b/probe/`
instead of `data/probe/`. Checkpointed every 250 words.

NOTE: this is a FRESH instrument, not the one behind the published 9B numbers. The journal
(E14, `docs/experiments-journal.md`) records that the published 9B run moved its probe to layer 24
by hand after a layer scan, having gated at layer 14. Layer choice here is whatever held-out R2
picks, so the sweep tests the same instrument CLASS (word-trained probe + EMA), not the exact
published probe.

Usage: python3 explore/polygon_sweep/build_9b_word_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import collect_word_states, train_probe, save_probe
from spiritbench.stimuli.phrase_bank import load_nrc

MODEL = "unsloth/gemma-2-9b-it"
OUT = REPO_ROOT / "data_gemma9b/probe"
CHUNK = 250


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "probe.pkl").exists():
        print("probe.pkl exists — skipping")
        return
    nrc = load_nrc(cfg["nrc_lexicon"])
    words = sorted(nrc)
    rng = np.random.RandomState(0)
    words = [words[i] for i in rng.choice(len(words), size=4000, replace=False)]

    chunks = OUT / "state_chunks"
    chunks.mkdir(exist_ok=True)
    cache = {}

    def model():
        if "m" not in cache:
            cache["m"] = HiddenStateModel(MODEL, device=cfg["device"])
        return cache["m"]

    parts = []
    for start in range(0, len(words), CHUNK):
        cpath = chunks / f"chunk_{start:05d}.npy"
        if cpath.exists():
            parts.append(np.load(cpath))
            continue
        s = collect_word_states(model(), words[start:start + CHUNK],
                                cfg["probe"]["carrier_templates"])
        tmp = chunks / f".tmp_{start:05d}.npy"
        np.save(tmp, s)
        tmp.replace(cpath)
        parts.append(s)
        print(f"chunk {start}-{start + len(s)}/{len(words)}", flush=True)
    states = np.concatenate(parts)

    v = np.array([nrc[w][0] for w in words])
    a = np.array([nrc[w][1] for w in words])
    probe = train_probe(states, v, a, alpha=cfg["probe"]["ridge_alpha"],
                        test_frac=cfg["probe"]["test_frac"])
    report = {"model": MODEL, "layer": probe.layer, "r2_v": probe.r2_v,
              "r2_a": probe.r2_a, "n_words": len(words)}
    print(report, flush=True)
    (OUT / "probe_report.json").write_text(json.dumps(report, indent=2))
    if probe.r2_v < cfg["probe"]["r2_gate_valence"]:
        sys.exit("PROBE GATE FAILED: valence R2 below gate")
    save_probe(probe, OUT / "probe.pkl")
    print("probe saved", flush=True)


if __name__ == "__main__":
    main()
