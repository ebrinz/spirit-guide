"""Train the NRC VA probe on the listener model. Exits 1 if the validity gate fails."""
import json
import sys
from pathlib import Path

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import collect_word_states, train_probe, save_probe
from spiritbench.stimuli.phrase_bank import load_nrc


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "probe.pkl").exists():
        print("probe.pkl exists — skipping")
        return
    nrc = load_nrc(cfg["nrc_lexicon"])
    words = sorted(nrc)  # ~20k; subsample for tractability
    rng = np.random.RandomState(0)
    words = [words[i] for i in rng.choice(len(words), size=4000, replace=False)]
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    states = collect_word_states(model, words, cfg["probe"]["carrier_templates"])
    v = np.array([nrc[w][0] for w in words])
    a = np.array([nrc[w][1] for w in words])
    probe = train_probe(states, v, a, alpha=cfg["probe"]["ridge_alpha"],
                        test_frac=cfg["probe"]["test_frac"])
    report = {"layer": probe.layer, "r2_v": probe.r2_v, "r2_a": probe.r2_a,
              "n_words": len(words)}
    print(report)
    with open(out_dir / "probe_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if probe.r2_v < cfg["probe"]["r2_gate_valence"]:
        print("PROBE GATE FAILED: valence R2 below gate — do not run the sweep")
        sys.exit(1)
    save_probe(probe, out_dir / "probe.pkl")
    print("probe saved")


if __name__ == "__main__":
    main()
