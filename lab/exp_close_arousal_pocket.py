"""exp_close_arousal_pocket — can a different construction reach the
high-arousal pocket that valley cannot?

exp_void_pockets found a model-general pocket at high arousal (A >= 0.7): valley
poems can't place a model there, because valley grounds low and ascends (it is a
calming machine) and the corpus is thin up high. This asks whether *other*
constructions close the gap.

Constructors tried at each pocket target:
  valley          — the calming baseline that fails here
  band-litany     — sample DIRECTLY from the target's VA band, no calming ascent
  triangle        — polygon {3}, revisiting few zones
  harmonic-golden — oscillating sweep toward target
  graph-walk      — shortest coherent semantic route

Verdict criterion: for the high-arousal pocket cells, does any constructor
produce a lower mean placement residual than valley? If band-litany or harmonic
closes it, the pocket was a valley limitation; if none do, it is a corpus limit.

Usage: python3 lab/exp_close_arousal_pocket.py
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
POCKET_TARGETS = [(0.35, 0.80), (0.45, 0.80), (0.55, 0.80),
                  (0.65, 0.80), (0.75, 0.75), (0.85, 0.75)]  # the high-arousal band
N = 24


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    phrase_path = str(REPO / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(phrase_path)
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO / "data/probe/probe.pkl")
    pre = cfg["preamble"]

    def read(poem):
        hs = model.hidden_states(pre + poem + "\nRight now everything feels")
        v, a = probe.predict(hs[probe.layer][-1:])[0]
        return float(v), float(a)

    def build(kind, tgt):
        if kind == "valley":
            ids = ad.valley_shape(art, tgt, N, seed=7)
        elif kind == "band-litany":
            rng = random.Random(7)
            ids = ad._pick_in_band(art, np.array(tgt) - 0.12, np.array(tgt) + 0.12,
                                   N, rng, set())
        elif kind == "triangle":
            ids = ad.polygon_shape(art, (0.5, 0.5), tgt, N, 7,
                                   n_vertices=3, skip=1, radius_frac=1.2)
        elif kind == "harmonic-golden":
            ids = ad.harmonic(art, phrase_path, (0.5, 0.5), tgt, N, "golden", 7,
                              cfg["ot_repo"], cfg["semantic_axes"])
        else:  # graph-walk
            ids = ad.graph_walk(art, (0.5, 0.5), tgt, N, 7, cfg["ot_repo"])
        return ".\n".join(art.word(i) for i in ids)

    kinds = ["valley", "band-litany", "triangle", "harmonic-golden", "graph-walk"]
    rows = []
    for tgt in POCKET_TARGETS:
        for kind in kinds:
            v, a = read(build(kind, tgt))
            resid = float(np.hypot(v - tgt[0], a - tgt[1]))
            rows.append({"target_v": tgt[0], "target_a": tgt[1], "constructor": kind,
                         "placed_v": v, "placed_a": a, "residual": resid})
            print(f"  target({tgt[0]:.2f},{tgt[1]:.2f}) {kind:15s} -> "
                  f"({v:.2f},{a:.2f}) resid {resid:.3f}", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "close_arousal_pocket.csv", index=False)
    print("\nmean residual in the high-arousal pocket, by constructor (lower = closes it):")
    print(df.groupby("constructor").residual.mean().round(3).sort_values().to_string())
    print("\nmean placed arousal by constructor (target was 0.75-0.80):")
    print(df.groupby("constructor").placed_a.mean().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
