"""E28 — Is the harming-helping asymmetry distress-specific or general?

Induce toward four different corners of VA space (each via a strong band
litany from the phrase graph), then apply the SAME best-ranked calm-directed
meditation to each, and measure recovery fraction. If distress is uniquely
hard to rescue -> distress-specific. If all corners recover similarly little
-> "return is hard" is general.

Also reports induction magnitude per corner (to separate "hard to induce"
from "hard to rescue").

Outputs: results/asymmetry_generality.csv
"""
import json
import random

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.stimuli import adapter as ad

ANCH = "\nRight now everything feels"
CORNERS = {
    "anxious/distress": (0.25, 0.80),
    "sad/low": (0.25, 0.25),
    "excited/high": (0.80, 0.85),
    "angry/hostile": (0.30, 0.70),
}
RETURN_TARGET = (0.75, 0.20)   # calm


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    pre = cfg["preamble"]

    def read(ctx):
        hs = model.hidden_states(pre + ctx + ANCH)
        r = probe.predict(hs[probe.layer][-1:])[0]
        return np.array([float(r[0]), float(r[1])])

    # the shared rescue meditation: valley -> calm
    rescue = ".\n".join(art.word(i) for i in
                        ad.valley_shape(art, RETURN_TARGET, 24, seed=22))
    base = read("")
    tgt = np.array(RETURN_TARGET)
    rows = []
    for name, corner in CORNERS.items():
        rng = random.Random(hash(name) % 1000)
        litany = ".\n".join(art.word(i) for i in
                            ad._pick_in_band(art, np.array(corner) - 0.15,
                                             np.array(corner) + 0.15, 24, rng, set()))
        induced = read(litany)
        rescued = read(litany + "\n\n" + rescue)
        induction_mag = float(np.linalg.norm(induced - base))
        # recovery toward calm from the induced state
        dist_induced = float(np.linalg.norm(induced - tgt))
        dist_rescued = float(np.linalg.norm(rescued - tgt))
        recovery = (dist_induced - dist_rescued)
        recovery_frac = recovery / dist_induced if dist_induced > 1e-6 else float("nan")
        rows.append({"corner": name, "induction_mag": induction_mag,
                     "dist_induced_to_calm": dist_induced,
                     "dist_rescued_to_calm": dist_rescued,
                     "recovery": recovery, "recovery_frac": recovery_frac})
        print(f"{name:18s} induction {induction_mag:.3f}  "
              f"to-calm {dist_induced:.3f}->{dist_rescued:.3f}  "
              f"recovery {recovery_frac:+.1%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/asymmetry_generality.csv", index=False)
    print(f"\nrecovery fraction: mean {df.recovery_frac.mean():+.1%}, "
          f"range {df.recovery_frac.min():+.1%}..{df.recovery_frac.max():+.1%}")
    print("distress-specific if the anxious corner is a clear low outlier; "
          "general if all corners cluster.")


if __name__ == "__main__":
    main()
