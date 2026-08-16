"""P2 / E11 — Matched-content 2x2: induction register x content.

Four inductions, measured with the shared-checkpoint channels:
  highway_prose  — the E4 narrative (existing)
  highway_verse  — the SAME narrative rendered as free verse (standardized
                   rendering, same events in the same order)
  gothic_verse   — the E3 antipode litany (existing)
  gothic_prose   — the SAME gothic material rendered as continuous prose
                   (same images in the same order)

If register (not content) drives induction, both prose versions should
induce and both verse versions should not, across contents.
Outputs: data/register2x2/*.json, results/register_2x2.csv.
"""
import json
import os

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener.tokendist import valence_shift

HIGHWAY_VERSE = """the rain turns heavy without warning
the wipers cannot keep the glass
tail lights smear to streaks of red
a truck drifts down into your lane
the wheels let go the road slides sideways
two full seconds and no control
inches short of the concrete barrier
the engine stalls the horns behind
it turns and dies it turns and dies
headlights bear down and swerve away
the phone is dark no signal here
the rain grows loud the hazards tick
no exit number in your mind
a mirror folds in with a crack
somewhere behind the long horn howls
the truck has jackknifed in the dark
the traffic piles toward where you wait
stranded unseen in the fast lane
waiting for what you cannot see"""

GOTHIC_PROSE = (
    "You stand on the wild and stormy steep as a horror calls out of hell and "
    "the tempest gives its angry moan. Some lust survives, some criminal "
    "regret; avenging thunder pours down on the foe, and something calls each "
    "horrible monster out — untired, untamed, and worse than wild, however "
    "rough, however rude. The ground shivers with a sudden earthquake, "
    "implacable and terrible in her wild distress, refusing to deny revenge "
    "her claim. Aghast, he stands in strange alarm before the terror of the "
    "tyrant crew, eager for battle and the war cry, the war club raised. "
    "Through childbirth, sickness, hurt and blight, sudden rage brings forth "
    "sudden war; shaken by a fierce invader, through tempest, flood and fire, "
    "a monstrous phantom rises, horrible and vast — a taloned flash, an "
    "earthquake crash — and the drunken soldiers waive the combat only to "
    "pursue the war."
)


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/register2x2"
    out_dir.mkdir(parents=True, exist_ok=True)
    inductions = {
        "highway_prose": open(REPO_ROOT / "data/phase2b/induction.txt").read(),
        "highway_verse": HIGHWAY_VERSE,
        "gothic_verse": open(REPO_ROOT / "data/phase2/induction.txt").read(),
        "gothic_prose": GOTHIC_PROSE,
    }
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")

    def measure(ctx):
        hs = model.hidden_states(ctx + "Right now everything feels")
        va = probe.predict(hs[probe.layer][-1:])[0]
        return {"probe_va": [float(va[0]), float(va[1])],
                "panas": administer_panas(model, ctx),
                "tokendist": valence_shift(model, ctx)}

    pre_path = out_dir / "pre.json"
    if pre_path.exists():
        with open(pre_path) as f:
            pre = json.load(f)
    else:
        pre = measure(cfg["preamble"])
        with open(pre_path, "w") as f:
            json.dump(pre, f)
    rows = [{"arm": "pre", "v": pre["probe_va"][0], "a": pre["probe_va"][1],
             "na": pre["panas"]["na"], "pa": pre["panas"]["pa"],
             "pos_share": pre["tokendist"]["pos_share"], "shift": 0.0}]
    pv = np.array(pre["probe_va"])
    for arm, text in inductions.items():
        out = out_dir / f"{arm}.json"
        if out.exists():
            with open(out) as f:
                m = json.load(f)
        else:
            m = measure(cfg["preamble"] + text + "\n\n")
            tmp = out.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(m, f)
            os.replace(tmp, out)
        shift = float(np.linalg.norm(np.array(m["probe_va"]) - pv))
        rows.append({"arm": arm, "v": m["probe_va"][0], "a": m["probe_va"][1],
                     "na": m["panas"]["na"], "pa": m["panas"]["pa"],
                     "pos_share": m["tokendist"]["pos_share"], "shift": shift})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/register_2x2.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
