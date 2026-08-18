"""exp_passage_probe_pocket — is the high-arousal pocket an instrument artifact
(probe range shrinkage) or a genuine model attractor?

The pocket arc concluded the high-arousal ceiling (placed_a ~0.47 against
targets of 0.75-0.80) is structural. But exp_close_arousal_pocket left one
confound open: the word-trained probe's arousal ridge may simply be unable to
OUTPUT high values (ruler shrinkage, the same instrument artifact that capped
valence at ~0.57 in E20). E20's fix was a passage-calibrated probe
(scripts/19_passage_probe.py): trained on whole passages read at the deployment
pathway (anchor final token) and labeled by NRC-mean, so its ruler reaches
nominal targets the word probe cannot.

Test. Re-read the SAME 7x7 valley-poem placements with BOTH probes off a SINGLE
forward pass per cell — identical hidden states, only the ruler differs. This is
the clean isolation: if the pocket is shrinkage, swapping to the calibrated
ruler collapses the high-arousal residual and lifts placed_a toward target. If
the pocket is a genuine model attractor, the poems still cannot be pushed high
even with a well-calibrated ruler, and the residual holds under both.

Verdict criterion (focus on the high-arousal band, target A >= 0.65):
  - passage placed_a rises toward target AND residual collapses  => INSTRUMENT
    artifact; the structural conclusion must be softened.
  - passage placed_a still stuck ~0.47, residual holds           => genuine
    MODEL attractor; the structural conclusion stands.

Llama-1B, forward-only. Usage: python3 lab/exp_passage_probe_pocket.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
GRID = 7
V_RANGE = (0.25, 0.85)
A_RANGE = (0.20, 0.80)
N_LINES = 24
HI_A = 0.65               # "high-arousal band" cutoff for the verdict


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    word = load_probe(REPO / "data/probe/probe.pkl")
    pas = load_probe(REPO / "data/passage_probe/probe_passage.pkl")
    pre = cfg["preamble"]
    print(f"word probe layer {word.layer}  |  passage probe layer {pas.layer} "
          f"(held-out r2_v={pas.r2_v:.3f} r2_a={pas.r2_a:.3f})", flush=True)

    vs = np.linspace(*V_RANGE, GRID)
    as_ = np.linspace(*A_RANGE, GRID)
    rows = []
    for iv, v in enumerate(vs):
        for ia, a in enumerate(as_):
            ids = ad.valley_shape(art, (float(v), float(a)), N_LINES, seed=7)
            poem = ".\n".join(art.word(i) for i in ids)
            hs = model.hidden_states(pre + poem + "\nRight now everything feels")
            wv, wa = word.predict(hs[word.layer][-1:])[0]
            pv, pa = pas.predict(hs[pas.layer][-1:])[0]
            rows.append({"iv": iv, "ia": ia, "target_v": float(v), "target_a": float(a),
                         "placed_v_word": float(wv), "placed_a_word": float(wa),
                         "placed_v_pass": float(pv), "placed_a_pass": float(pa),
                         "resid_word": float(np.hypot(wv - v, wa - a)),
                         "resid_pass": float(np.hypot(pv - v, pa - a))})
            print(f"  cell({iv},{ia}) tgt({v:.2f},{a:.2f})  "
                  f"word->({wv:.2f},{wa:.2f}) r{rows[-1]['resid_word']:.2f}   "
                  f"pass->({pv:.2f},{pa:.2f}) r{rows[-1]['resid_pass']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "passage_probe_pocket.csv", index=False)

    hi = df[df.target_a >= HI_A]
    print(f"\n=== overall (n={len(df)}) ===")
    print(f"  mean residual   word {df.resid_word.mean():.3f}   pass {df.resid_pass.mean():.3f}")
    print(f"\n=== high-arousal band, target A>={HI_A} (n={len(hi)}) ===")
    print(f"  mean residual   word {hi.resid_word.mean():.3f}   pass {hi.resid_pass.mean():.3f}")
    print(f"  mean placed_a   word {hi.placed_a_word.mean():.3f}   pass {hi.placed_a_pass.mean():.3f}"
          f"   (targets {hi.target_a.min():.2f}-{hi.target_a.max():.2f})")
    print(f"  max  placed_a   word {hi.placed_a_word.max():.3f}   pass {hi.placed_a_pass.max():.3f}")

    lift = hi.placed_a_pass.mean() - hi.placed_a_word.mean()
    drop = hi.resid_word.mean() - hi.resid_pass.mean()
    reach = hi.placed_a_pass.mean() / hi.target_a.mean()
    print(f"\n  passage ruler lifts high-A placement by {lift:+.3f}, "
          f"cuts residual by {drop:+.3f}")
    print(f"  passage placed_a reaches {reach*100:.0f}% of mean high-A target")
    if reach >= 0.85 and drop > 0.10:
        print("  VERDICT: INSTRUMENT artifact — calibrated ruler reaches the band; "
              "soften the structural claim.")
    elif hi.placed_a_pass.mean() < 0.60:
        print("  VERDICT: genuine MODEL attractor — even the calibrated ruler "
              "cannot place the poems high; structural conclusion stands.")
    else:
        print("  VERDICT: PARTIAL — ruler helps but does not fully reach; "
              "pocket is part instrument, part model.")
    print(f"\nwrote {RESULTS}/passage_probe_pocket.csv")


if __name__ == "__main__":
    main()
