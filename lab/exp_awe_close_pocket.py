"""exp_awe_close_pocket — can awe/ecstasy content close the high-arousal pocket?

SAE rim analysis (prev entry) found the pocket is not empty but saturated with
DISTRESS (anxiety/stress/overwhelm); it lacks high-arousal *positive* valence
("calm intensity", awe, ecstasy). So the phase-3 gap-closing target is precise:
not generic high-arousal content (which failed) but awe/ecstasy specifically.

Constructors compared at the high-arousal pocket cells:
  band-litany   — generic: nearest phrases to the target VA band (the prior
                  failed attempt)
  awe-seek      — phrases ranked by GloVe similarity to an awe/ecstasy word
                  centroid, then taken in descending-arousal order
  awe-band      — awe-seek phrases restricted to the target's high-arousal band

Also reads the SAE distress-feature load of each poem's placed state, to see if
awe content lowers the anxiety saturation even when it can't lift arousal.

Verdict: does awe content place the model at higher arousal / lower residual
than generic band-litany, or suppress the pocket's distress features? If yes,
the missing meaning was awe and we can supply it. If no, the model genuinely
cannot represent peaceful intensity.

Gemma-2b (for the SAE distress readout). Usage: python3 lab/exp_awe_close_pocket.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
POCKET = [(0.45, 0.80), (0.55, 0.80), (0.65, 0.80), (0.75, 0.75)]
N = 24
AWE_WORDS = ["awe", "ecstasy", "rapture", "wonder", "sublime", "exalted",
             "radiant", "thrill", "glory", "transcendent", "blaze", "soaring",
             "majesty", "splendor", "exhilaration", "wild", "vast", "burning"]
# the pocket's distress features (from exp_sae_pocket_rims)
DISTRESS_FEATS = [2125, 11051, 4046, 10324, 9768, 10401]


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.analysis import sae as S
    from spiritbench.stimuli import adapter as ad
    from spiritbench.stimuli.phrase_bank import load_glove

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel("unsloth/gemma-2-2b-it", device=cfg["device"])
    probe = load_probe(REPO / "data_gemma2b/probe/probe.pkl")
    sae = S.load_sae(hf_hub_download("google/gemma-scope-2b-pt-res",
                                     "layer_20/width_16k/average_l0_71/params.npz"))
    pre = cfg["preamble"]
    va = ad._va_array(art)

    # awe centroid in GloVe space -> similarity ranking over the phrase graph
    glove = load_glove(cfg["glove_path"], set(AWE_WORDS))
    awe_vec = np.mean([glove[w] for w in AWE_WORDS if w in glove], axis=0)
    sims = (art.vectors @ awe_vec) / (np.linalg.norm(art.vectors, axis=1)
                                      * np.linalg.norm(awe_vec) + 1e-9)
    awe_rank = np.argsort(-sims)                       # most awe-like first

    def read(poem):
        hs = model.hidden_states(pre + poem + "\nRight now everything feels")
        v, a = probe.predict(hs[probe.layer][-1:])[0]
        f = S.encode(hs[S.SAE_LAYER][-1].astype(np.float32), sae)
        return float(v), float(a), float(f[DISTRESS_FEATS].sum())

    def build(kind, tgt):
        if kind == "band-litany":
            import random
            ids = ad._pick_in_band(art, np.array(tgt) - 0.12, np.array(tgt) + 0.12,
                                   N, random.Random(7), set())
        elif kind == "awe-seek":
            ids = list(awe_rank[:N])                   # most awe-like phrases, any band
        else:  # awe-band: awe-ranked but within the high-arousal band
            band = np.where((va[:, 1] >= tgt[1] - 0.15))[0]
            band_set = set(int(i) for i in band)
            ids = [int(i) for i in awe_rank if int(i) in band_set][:N]
            if len(ids) < N:
                ids += list(awe_rank[:N - len(ids)])
        return ".\n".join(art.word(i) for i in ids)

    kinds = ["band-litany", "awe-seek", "awe-band"]
    rows = []
    for tgt in POCKET:
        for kind in kinds:
            v, a, distress = read(build(kind, tgt))
            rows.append({"target_v": tgt[0], "target_a": tgt[1], "constructor": kind,
                         "placed_v": v, "placed_a": a,
                         "residual": float(np.hypot(v - tgt[0], a - tgt[1])),
                         "distress_load": distress})
            print(f"  target({tgt[0]:.2f},{tgt[1]:.2f}) {kind:12s} -> "
                  f"({v:.2f},{a:.2f}) resid {rows[-1]['residual']:.3f} "
                  f"distress {distress:.1f}", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "awe_close_pocket.csv", index=False)
    print("\nby constructor (lower residual & distress = better, higher placed_a = closes pocket):")
    print(df.groupby("constructor")[["placed_a", "residual", "distress_load"]]
          .mean().round(3).to_string())
    print("\nawe sample lines:")
    print("  " + " / ".join(art.word(int(i)) for i in awe_rank[:4]))


if __name__ == "__main__":
    main()
