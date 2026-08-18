"""exp_sae_pocket_rims — label the rim of the high-arousal pocket with SAE
features, alongside the VAD probe.

The VAD probe locates a pocket (high residual = un-reachable) but only in 2-D.
Gemma-Scope SAE features add an interpretable readout: place the model at each
grid cell, read the probe VAD (which cells are pocket vs reachable rim) AND the
layer-20 SAE feature vector (what concepts are active). Then contrast the SAE
signature of the rim (reachable cells bordering the pocket) against the pocket
interior. Features present at the rim but suppressed in the pocket are the
*missing meanings* of the pocket — candidate content to inject to close it.

Gemma-2b only (Gemma-Scope availability).

Verdict criterion: a ranked, Neuronpedia-labelled list of features that
distinguish the reachable rim from the un-reachable pocket. Coherent, affect-
relevant labels => the pocket has an interpretable "missing concept"; noise =>
it is a probe-range artifact, not a semantic hole.

Usage: python3 lab/exp_sae_pocket_rims.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
GRID = 7
V_RANGE = (0.25, 0.85)
A_RANGE = (0.20, 0.80)
POCKET_PCTL = 70
N = 24


def label(idx):
    try:
        u = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{idx}"
        d = json.load(urllib.request.urlopen(u, timeout=12))
        e = d.get("explanations", [])
        return e[0]["description"] if e else "(no label)"
    except Exception:
        return "(label unavailable)"


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.analysis import sae as S
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel("unsloth/gemma-2-2b-it", device=cfg["device"])
    probe = load_probe(REPO / "data_gemma2b/probe/probe.pkl")
    sae = S.load_sae(hf_hub_download("google/gemma-scope-2b-pt-res",
                                     "layer_20/width_16k/average_l0_71/params.npz"))
    pre = cfg["preamble"]

    vs = np.linspace(*V_RANGE, GRID)
    as_ = np.linspace(*A_RANGE, GRID)
    rows, feats = [], []
    for iv, v in enumerate(vs):
        for ia, a in enumerate(as_):
            ids = ad.valley_shape(art, (float(v), float(a)), N, seed=7)
            poem = ".\n".join(art.word(i) for i in ids)
            hs = model.hidden_states(pre + poem + "\nRight now everything feels")
            pv, pa = probe.predict(hs[probe.layer][-1:])[0]
            resid = float(np.hypot(pv - v, pa - a))
            f = S.encode(hs[S.SAE_LAYER][-1].astype(np.float32), sae)
            rows.append({"iv": iv, "ia": ia, "target_v": float(v), "target_a": float(a),
                         "residual": resid})
            feats.append(f)
            print(f"  cell({iv},{ia}) A={a:.2f} resid {resid:.3f}", flush=True)
    df = pd.DataFrame(rows)
    X = np.stack(feats)

    thresh = np.percentile(df.residual, POCKET_PCTL)
    df["is_pocket"] = df.residual >= thresh
    # rim = reachable cells 4-adjacent to a pocket cell
    pocket_set = set(zip(df[df.is_pocket].iv, df[df.is_pocket].ia))
    rim_idx = []
    for k, r in df.iterrows():
        if r.is_pocket:
            continue
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r.iv + dy, r.ia + dx) in pocket_set:
                rim_idx.append(k); break
    pocket_idx = df.index[df.is_pocket].tolist()
    print(f"\npocket cells: {len(pocket_idx)}, rim cells: {len(rim_idx)}")

    rim_mean = X[rim_idx].mean(0)
    pocket_mean = X[pocket_idx].mean(0)
    diff = rim_mean - pocket_mean          # + = present at rim, suppressed in pocket
    top_rim = np.argsort(-diff)[:12]       # concepts the pocket LACKS
    top_pocket = np.argsort(diff)[:8]      # concepts unique to the pocket

    out = []
    print("\n=== features PRESENT at the rim but SUPPRESSED in the pocket (the missing meanings) ===")
    for i in top_rim:
        lab = label(int(i))
        print(f"  f{int(i):5d}  Δ+{diff[i]:.2f}  {lab}")
        out.append({"feature": int(i), "delta_rim_minus_pocket": float(diff[i]),
                    "kind": "rim", "label": lab})
    print("\n=== features unique to the pocket interior ===")
    for i in top_pocket:
        lab = label(int(i))
        print(f"  f{int(i):5d}  Δ{diff[i]:.2f}  {lab}")
        out.append({"feature": int(i), "delta_rim_minus_pocket": float(diff[i]),
                    "kind": "pocket", "label": lab})
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "sae_pocket_cells.csv", index=False)
    pd.DataFrame(out).to_csv(RESULTS / "sae_pocket_rim_features.csv", index=False)
    print(f"\nwrote {RESULTS}/sae_pocket_rim_features.csv")


if __name__ == "__main__":
    main()
