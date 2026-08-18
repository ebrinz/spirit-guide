"""E30 — Corroborate VAD placement with SAE features (convergent validity).

Two independent readouts of the same states:
  probe  — the linear VAD valence probe (our placement instrument)
  sae    — Gemma-Scope layer-20 features

For a set of poems, read both. Find SAE features whose activation correlates
with the probe valence across poems; build an SAE-only valence estimate
(ridge on the top features) and test whether it agrees with the probe's
valence out-of-sample. Agreement = the placement isn't an artifact of one
instrument. Gemma-2b only (SAE availability).

Outputs: results/sae_corroboration.csv
"""
import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.analysis import sae as S
from spiritbench.analysis.sae import NEURONPEDIA_URL

ANCH = "\nRight now everything feels"


def main():
    cfg = load_config()
    model = HiddenStateModel("unsloth/gemma-2-2b-it", device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data_gemma2b/probe/probe.pkl")
    sae = S.load_sae(hf_hub_download("google/gemma-scope-2b-pt-res",
                                     "layer_20/width_16k/average_l0_71/params.npz"))
    pre = cfg["preamble"]
    L = probe.layer

    stims = [json.loads(l) for l in open(REPO_ROOT / "data/stimuli/stimuli.jsonl")]
    stims = [s for s in stims if s["generator"] == "psg"][:50]

    probe_v, feats = [], []
    for s in stims:
        hs = model.hidden_states(pre + s["text"] + ANCH)
        probe_v.append(float(probe.predict(hs[L][-1:])[0][0]))
        feats.append(S.encode(hs[S.SAE_LAYER][-1].astype(np.float32), sae))
        print(f"{s['constructor']:14s} probe_v {probe_v[-1]:.2f}", flush=True)
    X = np.stack(feats)          # [n, 16384]
    y = np.array(probe_v)

    # correlate each active feature with probe valence
    active = np.where((X > 0).sum(0) >= 5)[0]
    corr = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in active])
    order = active[np.argsort(-np.abs(corr))]
    top = order[:30]

    # SAE-only valence estimate, cross-validated: does SAE predict the probe?
    ridge = Ridge(alpha=1.0)
    yhat = cross_val_predict(ridge, X[:, top], y, cv=5)
    from scipy.stats import pearsonr
    r, p = pearsonr(y, yhat)
    print(f"\nSAE-only valence estimate vs probe valence (5-fold CV): "
          f"r={r:.3f} p={p:.4f} (n={len(y)})")
    print("\ntop probe-valence-correlated SAE features:")
    labs = []
    for i in top[:8]:
        c = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  f{int(i):5d}  corr {c:+.2f}  {NEURONPEDIA_URL.format(idx=int(i))}")
        labs.append({"feature": int(i), "corr_with_probe_valence": float(c)})
    pd.DataFrame({"probe_valence": y, "sae_predicted_valence": yhat}).to_csv(
        REPO_ROOT / "results/sae_corroboration.csv", index=False)
    pd.DataFrame(labs).to_csv(REPO_ROOT / "results/sae_valence_features.csv", index=False)
    print("\nwrote results/sae_corroboration.csv")


if __name__ == "__main__":
    main()
