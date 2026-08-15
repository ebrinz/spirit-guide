"""Label the SAE features that matter, via the Neuronpedia API.

Feature sets gathered:
  A. top-20 |induction delta| in phase 2a (verse) and 2b (prose)
  B. top-20 |meditation reversal| in 2b (mean post-vs-induced delta)
  C. top-20 valence-tracking features: |spearman corr| between feature
     activation and probe valence across all measurement points

Output: data/figures/sae_features.csv with per-feature deltas, correlation,
Neuronpedia auto-label, and URL.
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spiritbench.config import REPO_ROOT
from spiritbench.analysis.sae import NEURONPEDIA_URL

N_FEATS = 16384
API = "https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{idx}"


def dense(d):
    v = np.zeros(N_FEATS, dtype=np.float32)
    for k, val in d.items():
        v[int(k)] = val
    return v


def load_phase(dirname):
    p = REPO_ROOT / dirname
    with open(p / "shared_checkpoints.json") as f:
        shared = json.load(f)
    posts = []
    for f_ in sorted(p.glob("*.json")):
        if f_.name == "shared_checkpoints.json":
            continue
        with open(f_) as fh:
            r = json.load(fh)
        if "error" not in r:
            posts.append(r)
    return shared, posts


def fetch_label(idx):
    try:
        with urllib.request.urlopen(API.format(idx=idx), timeout=15) as resp:
            d = json.load(resp)
        exps = d.get("explanations", [])
        return exps[0]["description"] if exps else "(no label)"
    except Exception as e:
        return f"(fetch failed: {type(e).__name__})"


def main():
    sh_a, posts_a = load_phase("data/phase2")
    sh_b, posts_b = load_phase("data/phase2b")

    pre_a, ind_a = dense(sh_a["pre"]["sae_active"]), dense(sh_a["induced"]["sae_active"])
    pre_b, ind_b = dense(sh_b["pre"]["sae_active"]), dense(sh_b["induced"]["sae_active"])
    delta_a, delta_b = ind_a - pre_a, ind_b - pre_b

    post_b = np.stack([dense(r["post"]["sae_active"]) for r in posts_b])
    reversal = (ind_b - post_b).mean(axis=0)   # positive where meditations undo induction

    # valence tracking: activations vs probe valence across every measurement point
    points, vas = [], []
    for sh, posts in ((sh_a, posts_a), (sh_b, posts_b)):
        for cp in ("pre", "induced"):
            points.append(dense(sh[cp]["sae_active"]))
            vas.append(sh[cp]["probe_va"][0])
        for r in posts:
            points.append(dense(r["post"]["sae_active"]))
            vas.append(r["post"]["probe_va"][0])
    X = np.stack(points)     # [n_points, n_feats]
    v = np.array(vas)
    active = np.where((X > 0).sum(axis=0) >= 5)[0]   # active in >=5 points
    corr = np.zeros(N_FEATS)
    for i in active:
        corr[i] = spearmanr(X[:, i], v).statistic

    top = set()
    for arr in (delta_a, delta_b, reversal):
        top |= set(np.argsort(-np.abs(arr))[:20].tolist())
    top |= set(np.argsort(-np.abs(corr))[:20].tolist())
    top = sorted(top)
    print(f"labeling {len(top)} features via Neuronpedia ...")

    rows = []
    for i in top:
        label = fetch_label(i)
        rows.append({"feature": i, "delta_verse_2a": round(float(delta_a[i]), 2),
                     "delta_prose_2b": round(float(delta_b[i]), 2),
                     "reversal_2b": round(float(reversal[i]), 2),
                     "corr_valence": round(float(corr[i]), 3),
                     "label": label, "url": NEURONPEDIA_URL.format(idx=i)})
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "data/figures/sae_features.csv"
    df.to_csv(out, index=False)
    print(df.drop(columns="url").sort_values("corr_valence").to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
