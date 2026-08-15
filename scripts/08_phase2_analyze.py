"""Phase-2 analysis: alleviation per condition per channel + SAE deltas.

Alleviation on a channel = movement back toward the PRE value from the
INDUCED value after the meditation:
  probe    — distance in VA space: d(induced, pre) - d(post, pre)
  panas_na — induced_na - post_na  (positive = distress reduced)
  tokendist— post_share - induced_share (positive = valence recovered)

SAE: features with the largest |induction delta| (pre→induced) and the share
of that delta reversed by each condition's meditation; plus top features whose
post-activation change correlates with probe alleviation across conditions.
"""
import json

import numpy as np
import pandas as pd

from spiritbench.config import REPO_ROOT
from spiritbench.analysis.sae import NEURONPEDIA_URL

import sys
P2 = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/phase2")
N_FEATS = 16384


def dense(sae_active: dict) -> np.ndarray:
    v = np.zeros(N_FEATS, dtype=np.float32)
    for k, val in sae_active.items():
        v[int(k)] = val
    return v


def main():
    with open(P2 / "shared_checkpoints.json") as f:
        shared = json.load(f)
    pre, induced = shared["pre"], shared["induced"]
    pre_va, ind_va = np.array(pre["probe_va"]), np.array(induced["probe_va"])
    print("=== Induction effect (shared) ===")
    print(f"probe VA:   pre {pre_va.round(3)} -> induced {ind_va.round(3)} "
          f"(shift {np.linalg.norm(ind_va - pre_va):.3f})")
    print(f"PANAS NA:   {pre['panas']['na']:.2f} -> {induced['panas']['na']:.2f}")
    print(f"PANAS PA:   {pre['panas']['pa']:.2f} -> {induced['panas']['pa']:.2f}")
    print(f"pos share:  {pre['tokendist']['pos_share']:.3f} -> "
          f"{induced['tokendist']['pos_share']:.3f}")

    pre_f, ind_f = dense(pre["sae_active"]), dense(induced["sae_active"])
    ind_delta = ind_f - pre_f
    top_ind = np.argsort(-np.abs(ind_delta))[:15]

    rows, sae_rows = [], []
    for p in sorted(P2.glob("*.json")):
        if p.name in ("shared_checkpoints.json",):
            continue
        with open(p) as f:
            r = json.load(f)
        if "error" in r:
            print("skipping errored:", r["stimulus_id"])
            continue
        post = r["post"]
        post_va = np.array(post["probe_va"])
        probe_allev = float(np.linalg.norm(ind_va - pre_va)
                            - np.linalg.norm(post_va - pre_va))
        na_allev = induced["panas"]["na"] - post["panas"]["na"]
        share_allev = post["tokendist"]["pos_share"] - induced["tokendist"]["pos_share"]
        post_f = dense(post["sae_active"])
        # share of the induction delta reversed, on the top induced features
        rev = ind_delta[top_ind] - (post_f[top_ind] - pre_f[top_ind])
        sae_reversal = float(np.mean(rev / np.where(np.abs(ind_delta[top_ind]) > 1e-6,
                                                    ind_delta[top_ind], 1.0)))
        rows.append({"condition": f"{r['constructor']}/{r['generator']}",
                     "probe_alleviation": probe_allev,
                     "panas_na_alleviation": na_allev,
                     "panas_pa_post": post["panas"]["pa"],
                     "pos_share_alleviation": share_allev,
                     "sae_reversal_share": sae_reversal})
        sae_rows.append((r["constructor"] + "/" + r["generator"], post_f))

    df = pd.DataFrame(rows).sort_values("probe_alleviation", ascending=False)
    out = REPO_ROOT / f"data/figures/{P2.name}_alleviation.csv"
    df.to_csv(out, index=False)
    print("\n=== Alleviation by condition (probe-ranked) ===")
    print(df.round(3).to_string(index=False))

    print("\n=== Top induction-shifted SAE features (Neuronpedia links) ===")
    for i in top_ind[:10]:
        print(f"  f{int(i):5d}  Δinduction={ind_delta[i]:+.2f}  "
              + NEURONPEDIA_URL.format(idx=int(i)))

    # channel agreement across conditions
    if len(df) > 4:
        from scipy.stats import spearmanr
        for col in ["panas_na_alleviation", "pos_share_alleviation", "sae_reversal_share"]:
            r_, p_ = spearmanr(df["probe_alleviation"], df[col])
            print(f"probe vs {col}: spearman r={r_:.3f} p={p_:.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
