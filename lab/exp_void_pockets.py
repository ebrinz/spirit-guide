"""exp_void_pockets — do small Llama and small Gemma share the same
un-reachable *pockets* of the VAD ontology?

Idea. A "pocket" is not a tiny interstitial gap in activation space (those are
model-specific and incomparable across architectures). A pocket is a region of
the shared VAD / meaning map that the model *cannot be linguistically placed
into*: grid the emotional map, try to place the model at each cell with a
constructed poem, and read the residual (how far it actually landed from the
target). Contiguous high-residual regions are pockets. Because both models are
scored against the *same* target grid, their pocket maps are directly
comparable.

Verdict criterion:
  - phase 1 (per model): a residual surface over the VAD grid, with contiguous
    high-residual pockets identified (flood-fill), not isolated cells.
  - phase 2 (--correlate): Spearman of per-cell residual between Llama and
    Gemma; shared pockets = cells high-residual in both. Positive correlation
    + overlapping pockets => the ontology has model-general holes.

Anchored to VAD (the probe's 2-D readout), so this maps holes in the *emotional*
projection of the ontology; full-meaning-space pockets are future work.

Usage:
  python3 lab/exp_void_pockets.py --model unsloth/Llama-3.2-1B-Instruct --probe data/probe/probe.pkl --tag llama1b
  python3 lab/exp_void_pockets.py --model unsloth/gemma-2-2b-it        --probe data_gemma2b/probe/probe.pkl --tag gemma2b
  python3 lab/exp_void_pockets.py --correlate llama1b gemma2b
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
GRID = 7                 # 7x7 target cells across the reachable map
V_RANGE = (0.25, 0.85)
A_RANGE = (0.20, 0.80)
POCKET_PCTL = 70         # a cell is "hard" if its residual is above this percentile
N_LINES = 24


def cells():
    vs = np.linspace(*V_RANGE, GRID)
    as_ = np.linspace(*A_RANGE, GRID)
    for iv, v in enumerate(vs):
        for ia, a in enumerate(as_):
            yield iv, ia, float(v), float(a)


def flood_pockets(resid, thresh):
    """4-connected components of cells with residual >= thresh."""
    hard = resid >= thresh
    seen = np.zeros_like(hard, dtype=bool)
    pockets = []
    for i in range(hard.shape[0]):
        for j in range(hard.shape[1]):
            if hard[i, j] and not seen[i, j]:
                stack, comp = [(i, j)], []
                seen[i, j] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hard.shape[0] and 0 <= nx < hard.shape[1] \
                                and hard[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if len(comp) >= 2:          # a pocket, not a lone cell
                    pockets.append(comp)
    return pockets


def map_model(model_id, probe_path, tag):
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(model_id, device=cfg["device"])
    probe = load_probe(REPO / probe_path)
    pre = cfg["preamble"]

    rows = []
    for iv, ia, v, a in cells():
        ids = ad.valley_shape(art, (v, a), N_LINES, seed=7)
        poem = ".\n".join(art.word(i) for i in ids)
        hs = model.hidden_states(pre + poem + "\nRight now everything feels")
        pv, pa = probe.predict(hs[probe.layer][-1:])[0]
        resid = float(np.hypot(pv - v, pa - a))
        rows.append({"iv": iv, "ia": ia, "target_v": v, "target_a": a,
                     "placed_v": float(pv), "placed_a": float(pa), "residual": resid})
        print(f"  cell({iv},{ia}) target({v:.2f},{a:.2f}) -> ({pv:.2f},{pa:.2f}) resid {resid:.3f}",
              flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"pockets_{tag}.csv", index=False)

    resid = df.pivot(index="iv", columns="ia", values="residual").values
    thresh = np.percentile(resid, POCKET_PCTL)
    pockets = flood_pockets(resid, thresh)
    print(f"\n[{tag}] residual mean {resid.mean():.3f}, "
          f"pocket threshold(p{POCKET_PCTL}) {thresh:.3f}, "
          f"{len(pockets)} pockets (>=2 contiguous hard cells)")
    for k, comp in enumerate(pockets):
        vv = [df[(df.iv == y) & (df.ia == x)].target_v.iloc[0] for y, x in comp]
        aa = [df[(df.iv == y) & (df.ia == x)].target_a.iloc[0] for y, x in comp]
        print(f"  pocket {k}: {len(comp)} cells, centered ~(V {np.mean(vv):.2f}, A {np.mean(aa):.2f})")
    print(f"wrote {RESULTS}/pockets_{tag}.csv")


def correlate(tag_a, tag_b):
    from scipy.stats import spearmanr
    a = pd.read_csv(RESULTS / f"pockets_{tag_a}.csv")
    b = pd.read_csv(RESULTS / f"pockets_{tag_b}.csv")
    m = a.merge(b, on=["iv", "ia"], suffixes=(f"_{tag_a}", f"_{tag_b}"))
    r, p = spearmanr(m[f"residual_{tag_a}"], m[f"residual_{tag_b}"])
    print(f"per-cell residual correlation {tag_a} vs {tag_b}: spearman r={r:.3f} p={p:.4f} (n={len(m)})")
    # shared pockets: cells hard in BOTH
    ta = np.percentile(m[f"residual_{tag_a}"], POCKET_PCTL)
    tb = np.percentile(m[f"residual_{tag_b}"], POCKET_PCTL)
    shared = m[(m[f"residual_{tag_a}"] >= ta) & (m[f"residual_{tag_b}"] >= tb)]
    tvcol, tacol = f"target_v_{tag_a}", f"target_a_{tag_a}"
    print(f"shared hard cells (top-30% in both): {len(shared)}")
    if len(shared):
        print(f"  centered ~(V {shared[tvcol].mean():.2f}, A {shared[tacol].mean():.2f})")
        for _, row in shared.iterrows():
            print(f"    (V {row[tvcol]:.2f}, A {row[tacol]:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--probe")
    ap.add_argument("--tag")
    ap.add_argument("--correlate", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()
    if args.correlate:
        correlate(*args.correlate)
    else:
        map_model(args.model, args.probe, args.tag)


if __name__ == "__main__":
    main()
