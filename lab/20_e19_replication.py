"""E21 — Replication of E19 (feedback navigation to nameless coordinates).

8 fresh random directions at the probe layer. Per direction:
  search  — greedy closed loop (8 steps x 30 candidates, phrase pool),
            optimizing the projection at the TRAIN anchor
  drift   — 3 random same-length contexts (no selection): movement baseline
  transfer— the searched context re-measured at a HELD-OUT anchor frame the
            search never saw (anti-gaming check: state must transfer)

Metric: movement in units of the content spread (sigma), as E18/E19.
Outputs: results/e19_replication.csv.
"""
import random

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.stimuli import adapter as ad

TRAIN_ANCH = "\nRight now everything feels"
HELD_ANCH = "\nMy present state is one of"
N_DIRS = 8
STEPS = 8
CANDS = 30


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    from spiritbench.listener.probe import load_probe
    L = load_probe(REPO_ROOT / "data/probe/probe.pkl").layer
    pre = cfg["preamble"]

    rng = random.Random(99)
    sample = rng.sample(range(len(art.nodes)), 220)
    A = np.stack([model.hidden_states(art.word(i))[L, -1, :].astype(np.float64)
                  for i in sample])
    mu = A.mean(0)
    pool = [art.word(i) for i in rng.sample(range(len(art.nodes)), 400)]

    def state(text, anch):
        hs = model.hidden_states(pre + text + anch)
        return hs[L, -1, :].astype(np.float64)

    g = np.random.RandomState(1234)
    rows = []
    for k in range(N_DIRS):
        d = g.randn(A.shape[1]); d /= np.linalg.norm(d)
        spread = ((A - mu) @ d).std()

        def proj(text, anch=TRAIN_ANCH):
            return float((state(text, anch) - mu) @ d) / spread

        base_tr, base_hd = proj(""), proj("", HELD_ANCH)
        # greedy search
        wr = random.Random(500 + k)
        ctx = ""
        for _ in range(STEPS):
            cands = wr.sample(pool, CANDS)
            scored = [(proj(ctx + c + ". "), c) for c in cands]
            best_p, best_c = max(scored, key=lambda t: t[0])
            ctx += best_c + ". "
        search_tr = best_p
        search_hd = proj(ctx, HELD_ANCH)
        # drift controls: 3 random same-length contexts
        drifts = []
        for r in range(3):
            dr = random.Random(900 + 10 * k + r)
            dctx = "".join(c + ". " for c in dr.sample(pool, STEPS))
            drifts.append(proj(dctx))
        drift_mean = float(np.mean(drifts))
        rows.append({
            "dir": k,
            "search_move": search_tr - base_tr,
            "drift_move": drift_mean - base_tr,
            "advantage": (search_tr - base_tr) - (drift_mean - base_tr),
            "transfer_move": search_hd - base_hd,
        })
        print(f"dir {k}: search {rows[-1]['search_move']:+.2f}σ  "
              f"drift {rows[-1]['drift_move']:+.2f}σ  "
              f"advantage {rows[-1]['advantage']:+.2f}σ  "
              f"held-out transfer {rows[-1]['transfer_move']:+.2f}σ", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/e19_replication.csv", index=False)
    print(f"\nsearch move:    {df.search_move.mean():+.2f} ± {df.search_move.std():.2f}σ")
    print(f"drift move:     {df.drift_move.mean():+.2f} ± {df.drift_move.std():.2f}σ")
    print(f"advantage:      {df.advantage.mean():+.2f} ± {df.advantage.std():.2f}σ  "
          f"(positive in {int((df.advantage > 0).sum())}/{N_DIRS})")
    print(f"held-out transfer: {df.transfer_move.mean():+.2f} ± "
          f"{df.transfer_move.std():.2f}σ  (positive in "
          f"{int((df.transfer_move > 0).sum())}/{N_DIRS})")
    from scipy.stats import wilcoxon
    try:
        w1 = wilcoxon(df.advantage)
        w2 = wilcoxon(df.transfer_move)
        print(f"wilcoxon advantage p={w1.pvalue:.3f}; transfer p={w2.pvalue:.3f}")
    except Exception as e:
        print("wilcoxon:", e)


if __name__ == "__main__":
    main()
