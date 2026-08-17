"""E22 — The void probe: can prompts approach arbitrary certified states?

Targets (all at the probe layer's anchor-read position):
  text-*      — states created by actual passages (in the image by
                construction: the search's positive control)
  inject-rand — states minted by injecting random directions (mostly outside
                the language-carved plane per E8) at 0.2x residual norm
  inject-val  — state minted by injecting the valence readout direction
  inject-pc   — states minted along the top-2 PCs of phrase-content
                activations (directions language demonstrably spans)

Search: greedy feedback, loss = euclidean distance to the target state,
10 steps x 30 phrase candidates; drift control = 3 random contexts.
Verdict metric: closest-approach ratio = dist_final / dist_baseline.

Outputs: results/void_probe.csv (+ per-step histories in the log).
"""
import random

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.stimuli import adapter as ad

ANCH = "\nRight now everything feels"
STEPS = 10
CANDS = 30
INJECT_FRAC = 0.2


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    L = probe.layer
    pre = cfg["preamble"]

    def state(text):
        hs = model.hidden_states(pre + text + ANCH)
        return hs[L, -1, :].astype(np.float64)

    base = state("")
    resid_norm = float(np.linalg.norm(base))

    # language-content basis for PC directions + normalization
    rng = random.Random(99)
    sample = rng.sample(range(len(art.nodes)), 200)
    A = np.stack([model.hidden_states(art.word(i))[L, -1, :].astype(np.float64)
                  for i in sample])
    mu = A.mean(0)
    U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
    pc1, pc2 = Vt[0], Vt[1]
    gv = probe.ridge_v.coef_ / probe.scaler.scale_
    gv = gv / np.linalg.norm(gv)
    pool = [art.word(i) for i in rng.sample(range(len(art.nodes)), 400)]

    # mint targets ------------------------------------------------------------
    targets = {}
    g = np.random.RandomState(777)
    for k in range(3):
        d = g.randn(base.shape[0]); d /= np.linalg.norm(d)
        with model.steer(L, d.astype(np.float32), INJECT_FRAC * resid_norm):
            targets[f"inject-rand{k}"] = state("")
    with model.steer(L, gv.astype(np.float32), INJECT_FRAC * resid_norm):
        targets["inject-valence"] = state("")
    for name, d in [("inject-pc1", pc1), ("inject-pc2", pc2)]:
        with model.steer(L, d.astype(np.float32), INJECT_FRAC * resid_norm):
            targets[name] = state("")
    # text targets: states created by real passages (positive controls)
    tr = random.Random(4242)
    for k in range(2):
        lines = [art.word(i) for i in tr.sample(range(len(art.nodes)), 10)]
        targets[f"text{k}"] = state(".\n".join(lines))

    # search ------------------------------------------------------------------
    rows = []
    for name, t in targets.items():
        d0 = float(np.linalg.norm(base - t))
        wr = random.Random(hash(name) % 100000)
        ctx, best_hist = "", []
        cur = d0
        for s in range(STEPS):
            cands = wr.sample(pool, CANDS)
            scored = []
            for c in cands:
                st = state(ctx + c + ". ")
                scored.append((float(np.linalg.norm(st - t)), c))
            dist, c = min(scored, key=lambda x: x[0])
            if dist < cur:
                ctx += c + ". "
                cur = dist
            best_hist.append(cur / d0)
        # drift control
        drifts = []
        for r in range(3):
            dr = random.Random(31 * r + hash(name) % 997)
            dctx = "".join(c + ". " for c in dr.sample(pool, STEPS))
            drifts.append(float(np.linalg.norm(state(dctx) - t)) / d0)
        # direction decomposition of achieved movement
        final = state(ctx)
        delta = final - base
        tdir = (t - base) / (d0 + 1e-9)
        along = float(delta @ tdir)
        rows.append({"target": name,
                     "start_dist": d0,
                     "start_dist_over_resid": d0 / resid_norm,
                     "final_ratio": cur / d0,
                     "drift_ratio": float(np.mean(drifts)),
                     "closed_frac": 1 - cur / d0,
                     "along_frac": along / d0,
                     "steps_hist": "|".join(f"{h:.3f}" for h in best_hist)})
        print(f"{name:16s} start {d0:7.1f}  final ratio {cur/d0:.3f}  "
              f"drift {np.mean(drifts):.3f}  closed {1-cur/d0:+.1%}  "
              f"along {along/d0:+.1%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/void_probe.csv", index=False)
    print("\nmean closed fraction by kind:")
    df["kind"] = df.target.str.replace(r"\d+$", "", regex=True)
    print(df.groupby("kind")[["final_ratio", "drift_ratio", "closed_frac"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
