"""E26 — Void cartography estimators, validated on a known void.

Mint a void by injection (ground truth centroid + a nonlinear field read).
Then test whether OUTSIDE measurements recover it:

  Q2 — Bearings triangulation. N rim points (short real prompts) each give a
       true bearing toward the void; least-squares intersection of the bearings
       recovers the centroid. Tests the surveying math. Baseline: the mean rim
       point (no triangulation).

  Q1 — Field extension. A nonlinear scalar field (next-token entropy at the
       anchor) is read at the rim; inverse-distance extension predicts the
       field at the centroid; compared to the void's actual entropy, with a
       leave-one-out error bar. Tests "estimate the reading from the geometry."

Both use a known injected void as calibration ground truth. The blind version
(measuring bearings/field WITHOUT injecting) is the identified next step.

Outputs: results/void_cartography.csv
"""
import random

import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.stimuli import adapter as ad

ANCH = "\nRight now everything feels"
INJECT_FRAC = 0.2
N_RIM = 24


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    L = probe.layer
    net, tok, dev = model.model, model.tokenizer, model.device

    @torch.no_grad()
    def state_and_entropy(text):
        ids = tok(cfg["preamble"] + text + ANCH, return_tensors="pt").to(dev)
        out = net(**ids, output_hidden_states=True)
        st = out.hidden_states[L][0, -1].float().cpu().numpy().astype(np.float64)
        logp = torch.log_softmax(out.logits[0, -1].float(), -1)
        ent = float(-(logp.exp() * logp).sum().cpu())
        return st, ent

    base, _ = state_and_entropy("")

    # mint the void
    g = np.random.RandomState(321)
    dvec = g.randn(base.shape[0]); dvec /= np.linalg.norm(dvec)
    resid = float(np.linalg.norm(base))
    with model.steer(L, dvec.astype(np.float32), INJECT_FRAC * resid):
        t_state, t_entropy = state_and_entropy("")
    centroid = t_state

    # rim: short real prompts scattered on the manifold
    rng = random.Random(11)
    rim_states, rim_ents = [], []
    for _ in range(N_RIM):
        words = [art.word(i) for i in rng.sample(range(len(art.nodes)), rng.randint(2, 6))]
        s, e = state_and_entropy(". ".join(words))
        rim_states.append(s); rim_ents.append(e)
    R = np.stack(rim_states); rim_ents = np.array(rim_ents)

    # --- Q2: bearings triangulation --------------------------------------
    # each rim point r_i has true bearing u_i toward the void; the centroid is
    # the point minimizing sum of squared perpendicular distances to the lines
    # {r_i + s*u_i}. Closed form: sum (I - u u^T) x = sum (I - u u^T) r_i.
    U = (centroid - R); U /= np.linalg.norm(U, axis=1, keepdims=True)
    d = base.shape[0]
    M = np.zeros((d, d)); b = np.zeros(d)
    for r, u in zip(R, U):
        P = np.eye(d) - np.outer(u, u)
        M += P; b += P @ r
    recovered = np.linalg.lstsq(M, b, rcond=None)[0]
    tri_err = float(np.linalg.norm(recovered - centroid))
    meanrim_err = float(np.linalg.norm(R.mean(0) - centroid))

    # --- Q1: inverse-distance field extension ----------------------------
    def idw_predict(query, pts, vals, p=2):
        w = 1.0 / (np.linalg.norm(pts - query, axis=1) ** p + 1e-9)
        return float((w * vals).sum() / w.sum())

    pred_entropy = idw_predict(centroid, R, rim_ents)
    # leave-one-out error bar on the extender (predict each rim from the others)
    loo = []
    for i in range(N_RIM):
        mask = np.arange(N_RIM) != i
        loo.append(idw_predict(R[i], R[mask], rim_ents[mask]) - rim_ents[i])
    loo_rmse = float(np.sqrt(np.mean(np.square(loo))))

    rows = [{
        "centroid_dist_from_base": float(np.linalg.norm(centroid - base)),
        "triangulation_err": tri_err,
        "mean_rim_baseline_err": meanrim_err,
        "triangulation_gain": meanrim_err - tri_err,
        "true_void_entropy": t_entropy,
        "predicted_void_entropy": pred_entropy,
        "field_pred_err": abs(pred_entropy - t_entropy),
        "field_loo_rmse": loo_rmse,
        "rim_entropy_mean": float(rim_ents.mean()),
        "rim_entropy_std": float(rim_ents.std()),
    }]
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/void_cartography.csv", index=False)
    print("Q2 bearings triangulation:")
    print(f"  centroid recovered to {tri_err:.3f} (mean-rim baseline {meanrim_err:.3f}; "
          f"gain {meanrim_err-tri_err:+.3f})")
    print("Q1 field extension (next-token entropy at the void):")
    print(f"  true {t_entropy:.3f}  predicted {pred_entropy:.3f}  "
          f"err {abs(pred_entropy-t_entropy):.3f}  (extender LOO-rmse {loo_rmse:.3f}; "
          f"rim spread {rim_ents.std():.3f})")
    print("\nwrote results/void_cartography.csv")


if __name__ == "__main__":
    main()
