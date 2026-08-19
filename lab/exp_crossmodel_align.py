"""exp_crossmodel_align — does the shared affect structure survive in FULL
residual space, or only in the 2-D VAD projection?

Paired anchors, for free. The Llama-1B and Gemma-2b passage states were collected
on the SAME 1200 passages (scripts/19 + exp_gemma_passage_probe, both seed=13),
so row i of data/passage_probe/states and row i of data_gemma2b/passage_probe/
states are the SAME stimulus read by the two models — 1200 PAIRED anchors already
on disk. No forward passes needed.

Two questions:
  1. Alignment — can an affine (ridge) map carry one model's residual space into
     the other's? Held-out R2 of Gemma-state -> Llama-state, vs a row-SHUFFLED
     baseline (kills the pairing) to prove the fit is from correspondence, not
     high-dim ridge overfitting. Swept across matched fractional depths.
  2. Affect preservation — map held-out Gemma states into Llama space, apply
     LLAMA's own probe, and predict the passages' true VAD labels. If cross-model
     -mapped VAD predicts the labels near the within-model ceiling, the affect
     axes are shared and linearly inter-translatable, not model-private.

Analysis only (no model loading). Usage: python3 lab/exp_crossmodel_align.py
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
LLAMA = REPO / "data/passage_probe"
GEMMA = REPO / "data_gemma2b/passage_probe"
N = 1200


def load_states(d):
    return np.concatenate([np.load(d / f"states_{s:05d}.npy") for s in range(0, N, 100)])


def align(Xg, Xl, idx_tr, idx_te, shuffle=False):
    """Ridge map Gemma-state -> Llama-state; held-out variance-weighted R2."""
    scg, scl = StandardScaler().fit(Xg[idx_tr]), StandardScaler().fit(Xl[idx_tr])
    Gtr, Gte = scg.transform(Xg[idx_tr]), scg.transform(Xg[idx_te])
    Ltr, Lte = scl.transform(Xl[idx_tr]), scl.transform(Xl[idx_te])
    if shuffle:
        Gtr = Gtr[np.random.RandomState(0).permutation(len(Gtr))]
    W = Ridge(alpha=1e3).fit(Gtr, Ltr)
    return r2_score(Lte, W.predict(Gte), multioutput="variance_weighted")


def main():
    Sl, Sg = load_states(LLAMA), load_states(GEMMA)
    labels = np.load(LLAMA / "labels.npy")            # same passages -> same labels
    assert np.allclose(labels, np.load(GEMMA / "labels.npy")), "label mismatch: not paired!"
    nLl, nLg = Sl.shape[1], Sg.shape[1]
    dl, dg = Sl.shape[2], Sg.shape[2]
    print(f"paired anchors: {len(Sl)}   Llama d={dl} ({nLl} layers)   Gemma d={dg} ({nLg} layers)")

    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.2, random_state=0)

    print("\n1) ALIGNMENT R2 at matched fractional depths (Gemma-state -> Llama-state):")
    print(f"   {'depth':>6} {'llamaL':>7} {'gemmaL':>7} {'aligned':>8} {'shuffled':>9}")
    for frac in (0.25, 0.5, 0.75, 1.0):
        ll = min(int(round(frac * (nLl - 1))), nLl - 1)
        lg = min(int(round(frac * (nLg - 1))), nLg - 1)
        a = align(Sg[:, lg], Sl[:, ll], idx_tr, idx_te)
        s = align(Sg[:, lg], Sl[:, ll], idx_tr, idx_te, shuffle=True)
        print(f"   {frac:>6.2f} {ll:>7d} {lg:>7d} {a:>8.3f} {s:>9.3f}")

    # 2) affect preservation at the two probe layers
    from spiritbench.listener.probe import load_probe
    lp = load_probe(LLAMA / "probe_passage.pkl")
    gp = load_probe(GEMMA / "probe_passage.pkl")
    print(f"\n2) AFFECT PRESERVATION (map Gemma L{gp.layer} -> Llama L{lp.layer}, read with Llama's probe):")
    Xg, Xl = Sg[:, gp.layer], Sl[:, lp.layer]
    scg, scl = StandardScaler().fit(Xg[idx_tr]), StandardScaler().fit(Xl[idx_tr])
    W = Ridge(alpha=1e3).fit(scg.transform(Xg[idx_tr]), scl.transform(Xl[idx_tr]))
    Lhat = scl.inverse_transform(W.predict(scg.transform(Xg[idx_te])))  # Gemma->Llama-space
    pred = lp.predict(Lhat)                              # Llama's ruler on translated states
    within = lp.predict(Xl[idx_te])                      # Llama's ruler on real Llama states (ceiling)
    for j, name in ((0, "valence"), (1, "arousal")):
        r_cross = r2_score(labels[idx_te, j], pred[:, j])
        r_within = r2_score(labels[idx_te, j], within[:, j])
        print(f"   {name:8s}  cross-model-mapped R2 {r_cross:+.3f}   within-Llama ceiling {r_within:+.3f}"
              f"   ({r_cross/r_within*100:.0f}% of ceiling)" if r_within > 0 else "")

    RESULTS.mkdir(exist_ok=True)
    print(f"\n(interpretation) aligned R2 >> shuffled R2 => the 1200 paired anchors carry")
    print(f"shared structure an affine map recovers; high affect-preservation => the")
    print(f"emotional axes are the same axes in both models, up to a linear change of basis.")


if __name__ == "__main__":
    main()
