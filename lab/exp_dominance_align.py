"""exp_dominance_align — does the DOMINANCE direction transfer cross-model as
cleanly as valence/arousal, or is D more model-private?

exp_crossmodel_align showed V/A are ~100% shared up to a linear map (read at 98%
of the within-Llama ceiling after mapping Gemma->Llama). Dominance was the third
lever that reached the arousal pocket (exp_dominance_axis), and it was ~40%
orthogonal to the V/A plane. Is that orthogonal-D structure ALSO shared between
architectures, or is it where the models diverge?

Method (mirrors the affect-preservation test, adding D). Same 1200 paired passage
anchors. Fit Llama ridge readouts for V, A, AND D at the Llama passage layer.
Fit a state-reconstruction map W: Gemma-state -> Llama-state (labels never seen).
For held-out passages, map Gemma->Llama and read V/A/D with LLAMA's own probes;
R2 vs the true labels, as a fraction of the within-Llama ceiling. If D transfers
at the same % of ceiling as V/A, dominance is equally architecture-invariant; a
lower % means D is more model-private.

Analysis only (no model loading). Usage: python3 lab/exp_dominance_align.py
"""
import random
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
LLAMA = REPO / "data/passage_probe"
GEMMA = REPO / "data_gemma2b/passage_probe"
N = 1200


def load_states(d):
    return np.concatenate([np.load(d / f"states_{s:05d}.npy") for s in range(0, N, 100)])


def d_labels(art, cfg):
    """Per-passage NRC-mean dominance, rebuilt from the seed=13 passages."""
    from spiritbench.stimuli import adapter as ad
    lex = {p[0]: float(p[3]) for p in
           (l.rstrip("\n").split("\t") for l in open(REPO / cfg["nrc_lexicon"])) if len(p) == 4}
    n = len(art.nodes)
    d_node = np.full(n, np.nan)
    for i in range(n):
        vals = [lex[t] for t in re.findall(r"[a-z']+", art.word(i).lower()) if t in lex]
        if vals:
            d_node[i] = np.mean(vals)
    rng = random.Random(13)
    va = ad._va_array(art)
    ids_per, va_lab = [], []
    for k in range(N):
        length = rng.randint(4, 16)
        if k % 5 == 0:
            ids = rng.sample(range(len(va)), length)
        else:
            c = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
            pool = list(np.argsort(np.linalg.norm(va - c, axis=1))[:250])
            ids = rng.sample(pool, min(length, len(pool)))
        ids_per.append(ids)
        va_lab.append(va[ids].mean(0))
    return (np.array([np.nanmean(d_node[ids]) for ids in ids_per]),
            np.array(va_lab))


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    Sl, Sg = load_states(LLAMA), load_states(GEMMA)
    labels = np.load(LLAMA / "labels.npy")
    dlab, va_rebuilt = d_labels(art, cfg)
    assert np.allclose(va_rebuilt, labels, atol=1e-6), "passage rebuild misaligned with saved states"
    lp, gp = load_probe(LLAMA / "probe_passage.pkl"), load_probe(GEMMA / "probe_passage.pkl")
    Ll, Lg = lp.layer, gp.layer
    print(f"paired anchors {N}; reading Gemma L{Lg} -> Llama L{Ll}")

    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.2, random_state=0)
    Xl, Xg = Sl[:, Ll], Sg[:, Lg]
    scl, scg = StandardScaler().fit(Xl[idx_tr]), StandardScaler().fit(Xg[idx_tr])
    Ltr, Lte = scl.transform(Xl[idx_tr]), scl.transform(Xl[idx_te])
    Gtr, Gte = scg.transform(Xg[idx_tr]), scg.transform(Xg[idx_te])

    # Llama readouts for V, A, D (standardized space); D fit here, V/A refit for parity
    targets = {"valence": labels[:, 0], "arousal": labels[:, 1], "dominance": dlab}
    heads = {k: Ridge(alpha=1e3).fit(Ltr, y[idx_tr]) for k, y in targets.items()}

    # state-reconstruction map (labels never seen), then read mapped states
    W = Ridge(alpha=1e3).fit(Gtr, Ltr)
    Lhat = W.predict(Gte)                       # Gemma states, in Llama's standardized space

    print(f"\n{'axis':>10} {'within-ceiling':>15} {'cross-mapped':>13} {'% of ceiling':>13}")
    for k, y in targets.items():
        within = r2_score(y[idx_te], heads[k].predict(Lte))
        cross = r2_score(y[idx_te], heads[k].predict(Lhat))
        pct = 100 * cross / within if within > 0 else float("nan")
        print(f"{k:>10} {within:>15.3f} {cross:>13.3f} {pct:>12.0f}%")

    print("\n(interpretation) if dominance transfers at ~the same % of ceiling as V/A,")
    print("the D axis is as architecture-invariant as valence/arousal; a lower %")
    print("means the orthogonal-D structure is more model-private.")


if __name__ == "__main__":
    main()
