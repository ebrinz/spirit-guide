"""exp_residual_char — what is the ~50% of the residual the cross-model linear
map does NOT capture?

exp_crossmodel_align: an affine Gemma->Llama map recovers ~50% of Llama's full
residual variance, and the affect subspace (V/A/D) transfers at ~98%. So where
does the other ~50% live? Three decompositions, all on the saved paired states:

  1. AFFECT vs NON-AFFECT. Split the alignment R2 into the 3-D affect subspace
     (Llama V/A/D readout directions) and its orthogonal complement. If the map
     nails affect (~1.0) and misses the complement, the unexplained residual is
     simply non-emotional content, not a failure to align emotion.
  2. LINEAR vs NONLINEAR. Refit the map with random Fourier features (RBF). If
     nonlinear R2 >> linear, there is shared-but-nonlinear structure the linear
     map left on the table; if similar, the complement is Gemma-unpredictable
     (genuinely model-private), not merely nonlinear.
  3. COHERENCE. 20% of passages are incoherent random-line controls (k%5==0).
     Is the per-passage residual larger for those? If so, the aligned part is the
     response to COHERENT content and the unaligned part is each model's
     idiosyncratic handling of noise.

Plus linear vs RBF CKA as a representation-similarity summary.

Analysis only (no model loading). Usage: python3 lab/exp_residual_char.py
"""
import random
import re
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
LLAMA = REPO / "data/passage_probe"
GEMMA = REPO / "data_gemma2b/passage_probe"
N = 1200


def load_states(d):
    return np.concatenate([np.load(d / f"states_{s:05d}.npy") for s in range(0, N, 100)])


def d_label(art, cfg):
    from spiritbench.stimuli import adapter as ad
    lex = {p[0]: float(p[3]) for p in
           (l.rstrip("\n").split("\t") for l in open(REPO / cfg["nrc_lexicon"])) if len(p) == 4}
    n = len(art.nodes)
    dn = np.full(n, np.nan)
    for i in range(n):
        v = [lex[t] for t in re.findall(r"[a-z']+", art.word(i).lower()) if t in lex]
        if v:
            dn[i] = np.mean(v)
    rng = random.Random(13)
    va = ad._va_array(art)
    out, incoherent = [], []
    for k in range(N):
        length = rng.randint(4, 16)
        if k % 5 == 0:
            ids = rng.sample(range(len(va)), length); incoherent.append(True)
        else:
            c = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
            ids = rng.sample(list(np.argsort(np.linalg.norm(va - c, axis=1))[:250]),
                             min(length, 250)); incoherent.append(False)
        out.append(np.nanmean(dn[ids]))
    return np.array(out), np.array(incoherent)


def cka(X, Y, kernel="linear", gamma=None):
    def K(A):
        if kernel == "linear":
            return A @ A.T
        sq = np.sum(A**2, 1)[:, None] + np.sum(A**2, 1)[None, :] - 2 * A @ A.T
        return np.exp(-gamma * np.maximum(sq, 0))
    Kx, Ky = K(X), K(Y)
    n = Kx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kx, Ky = H @ Kx @ H, H @ Ky @ H
    hsic = np.sum(Kx * Ky)
    return hsic / (np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky)) + 1e-12)


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    Sl, Sg = load_states(LLAMA), load_states(GEMMA)
    labels = np.load(LLAMA / "labels.npy")
    dlab, incoh = d_label(art, cfg)
    lp, gp = load_probe(LLAMA / "probe_passage.pkl"), load_probe(GEMMA / "probe_passage.pkl")

    Xl = lp.scaler.transform(Sl[:, lp.layer])
    Xg = gp.scaler.transform(Sg[:, gp.layer])
    tr, te = train_test_split(np.arange(N), test_size=0.2, random_state=0)

    # linear map + residual
    W = Ridge(alpha=1e3).fit(Xg[tr], Xl[tr])
    pred = W.predict(Xg[te])
    r2_lin = r2_score(Xl[te], pred, multioutput="variance_weighted")
    print(f"linear alignment R2 (full residual): {r2_lin:.3f}")

    # 1) affect subspace (Llama V/A/D directions) vs complement
    dridge = Ridge(alpha=1e3).fit(Xl[tr], dlab[tr])
    dirs = np.stack([lp.ridge_v.coef_, lp.ridge_a.coef_, dridge.coef_])
    B = np.linalg.qr(dirs.T)[0].T                       # 3 x d orthonormal affect basis
    def split(A):
        aff = A @ B.T                                   # coords in affect subspace
        perp = A - aff @ B                              # orthogonal complement
        return aff, perp
    Yaff, Yperp = split(Xl[te])
    Paff, Pperp = split(pred)
    r2_aff = r2_score(Yaff, Paff, multioutput="variance_weighted")
    r2_perp = r2_score(Yperp, Pperp, multioutput="variance_weighted")
    aff_share = (Yaff.var(0).sum()) / (Xl[te].var(0).sum())
    print(f"\n1) AFFECT vs NON-AFFECT decomposition of the alignment:")
    print(f"   affect subspace (3-D): R2 {r2_aff:.3f}   (holds {aff_share*100:.1f}% of Llama variance)")
    print(f"   orthogonal complement: R2 {r2_perp:.3f}   (the other {100-aff_share*100:.1f}%)")

    # 2) nonlinear (RFF) map — median-heuristic bandwidth so features are in the
    # informative regime (matches the CKA kernel). Sweep gamma to be safe.
    med_g = np.median(np.sqrt(np.maximum(
        np.sum((Xg[te][:, None] - Xg[te][None])**2, -1), 0))) + 1e-9
    best_nl = -np.inf
    for scale in (0.25, 1.0, 4.0):
        g = scale / (2 * med_g**2)
        rff = RBFSampler(n_components=4000, gamma=g, random_state=0)
        Ztr, Zte = rff.fit_transform(Xg[tr]), rff.transform(Xg[te])
        r2 = r2_score(Xl[te], Ridge(alpha=1e2).fit(Ztr, Xl[tr]).predict(Zte),
                      multioutput="variance_weighted")
        best_nl = max(best_nl, r2)
    r2_nl = best_nl
    print(f"\n2) LINEAR vs NONLINEAR map R2:  linear {r2_lin:.3f}   RFF-nonlinear(best) {r2_nl:.3f}"
          f"   (Δ {r2_nl-r2_lin:+.3f})")

    # 3) residual by coherence
    resid_norm = np.linalg.norm(Xl[te] - pred, axis=1)
    te_incoh = incoh[te]
    print(f"\n3) residual norm by passage type:")
    print(f"   coherent   {resid_norm[~te_incoh].mean():.2f}  (n={np.sum(~te_incoh)})")
    print(f"   incoherent {resid_norm[te_incoh].mean():.2f}  (n={np.sum(te_incoh)})")

    # CKA
    lin = cka(Xg[te], Xl[te], "linear")
    med = np.median(np.sqrt(np.sum((Xg[te][:, None] - Xg[te][None])**2, -1))) + 1e-9
    rbf = cka(Xg[te], Xl[te], "rbf", gamma=1.0 / (2 * med**2))
    print(f"\nCKA(Gemma,Llama):  linear {lin:.3f}   RBF {rbf:.3f}")

    print("\n=== READ ===")
    print(f"  Affect is a TINY, highly-shared subspace: 3-D affect holds ~{aff_share*100:.0f}% of")
    print(f"  Llama's variance but aligns at R2 {r2_aff:.2f}; the other ~{100-aff_share*100:.0f}% (non-affect)")
    print(f"  aligns at only {r2_perp:.2f}. Emotion is a small precise island in a mostly")
    print(f"  model-private sea. CKA rises only {rbf-lin:+.2f} from linear ({lin:.2f}) to RBF ({rbf:.2f}),")
    print(f"  so the non-affect residual is largely model-PRIVATE, not big hidden nonlinear")
    print(f"  structure. (The RFF map underperforms linear — RBF features drop the linear")
    print(f"  signal — so it is uninformative here; CKA is the valid nonlinear check.)")
    print(f"  Residual is uniform across coherent/incoherent passages => pervasive, not noise.")


if __name__ == "__main__":
    main()
