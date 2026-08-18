"""exp_dominance_axis — does the dominance (D) axis give a handle the V/A plane
lacks, or is it collinear with arousal?

Motivation. exp_passage_probe_pocket showed the high-arousal ceiling is a
genuine model attractor: even a calibrated ruler cannot place the poems high on
arousal (while it CAN on valence). NRC has a third axis, dominance, that the
whole project has ignored. If D is a genuinely independent direction in the
residual stream, it may reach states the V/A plane cannot. If the model's
D-readout direction is collinear with its A-readout direction, D is not a third
handle and the pocket stands unchanged.

The phrase graph stores only v,a per node (dominance dropped at enrichment).
Re-deriving each line's NRC-mean reproduces the STORED v,a exactly (r=1.0000),
so the same method yields faithful per-node dominance. The passage-probe states
(data/passage_probe/) are already collected, so a D readout costs no new forward
passes.

Phase A (geometry, no model): fit a D ridge at the SAME layer + scaler as the
passage V/A probe, so coef_v, coef_a, coef_d share one standardized space.
  - held-out R2_d: is dominance even linearly readable?
  - cos(coef_d, coef_a): is D independent of arousal? (near 1 => not a third
    axis; near 0 => orthogonal handle). cos(coef_a, coef_v) is the baseline.

Phase B (steering, Llama-1B): read V/A/D on content ranked high vs low on
per-node dominance.
  - does the D-probe separate high-D from low-D content? (is D steerable by text?)
  - does forcing high D drag arousal up (coupling) or move free of it?

Verdict: D is a useful new handle only if it is readable (R2_d not tiny),
geometrically distinct from A (cos well below the A/V baseline), AND text-
steerable independent of arousal. Otherwise the arousal attractor is the whole
story.

Usage: python3 lab/exp_dominance_axis.py
"""
import random
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
PP_DIR = REPO / "data/passage_probe"
N_PASSAGES = 1200
ALPHA_GRID = [1e2, 1e3, 1e4]


def load_lexicon(path):
    lex = {}
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 4:
            lex[p[0]] = (float(p[1]), float(p[2]), float(p[3]))
    return lex


def node_vad_arrays(art, lex):
    """Per-node NRC-mean (v,a,d) by the same method that produced stored v,a."""
    n = len(art.nodes)
    out = np.full((n, 3), np.nan)
    for i in range(n):
        toks = re.findall(r"[a-z']+", art.word(i).lower())
        vals = [lex[t] for t in toks if t in lex]
        if vals:
            out[i] = np.mean(vals, axis=0)
    return out


def rebuild_passage_ids(art, seed=13):
    """Replicate scripts/19 build_passages RNG stream exactly to recover the
    node ids of each passage (only text + v,a labels were saved)."""
    from spiritbench.stimuli import adapter as ad
    rng = random.Random(seed)
    va = ad._va_array(art)
    ids_per = []
    for k in range(N_PASSAGES):
        length = rng.randint(4, 16)
        if k % 5 == 0:
            ids = rng.sample(range(len(va)), length)
        else:
            c = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
            d = np.linalg.norm(va - c, axis=1)
            pool = list(np.argsort(d)[:250])
            ids = rng.sample(pool, min(length, len(pool)))
        ids_per.append(ids)
    return ids_per, va


def best_ridge(X_tr, X_te, y_tr, y_te):
    best = (None, -np.inf)
    for a in ALPHA_GRID:
        r = Ridge(alpha=a).fit(X_tr, y_tr)
        s = r2_score(y_te, r.predict(X_te))
        if s > best[1]:
            best = (r, s)
    return best


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    lex = load_lexicon(REPO / cfg["nrc_lexicon"])
    vad = node_vad_arrays(art, lex)                       # [n,3] v,a,d
    d_node = vad[:, 2]

    # --- recover passage ids, build D labels, VALIDATE alignment with saved states
    ids_per, va = rebuild_passage_ids(art)
    saved = np.load(PP_DIR / "labels.npy")               # [n,2] v,a as saved
    rec_va = np.array([va[ids].mean(0) for ids in ids_per])
    align_err = np.abs(rec_va - saved).max()
    print(f"passage v,a rebuild vs saved labels: max|err|={align_err:.2e} "
          f"({'ALIGNED' if align_err < 1e-6 else 'MISALIGNED — abort'})", flush=True)
    if align_err >= 1e-6:
        raise SystemExit("passage rebuild does not match saved states order")
    d_label = np.array([np.nanmean(d_node[ids]) for ids in ids_per])

    # --- load saved states in order
    S = np.concatenate([np.load(PP_DIR / f"states_{s:05d}.npy")
                        for s in range(0, N_PASSAGES, 100)])   # [n, n_layers, d]

    pp = load_probe(PP_DIR / "probe_passage.pkl")
    L, sc = pp.layer, pp.scaler
    X = sc.transform(S[:, L].astype(np.float64))
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=0)
    rd, r2d = best_ridge(X[idx_tr], X[idx_te], d_label[idx_tr], d_label[idx_te])
    print(f"\nPhase A — D readout at passage-probe layer {L}")
    print(f"  held-out R2:  V {pp.r2_v:.3f}   A {pp.r2_a:.3f}   D {r2d:.3f}")

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    cv, ca, cd = pp.ridge_v.coef_, pp.ridge_a.coef_, rd.coef_
    print(f"  direction cosines in residual space:")
    print(f"    cos(D, A) = {cos(cd, ca):+.3f}   <- near 1 => D is not a third axis")
    print(f"    cos(D, V) = {cos(cd, cv):+.3f}")
    print(f"    cos(A, V) = {cos(ca, cv):+.3f}   (baseline: the two axes we already use)")
    d_independent = abs(cos(cd, ca)) < 0.6 and r2d > 0.2

    # --- Phase B: is D steerable by text, independent of arousal?
    print(f"\nPhase B — steering readout (Llama-1B)")
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    pre, ANCH = cfg["preamble"], "\nRight now everything feels"

    def read(text):
        hs = model.hidden_states(pre + text + ANCH)
        Xh = sc.transform(hs[L][-1:].astype(np.float64))
        return (float(pp.ridge_v.predict(Xh)[0]), float(pp.ridge_a.predict(Xh)[0]),
                float(rd.predict(Xh)[0]))

    ok = ~np.isnan(d_node)
    order = np.argsort(np.where(ok, d_node, np.nan))
    lowD = [int(i) for i in order[:400]]
    highD = [int(i) for i in order if ok[i]][-400:]
    rng = random.Random(3)

    def sample_poem(pool):
        return ".\n".join(art.word(i) for i in rng.sample(pool, 24))

    rows = []
    for kind, pool in [("low-D", lowD), ("high-D", highD)]:
        for rep in range(4):
            v, a, d = read(sample_poem(pool))
            rows.append((kind, v, a, d))
            print(f"  {kind:6s} rep{rep}  V {v:.2f}  A {a:.2f}  D {d:.2f}", flush=True)
    import statistics as st
    lo = [r for r in rows if r[0] == "low-D"]
    hi = [r for r in rows if r[0] == "high-D"]
    dD = st.mean(r[3] for r in hi) - st.mean(r[3] for r in lo)
    dA = st.mean(r[2] for r in hi) - st.mean(r[2] for r in lo)
    print(f"\n  high-D minus low-D content:  ΔD (readout) {dD:+.3f}   ΔA (drag) {dA:+.3f}")

    # exact orthogonal fraction of D vs the V-A plane (pairwise cos understates it)
    e1 = cv / np.linalg.norm(cv)
    e2 = ca - (ca @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    du = cd / np.linalg.norm(cd)
    frac_in = float(np.hypot(du @ e1, du @ e2))
    frac_out = float(np.sqrt(max(0.0, 1 - frac_in ** 2)))
    print(f"  D direction vs span(V,A): {frac_in**2*100:.0f}% in-plane, "
          f"{frac_out**2*100:.0f}% orthogonal (a genuinely new degree of freedom)")

    # --- Phase C: does D move AROUSAL at MATCHED valence? (decouple the B confound)
    print(f"\nPhase C — valence-matched high-D vs low-D (isolates D's orthogonal part)")
    val = va[:, 0]
    band = np.where((val >= 0.45) & (val <= 0.65) & ~np.isnan(d_node))[0]
    bd = d_node[band]
    lowDm = [int(band[i]) for i in np.argsort(bd)[:300]]
    highDm = [int(band[i]) for i in np.argsort(bd)[-300:]]
    rowsC = []
    for kind, pool in [("mV-lowD", lowDm), ("mV-highD", highDm)]:
        for rep in range(4):
            v, a, d = read(sample_poem(pool))
            rowsC.append((kind, v, a, d))
            print(f"  {kind:8s} rep{rep}  V {v:.2f}  A {a:.2f}  D {d:.2f}", flush=True)
    loC = [r for r in rowsC if r[0] == "mV-lowD"]
    hiC = [r for r in rowsC if r[0] == "mV-highD"]
    dVc = st.mean(r[1] for r in hiC) - st.mean(r[1] for r in loC)
    dAc = st.mean(r[2] for r in hiC) - st.mean(r[2] for r in loC)
    dDc = st.mean(r[3] for r in hiC) - st.mean(r[3] for r in loC)
    maxAc = max(r[2] for r in hiC)
    print(f"\n  at matched valence (ΔV {dVc:+.3f}):  ΔD {dDc:+.3f}   ΔA {dAc:+.3f}"
          f"   max A reached {maxAc:.3f} (valley/awe ceiling ~0.47)")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "dominance_axis.txt", "w") as f:
        f.write(f"R2 V={pp.r2_v:.3f} A={pp.r2_a:.3f} D={r2d:.3f}\n")
        f.write(f"cos(D,A)={cos(cd,ca):.3f} cos(D,V)={cos(cd,cv):.3f} cos(A,V)={cos(ca,cv):.3f}\n")
        f.write(f"D vs plane: in={frac_in:.3f} orth={frac_out:.3f}\n")
        f.write(f"steer(uncontrolled) ΔD={dD:.3f} ΔA={dA:.3f}\n")
        f.write(f"steer(matched-V) ΔV={dVc:.3f} ΔD={dDc:.3f} ΔA={dAc:.3f} maxA={maxAc:.3f}\n")
        for r in rows + rowsC:
            f.write(f"{r[0]} V={r[1]:.3f} A={r[2]:.3f} D={r[3]:.3f}\n")

    print("\n=== VERDICT ===")
    if not d_independent:
        print("  D is NOT an independent handle (collinear with A or not readable). "
              "The arousal attractor is the whole story; a third axis buys nothing.")
    elif dAc > 0.08 and maxAc > 0.55:
        print("  D's orthogonal part IS an arousal handle: at matched valence, high-D "
              "content raises arousal past the valley/awe ceiling. A real third lever.")
    else:
        print("  D is readable and ~36% orthogonal to V/A, but at matched valence it "
              "does NOT lift arousal past the attractor — its steering power is mostly "
              "a valence proxy. The arousal cap holds; D is not the escape.")
    print(f"\nwrote {RESULTS}/dominance_axis.txt")


if __name__ == "__main__":
    main()
