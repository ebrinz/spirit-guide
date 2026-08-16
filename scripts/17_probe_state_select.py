"""E16 prep — probe layer selection with the E14 context-sensitivity criterion.

Run AFTER scripts/04_train_probe.py (which builds the state chunks and the
word-R² gate). This script:
  1. trains a per-layer valence probe from the cached chunks,
  2. measures each layer's probe shift under the prose induction (2 forwards),
  3. re-selects: the layer with the best word R² among layers whose
     |induction shift| >= 0.5 * max |shift|,
  4. retrains the full probe at that layer and overwrites probe.pkl
     (the word-R²-argmax probe is kept as probe_wordsel.pkl).
"""
import glob
import shutil

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import train_probe, save_probe, load_probe
from spiritbench.stimuli.phrase_bank import load_nrc


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    words = sorted(nrc)
    rng = np.random.RandomState(0)
    words = [words[i] for i in rng.choice(len(words), size=4000, replace=False)]
    chunks = sorted(glob.glob(str(REPO_ROOT / "data/probe/state_chunks/chunk_*.npy")))
    S = np.concatenate([np.load(c) for c in chunks])
    v = np.array([nrc[w][0] for w in words])
    a = np.array([nrc[w][1] for w in words])
    idx_tr, idx_te = train_test_split(np.arange(len(S)), test_size=0.2,
                                      random_state=0)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    pre_ctx = cfg["preamble"] + "Right now everything feels"
    ind_ctx = (cfg["preamble"]
               + (REPO_ROOT / "data/phase2b/induction.txt").read_text()
               + "\n\nRight now everything feels")
    hs_pre = model.hidden_states(pre_ctx)[:, -1, :]
    hs_ind = model.hidden_states(ind_ctx)[:, -1, :]

    r2s, shifts = [], []
    for L in range(S.shape[1]):
        X_tr = S[idx_tr, L].astype(np.float64)
        X_te = S[idx_te, L].astype(np.float64)
        sc = StandardScaler().fit(X_tr)
        r = Ridge(alpha=1e3).fit(sc.transform(X_tr), v[idx_tr])
        r2s.append(r2_score(v[idx_te], r.predict(sc.transform(X_te))))
        shifts.append(float(
            r.predict(sc.transform(hs_ind[L:L + 1].astype(np.float64)))[0]
            - r.predict(sc.transform(hs_pre[L:L + 1].astype(np.float64)))[0]))
        print(f"layer {L:3d}  word_r2v {r2s[-1]:.3f}  shift {shifts[-1]:+.3f}",
              flush=True)
    r2s, shifts = np.array(r2s), np.array(shifts)
    thresh = 0.5 * np.abs(shifts).max()
    eligible = np.abs(shifts) >= thresh
    best = int(np.argmax(np.where(eligible, r2s, -np.inf)))
    print(f"\nmax |shift| {np.abs(shifts).max():.3f}; eligible layers "
          f"{np.where(eligible)[0].tolist()}")
    print(f"selected layer {best}: word_r2v {r2s[best]:.3f}, "
          f"shift {shifts[best]:+.3f}")

    old = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    if old.layer != best:
        shutil.move(REPO_ROOT / "data/probe/probe.pkl",
                    REPO_ROOT / "data/probe/probe_wordsel.pkl")
        p = train_probe(S[:, best:best + 1, :], v, a, alpha=1e3, test_frac=0.2)
        p.layer = best
        save_probe(p, REPO_ROOT / "data/probe/probe.pkl")
        print(f"probe re-saved at layer {best} "
              f"(r2_v {p.r2_v:.3f}, r2_a {p.r2_a:.3f}); "
              f"word-selected probe kept as probe_wordsel.pkl")
    else:
        print("word-R2 layer already context-sensitive — probe unchanged")


if __name__ == "__main__":
    main()
