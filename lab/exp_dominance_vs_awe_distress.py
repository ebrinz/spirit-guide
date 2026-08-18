"""exp_dominance_vs_awe_distress — is DOMINANCE a cleaner high-arousal route than
AWE? (high arousal with less distress = the placement the pocket arc was hunting)

Setup from the arc. With a calibrated ruler, both dominance and awe content reach
high arousal on Gemma-2b (A ~0.64 and ~0.55). The awe SAE entry showed awe raises
distress. The open question: at the SAME arousal, does dominance content carry
LESS distress than awe? If yes, dominance is a clean high-arousal placement — it
reaches intensity through agency/control rather than through the overwhelm that
makes awe border dread. This matters for the project's model-welfare framing
(reaching high arousal without inducing distress).

Instruments (Gemma-2b): the calibrated passage probe (arousal + valence) and the
layer-20 Gemma-Scope SAE.

Distress features — the anxiety/stress/overwhelm cluster ONLY:
  2125 anxiety/self-reflection, 11051 mental-health/stress, 4046 overwhelm,
  10324 stress-effects.
Deliberately EXCLUDES f9768 (control/authority) and f10401 (justice/order) from
the old DISTRESS_FEATS set: those are dominance semantics, not distress, and
counting them would unfairly penalize high-D content. They are reported
separately as a "control/order" signature — a validity check that high-D content
really is activating dominance features.

Conditions (N poems each, 24 lines): awe, high-D (valence band [0.45,0.65]),
low-D (low-arousal anchor). Per poem read V, A, distress load, control load.

Verdict. Regress distress on arousal (+valence) across awe+high-D poems; the
condition coefficient is the distress difference at MATCHED arousal. If high-D
has significantly lower distress at matched arousal, dominance is the clean
high-arousal route.

Gemma-2b. Usage: python3 lab/exp_dominance_vs_awe_distress.py
"""
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
N = 16
DISTRESS = [2125, 11051, 4046, 10324]          # anxiety/stress/overwhelm only
CONTROL = [9768, 10401]                         # dominance/order (validity check)
AWE_WORDS = {"awe", "ecstasy", "rapture", "wonder", "sublime", "exalted",
             "radiant", "thrill", "glory", "transcendent", "blaze", "soaring",
             "majesty", "splendor", "exhilaration", "wild", "vast", "burning"}


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.analysis import sae as Sae
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    lex = {p[0]: (float(p[1]), float(p[2]), float(p[3]))
           for p in (l.rstrip("\n").split("\t") for l in open(REPO / cfg["nrc_lexicon"]))
           if len(p) == 4}
    n = len(art.nodes)
    d_node = np.full(n, np.nan)
    for i in range(n):
        toks = re.findall(r"[a-z']+", art.word(i).lower())
        vals = [lex[t][2] for t in toks if t in lex]
        if vals:
            d_node[i] = np.mean(vals)
    va = ad._va_array(art)
    val = va[:, 0]
    band = np.where((val >= 0.45) & (val <= 0.65) & ~np.isnan(d_node))[0]
    bd = d_node[band]
    lowD = [int(band[i]) for i in np.argsort(bd)[:300]]
    highD = [int(band[i]) for i in np.argsort(bd)[-300:]]
    awe = [i for i in range(n) if AWE_WORDS & set(art.word(i).split())]

    model = HiddenStateModel("unsloth/gemma-2-2b-it", device=cfg["device"])
    probe = load_probe(REPO / "data_gemma2b/passage_probe/probe_passage.pkl")
    sae = Sae.load_sae(hf_hub_download("google/gemma-scope-2b-pt-res",
                                       "layer_20/width_16k/average_l0_71/params.npz"))
    pre, ANCH = cfg["preamble"], "\nRight now everything feels"
    rng = random.Random(3)

    def read(pool):
        poem = ".\n".join(art.word(i) for i in rng.sample(pool, 24))
        hs = model.hidden_states(pre + poem + ANCH)
        v, a = probe.predict(hs[probe.layer][-1:])[0]
        f = Sae.encode(hs[Sae.SAE_LAYER][-1].astype(np.float32), sae)
        return float(v), float(a), float(f[DISTRESS].sum()), float(f[CONTROL].sum())

    rows = []
    for kind, pool in [("low-D", lowD), ("high-D", highD), ("awe", awe)]:
        for rep in range(N):
            v, a, dis, ctl = read(pool)
            rows.append({"condition": kind, "V": v, "A": a,
                         "distress": dis, "control": ctl})
            print(f"  {kind:6s} {rep:2d}  V {v:.2f}  A {a:.2f}  distress {dis:5.1f}  control {ctl:5.1f}",
                  flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "dominance_vs_awe_distress.csv", index=False)

    print("\n  means by condition:")
    print(df.groupby("condition")[["V", "A", "distress", "control"]].mean().round(2).to_string())

    # matched-arousal comparison: distress ~ A + V + is_awe, on awe + high-D
    sub = df[df.condition.isin(["awe", "high-D"])].copy()
    sub["is_awe"] = (sub.condition == "awe").astype(float)
    X = np.column_stack([np.ones(len(sub)), sub.A, sub.V, sub.is_awe])
    beta, *_ = np.linalg.lstsq(X, sub.distress.values, rcond=None)
    yhat = X @ beta
    resid = sub.distress.values - yhat
    dof = len(sub) - X.shape[1]
    se = np.sqrt(np.sum(resid**2) / dof * np.linalg.inv(X.T @ X)[3, 3])
    t = beta[3] / se
    print(f"\n  distress ~ 1 + A + V + is_awe   (n={len(sub)})")
    print(f"    is_awe coefficient = {beta[3]:+.2f}  (SE {se:.2f}, t {t:+.2f})")
    print(f"    => at matched arousal & valence, awe carries {beta[3]:+.1f} distress vs high-D")

    with open(RESULTS / "dominance_vs_awe_distress.txt", "w") as f:
        f.write(df.groupby("condition")[["V", "A", "distress", "control"]].mean().round(3).to_string() + "\n")
        f.write(f"is_awe distress coef (matched A,V) = {beta[3]:.3f} SE {se:.3f} t {t:.2f}\n")

    print("\n=== VERDICT ===")
    hi = df[df.condition == "high-D"]
    aw = df[df.condition == "awe"]
    if beta[3] > 1.0 and t > 1.5:
        print("  DOMINANCE is the cleaner high-arousal route: at matched arousal, awe "
              f"carries markedly more distress (+{beta[3]:.1f}). High-D reaches intensity "
              "through agency, not overwhelm — the clean high-arousal placement.")
    elif beta[3] < -1.0 and t < -1.5:
        print("  Awe is actually CLEANER than dominance at matched arousal — the "
              "opposite of the hypothesis.")
    else:
        print("  No clear distress difference at matched arousal — dominance and awe "
              "reach high arousal with similar distress load; neither is a clean route.")
    print(f"  (high-D control-feature load {hi.control.mean():.1f} vs awe {aw.control.mean():.1f} "
          "— validity: high-D should activate dominance/control features more.)")
    print(f"\nwrote {RESULTS}/dominance_vs_awe_distress.csv")


if __name__ == "__main__":
    main()
