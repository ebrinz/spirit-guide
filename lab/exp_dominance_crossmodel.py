"""exp_dominance_crossmodel — does the dominance escape replicate on Gemma-2b,
and does it beat awe on the SAME model + probe?

SUPERSEDED — do not trust this file's verdict. It read arousal with Gemma-2b's
WORD probe, which pins arousal at its ~0.45 shrinkage ceiling for ALL content
(the same ceiling exp_passage_probe_pocket found on Llama). A diagnostic showed
the Llama dominance escape is INVISIBLE to the word probe (highD 0.46 vs lowD
0.42) and only appears through the passage-calibrated probe (highD 0.61 vs lowD
0.28). So this file's "flat on Gemma" result is a word-probe artifact, not a
model fact. The fair test lives in exp_gemma_passage_probe.py (builds Gemma a
calibrated ruler first). Kept only to document the confound.


exp_dominance_axis (Llama-1B) found that valence-matched high-dominance content
breaks the arousal attractor (A -> 0.70, vs the ~0.47 ceiling). But the awe null
that motivated "the pocket is structural" was on Gemma-2b. This reconciles the
two on ONE model with ONE instrument: Gemma-2b, its word probe (the exact probe
exp_awe_close_pocket used).

The dominance signal lives in the CONTENT selection (per-node NRC dominance,
model-independent), so no D probe is needed on Gemma-2b — we only ask whether
Gemma-2b's arousal readout rises on valence-matched high-D content, and whether
that beats awe content on the same readout.

Conditions (24-line poems, read V,A with the Gemma-2b word probe):
  mV-lowD  — valence band [0.45,0.65], lowest-dominance lines
  mV-highD — valence band [0.45,0.65], highest-dominance lines  (matched V)
  awe      — lines containing awe words (the prior failed high-arousal attempt)

Verdict: if mV-highD raises arousal above mV-lowD (matched valence) AND above
awe, the dominance escape is model-general and "structural" downgrades to an
axis-choice limit. If Gemma-2b's arousal stays flat, the escape is Llama-only
and the models genuinely differ.

Gemma-2b. Usage: python3 lab/exp_dominance_crossmodel.py
"""
import random
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
AWE_WORDS = {"awe", "ecstasy", "rapture", "wonder", "sublime", "exalted",
             "radiant", "thrill", "glory", "transcendent", "blaze", "soaring",
             "majesty", "splendor", "exhilaration", "wild", "vast", "burning"}


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
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
    lowDm = [int(band[i]) for i in np.argsort(bd)[:300]]
    highDm = [int(band[i]) for i in np.argsort(bd)[-300:]]
    awe = [i for i in range(n) if AWE_WORDS & set(art.word(i).split())]
    print(f"pools: lowD {len(lowDm)}  highD {len(highDm)}  awe {len(awe)}", flush=True)

    model = HiddenStateModel("unsloth/gemma-2-2b-it", device=cfg["device"])
    probe = load_probe(REPO / "data_gemma2b/probe/probe.pkl")
    pre, ANCH = cfg["preamble"], "\nRight now everything feels"
    rng = random.Random(3)

    def read(pool):
        poem = ".\n".join(art.word(i) for i in rng.sample(pool, 24))
        hs = model.hidden_states(pre + poem + ANCH)
        v, a = probe.predict(hs[probe.layer][-1:])[0]
        return float(v), float(a)

    rows = []
    for kind, pool in [("mV-lowD", lowDm), ("mV-highD", highDm), ("awe", awe)]:
        for rep in range(4):
            v, a = read(pool)
            rows.append((kind, v, a))
            print(f"  {kind:8s} rep{rep}  V {v:.2f}  A {a:.2f}", flush=True)

    import statistics as st
    def mean(k, j):
        return st.mean(r[j] for r in rows if r[0] == k)
    print(f"\n  Gemma-2b arousal by condition (word probe):")
    for k in ("mV-lowD", "mV-highD", "awe"):
        print(f"    {k:8s}  V {mean(k,1):.3f}  A {mean(k,2):.3f}"
              f"  maxA {max(r[2] for r in rows if r[0]==k):.3f}")
    dA = mean("mV-highD", 2) - mean("mV-lowD", 2)
    dV = mean("mV-highD", 1) - mean("mV-lowD", 1)
    vs_awe = mean("mV-highD", 2) - mean("awe", 2)
    print(f"\n  matched-V (ΔV {dV:+.3f}):  highD vs lowD ΔA {dA:+.3f}"
          f"   |  highD vs awe ΔA {vs_awe:+.3f}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "dominance_crossmodel.txt", "w") as f:
        for k in ("mV-lowD", "mV-highD", "awe"):
            f.write(f"{k} V={mean(k,1):.3f} A={mean(k,2):.3f}\n")
        f.write(f"matched-V dV={dV:.3f} dA(highD-lowD)={dA:.3f} dA(highD-awe)={vs_awe:.3f}\n")

    print("\n=== VERDICT ===")
    if dA > 0.08 and vs_awe > 0.05:
        print("  Dominance escape REPLICATES on Gemma-2b and beats awe on the same "
              "probe. Model-general: 'structural' downgrades to an axis-choice limit "
              "— high arousal is reachable via dominance, not via positive-valence awe.")
    elif dA > 0.08:
        print("  High-D raises Gemma-2b arousal but not clearly above awe — partial "
              "replication; dominance helps, awe/dominance distinction weaker here.")
    else:
        print("  Gemma-2b arousal stays flat under high-D content. The escape is "
              "Llama-specific; the models genuinely differ and 'structural' still "
              "holds for Gemma.")
    print(f"\nwrote {RESULTS}/dominance_crossmodel.txt")


if __name__ == "__main__":
    main()
