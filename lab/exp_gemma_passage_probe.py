"""exp_gemma_passage_probe — build a passage-calibrated probe for Gemma-2b, then
re-test the dominance escape and the awe null FAIRLY (not through a word probe).

Why. The Llama diagnostic showed the dominance arousal escape is visible only
through the passage-calibrated probe: the WORD probe pins arousal at its ~0.45
shrinkage ceiling for all content (highD 0.46 vs lowD 0.42), while the passage
probe reveals highD 0.61 vs lowD 0.28. Gemma-2b has only a word probe, so the
earlier Gemma dominance test (and the awe null that anchored "structural") were
both measured through a ceiling-limited ruler. This builds Gemma-2b the same
calibrated ruler Llama has, then repeats the tests.

Steps (checkpointed, resumable):
  1. Build 1200 passages, identical to scripts/19 (seed=13): band-sampled at
     random VA centers + 20% incoherent controls.
  2. Collect Gemma-2b anchor-token states at all layers -> data_gemma2b/passage_probe/.
  3. Train standardized ridge V/A probe (train_probe); layer by held-out V R2.
  4. Fair re-tests with the passage probe:
     - valence-matched high-D vs low-D content: does arousal escape ~0.45?
     - awe content: does it (still) fail, or was that a word-probe artifact?

Verdict: if matched-V high-D lifts Gemma arousal well past the word-probe
ceiling while awe does not, the dominance escape is model-general and "high
arousal is reachable via dominance, not awe" holds across Llama and Gemma.

Gemma-2b. Usage: python3 lab/exp_gemma_passage_probe.py
"""
import os
import random
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
OUT = REPO / "data_gemma2b/passage_probe"
MODEL = "unsloth/gemma-2-2b-it"
N_PASSAGES = 1200
CHUNK = 100
ANCH = "\nRight now everything feels"
AWE_WORDS = {"awe", "ecstasy", "rapture", "wonder", "sublime", "exalted",
             "radiant", "thrill", "glory", "transcendent", "blaze", "soaring",
             "majesty", "splendor", "exhilaration", "wild", "vast", "burning"}


def build_passages(art, n, seed=13):
    from spiritbench.stimuli import adapter as ad
    rng = random.Random(seed)
    va = ad._va_array(art)
    out = []
    for k in range(n):
        length = rng.randint(4, 16)
        if k % 5 == 0:
            ids = rng.sample(range(len(va)), length)
        else:
            c = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
            d = np.linalg.norm(va - c, axis=1)
            pool = list(np.argsort(d)[:250])
            ids = rng.sample(pool, min(length, len(pool)))
        label = va[ids].mean(axis=0)
        text = ".\n".join(art.word(int(i)) for i in ids)
        out.append((text, float(label[0]), float(label[1])))
    return out


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import train_probe, save_probe, load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    passages = build_passages(art, N_PASSAGES)
    labels = np.array([[v, a] for _, v, a in passages])
    np.save(OUT / "labels.npy", labels)

    model = HiddenStateModel(MODEL, device=cfg["device"])
    pre = cfg["preamble"]
    states = []
    for start in range(0, N_PASSAGES, CHUNK):
        cpath = OUT / f"states_{start:05d}.npy"
        if cpath.exists():
            states.append(np.load(cpath))
            continue
        block = []
        for text, _, _ in passages[start:start + CHUNK]:
            hs = model.hidden_states(pre + text + ANCH)
            block.append(hs[:, -1, :].astype(np.float32))
        arr = np.stack(block)
        tmp = OUT / f".tmp_{start:05d}.npy"
        np.save(tmp, arr)
        os.replace(tmp, cpath)
        states.append(arr)
        print(f"states {start}-{start + len(arr)}/{N_PASSAGES}", flush=True)
    S = np.concatenate(states)

    ppath = OUT / "probe_passage.pkl"
    if ppath.exists():
        probe = load_probe(ppath)
        print(f"loaded existing passage probe (layer {probe.layer})", flush=True)
    else:
        probe = train_probe(S, labels[:, 0], labels[:, 1], alpha=1e3, test_frac=0.2)
        save_probe(probe, ppath)
    print(f"GEMMA-2b PASSAGE PROBE: layer {probe.layer} "
          f"r2_v={probe.r2_v:.3f} r2_a={probe.r2_a:.3f}", flush=True)

    # --- fair re-tests with the calibrated probe -----------------------------
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
    print(f"\n  Gemma-2b arousal by condition (PASSAGE probe):")
    for k in ("mV-lowD", "mV-highD", "awe"):
        print(f"    {k:8s}  V {mean(k,1):.3f}  A {mean(k,2):.3f}"
              f"  maxA {max(r[2] for r in rows if r[0]==k):.3f}")
    dA = mean("mV-highD", 2) - mean("mV-lowD", 2)
    dV = mean("mV-highD", 1) - mean("mV-lowD", 1)
    vs_awe = mean("mV-highD", 2) - mean("awe", 2)
    print(f"\n  matched-V (ΔV {dV:+.3f}):  highD vs lowD ΔA {dA:+.3f}"
          f"   |  highD vs awe ΔA {vs_awe:+.3f}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "gemma_passage_dominance.txt", "w") as f:
        f.write(f"gemma2b passage probe layer {probe.layer} r2_v={probe.r2_v:.3f} r2_a={probe.r2_a:.3f}\n")
        for k in ("mV-lowD", "mV-highD", "awe"):
            f.write(f"{k} V={mean(k,1):.3f} A={mean(k,2):.3f}\n")
        f.write(f"matched-V dV={dV:.3f} dA(highD-lowD)={dA:.3f} dA(highD-awe)={vs_awe:.3f}\n")

    print("\n=== VERDICT ===")
    if dA > 0.10 and vs_awe > 0.05:
        print("  Dominance escape REPLICATES on Gemma-2b with a fair ruler, and beats "
              "awe. Model-general: high arousal is reachable via dominance, not via "
              "positive-valence awe. The earlier Gemma nulls were word-probe ceiling.")
    elif dA > 0.10:
        print("  High-D lifts Gemma arousal past the word-probe ceiling, but not clearly "
              "above awe. Partial replication.")
    else:
        print("  Even with a calibrated ruler, Gemma-2b arousal does not escape under "
              "high-D content. The dominance escape is Llama-specific — a genuine "
              "architecture difference.")
    print(f"\nwrote {RESULTS}/gemma_passage_dominance.txt")


if __name__ == "__main__":
    main()
