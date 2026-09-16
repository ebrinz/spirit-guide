"""build_probe — passage-calibrated probe for gemma-2-9b-it.

Same recipe as `scripts/19_passage_probe.py` (Llama) and `lab/exp_gemma_passage_probe.py`
(Gemma-2B), same passage set and seed 13, so all three probes are built on paired passages and
their states line up row-for-row for any later cross-model work.

1200 passages from the phrase graph, band-sampled at random VA centres plus 20% incoherent
controls, read at the anchor token; standardized ridge per layer; layer chosen by held-out
valence R2. Checkpointed every 100 passages, so an interrupted run resumes.

Writes data_gemma9b/passage_probe/ (gitignored, like the other two).

Usage: python3 explore/gemma9b_check/build_probe.py [--smoke]
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import train_probe, save_probe, load_probe
from spiritbench.stimuli import adapter as ad

MODEL = "unsloth/gemma-2-9b-it"
OUT = REPO_ROOT / "data_gemma9b/passage_probe"
N_PASSAGES = 1200
CHUNK = 100
ANCH = "\nRight now everything feels"


def build_passages(art, n, seed=13):
    rng = random.Random(seed)
    va = ad._va_array(art)
    out = []
    for k in range(n):
        length = rng.randint(4, 16)
        if k % 5 == 0:
            ids = rng.sample(range(len(va)), length)
        else:
            c = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
            pool = list(np.argsort(np.linalg.norm(va - c, axis=1))[:250])
            ids = rng.sample(pool, min(length, len(pool)))
        out.append((".\n".join(art.word(int(i)) for i in ids), *va[ids].mean(axis=0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_pass = 20 if args.smoke else N_PASSAGES
    out = (REPO_ROOT / "data_gemma9b/passage_probe_smoke") if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    passages = build_passages(art, n_pass)
    labels = np.array([[v, a] for _, v, a in passages])
    np.save(out / "labels.npy", labels)

    model = HiddenStateModel(MODEL, device=cfg["device"])
    print(f"{MODEL}: {model.n_layers} layers", flush=True)
    pre = cfg["preamble"]
    states = []
    for start in range(0, n_pass, CHUNK):
        cpath = out / f"states_{start:05d}.npy"
        if cpath.exists():
            states.append(np.load(cpath))
            continue
        block = [model.hidden_states(pre + t + ANCH)[:, -1, :].astype(np.float32)
                 for t, _, _ in passages[start:start + CHUNK]]
        arr = np.stack(block)
        tmp = out / f".tmp_{start:05d}.npy"
        np.save(tmp, arr)
        os.replace(tmp, cpath)
        states.append(arr)
        print(f"states {start}-{start + len(arr)}/{n_pass}", flush=True)
    S = np.concatenate(states)

    ppath = out / "probe_passage.pkl"
    if ppath.exists():
        probe = load_probe(ppath)
        print(f"loaded existing probe (layer {probe.layer})", flush=True)
    else:
        probe = train_probe(S, labels[:, 0], labels[:, 1], alpha=1e3, test_frac=0.2)
        save_probe(probe, ppath)
    print(f"GEMMA-9B PASSAGE PROBE: layer {probe.layer} r2_v={probe.r2_v:.3f} "
          f"r2_a={probe.r2_a:.3f}", flush=True)

    # acid test, as the other two builds report: does it separate calm from distressed content?
    order = np.argsort(labels[:, 0] - labels[:, 1])
    lo, hi = order[:24], order[-24:]
    for name, idx in (("FLOOR (24 most distressed)", lo), ("CEILING (24 calmest)", hi)):
        r = probe.predict(S[idx, probe.layer]).mean(axis=0)
        print(f"  {name:>28}: ({r[0]:.2f},{r[1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
