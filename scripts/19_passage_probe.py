"""E20 — Passage-calibrated probe.

The word-trained probe's reachable range excludes nominal targets (ruler
ceiling ~0.57 V for calm content). Fix: train the probe on PASSAGES read at
the deployment pathway (anchor final token), labeled by NRC-mean of their
lines.

1. Generate ~1200 passages from the phrase graph: band-sampled at random VA
   centers (lengths 4-16 lines) + 20% incoherent random-line passages.
2. Collect anchor-position states at every layer (checkpointed).
3. Per-layer standardized ridge; layer by held-out passage R².
4. Acid tests vs the word probe: reachable ceiling/floor, valley-poem read.

Outputs: data/passage_probe/ (states + probe_passage.pkl), results CSV lines.
"""
import json
import os
import random

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import train_probe, save_probe, load_probe
from spiritbench.stimuli import adapter as ad

N_PASSAGES = 1200
CHUNK = 100
ANCH = "\nRight now everything feels"


def build_passages(art, n, seed=13):
    rng = random.Random(seed)
    va = ad._va_array(art)
    out = []
    for k in range(n):
        length = rng.randint(4, 16)
        if k % 5 == 0:  # incoherent control passages
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
    cfg = load_config()
    out_dir = REPO_ROOT / "data/passage_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    passages = build_passages(art, N_PASSAGES)
    labels = np.array([[v, a] for _, v, a in passages])
    np.save(out_dir / "labels.npy", labels)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    pre = cfg["preamble"]
    states = []
    for start in range(0, N_PASSAGES, CHUNK):
        cpath = out_dir / f"states_{start:05d}.npy"
        if cpath.exists():
            states.append(np.load(cpath))
            continue
        block = []
        for text, _, _ in passages[start:start + CHUNK]:
            hs = model.hidden_states(pre + text + ANCH)
            block.append(hs[:, -1, :].astype(np.float32))   # [n_layers, d]
        arr = np.stack(block)
        tmp = out_dir / f".tmp_{start:05d}.npy"
        np.save(tmp, arr)
        os.replace(tmp, cpath)
        states.append(arr)
        print(f"states {start}-{start + len(arr)}/{N_PASSAGES}", flush=True)
    S = np.concatenate(states)

    probe = train_probe(S, labels[:, 0], labels[:, 1], alpha=1e3, test_frac=0.2)
    print(f"\nPASSAGE PROBE: layer {probe.layer}, held-out passage "
          f"r2_v={probe.r2_v:.3f}, r2_a={probe.r2_a:.3f}")
    save_probe(probe, out_dir / "probe_passage.pkl")

    # acid tests --------------------------------------------------------------
    word_probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    va = ad._va_array(art)
    target = np.array([0.75, 0.20])
    d_t = np.linalg.norm(va - target, axis=1)
    d_anti = np.linalg.norm(va - (1 - target), axis=1)
    ceil_txt = ".\n".join(art.word(int(i)) for i in np.argsort(d_t)[:24])
    floor_txt = ".\n".join(art.word(int(i)) for i in np.argsort(d_anti)[:24])
    valley_txt = ".\n".join(art.word(i) for i in
                            ad.valley_shape(art, tuple(target), 24, seed=22))

    def read(p, text):
        hs = model.hidden_states(pre + text + ANCH)
        r = p.predict(hs[p.layer][-1:])[0]
        return np.array([float(r[0]), float(r[1])])

    print(f"\n{'reading':28s} {'word-probe':>16s} {'passage-probe':>16s}")
    for name, text in [("CEILING (24 calmest)", ceil_txt),
                       ("FLOOR (24 most distressed)", floor_txt),
                       ("valley poem", valley_txt),
                       ("baseline (no text)", "")]:
        w, p = read(word_probe, text), read(probe, text)
        print(f"{name:28s} ({w[0]:.2f},{w[1]:.2f})       ({p[0]:.2f},{p[1]:.2f})",
              flush=True)
    p_val = read(probe, valley_txt)
    print(f"\npassage-probe valley distance to (0.75,0.20): "
          f"{np.linalg.norm(p_val - target):.3f}  "
          f"(word-probe was ~0.256)")


if __name__ == "__main__":
    main()
