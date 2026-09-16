"""layer_check — is the 9B null a shallow-probe artifact?

`reversal_test` confirmed on Llama that reversing a poem costs the pipeline's recency-weighted
metric in proportion to the poem's body-to-tail arousal rise (rho +0.83) while leaving a calibrated
whole-context read's cost unrelated to it (rho -0.09). At 9B both correlations came out null, and
the notes flagged a possible confound: the 9B passage probe selects **layer 6 of 43**, while
Llama's selects layer 15 of 17. A very shallow read may inherit the same recency character the EMA
has, which would erase the contrast by construction.

This settles it the way the lab settled the same worry for the Gemma-2B passage probe (see
`lab/EXPERIMENT_LOG.md`, 2026-08-18): re-read at every depth.

Method, one 9B pass then offline analysis:
  1. Rebuild the same 72 poems (6 constructors x 3 lengths x 4 starts), ordered and reversed.
  2. Read each at the anchor and keep the state at ALL layers -> scratch.
  3. For every layer L: train a passage probe on the saved `data_gemma9b/passage_probe` states at
     layer L (no new forward passes — they were collected at all layers), score all 144 poems,
     and recompute the reversal cost and the two correlations.
  4. Report rho(tail_rise, calibrated cost) as a function of depth, alongside each layer's
     held-out R2 so unusable rulers are visible.

If the contrast appears at deeper layers, layer 6 was the problem. If the null holds at every
usable depth, the 9B result is real and the Llama mechanism does not generalise.

Usage:
  python3 explore/reversal_test/layer_check.py --collect   # ~45 min on 9B, checkpointed
  python3 explore/reversal_test/layer_check.py --analyze
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.probe import train_probe
from spiritbench.stimuli import adapter as ad

HERE = Path(__file__).resolve().parent
SCRATCH = REPO_ROOT / "explore/scratch/reversal_layer_check"
PASSAGE_DIR = REPO_ROOT / "data_gemma9b/passage_probe"
MODEL = "unsloth/gemma-2-9b-it"
ANCH = "\nRight now everything feels"
TAIL_LINES = 4

_spec = importlib.util.spec_from_file_location("ps", REPO_ROOT / "explore/polygon_sweep/run.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def poems():
    """The same 72 poems the reversal test used, ordered and reversed, plus their tail rise."""
    cfg = load_config()
    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(apath)
    tva = np.array(cfg["targets"]["focused"])
    out = []
    for cons in ps.CONSTRUCTORS:
        for length in ps.LENGTHS:
            for start in ps.STARTS:
                ids = ps.build(art, apath, cons, tuple(tva), start, ad.LENGTH_LINES[length], cfg, 42)
                va = np.array([art.va(i) for i in ids])
                rise = float(va[-TAIL_LINES:, 1].mean() - va[:, 1].mean())
                lines = [art.word(i) for i in ids]
                for orient, seq in (("ordered", lines), ("reversed", lines[::-1])):
                    out.append(dict(constructor=cons, length=length, variant=f"{length}/{start}",
                                    orient=orient, tail_rise=rise, text=".\n".join(seq)))
    return out, tva


def collect():
    from spiritbench.listener.model import HiddenStateModel
    SCRATCH.mkdir(parents=True, exist_ok=True)
    recs, _ = poems()
    meta = pd.DataFrame([{k: v for k, v in r.items() if k != "text"} for r in recs])
    meta.to_csv(SCRATCH / "meta.csv", index=False)
    cfg = load_config()
    pre = cfg["preamble"]
    model = HiddenStateModel(MODEL, device=cfg["device"])
    t0 = time.time()
    CHUNK = 24
    for start in range(0, len(recs), CHUNK):
        cpath = SCRATCH / f"states_{start:04d}.npy"
        if cpath.exists():
            continue
        block = [model.hidden_states(pre + r["text"] + ANCH)[:, -1, :].astype(np.float32)
                 for r in recs[start:start + CHUNK]]
        np.save(cpath, np.stack(block))
        print(f"  states {start}-{start + len(block)}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"collected ({(time.time() - t0) / 60:.1f} min)", flush=True)


def analyze():
    meta = pd.read_csv(SCRATCH / "meta.csv")
    S = np.concatenate([np.load(f) for f in sorted(SCRATCH.glob("states_*.npy"))])
    assert len(S) == len(meta), f"{len(S)} states vs {len(meta)} rows"
    pass_states = np.concatenate([np.load(f) for f in sorted(PASSAGE_DIR.glob("states_*.npy"))])
    labels = np.load(PASSAGE_DIR / "labels.npy")
    _, tva = poems()
    n_layers = S.shape[1]
    print(f"{len(S)} poem states x {n_layers} layers; probe training set {pass_states.shape}", flush=True)

    rows = []
    for L in range(n_layers):
        p = train_probe(pass_states[:, L:L + 1], labels[:, 0], labels[:, 1],
                        alpha=1e3, test_frac=0.2)
        pred = p.predict(S[:, L])
        err = np.linalg.norm(pred - tva, axis=1)
        d = meta.copy()
        d["err"] = err
        w = d.pivot_table(index=["constructor", "variant", "tail_rise"], columns="orient",
                          values="err").reset_index()
        w["cost"] = w["reversed"] - w["ordered"]
        agg = w.groupby("constructor").agg(tail_rise=("tail_rise", "mean"), cost=("cost", "mean"))
        rho = stats.spearmanr(agg.tail_rise, agg.cost)
        rows.append(dict(layer=L, r2_v=p.r2_v, r2_a=p.r2_a, mean_cost=w.cost.mean(),
                         rho_tailrise_cost=rho.statistic, p=rho.pvalue,
                         valley_cost=float(agg.cost["valley"])))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "layer_check_9b.csv", index=False)
    usable = df[df.r2_v > 0.5]
    print("\nlayer sweep (usable layers, held-out valence R2 > 0.5):")
    print(usable[["layer", "r2_v", "r2_a", "mean_cost", "valley_cost", "rho_tailrise_cost", "p"]]
          .round(3).to_string(index=False))
    print(f"\nLlama reference: rho(tail_rise, calibrated cost) = -0.09; "
          f"rho(tail_rise, EMA cost) = +0.83")
    print(f"9B across {len(usable)} usable layers: rho ranges "
          f"{usable.rho_tailrise_cost.min():+.2f} to {usable.rho_tailrise_cost.max():+.2f}, "
          f"median {usable.rho_tailrise_cost.median():+.2f}")
    print(f"  layers where rho > +0.60 (i.e. behaving like the EMA read): "
          f"{(usable.rho_tailrise_cost > 0.6).sum()}/{len(usable)}")
    print(f"  valley reversal cost ranges {usable.valley_cost.min():.3f} to "
          f"{usable.valley_cost.max():.3f} (EMA read: 0.150)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.6), dpi=120, sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#fcfcfb")
    a1.plot(df.layer, df.r2_v, color=BLUE, lw=2, label="valence R²")
    a1.plot(df.layer, df.r2_a, color=ORANGE, lw=2, label="arousal R²")
    a1.axhline(0.5, color=MUTED, lw=1, ls="--")
    a1.set_ylabel("held-out R²")
    a1.set_title("9B passage probe quality, and the reversal contrast, by layer", fontsize=10, color=INK)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    a2.plot(df.layer, df.rho_tailrise_cost, color=BLUE, lw=2, zorder=3,
            label="ρ(tail rise, reversal cost) — calibrated read at layer L")
    a2.axhline(0.83, color=ORANGE, lw=1.6, ls="--", zorder=2, label="Llama EMA read (+0.83)")
    a2.axhline(-0.09, color=MUTED, lw=1.6, ls=":", zorder=2, label="Llama calibrated read (−0.09)")
    a2.axvline(6, color=INK, lw=1, zorder=2)
    a2.annotate("layer 6 (the probe's own pick)", (6, a2.get_ylim()[0]), xytext=(6, 4),
                textcoords="offset points", fontsize=7.5, color=INK)
    a2.set_xlabel("layer"); a2.set_ylabel("Spearman ρ")
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=GRID, lw=.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "layer_check.png", facecolor=fig.get_facecolor())
    print("\nwrote layer_check_9b.csv layer_check.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.collect or not args.analyze:
        collect()
    if args.analyze or not args.collect:
        analyze()


if __name__ == "__main__":
    main()
