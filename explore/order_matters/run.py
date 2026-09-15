"""order_matters — does line ORDER affect probe placement, the pipeline's own metric?

`explore/line_ablation` found order irrelevant to the self-report battery. Placement is measured
differently: `scripts/05` reads the probe at every token, smooths with an EMA (alpha 0.1), and
scores the FINAL value's distance to target. That read is recency-weighted — 96% of its weight
sits in the last ~30 tokens, about the last 4 lines of a 24-line poem — so order ought to matter
a great deal. The committed results already contain a single shuffled control per constructor
(built by `scripts/02`, target calm only, one seed), and they say something more interesting:
shuffling costs the path constructors 0.04-0.10 placement error but costs `valley`, the published
winner, essentially nothing (+0.004 averaged over three models).

`zone_zoom` showed single-seed results in this project are not trustworthy, so this puts error
bars on that, extends it from calm to all three targets, and asks the mechanistic question.

Per constructor x target: the canonical medium poem, 8 random reorderings, and its exact reverse.
Two readouts on each, to separate the metric from the model:
  ema_error    — the canonical pipeline measure (probe every token, EMA, final value vs target)
  anchor_error — the whole-context read the explore tier uses (state at a fixed anchor after the
                 poem, single probe read) vs target
plus the mean NRC (V, A) of the last four lines, which is what the EMA actually looks at, to test
the recency explanation directly.

Canonical instrument: Llama-3.2-1B-Instruct + data/probe/probe.pkl (layer 10, R2_v 0.718).

Usage:
  python3 explore/order_matters/run.py            # ~6 min
  python3 explore/order_matters/run.py --smoke

Outputs (committed): order.csv, order.png, NOTES.md.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.analysis.metrics import ema, placement_error
from spiritbench.stimuli import adapter as ad

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
CONSTRUCTORS = ["valley", "harmonic-golden", "harmonic-prime", "graph-walk", "polygon-pca"]
N_SHUFFLES = 8
EMA_WINDOW_LINES = 4          # what the EMA effectively reads, from the weight calculation
SMOKE = dict(constructors=["valley", "harmonic-golden"], targets=["calm"], n_shuffles=2)


def build(art, apath, cons, target_va, start_va, n, cfg, seed=42):
    if cons == "valley":
        return ad.valley_shape(art, target_va, n, seed, ad.node_mask(art, None))
    if cons == "graph-walk":
        return ad.graph_walk(art, start_va, target_va, n, seed, cfg["ot_repo"])
    if cons == "polygon-pca":
        return ad.polygon_pca(art, start_va, target_va, n, seed)
    if cons.startswith("harmonic-"):
        return ad.harmonic(art, apath, start_va, target_va, n, cons.split("-")[1], seed,
                           cfg["ot_repo"], cfg["semantic_axes"])
    raise ValueError(cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_smoke" if args.smoke else ""
    cfg = load_config()
    cons_list = SMOKE["constructors"] if args.smoke else CONSTRUCTORS
    tgt_list = SMOKE["targets"] if args.smoke else list(cfg["targets"])
    n_shuf = SMOKE["n_shuffles"] if args.smoke else N_SHUFFLES

    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(apath)
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    pre = cfg["preamble"]
    n_pre = len(model.tokenizer(pre)["input_ids"])
    n_lines = ad.LENGTH_LINES["medium"]
    print(f"{cfg['listener_model']} probe layer {probe.layer} "
          f"(r2_v {probe.r2_v:.3f} r2_a {probe.r2_a:.3f})", flush=True)
    t0 = time.time()

    def measure(lines, target_va):
        """Canonical EMA read (scripts/05) + the anchor read the explore tier uses."""
        hs, _ = model.hidden_states_with_spans(pre, lines, sep=".\n")
        raw = probe.predict(hs[probe.layer])
        traj = ema(raw[n_pre:], cfg["ema_alpha"]) if len(raw) > n_pre else ema(raw, cfg["ema_alpha"])
        text = ".\n".join(lines)
        ah = model.hidden_states(pre + text + ANCH)
        av = probe.predict(ah[probe.layer][-1:])[0]
        return (placement_error(traj, target_va), float(np.linalg.norm(av - np.asarray(target_va))),
                traj[-1], av)

    rows = []
    for cons in cons_list:
        for tname in tgt_list:
            tva = tuple(cfg["targets"][tname])
            start = tuple(cfg["neutral_start"])
            ids = build(art, apath, cons, tva, start, n_lines, cfg)
            base = [art.word(i) for i in ids]
            va = np.array([art.va(i) for i in ids])
            variants = [("ordered", list(range(len(base))))]
            rng = random.Random(1234)
            for k in range(n_shuf):
                p = list(range(len(base)))
                rng.shuffle(p)
                variants.append((f"shuffle{k}", p))
            variants.append(("reversed", list(range(len(base)))[::-1]))
            for label, perm in variants:
                lines = [base[i] for i in perm]
                e_ema, e_anch, t_last, a_last = measure(lines, tva)
                tail = va[perm[-EMA_WINDOW_LINES:]]
                rows.append(dict(constructor=cons, target=tname, variant=label,
                                 kind="ordered" if label == "ordered" else
                                      ("reversed" if label == "reversed" else "shuffled"),
                                 ema_error=e_ema, anchor_error=e_anch,
                                 ema_v=float(t_last[0]), ema_a=float(t_last[1]),
                                 anchor_v=float(a_last[0]), anchor_a=float(a_last[1]),
                                 tail_v=float(tail[:, 0].mean()), tail_a=float(tail[:, 1].mean()),
                                 target_v=tva[0], target_a=tva[1]))
            o = rows[-len(variants)]
            sh = [r for r in rows[-len(variants):] if r["kind"] == "shuffled"]
            print(f"{cons:>16} {tname:>8}  ordered EMA {o['ema_error']:.3f} | "
                  f"shuffled {np.mean([r['ema_error'] for r in sh]):.3f} "
                  f"± {np.std([r['ema_error'] for r in sh]):.3f}  | "
                  f"anchor {o['anchor_error']:.3f} -> "
                  f"{np.mean([r['anchor_error'] for r in sh]):.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"order{tag}.csv", index=False)

    # --- summary: shuffle cost per constructor, both readouts ------------------
    print()
    out = []
    for cons in cons_list:
        d = df[df.constructor == cons]
        o = d[d.kind == "ordered"]
        s = d[d.kind == "shuffled"]
        r = d[d.kind == "reversed"]
        rec = dict(constructor=cons,
                   ema_ordered=o.ema_error.mean(), ema_shuffled=s.ema_error.mean(),
                   ema_cost=s.ema_error.mean() - o.ema_error.mean(),
                   ema_shuf_sd=s.groupby("target").ema_error.std().mean(),
                   ema_reversed_cost=r.ema_error.mean() - o.ema_error.mean(),
                   anch_ordered=o.anchor_error.mean(), anch_shuffled=s.anchor_error.mean(),
                   anch_cost=s.anchor_error.mean() - o.anchor_error.mean(),
                   tail_shift=abs(s.tail_v.mean() - o.tail_v.mean()) + abs(s.tail_a.mean() - o.tail_a.mean()))
        out.append(rec)
    sm = pd.DataFrame(out).set_index("constructor").sort_values("ema_cost")
    print(sm.round(3).to_string())
    sm.to_csv(HERE / f"order_summary{tag}.csv")
    from scipy import stats
    sh_all = df[df.kind == "shuffled"]
    for col, lab in (("ema_error", "EMA (pipeline)"), ("anchor_error", "anchor (explore)")):
        pairs = [(df[(df.constructor == c) & (df.target == t) & (df.kind == "ordered")][col].iloc[0],
                  sh_all[(sh_all.constructor == c) & (sh_all.target == t)][col].mean())
                 for c in cons_list for t in tgt_list]
        d = [b - a for a, b in pairs]
        w = stats.wilcoxon(d) if len(d) > 5 else None
        print(f"\n{lab}: shuffling changes error by {np.mean(d):+.3f} on average, "
              f"hurt in {sum(x > 0 for x in d)}/{len(d)} cells"
              + (f", wilcoxon p={w.pvalue:.3f}" if w else ""))

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    order = sm.index.tolist()
    for ax, col, title in ((axA, "ema_error", "EMA read (the pipeline's metric)"),
                           (axB, "anchor_error", "anchor read (whole context)")):
        for xi, cons in enumerate(order):
            d = df[df.constructor == cons]
            s = d[d.kind == "shuffled"]
            ax.scatter(np.full(len(s), xi) + np.linspace(-.16, .16, len(s)), s[col], s=22,
                       color=BLUE, alpha=.7, lw=0, zorder=3)
            ax.scatter([xi], [d[d.kind == "ordered"][col].mean()], marker="*", s=150,
                       color=ORANGE, lw=0, zorder=5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    axA.set_ylabel("placement error (lower is better)")
    axA.scatter([], [], color=BLUE, s=22, lw=0, label=f"{n_shuf} reorderings x {len(tgt_list)} targets")
    axA.scatter([], [], marker="*", s=150, color=ORANGE, lw=0, label="as constructed")
    axA.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Does line order affect placement? Constructors sorted by the cost of shuffling",
                 fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(HERE / f"order{tag}.png", facecolor=fig.get_facecolor())
    print(f"\nwrote order{tag}.csv order_summary{tag}.csv order{tag}.png  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
