"""polygon_sweep — is the polygon-pca inversion real, or an artifact of one poem?

`explore/gemma9b_check` found that at the `focused` target, polygon-pca ranks LAST of six
constructors under the pipeline's word-probe EMA read and FIRST under a passage-calibrated anchor
read, on both Llama-1B and Gemma-9B. That rested on one poem per constructor, and single-poem
results in this folder have not held up (`zone_zoom`, and my own mis-correction in
`calibrated_leaderboard`). So: resample.

**A construction-seed sweep is impossible, which is itself the first finding.** The `seed`
parameter is a no-op for all six constructors. `_pick_in_band` sorts band members by distance to
the band centre and takes the first k deterministically, consulting its RNG only in a fallback for
under-populated bands that a 50k-line phrase bank never triggers; `polygon_pca` builds
`np.random.RandomState(seed)` and never uses the object; `graph_walk` ignores the argument
entirely. The published `results/seed_expansion.csv` already shows this — its ordered rows for
seeds 43 and 44 are identical to 15 decimal places, and identical to the seed-42 leaderboard.

So this resamples over what does change a poem: **length** (8 / 24 / 56 lines) x **start
coordinate** (4 values), 12 variants per constructor, target `focused`. Note valley ignores
start_va by design, so its variation comes only from length. Each poem is read both ways on the
same model in the same run:
  ema_error     — word-trained probe at every token, EMA alpha 0.1, final value vs target
  passage_error — passage-calibrated probe read once at a fixed anchor after the poem

Models: Llama-1B (`data/probe`, `data/passage_probe`) and Gemma-9B (`data_gemma9b/probe` from
build_9b_word_probe.py, `data_gemma9b/passage_probe`). The 9B word probe is a fresh instrument,
not the one behind the published 9B table — see that script's docstring.

Verdict criterion, stated before running: the inversion is real if polygon-pca's mean rank under
the EMA read sits in the bottom half AND its mean rank under the calibrated read in the top half,
on both models, with the gap exceeding the variant-to-variant spread.

Usage:
  python3 explore/polygon_sweep/run.py --model llama     # ~3 min
  python3 explore/polygon_sweep/run.py --model gemma9b   # ~30 min
  python3 explore/polygon_sweep/run.py --analyze         # tables + figure from saved CSVs

Outputs (committed): sweep_<model>.csv, ranks.csv, sweep.png, NOTES.md.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.analysis.metrics import ema, placement_error
from spiritbench.stimuli import adapter as ad

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
CONSTRUCTORS = ["valley", "harmonic-golden", "harmonic-prime", "harmonic-organic",
                "graph-walk", "polygon-pca"]
# The seed is a NO-OP for every constructor here (see NOTES.md): _pick_in_band takes the k
# band-members nearest the band centre deterministically and only consults its RNG in a
# fallback that a 50k-line bank never triggers, and polygon_pca builds an RNG it never uses.
# So within-constructor variation has to come from parameters that actually change the poem.
SEEDS = [42]
LENGTHS = ["short", "medium", "long"]
STARTS = [(0.5, 0.5), (0.25, 0.80), (0.75, 0.25), (0.40, 0.65)]
TARGET = "focused"

MODELS = {
    "llama": dict(id=None, word="data/probe/probe.pkl", passage="data/passage_probe/probe_passage.pkl"),
    "gemma9b": dict(id="unsloth/gemma-2-9b-it", word="data_gemma9b/probe/probe.pkl",
                    passage="data_gemma9b/passage_probe/probe_passage.pkl"),
}


def build(art, apath, cons, target_va, start_va, n, cfg, seed):
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


def measure_model(name, smoke=False):
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    spec = MODELS[name]
    cfg = load_config()
    model_id = spec["id"] or cfg["listener_model"]
    cons_list = CONSTRUCTORS[:2] if smoke else CONSTRUCTORS

    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(apath)
    word = load_probe(REPO_ROOT / spec["word"])
    pas = load_probe(REPO_ROOT / spec["passage"])
    model = HiddenStateModel(model_id, device=cfg["device"])
    pre = cfg["preamble"]
    n_pre = len(model.tokenizer(pre)["input_ids"])
    tva = tuple(cfg["targets"][TARGET])
    start = tuple(cfg["neutral_start"])
    print(f"{model_id}: word probe layer {word.layer} (R2_v {word.r2_v:.3f}), "
          f"passage probe layer {pas.layer} (R2_v {pas.r2_v:.3f})", flush=True)
    t0 = time.time()

    lengths = LENGTHS[:1] if smoke else LENGTHS
    starts = STARTS[:1] if smoke else STARTS
    rows = []
    for cons in cons_list:
        for length in lengths:
            for start in starts:
                n_lines = ad.LENGTH_LINES[length]
                ids = build(art, apath, cons, tva, start, n_lines, cfg, SEEDS[0])
                lines = [art.word(i) for i in ids]
                hs, _ = model.hidden_states_with_spans(pre, lines, sep=".\n")
                raw = word.predict(hs[word.layer])
                traj = ema(raw[n_pre:], cfg["ema_alpha"]) if len(raw) > n_pre \
                    else ema(raw, cfg["ema_alpha"])
                text = ".\n".join(lines)
                ah = model.hidden_states(pre + text + ANCH)
                pv = pas.predict(ah[pas.layer][-1:])[0]
                rows.append(dict(model=name, constructor=cons, length=length,
                                 start_v=start[0], start_a=start[1], variant=f"{length}/{start}",
                                 ema_error=placement_error(traj, tva),
                                 passage_error=float(np.linalg.norm(pv - np.asarray(tva))),
                                 passage_v=float(pv[0]), passage_a=float(pv[1]),
                                 n_unique_lines=len(set(ids))))
                print(f"  {cons:>16} {length:>6} start {start}  EMA {rows[-1]['ema_error']:.3f}  "
                      f"calibrated {rows[-1]['passage_error']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"sweep_{name}.csv", index=False)
    print(f"wrote sweep_{name}.csv ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return df


def analyze():
    frames = [pd.read_csv(p) for p in sorted(HERE.glob("sweep_*.csv"))]
    if not frames:
        raise SystemExit("no sweep_*.csv yet")
    df = pd.concat(frames)
    out = []
    for m, g in df.groupby("model"):
        # rank constructors within each seed, then average the ranks
        r = g.copy()
        r["ema_rank"] = r.groupby("variant").ema_error.rank()
        r["cal_rank"] = r.groupby("variant").passage_error.rank()
        agg = r.groupby("constructor").agg(
            n=("variant", "size"), ema=("ema_error", "mean"), ema_sd=("ema_error", "std"),
            cal=("passage_error", "mean"), cal_sd=("passage_error", "std"),
            ema_rank=("ema_rank", "mean"), ema_rank_sd=("ema_rank", "std"),
            cal_rank=("cal_rank", "mean"), cal_rank_sd=("cal_rank", "std"))
        agg["rank_gap"] = agg.ema_rank - agg.cal_rank
        agg.insert(0, "model", m)
        out.append(agg.sort_values("ema_rank"))
        print(f"\n=== {m} ({g.variant.nunique()} seeds) ===")
        print(agg.sort_values("ema_rank").round(2).to_string())
        n_c = g.constructor.nunique()
        p = agg.loc["polygon-pca"]
        print(f"polygon-pca: EMA rank {p.ema_rank:.2f}±{p.ema_rank_sd:.2f}, "
              f"calibrated rank {p.cal_rank:.2f}±{p.cal_rank_sd:.2f} (of {n_c}); gap {p.rank_gap:+.2f}")
        pr = r[r.constructor == "polygon-pca"]
        w = stats.wilcoxon(pr.ema_rank, pr.cal_rank) if len(pr) > 5 else None
        if w:
            print(f"  wilcoxon on paired per-seed ranks: p={w.pvalue:.4f}")
        bottom = p.ema_rank > (n_c + 1) / 2
        top = p.cal_rank < (n_c + 1) / 2
        wide = abs(p.rank_gap) > p.ema_rank_sd          # the third clause of the stated criterion
        print(f"  verdict: EMA bottom half {bottom}, calibrated top half {top}, "
              f"gap {abs(p.rank_gap):.2f} > spread {p.ema_rank_sd:.2f} {wide} -> "
              f"{'INVERSION HOLDS' if bottom and top and wide else 'NOT SUPPORTED'}")
        # the same test for every constructor: who do the two readouts actually disagree about?
        print("  rank gap (EMA rank - calibrated rank) per constructor, gap vs spread:")
        for c, row in agg.sort_values("rank_gap").iterrows():
            pr_c = r[r.constructor == c]
            pv = stats.wilcoxon(pr_c.ema_rank, pr_c.cal_rank).pvalue if len(pr_c) > 5 else np.nan
            flag = "PASSES" if abs(row.rank_gap) > row.ema_rank_sd else "fails"
            print(f"    {c:>17}: {row.rank_gap:+.2f} (spread {row.ema_rank_sd:.2f}) "
                  f"p={pv:.3f}  {flag}")
        rho = stats.spearmanr(agg.ema, agg.cal)
        print(f"  constructor rank agreement between readouts: rho {rho.statistic:+.2f}")
    ranks = pd.concat(out)
    ranks.to_csv(HERE / "ranks.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5), dpi=120, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, m in zip(axes[0], models):
        g = df[df.model == m]
        order = g.groupby("constructor").ema_error.mean().sort_values().index.tolist()
        for xi, c in enumerate(order):
            s = g[g.constructor == c]
            col = ORANGE if c == "polygon-pca" else BLUE
            ax.scatter(np.full(len(s), xi) - 0.16, s.ema_error, s=26, color=col, alpha=.55, lw=0, zorder=3)
            ax.scatter(np.full(len(s), xi) + 0.16, s.passage_error, s=26, color=col, alpha=.55,
                       marker="s", lw=0, zorder=3)
            ax.plot([xi - 0.28, xi - 0.04], [s.ema_error.mean()] * 2, color=INK, lw=2, zorder=4)
            ax.plot([xi + 0.04, xi + 0.28], [s.passage_error.mean()] * 2, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{m} · {g.variant.nunique()} construction seeds · target {TARGET}",
                     fontsize=10, color=INK)
        ax.set_ylabel("placement error")
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0][0].scatter([], [], color=MUTED, s=26, lw=0, label="EMA read (circles, left)")
    axes[0][0].scatter([], [], color=MUTED, s=26, lw=0, marker="s", label="calibrated read (squares, right)")
    axes[0][0].scatter([], [], color=ORANGE, s=26, lw=0, label="polygon-pca")
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "sweep.png", facecolor=fig.get_facecolor())
    print("\nwrote ranks.csv sweep.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.model:
        measure_model(args.model, smoke=args.smoke)
    if args.analyze or not args.model:
        analyze()


if __name__ == "__main__":
    main()
