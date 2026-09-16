"""reversal_test — the causal test of why the pipeline's metric flatters valley.

`explore/polygon_sweep` §3b proposed a mechanism: the published placement metric weights the last
~30 tokens at 96%, valley is the only constructor that puts target-band content last (its NRC
arousal rises +0.206 from body to closing lines, more than double any other), so the metric reads
valley's ascent and the calibrated whole-context read does not.

That account makes a falsifiable prediction. **Reverse the poem** — same lines, same bag of content,
opposite order — and:
  1. the EMA read should get WORSE, in proportion to how much each constructor's arousal rises from
     body to tail (reversing moves the target band out of the metric's window);
  2. the calibrated read should barely move, since a whole-context read sees the same lines.
If instead reversal hurts both equally, the mechanism is wrong and something else drives the gap.

Design: 6 constructors x 3 lengths x 4 start coordinates = 72 poems per model, each measured
ordered and reversed, both readouts, target `focused`. Predictor is the ORDERED poem's body-to-tail
NRC arousal rise, computed from the phrase graph without the model.

Usage:
  python3 explore/reversal_test/run.py --model llama      # ~4 min
  python3 explore/reversal_test/run.py --model gemma9b    # ~45 min
  python3 explore/reversal_test/run.py --analyze

Outputs (committed): reversal_<model>.csv, reversal_summary.csv, reversal.png, NOTES.md.
"""
import argparse
import importlib.util
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
TAIL_LINES = 4          # the EMA's effective window at 24 lines

_spec = importlib.util.spec_from_file_location("ps", REPO_ROOT / "explore/polygon_sweep/run.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def measure_model(name, smoke=False):
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    spec = ps.MODELS[name]
    cfg = load_config()
    model_id = spec["id"] or cfg["listener_model"]
    cons_list = ps.CONSTRUCTORS[:2] if smoke else ps.CONSTRUCTORS
    lengths = ps.LENGTHS[:1] if smoke else ps.LENGTHS
    starts = ps.STARTS[:1] if smoke else ps.STARTS

    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(apath)
    word = load_probe(REPO_ROOT / spec["word"])
    pas = load_probe(REPO_ROOT / spec["passage"])
    model = HiddenStateModel(model_id, device=cfg["device"])
    pre = cfg["preamble"]
    n_pre = len(model.tokenizer(pre)["input_ids"])
    tva = np.array(cfg["targets"]["focused"])
    print(f"{model_id}: word layer {word.layer}, passage layer {pas.layer}", flush=True)
    t0 = time.time()

    def read(lines):
        hs, _ = model.hidden_states_with_spans(pre, lines, sep=".\n")
        raw = word.predict(hs[word.layer])
        traj = ema(raw[n_pre:], cfg["ema_alpha"]) if len(raw) > n_pre else ema(raw, cfg["ema_alpha"])
        ah = model.hidden_states(pre + ".\n".join(lines) + ANCH)
        pv = pas.predict(ah[pas.layer][-1:])[0]
        return (placement_error(traj, tva), float(np.linalg.norm(pv - tva)),
                float(traj[-1][1]), float(pv[1]))

    rows = []
    for cons in cons_list:
        for length in lengths:
            for start in starts:
                n = ad.LENGTH_LINES[length]
                ids = ps.build(art, apath, cons, tuple(tva), start, n, cfg, 42)
                va = np.array([art.va(i) for i in ids])
                rise = float(va[-TAIL_LINES:, 1].mean() - va[:, 1].mean())   # body -> tail arousal
                lines = [art.word(i) for i in ids]
                for orient, seq in (("ordered", lines), ("reversed", lines[::-1])):
                    e, p, ea, pa = read(seq)
                    rows.append(dict(model=name, constructor=cons, length=length,
                                     start_v=start[0], start_a=start[1], orient=orient,
                                     variant=f"{length}/{start}", tail_rise=rise,
                                     ema_error=e, passage_error=p, ema_a=ea, passage_a=pa))
                print(f"  {cons:>16} {length:>6} {str(start):>12}  "
                      f"EMA {rows[-2]['ema_error']:.3f}->{rows[-1]['ema_error']:.3f}  "
                      f"cal {rows[-2]['passage_error']:.3f}->{rows[-1]['passage_error']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"reversal_{name}.csv", index=False)
    print(f"wrote reversal_{name}.csv ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return df


def analyze():
    frames = [pd.read_csv(p) for p in sorted(HERE.glob("reversal_*.csv"))
              if "summary" not in p.name]
    if not frames:
        raise SystemExit("no reversal_*.csv yet")
    df = pd.concat(frames)
    wide = df.pivot_table(index=["model", "constructor", "variant", "tail_rise"],
                          columns="orient", values=["ema_error", "passage_error"]).reset_index()
    wide.columns = ["model", "constructor", "variant", "tail_rise",
                    "ema_ord", "ema_rev", "cal_ord", "cal_rev"]
    wide["d_ema"] = wide.ema_rev - wide.ema_ord
    wide["d_cal"] = wide.cal_rev - wide.cal_ord
    out = []
    for m, g in wide.groupby("model"):
        agg = g.groupby("constructor").agg(
            tail_rise=("tail_rise", "mean"), n=("variant", "size"),
            d_ema=("d_ema", "mean"), d_ema_sd=("d_ema", "std"),
            d_cal=("d_cal", "mean"), d_cal_sd=("d_cal", "std")).sort_values("tail_rise", ascending=False)
        agg.insert(0, "model", m)
        out.append(agg)
        print(f"\n=== {m} — cost of reversing (positive = worse when reversed) ===")
        print(agg.round(3).to_string())
        r1 = stats.spearmanr(agg.tail_rise, agg.d_ema)
        r2 = stats.spearmanr(agg.tail_rise, agg.d_cal)
        print(f"  prediction 1 — tail_rise vs EMA cost: rho {r1.statistic:+.2f} p={r1.pvalue:.3f}")
        print(f"  prediction 2 — tail_rise vs calibrated cost: rho {r2.statistic:+.2f} p={r2.pvalue:.3f}")
        w1 = stats.wilcoxon(g.d_ema)
        w2 = stats.wilcoxon(g.d_cal)
        print(f"  overall: EMA {g.d_ema.mean():+.3f} (p={w1.pvalue:.4f}), "
              f"calibrated {g.d_cal.mean():+.3f} (p={w2.pvalue:.4f}), "
              f"paired diff p={stats.wilcoxon(g.d_ema, g.d_cal).pvalue:.4f}")
        v = agg.loc["valley"]
        print(f"  valley specifically: EMA {v.d_ema:+.3f}±{v.d_ema_sd:.3f}, "
              f"calibrated {v.d_cal:+.3f}±{v.d_cal_sd:.3f}")
    summ = pd.concat(out)
    summ.to_csv(HERE / "reversal_summary.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 5), dpi=120, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, m in zip(axes[0], models):
        a = summ[summ.model == m]
        ax.axhline(0, color=MUTED, lw=1, zorder=2)
        ax.errorbar(a.tail_rise, a.d_ema, yerr=a.d_ema_sd / np.sqrt(a.n), fmt="o", color=BLUE,
                    ms=8, lw=1.4, capsize=3, zorder=4, label="EMA read (the metric)")
        ax.errorbar(a.tail_rise, a.d_cal, yerr=a.d_cal_sd / np.sqrt(a.n), fmt="s", color=ORANGE,
                    ms=8, lw=1.4, capsize=3, zorder=4, label="calibrated read")
        for c, row in a.iterrows():
            ax.annotate(c, (row.tail_rise, row.d_ema), xytext=(5, 6), textcoords="offset points",
                        fontsize=7.5, color=INK)
        ax.set_xlabel("body → tail arousal rise of the ordered poem")
        ax.set_ylabel("cost of reversing (placement error)")
        ax.set_title(f"{m}", fontsize=10, color=INK)
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=GRID, lw=.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Does reversing a poem cost the recency-weighted metric more than a whole-context read?",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "reversal.png", facecolor=fig.get_facecolor())
    print("\nwrote reversal_summary.csv reversal.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(ps.MODELS))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.model:
        measure_model(args.model, smoke=args.smoke)
    if args.analyze or not args.model:
        analyze()


if __name__ == "__main__":
    main()
