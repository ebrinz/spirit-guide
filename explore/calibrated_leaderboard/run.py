"""calibrated_leaderboard — the full leaderboard under the pipeline's metric and a calibrated read.

`explore/order_matters` found, on 15 constructor x target cells, that the published placement
metric (probe every token, EMA alpha 0.1, final value vs target) and a passage-calibrated
whole-context read rank the constructors differently (rho -0.10). This runs the comparison on the
full canonical stimulus set instead of 15 cells.

Two readouts per stimulus, same texts, same model:
  ema_error     — the published measure, reproduced here: data/probe/probe.pkl (word-trained,
                  layer 10) read at every token, EMA, final value's distance to target.
  passage_error — data/passage_probe/probe_passage.pkl (passage-trained, layer 15, held-out
                  R2_v 0.919) read once at a fixed anchor after the poem.

These are two different measures, not a correction of one by the other: the passage probe was
trained on anchor-token states, so it cannot be run per-token inside an EMA without going
off-distribution. The question is only whether the leaderboard's ORDERING depends on which is used.

Usage:
  python3 explore/calibrated_leaderboard/run.py            # ~5 min
  python3 explore/calibrated_leaderboard/run.py --smoke

Outputs (committed): per_stimulus.csv, leaderboard_compare.csv, by_target.csv,
leaderboard_compare.png, NOTES.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.analysis.metrics import ema, placement_error

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_smoke" if args.smoke else ""
    cfg = load_config()

    stims = []
    for name in ["data/stimuli/stimuli.jsonl", "data/stimuli/stimuli_additions.jsonl",
                 "data/renders/renders.jsonl"]:
        p = REPO_ROOT / name
        if p.exists():
            stims += [json.loads(l) for l in open(p)]
    if args.smoke:
        stims = stims[:6]
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    word = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    pas = load_probe(REPO_ROOT / "data/passage_probe/probe_passage.pkl")
    pre = cfg["preamble"]
    n_pre = len(model.tokenizer(pre)["input_ids"])
    print(f"{len(stims)} stimuli · word probe layer {word.layer} (R2_v {word.r2_v:.3f}) · "
          f"passage probe layer {pas.layer} (R2_v {pas.r2_v:.3f})", flush=True)
    t0 = time.time()

    rows = []
    for i, s in enumerate(stims):
        lines = s["lines"] if s["lines"] else [s["text"]]
        sep = ".\n" if s["text"] == ".\n".join(lines) else "\n"
        tva = tuple(s["target_va"])
        hs, _ = model.hidden_states_with_spans(pre, lines, sep=sep)
        raw = word.predict(hs[word.layer])
        traj = ema(raw[n_pre:], cfg["ema_alpha"]) if len(raw) > n_pre else ema(raw, cfg["ema_alpha"])
        ah = model.hidden_states(pre + s["text"] + ANCH)
        pv = pas.predict(ah[pas.layer][-1:])[0]
        rows.append(dict(id=s["id"], constructor=s["constructor"], generator=s["generator"],
                         target=s["target"], length=s["params"].get("length"),
                         intensity=s["params"].get("intensity"), style=s["params"].get("style"),
                         ema_error=placement_error(traj, tva),
                         passage_error=float(np.linalg.norm(pv - np.asarray(tva))),
                         ema_v=float(traj[-1][0]), ema_a=float(traj[-1][1]),
                         passage_v=float(pv[0]), passage_a=float(pv[1]),
                         target_v=tva[0], target_a=tva[1], n_lines=len(lines)))
        if (i + 1) % 20 == 0 or i == len(stims) - 1:
            print(f"  [{i + 1}/{len(stims)}] {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"per_stimulus{tag}.csv", index=False)

    # --- leaderboard, grouped as the published table is ------------------------
    lb = df.groupby(["constructor", "generator"]).agg(
        n=("id", "size"), ema=("ema_error", "mean"), passage=("passage_error", "mean")).reset_index()
    lb["ema_rank"] = lb.ema.rank()
    lb["passage_rank"] = lb.passage.rank()
    lb["rank_shift"] = lb.ema_rank - lb.passage_rank
    lb = lb.sort_values("ema")
    lb.to_csv(HERE / f"leaderboard_compare{tag}.csv", index=False)
    print("\n" + lb.round(3).to_string(index=False), flush=True)
    rho = stats.spearmanr(lb.ema, lb.passage)
    print(f"\nrank agreement across {len(lb)} constructor x generator cells: "
          f"spearman {rho.statistic:+.3f} p={rho.pvalue:.4f}", flush=True)

    # psg only — the row set the README's table reports
    psg = lb[lb.generator == "psg"].sort_values("ema")
    if len(psg) > 2:
        r2 = stats.spearmanr(psg.ema, psg.passage)
        print(f"psg (found poetry) only, {len(psg)} constructors: spearman {r2.statistic:+.3f} "
              f"p={r2.pvalue:.4f}", flush=True)

    # --- per target ------------------------------------------------------------
    bt = df[df.generator == "psg"].pivot_table(index="constructor", columns="target",
                                               values=["ema_error", "passage_error"])
    bt.to_csv(HERE / f"by_target{tag}.csv")
    print("\npsg only, by target:\n" + bt.round(3).to_string(), flush=True)

    # --- figure ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=120,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#fcfcfb")
    axA.scatter(df.ema_error, df.passage_error, s=26, color=BLUE, alpha=.7, lw=0, zorder=3)
    r = stats.spearmanr(df.ema_error, df.passage_error)
    axA.set_xlabel("placement error, pipeline EMA read")
    axA.set_ylabel("placement error, calibrated anchor read")
    axA.set_title(f"Per stimulus, n={len(df)}  ·  spearman {r.statistic:+.2f}", fontsize=10, color=INK)
    lim = [0, max(df.ema_error.max(), df.passage_error.max()) * 1.05]
    axA.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=2)
    axA.set_xlim(lim); axA.set_ylim(lim)
    # slope chart of the psg ranking
    p = psg.sort_values("ema").reset_index(drop=True)
    for i, row in p.iterrows():
        e_r, p_r = row.ema_rank, row.passage_rank
        axB.plot([0, 1], [e_r, p_r], color=ORANGE if e_r != p_r else MUTED, lw=1.8, zorder=3)
        axB.scatter([0, 1], [e_r, p_r], s=42, color=INK, zorder=4, lw=0)
        axB.annotate(f"{row.constructor}  {row.ema:.3f}", (0, e_r), xytext=(-8, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=8, color=INK)
        axB.annotate(f"{row.passage:.3f}  {row.constructor}", (1, p_r), xytext=(8, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=8, color=INK)
    axB.set_xlim(-.85, 1.85); axB.invert_yaxis()
    axB.set_xticks([0, 1]); axB.set_xticklabels(["pipeline EMA", "calibrated anchor"], fontsize=9)
    axB.set_yticks([])
    axB.set_title("Found-poetry constructor ranking, by readout", fontsize=10, color=INK)
    for ax in (axA, axB):
        ax.set_facecolor("#fcfcfb")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    axA.grid(color=GRID, lw=.6, zorder=0)
    for s_ in ("left", "bottom"):
        axB.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / f"leaderboard_compare{tag}.png", facecolor=fig.get_facecolor())
    print(f"\nwrote per_stimulus{tag}.csv leaderboard_compare{tag}.csv by_target{tag}.csv "
          f"leaderboard_compare{tag}.png  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
