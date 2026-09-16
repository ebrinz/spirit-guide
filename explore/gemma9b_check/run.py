"""gemma9b_check — does the readout disagreement replicate at 9B?

`explore/calibrated_leaderboard` found, on Llama-1B, that the published placement metric
(word probe, per-token, EMA) and a passage-calibrated anchor read agree about which TARGETS are
hard but not about how constructors rank within a target (mean within-target rho -0.05: +0.89 at
calm, -0.60 at focused, -0.43 at excited), and that the word-probe read barely moves when the
target moves. This asks whether that is a Llama-specific instrument quirk or holds at 9B.

No 9B word probe is retrained. The published per-stimulus placement errors in
`results/leaderboard_9b.csv` ARE the 9B EMA read, and the stimulus ids are deterministic — all 74
freshly built stimuli match committed ids exactly — so the EMA side is joined from there and only
the calibrated side is computed here, with the probe from build_probe.py.

Usage:
  python3 explore/gemma9b_check/build_probe.py      # first, ~40 min
  python3 explore/gemma9b_check/run.py              # then, ~10 min

Outputs (committed): per_stimulus_9b.csv, within_target_9b.csv, compare_9b.csv, NOTES.md.
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

HERE = Path(__file__).resolve().parent
MODEL = "unsloth/gemma-2-9b-it"
PROBE = REPO_ROOT / "data_gemma9b/passage_probe/probe_passage.pkl"
PUBLISHED = REPO_ROOT / "results/leaderboard_9b.csv"
ANCH = "\nRight now everything feels"
TARGETS = ["calm", "focused", "excited"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_smoke" if args.smoke else ""
    cfg = load_config()

    stims = []
    for n in ["data/stimuli/stimuli.jsonl", "data/stimuli/stimuli_additions.jsonl"]:
        p = REPO_ROOT / n
        if p.exists():
            stims += [json.loads(l) for l in open(p)]
    pub = pd.read_csv(PUBLISHED).drop_duplicates("id").set_index("id")
    stims = [s for s in stims if s["id"] in pub.index]
    if args.smoke:
        stims = stims[:4]
    print(f"{len(stims)} stimuli with published 9B placement errors", flush=True)

    model = HiddenStateModel(MODEL, device=cfg["device"])
    probe = load_probe(PROBE)
    pre = cfg["preamble"]
    print(f"9B passage probe layer {probe.layer} (R2_v {probe.r2_v:.3f} R2_a {probe.r2_a:.3f})",
          flush=True)
    t0 = time.time()

    rows = []
    for i, s in enumerate(stims):
        tva = np.asarray(s["target_va"])
        hs = model.hidden_states(pre + s["text"] + ANCH)
        pv = probe.predict(hs[probe.layer][-1:])[0]
        rows.append(dict(id=s["id"], constructor=s["constructor"], generator=s["generator"],
                         target=s["target"], length=s["params"].get("length"),
                         intensity=s["params"].get("intensity"), style=s["params"].get("style"),
                         ema_error=float(pub.placement_error[s["id"]]),   # published 9B EMA read
                         passage_error=float(np.linalg.norm(pv - tva)),
                         passage_v=float(pv[0]), passage_a=float(pv[1]),
                         target_v=float(tva[0]), target_a=float(tva[1])))
        if (i + 1) % 20 == 0 or i == len(stims) - 1:
            print(f"  [{i + 1}/{len(stims)}] {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"per_stimulus_9b{tag}.csv", index=False)

    psg = df[(df.generator == "psg") & (~df.constructor.str.startswith("shuffled"))].copy()
    core = psg[(psg["length"] == "medium") & (psg["intensity"] == "plain")
               & (psg["style"] == "unfiltered") & (psg["target"] != "rescue")]

    out = [dict(scope="pooled, all psg", n=len(psg),
                rho=stats.spearmanr(psg.ema_error, psg.passage_error).statistic)]
    for t in TARGETS:
        s = core[core.target == t]
        if len(s) < 3:
            continue
        out.append(dict(scope=f"within target: {t}", n=len(s),
                        rho=stats.spearmanr(s.ema_error, s.passage_error).statistic,
                        ema_best=s.loc[s.ema_error.idxmin(), "constructor"],
                        cal_best=s.loc[s.passage_error.idxmin(), "constructor"],
                        cal_lands=f"({s.passage_v.mean():.2f},{s.passage_a.mean():.2f})",
                        target=f"({s.target_v.iloc[0]:.2f},{s.target_a.iloc[0]:.2f})"))
    wt = pd.DataFrame(out)
    wt.to_csv(HERE / f"within_target_9b{tag}.csv", index=False)
    print("\n" + wt.round(3).to_string(index=False), flush=True)
    within = [r["rho"] for r in out[1:]]
    if within:
        print(f"\nmean within-target rank agreement (9B): {np.mean(within):+.3f}", flush=True)

    lb = psg.groupby("constructor").agg(n=("id", "size"), ema=("ema_error", "mean"),
                                        cal=("passage_error", "mean"))
    lb["ema_rank"] = lb.ema.rank().astype(int)
    lb["cal_rank"] = lb.cal.rank().astype(int)
    lb.sort_values("ema").to_csv(HERE / f"compare_9b{tag}.csv")
    print("\nfound-poetry constructors, 9B:\n" + lb.round(3).sort_values("ema").to_string())

    print("\ndoes the calibrated 9B read move when the target moves?")
    for t in TARGETS:
        s = core[core.target == t]
        if len(s):
            print(f"  target ({s.target_v.iloc[0]:.2f},{s.target_a.iloc[0]:.2f}) -> "
                  f"calibrated ({s.passage_v.mean():.2f},{s.passage_a.mean():.2f})")
    print(f"\nwrote per_stimulus_9b{tag}.csv within_target_9b{tag}.csv compare_9b{tag}.csv "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
