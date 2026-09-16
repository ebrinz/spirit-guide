"""why_polygon — why polygon-pca rises under the calibrated read, and why my pooled
rank-agreement number was misleading.

Analysis only, no model: reads per_stimulus.csv from run.py. Three questions.

  1. The pooled per-stimulus agreement between the two readouts (rho +0.79) counts every stimulus
     together. Both readouts agree that 'excited' is hard and 'focused' is easy, so pooling across
     targets manufactures agreement that says nothing about how constructors are RANKED. Split it.
  2. Within a target, at matched conditions, do the two readouts rank the six constructors the same?
  3. If not, is it because the word-probe EMA read barely moves? Measure the coordinate range each
     readout actually uses, and how far apart it places the six constructors within a target.

Usage: python3 explore/calibrated_leaderboard/why_polygon.py
Writes: within_target.csv, readout_range.csv (committed).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TARGETS = ["calm", "focused", "excited"]


def main():
    d = pd.read_csv(HERE / "per_stimulus.csv").drop_duplicates("id")
    psg = d[(d.generator == "psg") & (~d.constructor.str.startswith("shuffled"))].copy()
    core = psg[(psg["length"] == "medium") & (psg["intensity"] == "plain")
               & (psg["style"] == "unfiltered") & (psg["target"] != "rescue")]

    # 1 + 2 --------------------------------------------------------------------
    rows = [dict(scope="pooled, all psg stimuli", n=len(psg),
                 rho=stats.spearmanr(psg.ema_error, psg.passage_error).statistic)]
    for t in TARGETS:
        s = core[core.target == t]
        rho = stats.spearmanr(s.ema_error, s.passage_error).statistic
        rows.append(dict(scope=f"within target: {t}", n=len(s), rho=rho,
                         ema_best=s.loc[s.ema_error.idxmin(), "constructor"],
                         cal_best=s.loc[s.passage_error.idxmin(), "constructor"],
                         ema_spread=float(np.hypot(s.ema_v.std(), s.ema_a.std())),
                         cal_spread=float(np.hypot(s.passage_v.std(), s.passage_a.std())),
                         ema_lands=f"({s.ema_v.mean():.2f},{s.ema_a.mean():.2f})",
                         cal_lands=f"({s.passage_v.mean():.2f},{s.passage_a.mean():.2f})",
                         target=f"({s.target_v.iloc[0]:.2f},{s.target_a.iloc[0]:.2f})"))
    wt = pd.DataFrame(rows)
    wt.to_csv(HERE / "within_target.csv", index=False)
    print(wt.round(3).to_string(index=False))
    within = [r["rho"] for r in rows[1:]]
    print(f"\nmean within-target rank agreement: {np.mean(within):+.3f}  "
          f"(6 constructors per target; |rho| > 0.83 would be p < 0.05, so only calm's is)")

    # polygon's rank per target ------------------------------------------------
    print("\npolygon-pca rank out of 6, by target:")
    for t in TARGETS:
        s = core[core.target == t].set_index("constructor")
        print(f"  {t:>8}: EMA {s.ema_error['polygon-pca']:.3f} (rank "
              f"{int(s.ema_error.rank()['polygon-pca'])})   calibrated "
              f"{s.passage_error['polygon-pca']:.3f} (rank {int(s.passage_error.rank()['polygon-pca'])})")

    # 3 ------------------------------------------------------------------------
    rng = pd.DataFrame([
        dict(readout="EMA + word probe", v_lo=psg.ema_v.min(), v_hi=psg.ema_v.max(),
             a_lo=psg.ema_a.min(), a_hi=psg.ema_a.max()),
        dict(readout="anchor + passage probe", v_lo=psg.passage_v.min(), v_hi=psg.passage_v.max(),
             a_lo=psg.passage_a.min(), a_hi=psg.passage_a.max())])
    rng["v_range"] = rng.v_hi - rng.v_lo
    rng["a_range"] = rng.a_hi - rng.a_lo
    rng["area"] = rng.v_range * rng.a_range
    rng.to_csv(HERE / "readout_range.csv", index=False)
    print("\ncoordinate range each readout uses across the 40 found-poetry stimuli:")
    print(rng.round(3).to_string(index=False))
    print("\ndoes the readout move when the TARGET moves?")
    for t in TARGETS:
        s = core[core.target == t]
        print(f"  target ({s.target_v.iloc[0]:.2f},{s.target_a.iloc[0]:.2f}) -> "
              f"EMA ({s.ema_v.mean():.2f},{s.ema_a.mean():.2f})   "
              f"calibrated ({s.passage_v.mean():.2f},{s.passage_a.mean():.2f})")


if __name__ == "__main__":
    main()
