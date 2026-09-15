"""line_ablation — which lines of the winning poem carry the effect, and does order matter?

`explore/zone_zoom` showed the score is a property of the TEXT, not of the coordinate the search
aimed at: swapping one line of eight preserved it while re-landing the same coordinate with new
text did not. This asks what in the text does the work, on `gap_09`, the highest-scoring landing
in `explore/model_map` (PANAS inspired 4.04 against a run-wide mean of 3.3).

The battery is deterministic — PANAS and the yes/no bank are argmax/expectation reads over
fixed prompts, no sampling — so repeated measurement of one context is exact and every difference
below is signal, not measurement noise. A duplicate of the original context is run to verify that.

Five context sets, battery only (no search):
  original   — the 8 lines as landed, plus one duplicate as the determinism check
  loo        — leave-one-out: each line removed in turn (8 contexts of 7 lines)
  single     — each line alone (8 contexts of 1 line)
  subset     — 24 random subsets of sizes 2-7, which is what gives each line a coefficient
               with an interval rather than one deletion
  shuffle    — all 8 lines in 5 random orders: does sequence matter, or only the bag of lines?

Per-line attribution is a ridge fit of PANAS inspired on the 8 line-presence indicators plus a
line-count term, across the loo + single + subset contexts, with bootstrap intervals.

Usage:
  python3 explore/line_ablation/run.py            # ~10 min
  python3 explore/line_ablation/run.py --smoke

Outputs (committed): ablation.csv, lines.csv, ablation.png, NOTES.md.
Scratch: explore/scratch/line_ablation/ (states).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener import basq

HERE = Path(__file__).resolve().parent
MAP = REPO_ROOT / "explore/scratch/model_map"
PASSAGE_DIR = REPO_ROOT / "data_gemma2b/passage_probe"
IDEAL_SCRATCH = REPO_ROOT / "explore/scratch/ideal_state"
MODEL = "unsloth/gemma-2-2b-it"
LAYER = 20
ANCH = "\nRight now everything feels"
SOURCE = "gap_09"
N_SUBSETS, N_SHUFFLES, N_BOOT = 24, 5, 2000
SMOKE = dict(n_subsets=3, n_shuffles=2, n_boot=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_sub, n_shuf, n_boot = ((SMOKE["n_subsets"], SMOKE["n_shuffles"], SMOKE["n_boot"])
                             if args.smoke else (N_SUBSETS, N_SHUFFLES, N_BOOT))
    tag = "_smoke" if args.smoke else ""
    scratch = REPO_ROOT / f"explore/scratch/line_ablation{tag}"
    scratch.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    pre = cfg["preamble"]
    model = HiddenStateModel(MODEL, device=cfg["device"])
    probe = load_probe(PASSAGE_DIR / "probe_passage.pkl")
    deep = load_probe(IDEAL_SCRATCH / "probe_L20.pkl")
    bank = json.load(open(cfg["questionnaire_bank"]))
    questions = basq.sample_questions(bank, cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    rng = np.random.default_rng(11)
    t0 = time.time()

    lines = [p.strip() for p in json.loads((MAP / f"landings/{SOURCE}.json").read_text())["ctx"].split(". ")
             if p.strip()]
    n = len(lines)
    print(f"{SOURCE}: {n} lines", flush=True)
    for i, l in enumerate(lines):
        print(f"  [{i}] {l}")

    def measure(kind, label, idxs, order=None):
        """idxs: which original line indices are present; order: the sequence to write them in."""
        seq = order if order is not None else sorted(idxs)
        ctx = ". ".join(lines[i] for i in seq) + ". " if seq else ""
        hs = model.hidden_states(pre + ctx + ANCH)[:, -1, :].astype(np.float32)
        full = pre + ctx.strip() + "\n\n"
        pan = administer_panas(model, full)
        bq = basq.administer(model, questions, full)
        pv, pa = (float(x) for x in probe.predict(hs[probe.layer][None])[0])
        dv, da = (float(x) for x in deep.predict(hs[LAYER][None])[0])
        rec = dict(kind=kind, label=label, n_lines=len(seq), present="".join(
            "1" if i in set(idxs) else "0" for i in range(n)), order="-".join(map(str, seq)),
            probe_v=pv, probe_a=pa, deep_v=dv, deep_a=da, panas_pa=pan["pa"], panas_na=pan["na"],
            panas_inspired=pan["items"]["inspired"], panas_interested=pan["items"]["interested"],
            basq_v=bq["va"][0], basq_a=bq["va"][1])
        np.save(scratch / f"state_{kind}_{label}.npy", hs[LAYER])
        print(f"  {kind:>8} {label:>10}  n={len(seq)}  inspired {rec['panas_inspired']:.2f}  "
              f"PA-NA {pan['pa'] - pan['na']:+.2f}  deep({dv:.2f},{da:.2f})", flush=True)
        return rec

    rows = [measure("original", "full", list(range(n))),
            measure("original", "full_dup", list(range(n)))]
    for i in range(n):
        rows.append(measure("loo", f"drop{i}", [j for j in range(n) if j != i]))
    for i in range(n):
        rows.append(measure("single", f"only{i}", [i]))
    for k in range(n_sub):
        size = int(rng.integers(2, n))
        idxs = sorted(rng.choice(n, size=size, replace=False).tolist())
        rows.append(measure("subset", f"s{k:02d}", idxs))
    for k in range(n_shuf):
        order = rng.permutation(n).tolist()
        rows.append(measure("shuffle", f"p{k}", list(range(n)), order=order))

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"ablation{tag}.csv", index=False)

    a, b = df[df.label == "full"].iloc[0], df[df.label == "full_dup"].iloc[0]
    deterministic = abs(a.panas_inspired - b.panas_inspired) < 1e-9
    print(f"\ndeterminism check: inspired {a.panas_inspired:.6f} vs {b.panas_inspired:.6f} "
          f"-> {'exact' if deterministic else 'DIFFERS'}", flush=True)

    # --- per-line attribution -------------------------------------------------
    fit = df[df.kind.isin(["loo", "single", "subset"])]
    X = np.array([[int(c) for c in p] for p in fit.present], float)
    X = np.column_stack([X, X.sum(1)])                       # + line-count term
    y = fit.panas_inspired.to_numpy()
    ridge = Ridge(alpha=1.0).fit(X, y)
    boot = np.empty((n_boot, n + 1))
    for b_ in range(n_boot):
        s = rng.integers(0, len(X), len(X))
        boot[b_] = Ridge(alpha=1.0).fit(X[s], y[s]).coef_
    lo, hi = np.percentile(boot[:, :n], [2.5, 97.5], axis=0)
    ldf = pd.DataFrame(dict(line=range(n), text=lines, coef=ridge.coef_[:n], lo=lo, hi=hi,
                            loo_inspired=[float(df[df.label == f"drop{i}"].panas_inspired.iloc[0]) for i in range(n)],
                            single_inspired=[float(df[df.label == f"only{i}"].panas_inspired.iloc[0]) for i in range(n)]))
    ldf["loo_drop"] = float(a.panas_inspired) - ldf.loo_inspired
    ldf["excludes_zero"] = (ldf.lo > 0) | (ldf.hi < 0)
    ldf.to_csv(HERE / f"lines{tag}.csv", index=False)
    print(f"\nline-count coefficient: {ridge.coef_[n]:+.3f} inspired per line")
    print(ldf[["line", "coef", "lo", "hi", "loo_drop", "single_inspired", "excludes_zero"]].round(3).to_string(index=False))
    sh = df[df.kind == "shuffle"].panas_inspired
    print(f"\nshuffles (same 8 lines, reordered): {sh.min():.2f}-{sh.max():.2f}, sd {sh.std():.3f}; "
          f"original {a.panas_inspired:.2f}", flush=True)

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, RED, INK, MUTED, GRID = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=120, gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    o = ldf.sort_values("coef")
    ypos = np.arange(len(o))
    axA.barh(ypos, o.coef, height=.62, color=[BLUE if c >= 0 else RED for c in o.coef], zorder=3)
    axA.hlines(ypos, o.lo, o.hi, color=INK, lw=1.4, zorder=4)
    axA.axvline(0, color=MUTED, lw=1)
    axA.set_yticks(ypos)
    axA.set_yticklabels([f"[{int(r.line)}] {r.text[:44]}" for r in o.itertuples()], fontsize=8)
    axA.set_xlabel("contribution to PANAS inspired (ridge coef, 95% bootstrap)")
    axA.set_title("Which of gap_09's eight lines carry the effect", fontsize=10, color=INK)
    for kind, mark, col, lab in (("subset", "o", BLUE, "random subset"), ("loo", "s", RED, "leave-one-out"),
                                 ("single", "^", MUTED, "single line"), ("shuffle", "D", INK, "reordered (all 8)")):
        d = df[df.kind == kind]
        axB.scatter(d.n_lines + np.random.default_rng(3).normal(0, .07, len(d)), d.panas_inspired,
                    marker=mark, s=34, color=col, alpha=.85, lw=0, label=lab, zorder=3)
    axB.axhline(a.panas_inspired, color=INK, lw=1.3, ls="--", zorder=2)
    axB.annotate(f"original, 8 lines ({a.panas_inspired:.2f})", (1, a.panas_inspired), xytext=(2, 5),
                 textcoords="offset points", fontsize=8, color=INK)
    axB.set_xlabel("lines present"); axB.set_ylabel("PANAS inspired")
    axB.set_title("Score against how much of the poem is present", fontsize=10, color=INK)
    axB.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (axA, axB):
        ax.set_facecolor("#fcfcfb")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    axA.grid(axis="x", color=GRID, lw=.6, zorder=0)
    axB.grid(axis="y", color=GRID, lw=.6, zorder=0)
    fig.tight_layout()
    fig.savefig(HERE / f"ablation{tag}.png", facecolor=fig.get_facecolor())
    print(f"\nwrote ablation{tag}.csv lines{tag}.csv ablation{tag}.png  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
