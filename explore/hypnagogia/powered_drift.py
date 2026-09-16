"""powered_drift — the properly powered comparison. No search, fixed poems, 100 continuations each.

Two searches failed because the objective's standard error exceeded the effect: optimising a
quantity that noisy just fits sampling noise, and both did. The fix is not a better search. It is to
stop searching, fix a small set of candidates, and measure each one enough times to know whether the
differences are real.

**Six conditions**, all previously built, nothing new:
    baseline            — no poem
    semantic            — the w=0.3 hypnagogia poem, about the state; led on drift in both prior runs
    random_screened     — 16 random lines from the screened pool
    drift_searched      — the output of the failed direct search
    flow                — the opposite affective pole
    neutral_prose       — matched-length declarative prose

**100 continuations per condition**, 600 generations total, fixed seeds shared across conditions so
every condition meets the same sampling noise.

**Clustered bootstrap.** Sentence-to-sentence hops within one continuation are correlated — they come
from the same sample — so resampling hops would badly understate the interval. The unit of
resampling is the *generation*: each contributes one mean-hop value, and the 95% interval is taken
over 10,000 resamples of those 100 values. Pairwise differences are bootstrapped the same way.

**Pre-registered question.** Does the semantic poem's drift advantage (0.290 and 0.287 in two
underpowered runs, against ~0.244 baseline) survive? A difference of ~0.04 is what we are trying to
resolve, so the interval on each condition needs to be well inside that.

Usage: python3 explore/hypnagogia/powered_drift.py [--smoke]
Outputs (committed): powered_drift.csv, powered_drift_summary.csv, powered_drift.png
"""
import argparse
import importlib.util
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
SCRATCH = REPO_ROOT / "explore/scratch/powered_drift"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
WORD = re.compile(r"[a-z']+")
# 55 tokens left most continuations with <2 sentences, so drift was undefined and
# two-thirds of samples were discarded. 110 gives 3+ sentences in nearly all of them.
N_GEN, GEN_TOK, N_BOOT = 100, 110, 10000

_r = importlib.util.spec_from_file_location("hr", HERE / "run.py")
hr = importlib.util.module_from_spec(_r); _r.loader.exec_module(hr)
_w = importlib.util.spec_from_file_location("ws", HERE / "weighted_selection.py")
ws = importlib.util.module_from_spec(_w); _w.loader.exec_module(ws)
_d = importlib.util.spec_from_file_location("ds", HERE / "drift_search.py")
ds = importlib.util.module_from_spec(_d); _d.loader.exec_module(ds)
cp, hy, fl, rb = hr.cp, hr.hy, hr.fl, hr.rb

NEUTRAL = ("The room contains a table and two chairs. A window faces the street. Papers are stacked "
           "on the desk. The clock shows the hour. Outside a car passes. The shelves hold books in "
           "no particular order. A cup sits near the lamp. The floor is wooden. Light comes from "
           "the left. The door stays closed. A calendar hangs on the wall. The air is still.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_gen = 6 if args.smoke else N_GEN
    SCRATCH.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    screen_ok = np.array([not ds.screened(art.word(i)) for i in range(len(art.nodes))])
    pool = np.where(ok & no_minor & screen_ok)[0]
    cen = lambda F: tuple(np.mean([[nrc[w][0], nrc[w][1]]                      # noqa: E731
                                   for wss in F.values() for w in wss if w in nrc], axis=0))
    hyp_mask = ok & no_minor & screen_ok & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    flow_mask = ok & no_minor & screen_ok & hr.facet_union(art, fl.FACETS, vocab, cfg["glove_path"])

    import random as _r2
    rng = _r2.Random(5)
    line = lambda ids: ".\n".join(art.word(i) for i in ids)                     # noqa: E731
    conditions = {
        "baseline": "",
        "semantic": line(ws.walk_weighted(art, hyp_mask, cen(hy.FACETS), 16, 0.3)),
        "random_screened": line(rng.sample(list(pool), 16)),
        "flow": line(ws.walk_weighted(art, flow_mask, cen(fl.FACETS), 16, 0.3)),
        "neutral_prose": NEUTRAL,
    }
    dp = HERE / "drift_poems.md"
    if dp.exists():
        t = dp.read_text()
        if "## drift-searched (max)" in t:
            conditions["drift_searched"] = t.split("## drift-searched (max)")[1].split("```")[1].strip()

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    pre = cfg["preamble"]
    WV, WK = ds.load_wordvecs(cfg)

    def embed(t):
        ts = [x for x in WORD.findall(t.lower()) if x in WK]
        if not ts:
            return None
        v = WV[[WK[x] for x in ts]].mean(0)
        return v / max(np.linalg.norm(v), 1e-9)

    t0 = time.time()
    rows = []
    for name, body in conditions.items():
        cpath = SCRATCH / f"{name}.csv"
        if cpath.exists():
            rows.append(pd.read_csv(cpath))
            print(f"  {name:<16} cached", flush=True)
            continue
        ctx = pre + (body + "\n\n" if body else "") + GEN_PROMPT
        ii = tok(ctx, return_tensors="pt").to(dev)
        recs = []
        for g in range(n_gen):
            torch.manual_seed(5000 + g)                    # shared across conditions
            with torch.no_grad():
                o = net.generate(**ii, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
            txt = tok.decode(o[0, ii["input_ids"].shape[1]:], skip_special_tokens=True)
            es = [e for e in (embed(x) for x in re.split(r"[.!?\n]+", txt) if len(x.split()) > 2)
                  if e is not None]
            hops = [1 - es[i] @ es[i + 1] for i in range(len(es) - 1)]
            toks = WORD.findall(txt.lower())
            recs.append(dict(condition=name, gen=g, n_hops=len(hops),
                             drift=float(np.mean(hops)) if hops else np.nan,
                             ttr=len(set(toks)) / len(toks) if toks else np.nan,
                             n_tokens=len(toks)))
            if (g + 1) % 25 == 0:
                print(f"    {name}: {g + 1}/{n_gen}  [{time.time() - t0:.0f}s]", flush=True)
        d = pd.DataFrame(recs)
        d.to_csv(cpath, index=False)
        rows.append(d)
        print(f"  {name:<16} drift {d.drift.mean():.4f} (n={d.drift.notna().sum()} usable)",
              flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(HERE / "powered_drift.csv", index=False)

    # clustered bootstrap: the generation is the unit of resampling
    boot = np.random.default_rng(0)
    summ = []
    vals = {}
    for name, g in df.groupby("condition"):
        v = g.drift.dropna().to_numpy()
        vals[name] = v
        bs = np.array([boot.choice(v, len(v), replace=True).mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        summ.append(dict(condition=name, n=len(v), drift=v.mean(), ci_lo=lo, ci_hi=hi,
                         sd=v.std(ddof=1), se=v.std(ddof=1) / np.sqrt(len(v)),
                         mean_hops=g.n_hops.mean(), ttr=g.ttr.mean()))
    s = pd.DataFrame(summ).sort_values("drift", ascending=False)
    s.to_csv(HERE / "powered_drift_summary.csv", index=False)
    print("\n" + s.round(4).to_string(index=False))

    print("\npairwise differences vs each reference, bootstrapped over generations:")
    for ref in ("baseline", "random_screened"):
        if ref not in vals:
            continue
        print(f"  against {ref}:")
        for name, v in vals.items():
            if name == ref:
                continue
            d = np.array([boot.choice(v, len(v), True).mean()
                          - boot.choice(vals[ref], len(vals[ref]), True).mean()
                          for _ in range(N_BOOT)])
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"    {name:<16} {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=120)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    o = s.sort_values("drift")
    yy = np.arange(len(o))
    ax.hlines(yy, o.ci_lo, o.ci_hi, color=INK, lw=1.6, zorder=3)
    ax.scatter(o.drift, yy, s=54, color=BLUE, zorder=4, lw=0)
    bl = s[s.condition == "baseline"]
    if len(bl):
        ax.axvline(bl.drift.iloc[0], color=MUTED, ls="--", lw=1.2, zorder=2)
    ax.set_yticks(yy); ax.set_yticklabels(o.condition, fontsize=9)
    ax.set_xlabel("associative drift (mean sentence-to-sentence distance), 95% CI")
    ax.set_title(f"{int(s.n.iloc[0])} continuations per condition, bootstrapped over generations",
                 fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=.6, zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "powered_drift.png", facecolor=fig.get_facecolor())
    print(f"\nwrote powered_drift.csv powered_drift_summary.csv powered_drift.png "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
