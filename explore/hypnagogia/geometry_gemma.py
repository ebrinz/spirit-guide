"""geometry_gemma — does the selection-geometry effect transfer to a second model?

`coherence_collinearity` established three separable levers on Llama-3.2-1B: line coherence
(-0.109/unit), poem lexical diversity (-0.117/unit), and a selection-geometry term of +0.016 that
survives controlling for both. Its stated bound was that all of it lives inside one semantic mask on
one model. This replays the identical stimuli through Gemma-2-2B.

**The stimuli are model-independent**, built from the phrase graph with no model in the loop, so
this is a clean transfer test: same 24 poems, same seeds, same drift measure, different listener.
A no-poem baseline is added because the headline bound — nothing beats baseline — has to be
re-established per model rather than assumed.

**Why the base model and not gemma-2-2b-it.** Probed first, because `explore/README.md` already logs
"instruct-tuned models analyse the poem instead of inhabiting it". Gemma-2-2B-it does exactly that:
6 of 6 continuations after a poem came back as literary criticism with markdown headers and
discussion questions ("**Here's why this poem isn't easily summarized**"), while the no-poem baseline
produced ordinary meditation prose. That failure is *differential by condition*, so it would not
cancel — and screening the analysis-mode generations out would condition on an outcome that differs
by condition, which is the trap this folder keeps falling into. Three prompt variants were tried;
the best (an explicit "do not explain or analyse" instruction) still left 2 of 6 contaminated.

The base model has no chat behaviour to suppress. `ANALYSIS` below is retained as a **diagnostic**:
its rate is reported per condition, and if it is not near zero the measure is not valid here and the
run says so rather than quietly reporting numbers.

**Batched generation.** 4 batches of 25 rather than 100 sequential calls. Measured on this machine
after warm-up: 0.70 s/sequence batched against 4.24 s/sequence one at a time, which is the difference
between 30 minutes and 4.8 hours. Every condition draws the same four batch seeds, so conditions
still meet matched sampling noise. This does change the RNG stream relative to the sequential Llama
runs, so absolute drift values are not comparable across the two scripts — only the *effects* are,
which is what transfers or fails to.

Usage: python3 explore/hypnagogia/geometry_gemma.py [--smoke] [--model NAME]
Outputs (committed): geometry_gemma.csv, geometry_gemma_summary.csv, geometry_gemma.png
"""
import argparse
import importlib.util
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
SCRATCH = REPO_ROOT / "explore/scratch/geometry_gemma"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
WORD = re.compile(r"[a-z']+")
N_GEN, GEN_TOK, BATCH, N_BOOT, N_LINES = 100, 110, 25, 10000, 16
STARTS = {"S1": (0.50, 0.50), "S2": (0.60, 0.25), "S3": (0.40, 0.45), "S4": (0.70, 0.30)}
WEIGHTS = [0.0, 0.04, 0.07, 0.1, 0.3]
DEFAULT_MODEL = "unsloth/gemma-2-2b"
# diagnostic only: markers of a model critiquing the poem rather than continuing from it
ANALYSIS = re.compile(r"\*\*|^\s*[\*\-]\s|\b(poem|stanza|speaker|imagery|metaphor|evokes?|"
                      r"the author|these lines|this piece|reflects on|symboli)\b", re.I | re.M)


def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


hr = _mod("hr", HERE / "run.py")
ws = _mod("ws", HERE / "weighted_selection.py")
ds = _mod("ds", HERE / "drift_search.py")
cp, hy, rb = hr.cp, hr.hy, hr.rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    n_gen = BATCH if args.smoke else N_GEN
    SCRATCH.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    screen_ok = np.array([not ds.screened(art.word(i)) for i in range(len(art.nodes))])
    mask = ok & no_minor & screen_ok & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    tva = tuple(np.mean([[nrc[w][0], nrc[w][1]]
                         for wss in hy.FACETS.values() for w in wss if w in nrc], axis=0))

    builds = {}
    for k, s in STARTS.items():
        builds[f"polygon_{k}"] = (
            ad.apply_mask_to_path(art, ad.polygon_pca(art, s, tva, N_LINES, 42), mask),
            "polygon", np.nan, k)
        for w in WEIGHTS:
            nm = f"walk_{k}" if w == 0.3 else f"walk_w{int(w * 1000):03d}_{k}"
            builds[nm] = (ws.walk_weighted(art, mask, tva, N_LINES, w, start_va=s), "walk", w, k)

    texts, meta = {}, {}
    for nm, (ids, fam, w, k) in builds.items():
        texts[nm] = ".\n".join(art.word(i) for i in ids)
        tk = [t for i in ids for t in WORD.findall(art.word(i).lower())]
        meta[nm] = dict(build=nm, family=fam, w=w, origin=k,
                        coherence=hr.line_coherence(art, ids),
                        poem_ttr=len(set(tk)) / len(tk), n_distinct=len(set(ids)))
    texts["baseline"] = ""
    meta["baseline"] = dict(build="baseline", family="baseline", w=np.nan, origin="-",
                            coherence=np.nan, poem_ttr=np.nan, n_distinct=0)
    order = ["baseline"] + list(builds)
    print(f"{args.model} · {len(order)} conditions ({len(builds)} poems + baseline)\n", flush=True)

    model = HiddenStateModel(args.model, device=cfg["device"])
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
    for nm in order:
        cpath = SCRATCH / f"{nm}.csv"
        if cpath.exists() and not args.smoke:
            d = pd.read_csv(cpath)
            d["build"] = nm
            rows.append(d)
            print(f"  {nm:<18} cached  drift {d.drift.mean():.4f}", flush=True)
            continue
        body = texts[nm]
        ii = tok(pre + (body + "\n\n" if body else "") + GEN_PROMPT, return_tensors="pt").to(dev)
        recs = []
        for b0 in range(0, n_gen, BATCH):
            n = min(BATCH, n_gen - b0)
            torch.manual_seed(5000 + b0 // BATCH)      # same batch seeds for every condition
            with torch.no_grad():
                o = net.generate(**ii, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id,
                                 num_return_sequences=n)
            for j in range(n):
                txt = tok.decode(o[j, ii["input_ids"].shape[1]:], skip_special_tokens=True)
                es = [e for e in (embed(x) for x in re.split(r"[.!?\n]+", txt) if len(x.split()) > 2)
                      if e is not None]
                hops = [1 - es[i] @ es[i + 1] for i in range(len(es) - 1)]
                toks = WORD.findall(txt.lower())
                recs.append(dict(condition=nm, gen=b0 + j, n_hops=len(hops),
                                 drift=float(np.mean(hops)) if hops else np.nan,
                                 ttr=len(set(toks)) / len(toks) if toks else np.nan,
                                 n_tokens=len(toks),
                                 analysis_mode=bool(ANALYSIS.search(txt))))
        d = pd.DataFrame(recs)
        if not args.smoke:
            d.to_csv(cpath, index=False)
        d["build"] = nm
        rows.append(d)
        print(f"  {nm:<18} drift {d.drift.mean():.4f} ({d.drift.notna().sum()} usable, "
              f"analysis-mode {d.analysis_mode.mean():.0%}) [{time.time() - t0:.0f}s]", flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(HERE / "geometry_gemma.csv", index=False)

    amr = df.analysis_mode.mean()
    print(f"\nVALIDITY CHECK — analysis-mode rate overall {amr:.1%} "
          f"(poems {df[df.build != 'baseline'].analysis_mode.mean():.1%}, "
          f"baseline {df[df.build == 'baseline'].analysis_mode.mean():.1%})")
    if amr > 0.15:
        print("  *** HIGH: this model is critiquing the poems rather than continuing from them.")
        print("  *** Drift here measures how far a literary analysis wanders. Treat as INVALID.")
    else:
        print("  low — continuations are free-associative, so drift means what it means on Llama")

    boot = np.random.default_rng(0)
    per, summ = {}, []
    for nm in order:
        v = df[df.build == nm].drift.dropna().to_numpy()
        per[nm] = v
        bs = np.array([boot.choice(v, len(v), True).mean() for _ in range(N_BOOT)])
        summ.append(dict(meta[nm], n=len(v), drift=v.mean(),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         mean_hops=df[df.build == nm].n_hops.mean(),
                         analysis_rate=df[df.build == nm].analysis_mode.mean()))
    s = pd.DataFrame(summ)
    s.to_csv(HERE / "geometry_gemma_summary.csv", index=False)
    print("\n" + s.sort_values("drift", ascending=False).round(4).to_string(index=False))

    base = s[s.family == "baseline"].drift.iloc[0]
    poems = s[s.family != "baseline"]
    print(f"\nbaseline {base:.4f} · best poem {poems.drift.max():.4f} "
          f"({poems.loc[poems.drift.idxmax(), 'build']}) · "
          f"{'a poem BEATS baseline here' if poems.drift.max() > base else 'nothing beats baseline, as on Llama'}")

    wk, pg = poems[poems.family == "walk"], poems[poems.family == "polygon"]
    lr = stats.linregress(wk.coherence, wk.drift)
    print(f"\nwalk coherence curve (n={len(wk)}): slope {lr.slope:+.4f}, r={lr.rvalue:+.2f}, "
          f"p={lr.pvalue:.4f}")
    pred = lr.intercept + lr.slope * pg.coherence
    resid = pg.drift.to_numpy() - pred.to_numpy()
    rb_ = np.array([np.mean([boot.choice(per[nm], len(per[nm]), True).mean() for nm in pg.build]
                            - pred.to_numpy()) for _ in range(N_BOOT)])
    lo, hi = np.percentile(rb_, [2.5, 97.5])
    print(f"polygon residual from that curve: {resid.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"({int((resid > 0).sum())}/{len(resid)} positive)")

    def ols(d, cols, label):
        X = np.column_stack([np.ones(len(d))] + [
            (d.family == "polygon").astype(float).to_numpy() if c == "fam" else d[c].to_numpy()
            for c in cols])
        y = d.drift.to_numpy()
        be, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ be
        dof = len(d) - X.shape[1]
        se = np.sqrt(np.sum(r ** 2) / dof * np.diag(np.linalg.inv(X.T @ X)))
        i = cols.index("fam") + 1
        tv = be[i] / se[i]
        print(f"  {label:<44} {be[i]:+.4f}  se {se[i]:.4f}  p {2 * stats.t.sf(abs(tv), dof):.4f}")
        return be, se, cols

    print("\nGEOMETRY TERM (Llama gave +0.0160 to +0.0220 across these):")
    ols(poems, ["coherence", "fam"], "24 builds ~ coherence")
    be, se, cols = ols(poems, ["coherence", "poem_ttr", "fam"], "24 builds ~ coherence + poem TTR")
    ols(poems, ["coherence", "poem_ttr", "mean_hops", "fam"], "24 builds ~ + mean hops")
    O = pd.get_dummies(poems.origin, prefix="o", drop_first=True).astype(float)
    ols(pd.concat([poems.reset_index(drop=True), O], axis=1),
        ["coherence", "poem_ttr", "o_S2", "o_S3", "o_S4", "fam"], "24 builds ~ + origin fixed effects")
    lo_, hi_ = max(wk.coherence.min(), pg.coherence.min()), min(wk.coherence.max(), pg.coherence.max())
    b = poems[(poems.coherence >= lo_) & (poems.coherence <= hi_)]
    t = stats.ttest_ind(b[b.family == "polygon"].drift, b[b.family == "walk"].drift)
    print(f"  {'matched band, raw difference':<44} "
          f"{b[b.family == 'polygon'].drift.mean() - b[b.family == 'walk'].drift.mean():+.4f}"
          f"{'':>17}p {t.pvalue:.4f}  (n={len(b)})")

    print(f"\ncoherence {be[1]:+.4f} (Llama -0.1085) · poem TTR {be[2]:+.4f} (Llama -0.1170)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=120)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    xs = np.linspace(poems.coherence.min() - .01, poems.coherence.max() + .01, 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, color=ORANGE, lw=1.8, zorder=2,
            label=f"walk curve (r={lr.rvalue:+.2f})")
    ax.axvspan(lo_, hi_, color=GRID, alpha=.55, zorder=0)
    ax.axhline(base, color=MUTED, ls="--", lw=1.2, zorder=1, label="no-poem baseline")
    for fam, c, mk in (("walk", ORANGE, "o"), ("polygon", BLUE, "D")):
        q = poems[poems.family == fam]
        ax.errorbar(q.coherence, q.drift, yerr=[q.drift - q.ci_lo, q.ci_hi - q.drift],
                    fmt=mk, ms=7, color=c, ecolor=INK, elinewidth=1.1, lw=0, zorder=4, label=fam)
    ax.set_xlabel("line coherence"); ax.set_ylabel("associative drift, 95% CI")
    ax.set_title(f"Does the geometry effect transfer? {args.model}", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "geometry_gemma.png", facecolor=fig.get_facecolor())
    print(f"\nwrote geometry_gemma.csv geometry_gemma_summary.csv geometry_gemma.png "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
