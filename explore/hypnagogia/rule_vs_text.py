"""rule_vs_text — separate "the construction rule did it" from "those two texts did it".

The powered run found polygon-pca at 0.2429 drift against the weighted walk's 0.1939, a difference
of +0.049 with the concept, target, semantic mask, content filters and length all held fixed. That
was **one build against one build**, so "different rule" and "different text" were perfectly
confounded. This separates them.

**It also fixes a confound in that comparison.** The two builds did not start from the same place:
`polygon_pca` was called with `neutral_start` (0.5, 0.5) while `walk_weighted` hardcoded its schedule
origin at (0.6, 0.25). So the earlier claim that *only* the selection rule differed was overstated —
the trajectory origin differed too. Here both rules are run from the same four origins.

**Design: 2 rules x 4 trajectory origins = 8 poems, fully crossed.**

    rule    polygon-pca   orbits a local PCA neighbourhood in the phrase bank's vector space
            walk (w=0.3)  picks from the affective band, weighted toward meaning

    origin  S1 (0.50, 0.50)   S2 (0.60, 0.25)   S3 (0.40, 0.45)   S4 (0.70, 0.30)

The origin is the one knob both rules genuinely expose: the `seed` argument is a no-op for
`polygon_pca` (it builds an RandomState and never reads it) and inert for the walk at w < 1, so
seed-sweeping would produce eight identical poems. Origin produces eight nearly disjoint ones —
within-rule Jaccard on chosen lines is 0.03-0.06, cross-rule at the same origin 0.00-0.03.

polygon@S1 and walk@S2 rebuild the two conditions from the powered run exactly. They are regenerated
rather than reused from its cache, and the script checks that they reproduce its numbers — a free
internal replication of the result being interrogated.

**The analysis that answers the question.** Pooling generations within a rule treats the generation
as the unit and would understate the standard error enormously, because generations from one poem
are not independent draws of "that rule". The unit here is the **poem**: resample 4 poems with
replacement within each rule, then generations within each drawn poem, and take the rule difference
over 10,000 such draws. If the rule is the cause, the four poems of each family separate despite
sharing no lines. If the original result was a property of two particular texts, the families
overlap and the hierarchical interval covers zero.

**A rival explanation tested at the same time.** The rules differ systematically in line coherence
(polygon 0.65-0.76, walk 0.82-0.86). With eight poems spanning that range, drift can be regressed on
coherence directly to ask whether "the rule" is really "coherence".

Usage: python3 explore/hypnagogia/rule_vs_text.py [--smoke]
Outputs (committed): rule_vs_text.csv, rule_vs_text_summary.csv, rule_vs_text_poems.md,
                     rule_vs_text.png
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
SCRATCH = REPO_ROOT / "explore/scratch/rule_vs_text"
OLD = REPO_ROOT / "explore/scratch/powered_drift"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
WORD = re.compile(r"[a-z']+")
N_GEN, GEN_TOK, N_BOOT, N_LINES = 100, 110, 10000, 16
STARTS = {"S1": (0.50, 0.50), "S2": (0.60, 0.25), "S3": (0.40, 0.45), "S4": (0.70, 0.30)}
# the two builds the powered run already generated, and the condition name each reused from
REUSE = {("polygon", "S1"): "polygon_pca", ("walk", "S2"): "semantic"}


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
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_gen = 4 if args.smoke else N_GEN
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
    print(f"target ({tva[0]:.3f}, {tva[1]:.3f}) · {int(mask.sum())} eligible lines\n", flush=True)

    builds = {}
    for k, s in STARTS.items():
        builds[("polygon", k)] = ad.apply_mask_to_path(
            art, ad.polygon_pca(art, s, tva, N_LINES, 42), mask)
        builds[("walk", k)] = ws.walk_weighted(art, mask, tva, N_LINES, 0.3, start_va=s)

    texts, meta = {}, {}
    for (rule, k), ids in builds.items():
        texts[(rule, k)] = ".\n".join(art.word(i) for i in ids)
        va = np.array([art.va(i) for i in ids])
        meta[(rule, k)] = dict(
            rule=rule, start=k, start_v=STARTS[k][0], start_a=STARTS[k][1],
            n_distinct=len(set(ids)), coherence=hr.line_coherence(art, ids),
            affect_error=float(np.linalg.norm(va.mean(0) - np.asarray(tva))))
        m = meta[(rule, k)]
        print(f"  {rule}@{k:<3} distinct {m['n_distinct']:>2}/{N_LINES}  "
              f"coherence {m['coherence']:.3f}  affect-err {m['affect_error']:.3f}", flush=True)

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
    for (rule, k) in builds:
        name = f"{rule}_{k}"
        cpath = SCRATCH / f"{name}.csv"
        if cpath.exists():
            d = pd.read_csv(cpath)
            d["rule"], d["start"], d["build"] = rule, k, name
            rows.append(d)
            print(f"  {name:<12} cached ({d.drift.notna().sum()} usable)", flush=True)
            continue
        ctx = pre + texts[(rule, k)] + "\n\n" + GEN_PROMPT
        ii = tok(ctx, return_tensors="pt").to(dev)
        recs = []
        for g in range(n_gen):
            torch.manual_seed(5000 + g)                   # same seeds as the powered run
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
            if (g + 1) % 50 == 0:
                print(f"    {name}: {g + 1}/{n_gen}  [{time.time() - t0:.0f}s]", flush=True)
        d = pd.DataFrame(recs)
        d.to_csv(cpath, index=False)
        d["rule"], d["start"], d["build"] = rule, k, name
        rows.append(d)
        print(f"  {name:<12} drift {d.drift.mean():.4f} ({d.drift.notna().sum()} usable)",
              flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(HERE / "rule_vs_text.csv", index=False)

    # free internal replication: two of the eight are the powered run's own conditions
    prev_path = HERE / "powered_drift.csv"
    if prev_path.exists() and not args.smoke:
        prev = pd.read_csv(prev_path)
        print("\nreproduction check against the powered run (same seeds, rebuilt from scratch):")
        for (rule, k), old in REUSE.items():
            a = df[df.build == f"{rule}_{k}"].drift.dropna()
            b = prev[prev.condition == old].drift.dropna()
            if len(b):
                print(f"  {rule}@{k} vs {old}: {a.mean():.4f} (n={len(a)}) vs {b.mean():.4f} "
                      f"(n={len(b)})  delta {a.mean() - b.mean():+.5f}")

    boot = np.random.default_rng(0)
    per = {}
    summ = []
    for (rule, k) in builds:
        v = df[(df.rule == rule) & (df.start == k)].drift.dropna().to_numpy()
        per[(rule, k)] = v
        bs = np.array([boot.choice(v, len(v), True).mean() for _ in range(N_BOOT)])
        summ.append(dict(meta[(rule, k)], n=len(v), drift=v.mean(),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         mean_hops=df[(df.rule == rule) & (df.start == k)].n_hops.mean()))
    s = pd.DataFrame(summ).sort_values("drift", ascending=False)
    s.to_csv(HERE / "rule_vs_text_summary.csv", index=False)
    print("\n" + s.round(4).to_string(index=False))

    poly = np.array([per[("polygon", k)].mean() for k in STARTS])
    walk = np.array([per[("walk", k)].mean() for k in STARTS])
    print(f"\npolygon poems: {np.round(poly, 4)}  mean {poly.mean():.4f}  sd {poly.std(ddof=1):.4f}")
    print(f"walk poems:    {np.round(walk, 4)}  mean {walk.mean():.4f}  sd {walk.std(ddof=1):.4f}")
    sep = poly.min() > walk.max()
    print(f"families {'SEPARATE COMPLETELY' if sep else 'OVERLAP'}: "
          f"lowest polygon {poly.min():.4f} vs highest walk {walk.max():.4f}")

    # hierarchical bootstrap: resample POEMS within rule, then generations within poem
    ks = list(STARTS)
    hb = []
    for _ in range(N_BOOT):
        a = np.mean([boot.choice(per[("polygon", ks[i])], len(per[("polygon", ks[i])]), True).mean()
                     for i in boot.integers(0, 4, 4)])
        b = np.mean([boot.choice(per[("walk", ks[i])], len(per[("walk", ks[i])]), True).mean()
                     for i in boot.integers(0, 4, 4)])
        hb.append(a - b)
    hb = np.array(hb)
    lo, hi = np.percentile(hb, [2.5, 97.5])
    print(f"\nRULE EFFECT, poem as the unit of resampling: {hb.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          f"  {'SIGNIFICANT' if not (lo <= 0 <= hi) else 'not significant'}")
    flat = np.concatenate([per[("polygon", k)] for k in ks]).mean() \
        - np.concatenate([per[("walk", k)] for k in ks]).mean()
    print(f"  (pooling generations instead would give {flat:+.4f} on a far too narrow interval)")
    print(f"  between-poem sd within rule: polygon {poly.std(ddof=1):.4f}, walk "
          f"{walk.std(ddof=1):.4f}; rule gap {poly.mean() - walk.mean():+.4f}")
    t = stats.ttest_ind(poly, walk)
    print(f"  4-vs-4 t-test on poem means: t={t.statistic:.2f}, p={t.pvalue:.4f}")

    # is "the rule" really "coherence"?
    co = s.coherence.to_numpy(); dr = s.drift.to_numpy()
    r_all = stats.pearsonr(co, dr)
    print(f"\ncoherence vs drift across all 8 poems: r={r_all.statistic:+.2f} (p={r_all.pvalue:.3f})")
    for rule in ("polygon", "walk"):
        q = s[s.rule == rule]
        rr = stats.pearsonr(q.coherence, q.drift)
        print(f"  within {rule:<8} r={rr.statistic:+.2f} (p={rr.pvalue:.3f}, n=4)")
    print("  -> a rule effect that is really coherence would show a strong WITHIN-rule slope too")

    (HERE / "rule_vs_text_poems.md").write_text(
        "# The eight poems: 2 rules x 4 trajectory origins\n\n"
        "Same concept, target, semantic mask, content filters and length throughout. Only the "
        "selection rule and the origin of the affective trajectory vary.\n\n" + "\n".join(
            f"## {r}@{k}  (origin {STARTS[k]}, coherence {meta[(r, k)]['coherence']:.3f}, "
            f"drift {per[(r, k)].mean():.4f})\n\n```\n{texts[(r, k)]}\n```\n"
            for (r, k) in builds))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=120,
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    o = s.sort_values("drift")
    yy = np.arange(len(o))
    cols = [BLUE if r == "polygon" else ORANGE for r in o.rule]
    a1.hlines(yy, o.ci_lo, o.ci_hi, color=INK, lw=1.5, zorder=3)
    a1.scatter(o.drift, yy, s=58, c=cols, zorder=4, lw=0)
    a1.set_yticks(yy); a1.set_yticklabels([f"{r}@{k}" for r, k in zip(o.rule, o.start)], fontsize=9)
    a1.set_xlabel("associative drift, 95% CI")
    a1.set_title("Eight poems, two rules, nearly disjoint texts", fontsize=10, color=INK)
    for lab, c in (("polygon-pca", BLUE), ("walk w=0.3", ORANGE)):
        a1.scatter([], [], s=58, color=c, label=lab, lw=0)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    for rule, c in (("polygon", BLUE), ("walk", ORANGE)):
        q = s[s.rule == rule]
        a2.scatter(q.coherence, q.drift, s=58, color=c, lw=0, zorder=4, label=rule)
    a2.set_xlabel("line coherence"); a2.set_ylabel("associative drift")
    a2.set_title("Is the rule effect really coherence?", fontsize=10, color=INK)
    a2.legend(frameon=False, fontsize=8)
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb"); ax.grid(color=GRID, lw=.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "rule_vs_text.png", facecolor=fig.get_facecolor())
    print(f"\nwrote rule_vs_text.csv rule_vs_text_summary.csv rule_vs_text_poems.md "
          f"rule_vs_text.png ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
