"""schedule_sweep — vary the affective schedule, which is where coherence actually comes from.

`coherence_sweep.py` established that line coherence predicts self-perplexity (rho -0.79 to -0.89
across three arms) but could only sample coherence 0.84-0.94, because its knob — how many
nearest-in-meaning candidates to choose among — turned out to be a weak lever. `valley` sits at
0.617, outside that range entirely, and the diagnosis was that low coherence comes from valley's
three-phase band schedule jumping between distant affective regions rather than from selection
breadth.

This tests that diagnosis and extends the curve. **Selection is held at k=1 (always the nearest in
meaning) for every schedule, so the affective path is the only thing that varies.**

Schedules, all ending at the same target:
  static      — every line drawn from the target band; no movement at all
  linear      — smooth ramp from a low-arousal ground to the target (the previous walk)
  descend     — the reverse: start at the target, descend to ground
  oscillate   — linear ramp with a sinusoidal deviation in arousal, as the harmonic constructors do
  valley3     — valley's own three-phase shape: ground band, three ascending sub-bands, target band
  random_band — the same bands as `linear`, visited in shuffled order: maximum jumping

Plus the six stock constructors as reference points, since they were built by other people for other
reasons and should sample the space differently.

**Prediction, stated before running.** If the diagnosis is right, schedules that jump between distant
bands (`valley3`, `random_band`) should produce markedly lower coherence than smooth ones, finally
spanning the range; and if the coherence→perplexity relationship is real rather than an artifact of a
narrow band, it should hold across the wider span.

Usage: python3 explore/hypnagogia/schedule_sweep.py
Outputs (committed): schedule_sweep.csv, schedule_sweep.png
"""
import importlib.util
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
CP = REPO_ROOT / "explore/creativity_poem"
ANCH = "\nRight now everything feels"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
N_LINES, N_GEN, GEN_TOK = 24, 4, 50

_r = importlib.util.spec_from_file_location("hr", HERE / "run.py")
hr = importlib.util.module_from_spec(_r); _r.loader.exec_module(hr)
cp, hy, rb = hr.cp, hr.hy, hr.rb


def schedule(name, target, n, rng):
    """(valence, arousal) waypoint per line. Ground is the low-arousal start valley uses."""
    tv, ta = target
    gv, ga = 0.6, 0.25
    lin = [(gv + (tv - gv) * j / (n - 1), ga + (ta - ga) * j / (n - 1)) for j in range(n)]
    if name == "static":
        return [(tv, ta)] * n
    if name == "linear":
        return lin
    if name == "descend":
        return lin[::-1]
    if name == "oscillate":
        return [(v, a + 0.18 * np.sin(2 * np.pi * 3 * j / n)) for j, (v, a) in enumerate(lin)]
    if name == "valley3":                      # valley's own three-phase shape
        n1, n3 = max(1, n // 3), max(1, n // 4)
        n2 = n - n1 - n3
        out = [(0.75, 0.20)] * n1              # grounding band centre
        for j in range(n2):
            f = (j + 1) / (n2 + 1)
            out.append((0.6 + f * (tv - 0.6), ga + f * (ta - ga)))
        return out + [(tv, ta)] * n3
    if name == "random_band":
        idx = rng.permutation(n)
        return [lin[i] for i in idx]
    raise ValueError(name)


def walk(art, mask, sched, seed=42):
    """k=1 selection: always the nearest in meaning within the band for this step."""
    rng = np.random.RandomState(seed)
    va = ad._va_array(art)
    V = art.vectors / np.maximum(np.linalg.norm(art.vectors, axis=1, keepdims=True), 1e-9)
    idx = np.where(mask)[0]
    used, out, cur = set(), [], None
    for tv, ta in sched:
        band = np.array([i for i in idx[(np.abs(va[idx, 0] - tv) < 0.18)
                                        & (np.abs(va[idx, 1] - ta) < 0.18)] if i not in used])
        if len(band) == 0:
            band = np.array([i for i in idx if i not in used])
        pick = band[rng.randint(len(band))] if cur is None else band[int(np.argmax(V[band] @ V[cur]))]
        out.append(int(pick)); used.add(int(pick)); cur = int(pick)
    return out


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    mask = ok & no_minor & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    tva = tuple(np.mean([[nrc[w][0], nrc[w][1]]
                         for ws in hy.FACETS.values() for w in ws if w in nrc], axis=0))
    start = tuple(cfg["neutral_start"])
    rng = np.random.RandomState(7)

    builds = {}
    for name in ("static", "linear", "descend", "oscillate", "valley3", "random_band"):
        builds[f"schedule:{name}"] = walk(art, mask, schedule(name, tva, N_LINES, rng))
    # stock constructors, masked where they accept one
    builds["stock:valley"] = ad.valley_shape(art, tva, N_LINES, 42, mask)
    for h in ("golden", "prime", "organic"):
        try:
            ids = ad.harmonic(art, apath, start, tva, N_LINES, h, 42, cfg["ot_repo"],
                              cfg["semantic_axes"])
            builds[f"stock:harmonic-{h}"] = ad.apply_mask_to_path(art, ids, mask)
        except Exception as e:                                   # noqa: BLE001
            print(f"  harmonic-{h} unavailable: {type(e).__name__}", flush=True)
    builds["stock:polygon-pca"] = ad.apply_mask_to_path(
        art, ad.polygon_pca(art, start, tva, N_LINES, 42), mask)
    try:
        builds["stock:graph-walk"] = ad.apply_mask_to_path(
            art, ad.graph_walk(art, start, tva, N_LINES, 42, cfg["ot_repo"]), mask)
    except Exception as e:                                       # noqa: BLE001
        print(f"  graph-walk unavailable: {type(e).__name__}", flush=True)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    pre = cfg["preamble"]
    t0 = time.time()
    ids_ = tok(pre + ANCH, return_tensors="pt").to(dev)
    with torch.no_grad():
        p0 = torch.softmax(net(**ids_).logits[0, -1].float(), -1)
    base_H = float(-(p0 * torch.log(p0 + 1e-12)).sum())
    print(f"baseline entropy {base_H:.3f}\n", flush=True)

    rows = []
    for name, ids in builds.items():
        text = ".\n".join(art.word(i) for i in ids)
        coh = hr.line_coherence(art, ids)
        ii = tok(pre + text + ANCH, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(net(**ii).logits[0, -1].float(), -1)
        H = float(-(p * torch.log(p + 1e-12)).sum())
        part = float(1.0 / (p ** 2).sum())
        ppls = []
        for g in range(N_GEN):
            torch.manual_seed(100 + g)
            jj = tok(pre + text + "\n\n" + GEN_PROMPT, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = net.generate(**jj, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
                kk = tok(tok.decode(o[0], skip_special_tokens=True), return_tensors="pt").to(dev)
                ppls.append(float(torch.exp(net(**kk, labels=kk["input_ids"]).loss)))
        rows.append(dict(build=name, kind=name.split(":")[0], line_coherence=coh, entropy=H,
                         d_entropy=H - base_H, participation=part,
                         self_perplexity=float(np.mean(ppls)), n_distinct=len(set(ids))))
        r = rows[-1]
        print(f"  {name:<24} coherence {coh:.3f}  H {H:.3f} ({r['d_entropy']:+.3f})  "
              f"part {part:6.1f}  ppl {r['self_perplexity']:6.1f}  distinct {r['n_distinct']}",
              flush=True)

    df = pd.DataFrame(rows).sort_values("line_coherence")
    df.to_csv(HERE / "schedule_sweep.csv", index=False)
    lo, hi = df.line_coherence.min(), df.line_coherence.max()
    print(f"\ncoherence now spans {lo:.3f}–{hi:.3f} "
          f"(the k-sweep managed 0.84–0.94)")
    for col in ("self_perplexity", "entropy", "participation"):
        r = stats.spearmanr(df.line_coherence, df[col])
        print(f"  coherence vs {col:>16}: rho {r.statistic:+.2f} (p={r.pvalue:.3f}, n={len(df)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, col, lab in ((a1, "self_perplexity", "self-perplexity of continuations"),
                         (a2, "entropy", "next-token entropy")):
        for kind, col_ in (("schedule", BLUE), ("stock", ORANGE)):
            g = df[df.kind == kind]
            ax.scatter(g.line_coherence, g[col], s=52, color=col_, lw=0, zorder=3,
                       label="schedule variant" if kind == "schedule" else "stock constructor")
        for r in df.itertuples():
            ax.annotate(r.build.split(":")[1], (r.line_coherence, getattr(r, col)),
                        xytext=(4, 4), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_xlabel("line-to-line coherence"); ax.set_ylabel(lab)
        ax.set_facecolor("#fcfcfb"); ax.grid(color=GRID, lw=.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    a2.axhline(base_H, color=MUTED, ls="--", lw=1.2, zorder=2)
    a1.legend(frameon=False, fontsize=8)
    fig.suptitle("Varying the affective schedule spans the coherence range the k-knob could not",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "schedule_sweep.png", facecolor=fig.get_facecolor())
    print(f"\nwrote schedule_sweep.csv schedule_sweep.png ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
