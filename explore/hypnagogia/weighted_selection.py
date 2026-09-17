"""weighted_selection — a third attempt at a coherence knob, on the variable that actually matters.

Two knobs have failed. Selection breadth (`coherence_sweep.py`) moved coherence only 0.84–0.94 over a
1000-fold change in k. The affective schedule (`schedule_sweep.py`) moved it not at all: six paths
from static to shuffled-bands all landed 0.925–0.956, and the most disjointed path gave the *highest*
coherence. What separates the stock constructors, which span 0.617–0.947, is that some never consult
meaning: `valley` picks by distance to a band centre, my walk picks by nearest in meaning.

So this makes that the explicit parameter. At each step, every eligible line in the band is scored

    score = w · (similarity in meaning to the previous line) + (1 − w) · (proximity to band centre)

with both terms min–max normalised within the band so the weight is meaningful. **w = 1** reproduces
the coherent walk; **w = 0** reproduces valley's rule, selecting purely on affect and ignoring what
the line is about. Schedule (linear ramp), mask, target, length and seed are all held fixed.

**Prediction, stated before running.** If the diagnosis is right this third knob should finally span
the range monotonically, and with a clean span the coherence→self-perplexity relationship can be
tested properly — it was ρ = −0.79 to −0.89 inside a narrow band and fell to −0.54 when the span was
widened by mixing in constructors that differ in many ways at once.

Usage: python3 explore/hypnagogia/weighted_selection.py
Outputs (committed): weighted_selection.csv, weighted_selection.png
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
ANCH = "\nRight now everything feels"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
N_LINES, N_GEN, GEN_TOK = 24, 4, 50
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]

_r = importlib.util.spec_from_file_location("hr", HERE / "run.py")
hr = importlib.util.module_from_spec(_r); _r.loader.exec_module(hr)
cp, hy, rb = hr.cp, hr.hy, hr.rb


def norm(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def walk_weighted(art, mask, target_va, n_lines, w, seed=42, start_va=(0.6, 0.25)):
    """w=1: always the nearest in meaning (the coherent walk).
       w=0: always the nearest to the band centre (valley's rule), meaning ignored.

    start_va is where the affective schedule begins. It defaults to the value this
    was written with, so every existing caller is unchanged; `rule_vs_text.py` varies
    it to get several builds from one rule."""
    rng = np.random.RandomState(seed)
    va = ad._va_array(art)
    V = art.vectors / np.maximum(np.linalg.norm(art.vectors, axis=1, keepdims=True), 1e-9)
    idx = np.where(mask)[0]
    tv0, ta0 = start_va
    sched = [(tv0 + (target_va[0] - tv0) * j / (n_lines - 1),
              ta0 + (target_va[1] - ta0) * j / (n_lines - 1)) for j in range(n_lines)]
    used, out, cur = set(), [], None
    for tv, ta in sched:
        band = np.array([i for i in idx[(np.abs(va[idx, 0] - tv) < 0.18)
                                        & (np.abs(va[idx, 1] - ta) < 0.18)] if i not in used])
        if len(band) == 0:
            band = np.array([i for i in idx if i not in used])
        centre = norm(-np.linalg.norm(va[band] - np.array([tv, ta]), axis=1))
        if cur is None:
            pick = band[int(np.argmax(centre))] if w < 1 else band[rng.randint(len(band))]
        else:
            sim = norm(V[band] @ V[cur])
            pick = band[int(np.argmax(w * sim + (1 - w) * centre))]
        out.append(int(pick)); used.add(int(pick)); cur = int(pick)
    return out


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    mask = ok & no_minor & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    tva = tuple(np.mean([[nrc[w][0], nrc[w][1]]
                         for ws in hy.FACETS.values() for w in ws if w in nrc], axis=0))

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    pre = cfg["preamble"]
    t0 = time.time()
    ii = tok(pre + ANCH, return_tensors="pt").to(dev)
    with torch.no_grad():
        p0 = torch.softmax(net(**ii).logits[0, -1].float(), -1)
    base_H = float(-(p0 * torch.log(p0 + 1e-12)).sum())
    print(f"baseline entropy {base_H:.3f}\n", flush=True)

    rows, texts = [], {}
    for w in WEIGHTS:
        ids = walk_weighted(art, mask, tva, N_LINES, w)
        text = ".\n".join(art.word(i) for i in ids)
        texts[w] = text
        coh = hr.line_coherence(art, ids)
        va = np.array([art.va(i) for i in ids])
        aff_err = float(np.linalg.norm(va.mean(0) - np.array(tva)))
        jj = tok(pre + text + ANCH, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(net(**jj).logits[0, -1].float(), -1)
        H = float(-(p * torch.log(p + 1e-12)).sum())
        part = float(1.0 / (p ** 2).sum())
        ppls = []
        for g in range(N_GEN):
            torch.manual_seed(100 + g)
            kk = tok(pre + text + "\n\n" + GEN_PROMPT, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = net.generate(**kk, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
                mm = tok(tok.decode(o[0], skip_special_tokens=True), return_tensors="pt").to(dev)
                ppls.append(float(torch.exp(net(**mm, labels=mm["input_ids"]).loss)))
        rows.append(dict(w=w, line_coherence=coh, affect_error=aff_err, entropy=H,
                         d_entropy=H - base_H, participation=part,
                         self_perplexity=float(np.mean(ppls)), n_distinct=len(set(ids))))
        r = rows[-1]
        print(f"  w={w:<5} coherence {coh:.3f}  affect-err {aff_err:.3f}  H {H:.3f} "
              f"({r['d_entropy']:+.3f})  part {part:6.1f}  ppl {r['self_perplexity']:6.1f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "weighted_selection.csv", index=False)
    lo, hi = df.line_coherence.min(), df.line_coherence.max()
    print(f"\ncoherence spans {lo:.3f}–{hi:.3f}  (k-sweep 0.84–0.94; schedule sweep 0.93–0.96; "
          f"stock constructors 0.62–0.95)")
    print(f"  w vs coherence:            rho {stats.spearmanr(df.w, df.line_coherence).statistic:+.2f} "
          f"(p={stats.spearmanr(df.w, df.line_coherence).pvalue:.3f})")
    for c in ("self_perplexity", "entropy", "participation", "affect_error"):
        r = stats.spearmanr(df.line_coherence, df[c])
        print(f"  coherence vs {c:>16}: rho {r.statistic:+.2f} (p={r.pvalue:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    a1.plot(df.w, df.line_coherence, "o-", color=BLUE, lw=2, ms=6, zorder=3)
    a1.set_xlabel("w  (0 = affect only, 1 = meaning only)"); a1.set_ylabel("line coherence")
    a1.set_title("Does the knob work?", fontsize=10, color=INK)
    a2.plot(df.line_coherence, df.self_perplexity, "o-", color=BLUE, lw=2, ms=6, zorder=3)
    a2.set_xlabel("line coherence"); a2.set_ylabel("self-perplexity")
    a2.set_title("Continuation difficulty", fontsize=10, color=INK)
    a3.plot(df.line_coherence, df.entropy, "o-", color=ORANGE, lw=2, ms=6, zorder=3)
    a3.axhline(base_H, color=MUTED, ls="--", lw=1.2, zorder=2)
    a3.set_xlabel("line coherence"); a3.set_ylabel("next-token entropy")
    a3.set_title("Distribution flatness", fontsize=10, color=INK)
    for ax in (a1, a2, a3):
        ax.set_facecolor("#fcfcfb"); ax.grid(color=GRID, lw=.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Weighting meaning against affect in the selection rule", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "weighted_selection.png", facecolor=fig.get_facecolor())
    (HERE / "weighted_poems.md").write_text("# Poems by selection weight\n\n" + "\n".join(
        f"## w = {w}  (coherence {df[df.w == w].line_coherence.iloc[0]:.3f})\n\n```\n{t}\n```\n"
        for w, t in texts.items()))
    print(f"\nwrote weighted_selection.csv weighted_poems.md weighted_selection.png "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
