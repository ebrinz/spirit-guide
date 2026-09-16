"""coherence_sweep — how does line-to-line discontinuity change the model's output distribution?

`run.py` found that every poem flattens the next-token distribution enormously while matched-length
prose sharpens it, and that the affective target contributes little on top. The natural reading was
"discontinuity drives it". But the two builds compared there disagreed with that:

    valley, affective band only   line coherence 0.617   entropy +1.35
    coherent walk                 line coherence 0.935   entropy +2.05

More coherent, *more* entropy — the opposite of the naive prediction. Those two also differ in their
affective schedule (valley's three-phase grounding vs a linear ramp), so it was not a clean test.

This is the clean test. One walk algorithm, one knob: at each step, choose uniformly among the **k
most similar** eligible lines to the previous one. k = 1 is maximally coherent; large k approaches
random selection within the affective band. Everything else is held fixed.

Swept on **both** the hypnagogia and flow targets, because `run.py` concluded the effect tracks
register rather than affective target — if that is right the two curves should have the same shape.

**Prediction: none.** The one prior data point points the wrong way for the obvious story, and I do
not have a confident direction. That is stated so the result cannot be retrofitted.

Usage: python3 explore/hypnagogia/coherence_sweep.py [--smoke]
Outputs (committed): coherence_sweep.csv, coherence_sweep.png
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
CP = REPO_ROOT / "explore/creativity_poem"
ANCH = "\nRight now everything feels"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
N_LINES, N_GEN, GEN_TOK = 24, 4, 50
KS = [1, 2, 3, 5, 10, 25, 50, 100, 250, 1000]

_r = importlib.util.spec_from_file_location("hr", HERE / "run.py")
hr = importlib.util.module_from_spec(_r); _r.loader.exec_module(hr)
cp, hy, fl, rb = hr.cp, hr.hy, hr.fl, hr.rb


def walk_topk(art, mask, target_va, n_lines, k, seed=42):
    """Identical to run.py's coherent_walk except the next line is chosen uniformly from the k
    nearest in meaning rather than always the single nearest. k=1 reproduces it exactly."""
    rng = np.random.RandomState(seed)
    va = ad._va_array(art)
    V = art.vectors / np.maximum(np.linalg.norm(art.vectors, axis=1, keepdims=True), 1e-9)
    idx = np.where(mask)[0]
    sched = [(0.6 + (target_va[0] - 0.6) * j / (n_lines - 1),
              0.25 + (target_va[1] - 0.25) * j / (n_lines - 1)) for j in range(n_lines)]
    used, out, cur = set(), [], None
    for tv, ta in sched:
        band = np.array([i for i in idx[(np.abs(va[idx, 0] - tv) < 0.18)
                                        & (np.abs(va[idx, 1] - ta) < 0.18)] if i not in used])
        if len(band) == 0:
            band = np.array([i for i in idx if i not in used])
        if cur is None:
            pick = band[rng.randint(len(band))]
        else:
            sims = V[band] @ V[cur]
            top = band[np.argsort(-sims)[:min(k, len(band))]]
            pick = top[rng.randint(len(top))]
        out.append(int(pick)); used.add(int(pick)); cur = int(pick)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ks = KS[:3] if args.smoke else KS
    n_gen = 2 if args.smoke else N_GEN

    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    cen = lambda ws: tuple(np.mean([[nrc[w][0], nrc[w][1]] for w in ws if w in nrc], axis=0))  # noqa: E731
    hyp_t = cen([w for ws in hy.FACETS.values() for w in ws])
    targets = {
        "hypnagogia": (hyp_t,
                       ok & no_minor & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])),
        "flow": (cen([w for ws in fl.FACETS.values() for w in ws]),
                 ok & no_minor & hr.facet_union(art, fl.FACETS, vocab, cfg["glove_path"])),
        # The semantic mask compresses the coherence range: even k=1000 within it stays near 0.85,
        # while valley's band-centre picks sit at 0.62. This arm drops the mask so the knob can
        # actually reach the discordant end, at the cost of the poem no longer being on-topic.
        "no_mask": (hyp_t, ok & no_minor),
    }

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    pre = cfg["preamble"]
    t0 = time.time()

    # baseline, once
    ids_ = tok(pre + ANCH, return_tensors="pt").to(dev)
    with torch.no_grad():
        p0 = torch.softmax(net(**ids_).logits[0, -1].float(), -1)
    base_H = float(-(p0 * torch.log(p0 + 1e-12)).sum())
    base_part = float(1.0 / (p0 ** 2).sum())
    print(f"baseline: H {base_H:.3f} participation {base_part:.1f}", flush=True)

    rows = []
    for tname, (tva, mask) in targets.items():
        for k in ks:
            ids = walk_topk(art, mask, tva, N_LINES, k)
            text = ".\n".join(art.word(i) for i in ids)
            coh = hr.line_coherence(art, ids)
            ii = tok(pre + text + ANCH, return_tensors="pt").to(dev)
            with torch.no_grad():
                p = torch.softmax(net(**ii).logits[0, -1].float(), -1)
            H = float(-(p * torch.log(p + 1e-12)).sum())
            part = float(1.0 / (p ** 2).sum())
            top1 = float(torch.sort(p, descending=True).values[0])
            ppls = []
            for g in range(n_gen):
                torch.manual_seed(100 + g)
                ctx = pre + text + "\n\n" + GEN_PROMPT
                jj = tok(ctx, return_tensors="pt").to(dev)
                with torch.no_grad():
                    o = net.generate(**jj, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                     top_p=0.95, repetition_penalty=1.2,
                                     pad_token_id=tok.eos_token_id)
                    kk = tok(tok.decode(o[0], skip_special_tokens=True), return_tensors="pt").to(dev)
                    ppls.append(float(torch.exp(net(**kk, labels=kk["input_ids"]).loss)))
            rows.append(dict(target=tname, k=k, line_coherence=coh, entropy=H,
                             d_entropy=H - base_H, participation=part, top1_mass=top1,
                             self_perplexity=float(np.mean(ppls)),
                             n_distinct=len(set(ids))))
            r = rows[-1]
            print(f"  {tname:>10} k={k:<5} coherence {coh:.3f}  H {H:.3f} ({r['d_entropy']:+.3f})  "
                  f"part {part:6.1f}  top1 {top1:.3f}  ppl {r['self_perplexity']:.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "coherence_sweep.csv", index=False)
    from scipy import stats
    print()
    for tname, g in df.groupby("target"):
        r1 = stats.spearmanr(g.line_coherence, g.entropy)
        r2 = stats.spearmanr(g.line_coherence, g.self_perplexity)
        print(f"{tname}: coherence vs entropy rho {r1.statistic:+.2f} (p={r1.pvalue:.3f}); "
              f"vs self-perplexity rho {r2.statistic:+.2f} (p={r2.pvalue:.3f})")
    a = df[df.target == "hypnagogia"].set_index("k").entropy
    b = df[df.target == "flow"].set_index("k").entropy
    print(f"curve shape agreement across targets: rho {stats.spearmanr(a, b).statistic:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e2e1dc"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    for (tname, g), col in zip(df.groupby("target"), (BLUE, ORANGE)):
        g = g.sort_values("line_coherence")
        a1.plot(g.line_coherence, g.entropy, "o-", color=col, lw=2, ms=6, label=tname, zorder=3)
        a2.plot(g.line_coherence, g.self_perplexity, "o-", color=col, lw=2, ms=6, label=tname, zorder=3)
    a1.axhline(base_H, color=MUTED, ls="--", lw=1.2, zorder=2)
    a1.annotate("baseline, no poem", (a1.get_xlim()[0], base_H), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
    a1.set_ylabel("next-token entropy"); a2.set_ylabel("self-perplexity of continuations")
    a1.set_title("Distribution flatness against line coherence", fontsize=10, color=INK)
    a2.set_title("How hard the model finds it to continue", fontsize=10, color=INK)
    for ax in (a1, a2):
        ax.set_xlabel("line-to-line coherence (higher = reads more continuously)")
        ax.set_facecolor("#fcfcfb"); ax.grid(color=GRID, lw=.6, zorder=0)
        ax.legend(frameon=False, fontsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "coherence_sweep.png", facecolor=fig.get_facecolor())
    print(f"\nwrote coherence_sweep.csv coherence_sweep.png ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
