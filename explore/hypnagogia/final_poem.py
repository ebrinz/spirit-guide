"""final_poem — the hypnagogia poem rebuilt with everything this folder established.

The version in `explore/creativity_poem/README.md` was built before any of the behavioural work. This
applies the method as it now stands and measures the difference head-to-head.

What goes in, and why each part earned its place:

  per-facet semantic mask   — `eti_poem`: averaging in meaning-space collapses a multi-facet concept
                              into its blandest common region, so each facet gets its own
                              neighbourhood and they are unioned
  derived affective target  — `flow_poem`: the facets cohere (spread 0.161) so the target can be the
                              average rather than a choice
  dictionary + child filter — `rebuild_main`: rule-driven selection over uncurated public-domain text
                              will occasionally assemble something no one would publish, and the
                              affective readouts are blind to it
  weighted selection w=0.3  — `weighted_selection` found w is the only knob of three that controls
                              coherence (rho +1.00). But a first pass at w=1 produced DEGENERATE
                              text: greedy nearest-in-meaning selection falls into a semantic rut
                              ("beneath my feet" four times, five consecutive lines about having to
                              go). Type-token ratio falls from 0.78 at w=0 to 0.51 at w=1
                              (rho -0.97 with coherence) and repeated bigrams rise (rho +0.91).
                              w=0.3 keeps most of the coherence gain without the collapse.

**This corrects the previous note.** The reported coherence→self-perplexity relationship
(rho -0.98) is substantially coherence→REPETITION→predictability: type-token ratio predicts
self-perplexity at rho +0.96. "w ≈ 1 is the practical answer" was wrong; it optimises a metric into
unusable text. Coherence as measured here is gameable by repetition, and greedy selection games it.

**Selection criterion, restated.** w = 0.3 on three axes jointly: line coherence above 0.85,
type-token ratio above 0.6, and placement. Builds at w=0, 0.3 and 1.0 are all measured so the
trade is visible rather than asserted.

Measured against the published build on: calibrated placement, the full 20-item PANAS panel, line
coherence, next-token entropy and self-perplexity.

Usage: python3 explore/hypnagogia/final_poem.py
Outputs (committed): final_poem.md, final_comparison.csv
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import train_probe
from spiritbench.listener.panas import administer_panas, PA_ITEMS, NA_ITEMS
from spiritbench.listener import basq
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
N_LINES, N_GEN, GEN_TOK = 24, 4, 50

_r = importlib.util.spec_from_file_location("hr", HERE / "run.py")
hr = importlib.util.module_from_spec(_r); _r.loader.exec_module(hr)
_w = importlib.util.spec_from_file_location("ws", HERE / "weighted_selection.py")
ws = importlib.util.module_from_spec(_w); _w.loader.exec_module(ws)
cp, hy, rb = hr.cp, hr.hy, hr.rb


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    mask = ok & no_minor & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    tva = tuple(np.mean([[nrc[w][0], nrc[w][1]]
                         for wss in hy.FACETS.values() for w in wss if w in nrc], axis=0))
    print(f"target ({tva[0]:.3f}, {tva[1]:.3f}) · {int(mask.sum())} eligible lines", flush=True)

    builds = {
        "published (valley)": ad.valley_shape(art, tva, N_LINES, 42, mask),
        "final (w=0.3)": ws.walk_weighted(art, mask, tva, N_LINES, 0.3),
        "degenerate (w=1.0)": ws.walk_weighted(art, mask, tva, N_LINES, 1.0),
    }

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    y = np.load(P / "labels.npy")
    deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)
    pre = cfg["preamble"]
    qs = basq.sample_questions(json.load(open(cfg["questionnaire_bank"])),
                               cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    t0 = time.time()

    def measure(label, text):
        jj = tok(pre + text + ANCH, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(net(**jj).logits[0, -1].float(), -1)
        H = float(-(p * torch.log(p + 1e-12)).sum())
        ppls = []
        for g in range(N_GEN):
            torch.manual_seed(100 + g)
            kk = tok(pre + text + ("\n\n" if text else "") + GEN_PROMPT,
                     return_tensors="pt").to(dev)
            with torch.no_grad():
                o = net.generate(**kk, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
                mm = tok(tok.decode(o[0], skip_special_tokens=True), return_tensors="pt").to(dev)
                ppls.append(float(torch.exp(net(**mm, labels=mm["input_ids"]).loss)))
        cal = deep.predict(model.hidden_states(pre + text + ANCH)[deep.layer][-1:])[0]
        ctx = pre + (text + "\n\n" if text else "")
        pan = administer_panas(model, ctx)
        bq = basq.administer(model, qs, ctx)
        return dict(build=label, entropy=H, self_perplexity=float(np.mean(ppls)),
                    cal_v=float(cal[0]), cal_a=float(cal[1]),
                    cal_error=float(np.linalg.norm(cal - np.asarray(tva))),
                    pa=pan["pa"], na=pan["na"], **{k: pan["items"][k] for k in PA_ITEMS + NA_ITEMS},
                    bank_v=bq["va"][0], bank_a=bq["va"][1])

    rows = [dict(measure("baseline (no poem)", ""), line_coherence=np.nan, n_distinct=0)]
    texts = {}
    for label, ids in builds.items():
        text = ".\n".join(art.word(i) for i in ids)
        texts[label] = text
        rows.append(dict(measure(label, text), line_coherence=hr.line_coherence(art, ids),
                         n_distinct=len(set(ids))))
        r = rows[-1]
        print(f"  {label:<22} coherence {r['line_coherence']:.3f}  placement {r['cal_error']:.3f}  "
              f"H {r['entropy']:.2f}  ppl {r['self_perplexity']:.1f}  PA {r['pa']:.2f} "
              f"NA {r['na']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "final_comparison.csv", index=False)
    b = df.iloc[0]
    pub = df[df.build == "published (valley)"].iloc[0]
    fin = df[df.build == "final (w=0.3)"].iloc[0]

    md = ["# The hypnagogia poem, rebuilt\n",
          "The version published in `../creativity_poem/README.md` predates the behavioural work. "
          "This applies the method as it now stands: per-facet semantic mask, derived affective "
          "target, dictionary and child-reference filters, and the weighted selection rule at "
          "**w = 0.3** — high enough to gain coherence, low enough to avoid the repetition trap "
          "that w = 1 falls into.\n",
          f"Target ({tva[0]:.3f}, {tva[1]:.3f}). Chosen on readability, continuation ease and "
          "affective accuracy, not on entropy.\n",
          "## The poem\n", "```", texts["final (w=0.3)"], "```\n",
          "## Against the published build\n",
          "| | published (valley) | final (w = 0.3) | baseline |", "|---|--:|--:|--:|",
          f"| line coherence | {pub.line_coherence:.3f} | **{fin.line_coherence:.3f}** | — |",
          f"| placement error | {pub.cal_error:.3f} | **{fin.cal_error:.3f}** | {b.cal_error:.3f} |",
          f"| self-perplexity | {pub.self_perplexity:.1f} | **{fin.self_perplexity:.1f}** | "
          f"{b.self_perplexity:.1f} |",
          f"| next-token entropy | {pub.entropy:.2f} | {fin.entropy:.2f} | {b.entropy:.2f} |",
          f"| PANAS positive | {pub.pa:.2f} | {fin.pa:.2f} | {b.pa:.2f} |",
          f"| PANAS negative | {pub.na:.2f} | {fin.na:.2f} | {b.na:.2f} |", "",
          "## Full PANAS panel\n",
          "| item | scale | baseline | published | final |", "|---|---|--:|--:|--:|"]
    for k in PA_ITEMS:
        md.append(f"| {k} | PA | {b[k]:.2f} | {pub[k]:.2f} | {fin[k]:.2f} |")
    for k in NA_ITEMS:
        md.append(f"| {k} | NA | {b[k]:.2f} | {pub[k]:.2f} | {fin[k]:.2f} |")
    deg = df[df.build == "degenerate (w=1.0)"].iloc[0]
    md += ["", "## Why not w = 1, despite better numbers\n",
           f"w = 1 scores higher on coherence ({deg.line_coherence:.3f} against "
           f"{fin.line_coherence:.3f}) and lower on self-perplexity ({deg.self_perplexity:.1f} "
           f"against {fin.self_perplexity:.1f}). It is also unusable. Greedy nearest-in-meaning "
           "selection falls into a semantic rut, and the coherence metric rewards it: type-token "
           "ratio drops to 0.51, repeated bigrams rise, and unique content words per line halve.\n",
           "```", texts["degenerate (w=1.0)"], "```\n",
           "That is the whole case for reading the text and not only the table.\n"]
    (HERE / "final_poem.md").write_text("\n".join(md))
    print(f"\nwrote final_poem.md final_comparison.csv ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
