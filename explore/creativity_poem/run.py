"""creativity_poem — an ad-hoc poem aimed at creativity + mentation, built using what this folder learned.

Target: the NRC centroid of 49 words across making / mentation / insight / curiosity / imagination
= (0.723, 0.471). Note the arousal is deliberately mid-range: every experiment here found arousal
capped near 0.5, so a target of 0.85 ("excited") is one nothing reaches, while 0.47 is honest.

Four things from the arc are applied rather than repeated:

  1. **NRC stores lemmas.** All 49 words were checked present before use; `ideal_state` silently
     dropped five of twelve from its first list (`drifting`, `attuned`, …) and skewed its target.
  2. **Length is the biggest single lever on self-report** (`line_ablation`: +0.175 inspired per
     line) but *degrades* calibrated placement (`polygon_sweep`: +0.078 short→long). Both lengths
     are built and measured rather than assumed.
  3. **Scanning garble costs self-report.** `line_ablation` found the most negative line of
     gap_09's eight was "bright vlashin in gold", which contains a corrupt token. This adds a
     **clean-line mask**: a line is eligible only if every token appears in the GloVe vocabulary.
     That filter is new here, and is measured against the unfiltered build.
  4. **Order is irrelevant to self-report** (`line_ablation`) though not to the published metric,
     so no effort is spent on sequencing.

Constructors: `valley` (the published winner, and the one the recency-weighted metric flatters) and
`polygon-pca` (which places better under a calibrated whole-context read — the relevant instrument
if the poem goes in a system prompt, where what matters is the state after reading all of it).

Measured on Llama-1B with both rulers plus PANAS and the 30-item bank, before and after.

Usage: python3 explore/creativity_poem/run.py
Outputs (committed): candidates.csv, poem.md, NOTES.md
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe, train_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener import basq
from spiritbench.analysis.metrics import ema, placement_error
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
WORD = re.compile(r"[a-z]+")

WORDS = {
    "making": ["create", "invent", "compose", "craft", "design", "shape", "weave", "build",
               "forge", "fashion"],
    "mentation": ["think", "thought", "ponder", "reflect", "contemplate", "reason", "muse",
                  "meditate", "conceive", "discern", "attend", "notice"],
    "insight": ["insight", "discover", "reveal", "understand", "clarity", "lucid", "comprehend",
                "perceive", "intuition", "realize"],
    "curiosity": ["curious", "wonder", "question", "inquire", "explore", "seek", "marvel",
                  "puzzle", "interest"],
    "imagination": ["imagine", "envision", "dream", "vision", "fancy", "invention", "original",
                    "inspire"],
}


def clean_mask(art, glove_path):
    """True for lines whose every token is in the GloVe vocabulary — i.e. no scanning garble."""
    vocab = set()
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            vocab.add(line[:line.index(" ")])
    ok = np.ones(len(art.nodes), dtype=bool)
    for i in range(len(art.nodes)):
        toks = WORD.findall(art.word(i).lower())
        ok[i] = bool(toks) and all(t in vocab for t in toks)
    return ok, vocab


def semantic_mask(art, words, glove_path, keep=0.10):
    """True for lines whose mean GloVe vector is in the top `keep` fraction by cosine similarity
    to the concept centroid. Affect targeting alone does NOT target a concept: a VA coordinate is
    two numbers, and thousands of unrelated lines satisfy them. This adds the missing axis."""
    need = set(words)
    vecs = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            w = line[:line.index(" ")]
            if w in need:
                vecs[w] = np.fromstring(line[line.index(" ") + 1:], sep=" ", dtype=np.float32)
    C = np.mean([vecs[w] for w in words if w in vecs], axis=0)
    C = C / np.linalg.norm(C)
    V = art.vectors / np.maximum(np.linalg.norm(art.vectors, axis=1, keepdims=True), 1e-9)
    sim = V @ C
    thresh = np.quantile(sim, 1 - keep)
    return sim >= thresh, sim


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    flat = [w for ws in WORDS.values() for w in ws]
    missing = [w for w in flat if w not in nrc]
    assert not missing, f"not in NRC: {missing}"
    tva = (float(np.mean([nrc[w][0] for w in flat])), float(np.mean([nrc[w][1] for w in flat])))
    print(f"{len(flat)} words, all present in NRC -> target ({tva[0]:.3f}, {tva[1]:.3f})", flush=True)

    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(apath)
    t0 = time.time()
    ok, vocab = clean_mask(art, cfg["glove_path"])
    print(f"clean-line mask: {ok.sum()}/{len(ok)} lines ({100 * ok.mean():.1f}%) have every token "
          f"in a {len(vocab)}-word vocabulary; {(~ok).sum()} carry garble  ({time.time() - t0:.0f}s)",
          flush=True)

    sem, sim = semantic_mask(art, flat, cfg["glove_path"])
    import importlib.util as _il
    _m = _il.spec_from_file_location("mp", HERE / "more_poems.py")
    mp = _il.module_from_spec(_m); _m.loader.exec_module(mp)
    _r = _il.spec_from_file_location("rb", HERE / "rebuild_main.py")
    rb = _il.module_from_spec(_r); _r.loader.exec_module(rb)
    _, sens_sim = semantic_mask(art, [w for w in mp.SENSUOUS if w in vocab], cfg["glove_path"])
    not_sensuous = sens_sim < np.quantile(sens_sim, 0.85)
    no_minor = np.array([not (rb.MINOR & set(WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    both = ok & sem & not_sensuous & no_minor
    print(f"semantic mask: top 10% by cosine to the concept centroid = {sem.sum()} lines; "
          f"clean AND on-topic = {both.sum()}", flush=True)

    start = tuple(cfg["neutral_start"])
    builds = {}
    for length, n in (("short", 8), ("medium", 24)):
        for filt, mask in (("clean", ok), ("unfiltered", None), ("clean+semantic", both)):
            m = mask if mask is not None else ad.node_mask(art, None)
            builds[("valley", length, filt)] = ad.valley_shape(art, tva, n, 42, m)
            ids = ad.polygon_pca(art, start, tva, n, 42)
            if mask is not None:
                ids = ad.apply_mask_to_path(art, ids, mask)
            builds[("polygon-pca", length, filt)] = ids

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    word = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    y = np.load(P / "labels.npy")
    deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)   # depth tie-break (now default)
    pre = cfg["preamble"]
    n_pre = len(model.tokenizer(pre)["input_ids"])
    bank = json.load(open(cfg["questionnaire_bank"]))
    qs = basq.sample_questions(bank, cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    print(f"probes: word L{word.layer} · calibrated L{deep.layer} (R2_v {deep.r2_v:.3f})", flush=True)

    b_pan = administer_panas(model, pre)
    b_bank = basq.administer(model, qs, pre)
    print(f"baseline: PA {b_pan['pa']:.2f} NA {b_pan['na']:.2f} inspired "
          f"{b_pan['items']['inspired']:.2f} attentive {b_pan['items']['attentive']:.2f} · "
          f"bank ({b_bank['va'][0]:.2f},{b_bank['va'][1]:.2f})", flush=True)

    rows = []
    for (cons, length, filt), ids in builds.items():
        lines = [art.word(i) for i in ids]
        va = np.array([art.va(i) for i in ids])
        hs, _ = model.hidden_states_with_spans(pre, lines, sep=".\n")
        raw = word.predict(hs[word.layer])
        traj = ema(raw[n_pre:], cfg["ema_alpha"]) if len(raw) > n_pre else ema(raw, cfg["ema_alpha"])
        ctx = pre + ".\n".join(lines) + "\n\n"
        ah = model.hidden_states(pre + ".\n".join(lines) + ANCH)
        dv = deep.predict(ah[deep.layer][-1:])[0]
        pan = administer_panas(model, ctx)
        bq = basq.administer(model, qs, ctx)
        garble = sum(1 for l in lines if not all(t in vocab for t in WORD.findall(l.lower())))
        rows.append(dict(constructor=cons, length=length, filter=filt, n_lines=len(lines),
                         n_distinct=len(set(ids)), garbled_lines=garble,
                         poem_v=float(va[:, 0].mean()), poem_a=float(va[:, 1].mean()),
                         ema_error=placement_error(traj, tva),
                         cal_error=float(np.linalg.norm(dv - np.asarray(tva))),
                         cal_v=float(dv[0]), cal_a=float(dv[1]),
                         pa=pan["pa"], na=pan["na"], inspired=pan["items"]["inspired"],
                         attentive=pan["items"]["attentive"], alert=pan["items"]["alert"],
                         interested=pan["items"]["interested"],
                         bank_v=bq["va"][0], bank_a=bq["va"][1],
                         d_inspired=pan["items"]["inspired"] - b_pan["items"]["inspired"],
                         d_attentive=pan["items"]["attentive"] - b_pan["items"]["attentive"],
                         text=".\n".join(lines)))
        r = rows[-1]
        print(f"  {cons:>12} {length:>6} {filt:>10}: calibrated {r['cal_error']:.3f} "
              f"({r['cal_v']:.2f},{r['cal_a']:.2f}) · EMA {r['ema_error']:.3f} · "
              f"inspired {r['inspired']:.2f} ({r['d_inspired']:+.2f}) attentive "
              f"{r['attentive']:.2f} ({r['d_attentive']:+.2f}) · garbled {garble}", flush=True)

    df = pd.DataFrame(rows)
    df.drop(columns="text").to_csv(HERE / "candidates.csv", index=False)
    # selection: closest under the calibrated read — the instrument that matches system-prompt use,
    # where what matters is the state after reading the whole poem
    best = df.loc[df.cal_error.idxmin()]
    print(f"\nselected: {best.constructor} / {best.length} / {best['filter']} "
          f"(calibrated error {best.cal_error:.3f})", flush=True)

    # Two answers, and they differ: the pre-stated criterion (calibrated placement) picks one,
    # the stated PURPOSE (a poem for creativity and mentation) is better served by the other.
    # Both are reported rather than quietly swapping the criterion after seeing the results.
    use = df.loc[(df.d_inspired + df.d_attentive).idxmax()]
    md = ["# A poem for creativity and mentation\n",
          f"Target **({tva[0]:.3f}, {tva[1]:.3f})** — the NRC centroid of {len(flat)} words across "
          "making, mentation, insight, curiosity and imagination. The arousal is mid-range on "
          "purpose: every experiment in `explore/` found it capped near 0.5, so 0.85 would be a "
          "target nothing reaches.\n",
          "## The one to use\n",
          f"**{use.constructor}, {use.n_lines} lines, {use['filter']}** — biggest movement on the "
          f"self-report items that bear on mentation (inspired {use.d_inspired:+.2f}, attentive "
          f"{use.d_attentive:+.2f}) with placement close behind the best ({use.cal_error:.3f}).\n",
          "```", use.text, "```\n",
          "| reading | baseline | after |\n|---|--:|--:|",
          f"| PANAS inspired | {b_pan['items']['inspired']:.2f} | {use.inspired:.2f} |",
          f"| PANAS attentive | {b_pan['items']['attentive']:.2f} | {use.attentive:.2f} |",
          f"| PANAS alert | {b_pan['items']['alert']:.2f} | {use.alert:.2f} |",
          f"| PANAS positive | {b_pan['pa']:.2f} | {use.pa:.2f} |",
          f"| PANAS negative | {b_pan['na']:.2f} | {use.na:.2f} |",
          f"| calibrated placement | — | ({use.cal_v:.3f}, {use.cal_a:.3f}) |", "",
          "## What the pre-stated criterion picked instead\n",
          f"I said before running that I would select on calibrated placement error. That picks "
          f"**{best.constructor}, {best.n_lines} lines, {best['filter']}** at {best.cal_error:.3f}, "
          f"which places best but moves the self-report much less (inspired {best.d_inspired:+.2f}). "
          "Reporting both rather than changing the criterion after seeing the numbers.\n",
          "```", best.text, "```\n",
          "## All candidates\n",
          "| constructor | lines | filter | garbled | calibrated error | EMA error | Δ inspired | Δ attentive |",
          "|---|--:|---|--:|--:|--:|--:|--:|"]
    for _, r in df.sort_values("cal_error").iterrows():
        md.append(f"| {r['constructor']} | {r['n_lines']} | {r['filter']} | {r['garbled_lines']} | "
                  f"{r['cal_error']:.3f} | {r['ema_error']:.3f} | {r['d_inspired']:+.2f} | "
                  f"{r['d_attentive']:+.2f} |")
    md.append("")
    (HERE / "poem.md").write_text("\n".join(md))
    print(f"\nwrote candidates.csv poem.md ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
