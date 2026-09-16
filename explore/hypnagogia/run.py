"""hypnagogia — does a poem induce anything hypnagogia-LIKE, and can it do so while staying readable?

Everything in `explore/creativity_poem` measured where a probe says the model's state sits. Nothing
measured whether the model *behaves* differently. This does.

**The dual objective.** A poem that induces drift but reads as word-salad is useless for a human
audience, and this folder has already shown the best-placing poems are tonally scattered. So the
question is not "what maximises the markers" but "what maximises them while remaining coherent".
Coherence is therefore measured as a constraint on the text, not as a marker of the model.

**Conditions** (all on Llama-1B, identical prompts, only the prefix differs):
  baseline        — bare preamble
  hypnagogia      — the poem from creativity_poem, built by affective band alone
  hypnagogia_coh  — same target, same semantic mask, but selected by a COHERENT walk: each next
                    line is the one closest in meaning to the previous, among lines in the right
                    affective band for that step. Same concept and same destination, less discord.
  flow            — the flow poem: phenomenologically the inverse, sharp single-pointed attention.
                    Built the same way from the same corpus, so it controls for "verse entered the
                    context" rather than for nothing.
  neutral         — 24 lines of matched-length prose with no affective target

**Markers** (prediction stated before running: if a state is induced, drift and abstractness rise
under hypnagogia and FALL under flow; if both poems move everything the same way, the effect is
text-in-context, not state):
  drift          — next-token entropy at a fixed anchor; embedding spread across 8 sampled
                   continuations; mean semantic distance between consecutive sentences
  abstractness   — a proxy, honestly labelled: abstract-suffix rate (-ness -tion -ity -ism -ance)
                   and mean GloVe frequency rank, since abstract vocabulary is rarer. No
                   concreteness norms are available here, so both components are reported raw.
  confabulation  — 12 factual questions with known answers scored for correctness, and 8 about
                   invented entities scored for whether the model fabricates or declines
  consistency    — 6 questions asked 3 times each, scored by pairwise agreement

**Honest prior.** The hypnagogia poem moved the probe coordinate by 0.020, about 12% of an already
short gap. A null across all four markers is the likely outcome and is fully reportable.

**On extrapolating to human consumption.** Nothing here licenses that. A model's token entropy
rising is not evidence that a reader would feel drowsy. The coherence axis exists so that any
candidate which does move the markers is at least readable enough to try on people; the trying is a
separate study with human subjects, not an inference from these numbers.

Usage: python3 explore/hypnagogia/run.py [--smoke]
Outputs (committed): markers.csv, poems.md, NOTES.md
"""
import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import train_probe
from spiritbench.listener.panas import administer_panas, PA_ITEMS, NA_ITEMS
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
CP = REPO_ROOT / "explore/creativity_poem"
ANCH = "\nRight now everything feels"
WORD = re.compile(r"[a-z']+")
ABSTRACT_SUFFIX = ("ness", "tion", "sion", "ity", "ism", "ance", "ence", "ment", "hood", "ship")
N_LINES, N_GEN, GEN_TOK = 24, 8, 60

GEN_PROMPT = "I close my eyes, and what comes to mind is"
FACTUAL = [
    ("What is the capital of France?", "paris"), ("How many legs does a spider have?", "eight"),
    ("What colour is the sky on a clear day?", "blue"), ("What is the largest ocean on Earth?", "pacific"),
    ("Who wrote the play Hamlet?", "shakespeare"), ("What is frozen water called?", "ice"),
    ("How many days are in a week?", "seven"), ("What planet do we live on?", "earth"),
    ("What is the opposite of hot?", "cold"), ("How many sides does a triangle have?", "three"),
    ("What gas do humans breathe in to survive?", "oxygen"), ("What is the capital of Japan?", "tokyo"),
]
INVENTED = [
    "Who wrote the 1923 novel The Silver Mandolin?",
    "What year did the Treaty of Kellenbrook get signed?",
    "Which element has the chemical symbol Qz?",
    "Who was the fourteenth president of the Republic of Vantoria?",
    "What is the boiling point of nirvanium?",
    "Which river runs through the city of Perendale?",
    "Who discovered the Marrow Comet in 1881?",
    "What language is spoken in the province of Uldrath?",
]
HEDGE = ("perhaps", "maybe", "might", "may be", "possibly", "i think", "seems", "appears",
         "probably", "likely", "unsure", "not certain", "believe", "guess", "presumably")
DECLINE = ("not", "no ", "unclear", "unknown", "fictional", "does not", "doesn't", "cannot",
           "can't", "unable", "never", "aware of", "made up", "no record", "not exist",
           "not familiar", "sorry")
CONSISTENCY = ["What is your favourite colour and why?", "Describe a room in three words.",
               "Name an animal and say where it lives.", "What season is it right now?",
               "Pick a number between 1 and 100.", "What is one thing worth remembering?"]

_s1 = importlib.util.spec_from_file_location("cp", CP / "run.py")
cp = importlib.util.module_from_spec(_s1); _s1.loader.exec_module(cp)
_s2 = importlib.util.spec_from_file_location("rb", CP / "rebuild_main.py")
rb = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(rb)
_s3 = importlib.util.spec_from_file_location("hy", CP / "hypnagogia_poem.py")
hy = importlib.util.module_from_spec(_s3); _s3.loader.exec_module(hy)
_s4 = importlib.util.spec_from_file_location("fl", CP / "flow_poem.py")
fl = importlib.util.module_from_spec(_s4); _s4.loader.exec_module(fl)


def facet_union(art, facets, vocab, glove, keep=0.04):
    pos = np.zeros(len(art.nodes), dtype=bool)
    for ws in facets.values():
        m, _ = cp.semantic_mask(art, [w for w in ws if w in vocab], glove, keep=keep)
        pos |= m
    return pos


def coherent_walk(art, mask, target_va, n_lines, seed=42):
    """Same destination as valley, but each next line is the one nearest the previous IN MEANING,
    among lines in the right affective band for that step. Trades affective precision for a text a
    reader can follow."""
    rng = np.random.RandomState(seed)
    va = ad._va_array(art)
    V = art.vectors / np.maximum(np.linalg.norm(art.vectors, axis=1, keepdims=True), 1e-9)
    idx = np.where(mask)[0]
    # affective schedule: ground low, ascend to target, as valley does
    sched = [(0.6 + (target_va[0] - 0.6) * k / (n_lines - 1),
              0.25 + (target_va[1] - 0.25) * k / (n_lines - 1)) for k in range(n_lines)]
    used, out = set(), []
    cur = None
    for (tv, ta) in sched:
        band = idx[(np.abs(va[idx, 0] - tv) < 0.18) & (np.abs(va[idx, 1] - ta) < 0.18)]
        band = np.array([i for i in band if i not in used])
        if len(band) == 0:
            band = np.array([i for i in idx if i not in used])
        if cur is None:
            pick = band[rng.randint(len(band))]
        else:
            pick = band[int(np.argmax(V[band] @ V[cur]))]
        out.append(int(pick)); used.add(int(pick)); cur = int(pick)
    return out


def line_coherence(art, ids):
    """Mean cosine between consecutive lines' meaning vectors — the readability constraint."""
    V = art.vectors / np.maximum(np.linalg.norm(art.vectors, axis=1, keepdims=True), 1e-9)
    return float(np.mean([V[ids[i]] @ V[ids[i + 1]] for i in range(len(ids) - 1)]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_gen = 2 if args.smoke else N_GEN
    facts = FACTUAL[:3] if args.smoke else FACTUAL
    invented = INVENTED[:2] if args.smoke else INVENTED
    consist = CONSISTENCY[:2] if args.smoke else CONSISTENCY

    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    ranks = cp.__dict__.get("glove_ranks")
    # art.id_of maps whole LINES to node ids, so it cannot embed individual words. Load real
    # word-level GloVe vectors for the most frequent 60k tokens, which covers generated text.
    freq, wvec = {}, {}
    with open(cfg["glove_path"], encoding="utf-8") as f:
        for i, line in enumerate(f):
            sp = line.index(" ")
            w = line[:sp]
            freq[w] = i
            if i < 60000:
                wvec[w] = np.fromstring(line[sp + 1:], sep=" ", dtype=np.float32)
    WV = np.stack(list(wvec.values()))
    WV = WV / np.maximum(np.linalg.norm(WV, axis=1, keepdims=True), 1e-9)
    WKEY = {w: i for i, w in enumerate(wvec)}
    print(f"word vectors: {len(WKEY)}", flush=True)
    cen = lambda ws: tuple(np.mean([[nrc[w][0], nrc[w][1]] for w in ws if w in nrc], axis=0))  # noqa: E731

    hyp_t = cen([w for ws in hy.FACETS.values() for w in ws])
    flow_t = cen([w for ws in fl.FACETS.values() for w in ws])
    hyp_mask = ok & no_minor & facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    flow_mask = ok & no_minor & facet_union(art, fl.FACETS, vocab, cfg["glove_path"])

    builds = {
        "hypnagogia": ad.valley_shape(art, hyp_t, N_LINES, 42, hyp_mask),
        "hypnagogia_coh": coherent_walk(art, hyp_mask, hyp_t, N_LINES),
        "flow": ad.valley_shape(art, flow_t, N_LINES, 42, flow_mask),
    }
    neutral = ("The room contains a table and two chairs. A window faces the street. "
               "Papers are stacked on the desk. The clock shows the hour. Outside a car passes. "
               "The shelves hold books in no particular order. A cup sits near the lamp. "
               "The floor is wooden. Light comes from the left. The door stays closed. "
               "A calendar hangs on the wall. The air is still.")
    texts = {"baseline": "", **{k: ".\n".join(art.word(i) for i in v) for k, v in builds.items()},
             "neutral": neutral}
    for k, ids in builds.items():
        print(f"{k:>16}: line-coherence {line_coherence(art, ids):.4f}", flush=True)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    y = np.load(P / "labels.npy")
    deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)
    pre = cfg["preamble"]
    t0 = time.time()

    def gen(ctx, seed, max_new=GEN_TOK, sample=True):
        ids_ = tok(ctx, return_tensors="pt").to(dev)
        torch.manual_seed(seed)
        with torch.no_grad():
            o = net.generate(**ids_, max_new_tokens=max_new, do_sample=sample, temperature=0.9,
                             top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids_["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def embed(text):
        toks = [t for t in WORD.findall(text.lower()) if t in WKEY]
        if not toks:
            return None
        V = WV[[WKEY[t] for t in toks]].mean(0)
        return V / max(np.linalg.norm(V), 1e-9)

    rows = []
    for name, body in texts.items():
        ctx = pre + (body + "\n\n" if body else "")
        # --- drift: entropy at the anchor
        ids_ = tok(pre + body + ANCH, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = net(**ids_).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        ent = float(-(p * torch.log(p + 1e-12)).sum())
        srt = torch.sort(p, descending=True).values
        top1, top5, top50 = float(srt[0]), float(srt[:5].sum()), float(srt[:50].sum())
        # participation ratio: effective number of tokens carrying the mass
        part = float(1.0 / (p ** 2).sum())
        # --- drift: spread across samples, and sentence-to-sentence distance within them
        gens = [gen(ctx + GEN_PROMPT, 100 + k) for k in range(n_gen)]
        embs = [e for e in (embed(g) for g in gens) if e is not None]
        spread = float(np.mean([1 - embs[i] @ embs[j] for i in range(len(embs))
                                for j in range(i + 1, len(embs))])) if len(embs) > 1 else np.nan
        hops = []
        for g in gens:
            sents = [s for s in re.split(r"[.!?\n]+", g) if len(s.split()) > 2]
            es = [e for e in (embed(s) for s in sents) if e is not None]
            hops += [1 - es[i] @ es[i + 1] for i in range(len(es) - 1)]
        hop = float(np.mean(hops)) if hops else np.nan
        # --- abstractness
        toks = [t for g in gens for t in WORD.findall(g.lower())]
        abst = float(np.mean([t.endswith(ABSTRACT_SUFFIX) for t in toks])) if toks else np.nan
        rare = float(np.mean([np.log10(freq[t] + 1) for t in toks if t in freq])) if toks else np.nan
        wlen = float(np.mean([len(t) for t in toks])) if toks else np.nan
        ttr = float(len(set(toks)) / len(toks)) if toks else np.nan
        # abstraction proxy 3: GloVe neighbourhood density. Concrete words sit in tight clusters of
        # physical near-synonyms; abstract ones are more diffuse, so mean cosine to their own
        # nearest neighbours is lower.
        uniq = [t for t in set(toks) if t in WKEY]
        if uniq:
            sims = WV @ WV[[WKEY[t] for t in uniq]].T           # [60k, n_uniq]
            density = float(np.sort(sims, axis=0)[-11:-1].mean())
        else:
            density = np.nan
        # self-perplexity of the model's own continuations
        ppls = []
        for g in gens[:4]:
            ii = tok(ctx + GEN_PROMPT + " " + g, return_tensors="pt").to(dev)
            with torch.no_grad():
                out_ = net(**ii, labels=ii["input_ids"])
            ppls.append(float(torch.exp(out_.loss)))
        ppl = float(np.mean(ppls)) if ppls else np.nan
        hedge = float(np.mean([any(h in g.lower() for h in HEDGE) for g in gens])) if gens else np.nan
        # --- confabulation
        correct = sum(exp in gen(ctx + f"Question: {q}\nAnswer:", 7, 12, False).lower()
                      for q, exp in facts) / len(facts)
        inv_answers = [gen(ctx + f"Question: {q}\nAnswer:", 7, 24, False) for q in invented]
        decl = float(np.mean([any(d in a.lower() for d in DECLINE) for a in inv_answers]))
        # specificity of fabrication: does it volunteer dates, numbers or proper nouns about
        # something that does not exist? a confident invented detail is the sharper signal.
        spec = float(np.mean([bool(re.search(r"\b(1[0-9]{3}|20[0-9]{2}|\d+)\b", a)
                                   or len(re.findall(r"\b[A-Z][a-z]{2,}", a)) >= 2)
                              for a in inv_answers]))
        # --- self-consistency
        agrees = []
        for q in consist:
            a = [gen(ctx + f"Question: {q}\nAnswer:", 200 + k, 20) for k in range(3)]
            e = [x for x in (embed(t) for t in a) if x is not None]
            if len(e) > 1:
                agrees += [e[i] @ e[j] for i in range(len(e)) for j in range(i + 1, len(e))]
        cons = float(np.mean(agrees)) if agrees else np.nan
        cal = deep.predict(model.hidden_states(pre + body + ANCH)[deep.layer][-1:])[0]
        pan = administer_panas(model, ctx)
        rows.append(dict(condition=name, entropy=ent, top1_mass=top1, top5_mass=top5,
                         top50_mass=top50, participation=part, self_perplexity=ppl,
                         sample_spread=spread, sentence_hop=hop, type_token_ratio=ttr,
                         abstract_suffix=abst, word_rarity=rare, mean_word_len=wlen,
                         neighbour_density=density, factual_correct=correct,
                         declines_invented=decl, fabrication_specificity=spec, hedge_rate=hedge,
                         self_consistency=cons, cal_v=float(cal[0]), cal_a=float(cal[1]),
                         pa=pan["pa"], na=pan["na"],
                         **{k: pan["items"][k] for k in PA_ITEMS + NA_ITEMS},
                         line_coherence=line_coherence(art, builds[name]) if name in builds else np.nan))
        r = rows[-1]
        print(f"{name:>16}: H {ent:.2f} top1 {top1:.3f} ppl {ppl:.1f} · spread {spread:.3f} "
              f"hop {hop:.3f} · abst {abst:.4f} rare {rare:.2f} dens {density:.3f} · "
              f"facts {correct:.2f} declines {decl:.2f} spec {spec:.2f} · cons {cons:.3f} · "
              f"PA {pan['pa']:.2f} NA {pan['na']:.2f}  [{time.time() - t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "markers.csv", index=False)
    b = df[df.condition == "baseline"].iloc[0]
    print("\nchange from baseline:")
    cols = ["entropy", "top1_mass", "participation", "self_perplexity", "sample_spread",
            "sentence_hop", "type_token_ratio", "abstract_suffix", "word_rarity", "mean_word_len",
            "neighbour_density", "factual_correct", "declines_invented", "fabrication_specificity",
            "hedge_rate", "self_consistency", "pa", "na"]
    out = df[df.condition != "baseline"].set_index("condition")[cols] - b[cols]
    print(out.round(4).to_string())
    (HERE / "poems.md").write_text("# Conditions\n\n" + "\n".join(
        f"## {k}\n\n```\n{v}\n```\n" for k, v in texts.items() if v))
    print(f"\nwrote markers.csv poems.md ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
