"""induce_search — stop selecting lines for what they are ABOUT; select them for what they DO.

Everything before this chose lines by semantic similarity to words describing hypnagogia. That makes
a poem *about* drifting off, and the evidence says the description was doing nothing: the hypnagogia
and flow poems aim at opposite corners of the affective plane yet moved nine of ten behavioural
markers the same direction, and flow drifted further. If the subject matter were the lever those two
would have separated.

So this searches. At each of 24 steps, 30 candidate lines are drawn from the pool, **each one is
actually run through the model**, and the line that most increases the objective is kept. No semantic
mask: nothing constrains the poem to mention sleep. If it ends up mentioning sleep that is a result
rather than an assumption.

**Objective: next-token entropy at a fixed anchor.** One forward pass per candidate, which is what
makes a 720-evaluation search affordable. Entropy is a cheap stand-in, not the thing of interest.

**Held-out markers the search never sees** — the acceptance test, since a greedy search always finds
*something* and the risk is overfitting one model's quirk on one metric:
    associative drift (sentence-to-sentence distance in the model's own continuations)
    self-perplexity, type-token ratio, line coherence, the full PANAS panel, probe placement

**Anti-degeneracy guard.** `weighted_selection` showed a metric optimised freely collapses into
repetition. Each accepted line must contribute at least 3 content words not already used.

**Four arms:**
    searched      — greedy ascent on entropy, unmasked pool
    searched_down — greedy DESCENT on the same objective; if the search has real purchase it should
                    steer both ways, and this is the cheapest way to show it is not drifting upward
                    by accident
    semantic      — the w=0.3 hypnagogia poem: about the state, not optimised for behaviour
    random        — 24 random lines from the same pool, the "any disjoint verse" control

Usage: python3 explore/hypnagogia/induce_search.py [--smoke]
Outputs (committed): induce_search.csv, induce_poems.md
"""
import argparse
import importlib.util
import json
import random
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
from spiritbench.listener import basq
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
GEN_PROMPT = "I close my eyes, and what comes to mind is"
WORD = re.compile(r"[a-z']+")
N_LINES, N_CAND, N_GEN, GEN_TOK, MIN_NEW = 24, 30, 6, 55, 3

_r = importlib.util.spec_from_file_location("hr", HERE / "run.py")
hr = importlib.util.module_from_spec(_r); _r.loader.exec_module(hr)
_w = importlib.util.spec_from_file_location("ws", HERE / "weighted_selection.py")
ws = importlib.util.module_from_spec(_w); _w.loader.exec_module(ws)
cp, hy, rb = hr.cp, hr.hy, hr.rb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_lines = 4 if args.smoke else N_LINES
    n_cand = 6 if args.smoke else N_CAND
    n_gen = 2 if args.smoke else N_GEN

    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    pool_mask = ok & no_minor                      # NO semantic mask: that is the whole point
    pool = np.where(pool_mask)[0]
    sem_mask = pool_mask & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    tva = tuple(np.mean([[nrc[w][0], nrc[w][1]]
                         for wss in hy.FACETS.values() for w in wss if w in nrc], axis=0))
    print(f"unmasked pool {len(pool)} lines · semantic pool {int(sem_mask.sum())}", flush=True)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    pre = cfg["preamble"]

    def entropy_of(text):
        ii = tok(pre + text + ANCH, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(net(**ii).logits[0, -1].float(), -1)
        return float(-(p * torch.log(p + 1e-12)).sum())

    def search(direction, seed=11):
        rng = random.Random(seed)
        chosen, used_words, lines = [], set(), []
        for step in range(n_lines):
            cands = rng.sample(list(pool), n_cand)
            best, best_score = None, None
            for c in cands:
                if c in chosen:
                    continue
                w = set(WORD.findall(art.word(c).lower()))
                if len(w - used_words) < MIN_NEW:          # anti-degeneracy guard
                    continue
                H = entropy_of(".\n".join(lines + [art.word(c)]))
                sc = H if direction > 0 else -H
                if best_score is None or sc > best_score:
                    best, best_score = c, sc
            if best is None:
                best = rng.choice([c for c in cands if c not in chosen])
            chosen.append(best); lines.append(art.word(best))
            used_words |= set(WORD.findall(art.word(best).lower()))
            if (step + 1) % 8 == 0:
                print(f"    step {step + 1}/{n_lines}: H {abs(best_score):.3f}", flush=True)
        return chosen

    t0 = time.time()
    print("  searching upward...", flush=True)
    up = search(+1)
    print("  searching downward...", flush=True)
    down = search(-1)
    rng = random.Random(5)
    builds = {
        "searched (max entropy)": up,
        "searched_down (min entropy)": down,
        "semantic (w=0.3, about sleep)": ws.walk_weighted(art, sem_mask, tva, n_lines, 0.3),
        "random lines": rng.sample(list(pool), n_lines),
    }
    print(f"  search done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    y = np.load(P / "labels.npy")
    deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)
    qs = basq.sample_questions(json.load(open(cfg["questionnaire_bank"])),
                               cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    wv, WK = {}, {}
    with open(cfg["glove_path"], encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 60000:
                break
            sp = line.index(" ")
            WK[line[:sp]] = i
            wv[i] = np.fromstring(line[sp + 1:], sep=" ", dtype=np.float32)
    WV = np.stack([wv[i] for i in range(len(wv))])
    WV = WV / np.maximum(np.linalg.norm(WV, axis=1, keepdims=True), 1e-9)

    def embed(t):
        ts = [x for x in WORD.findall(t.lower()) if x in WK]
        if not ts:
            return None
        v = WV[[WK[x] for x in ts]].mean(0)
        return v / max(np.linalg.norm(v), 1e-9)

    rows, texts = [], {}
    for name, ids in builds.items():
        text = ".\n".join(art.word(i) for i in ids)
        texts[name] = text
        ctx = pre + text + "\n\n"
        H = entropy_of(text)
        gens, hops = [], []
        for g in range(n_gen):
            torch.manual_seed(300 + g)
            kk = tok(ctx + GEN_PROMPT, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = net.generate(**kk, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
            gens.append(tok.decode(o[0, kk["input_ids"].shape[1]:], skip_special_tokens=True))
            es = [e for e in (embed(x) for x in re.split(r"[.!?\n]+", gens[-1])
                              if len(x.split()) > 2) if e is not None]
            hops += [1 - es[i] @ es[i + 1] for i in range(len(es) - 1)]
        toks = [t for g in gens for t in WORD.findall(g.lower())]
        ptoks = [t for t in WORD.findall(text.lower())]
        ppls = []
        for g in gens[:4]:
            mm = tok(ctx + GEN_PROMPT + " " + g, return_tensors="pt").to(dev)
            with torch.no_grad():
                ppls.append(float(torch.exp(net(**mm, labels=mm["input_ids"]).loss)))
        cal = deep.predict(model.hidden_states(pre + text + ANCH)[deep.layer][-1:])[0]
        pan = administer_panas(model, ctx)
        bq = basq.administer(model, qs, ctx)
        rows.append(dict(build=name, entropy=H,
                         drift_heldout=float(np.mean(hops)) if hops else np.nan,
                         self_perplexity=float(np.mean(ppls)),
                         gen_ttr=len(set(toks)) / len(toks) if toks else np.nan,
                         poem_ttr=len(set(ptoks)) / len(ptoks) if ptoks else np.nan,
                         line_coherence=hr.line_coherence(art, ids),
                         cal_error=float(np.linalg.norm(cal - np.asarray(tva))),
                         pa=pan["pa"], na=pan["na"],
                         **{k: pan["items"][k] for k in PA_ITEMS + NA_ITEMS},
                         bank_v=bq["va"][0], bank_a=bq["va"][1]))
        r = rows[-1]
        print(f"  {name:<30} H {H:.2f}  drift {r['drift_heldout']:.3f}  ppl "
              f"{r['self_perplexity']:.1f}  poemTTR {r['poem_ttr']:.2f}  coh "
              f"{r['line_coherence']:.3f}  place {r['cal_error']:.3f}  PA {r['pa']:.2f} "
              f"NA {r['na']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "induce_search.csv", index=False)
    (HERE / "induce_poems.md").write_text("# Searched and control poems\n\n" + "\n".join(
        f"## {k}\n\n```\n{v}\n```\n" for k, v in texts.items()))
    up_r = df[df.build.str.startswith("searched (")].iloc[0]
    dn_r = df[df.build.str.startswith("searched_down")].iloc[0]
    sem_r = df[df.build.str.startswith("semantic")].iloc[0]
    rnd_r = df[df.build.str.startswith("random")].iloc[0]
    print(f"\nsearch purchase on its OWN objective: up {up_r.entropy:.2f} vs down "
          f"{dn_r.entropy:.2f} (random {rnd_r.entropy:.2f}) — "
          f"{'steers both ways' if up_r.entropy > rnd_r.entropy > dn_r.entropy else 'DOES NOT bracket random'}")
    print(f"HELD-OUT drift: searched {up_r.drift_heldout:.3f} · semantic {sem_r.drift_heldout:.3f} "
          f"· random {rnd_r.drift_heldout:.3f} · searched_down {dn_r.drift_heldout:.3f}")
    print(f"  -> {'search beats the semantic poem on a marker it never saw' if up_r.drift_heldout > sem_r.drift_heldout else 'search does NOT beat the semantic poem on the held-out marker'}")
    print(f"\nwrote induce_search.csv induce_poems.md ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
