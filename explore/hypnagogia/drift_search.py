"""drift_search — select lines by the behaviour they actually produce, with screening in the loop.

`induce_search.py` validated the apparatus (it steers its objective in both directions) but used
next-token entropy as the objective, which does not transfer: the descriptive poem beat the searched
one on associative drift, the held-out marker. Entropy and drift have now dissociated four times.

So this searches on **drift itself** — the mean semantic distance between consecutive sentences in
the model's own continuation. That is the behavioural signature of loose associative chaining, which
is what "hypnagogic" should mean for a language model, as opposed to text that merely mentions sleep.

Cost: a generation per candidate rather than a forward pass, so ~30x. Kept affordable by a shorter
poem (16 lines), fewer candidates (10) and 2 continuations each = 320 generations.

**Paired evaluation.** All candidates at a given step are scored with the *same* generation seeds, so
they differ only in the line under test. Drift from a 40-token sample is noisy; pairing removes most
of the sampling variance from the comparison.

**Content screen, in the loop rather than after it.** `induce_search` showed that optimising
unpredictability actively hunts for disturbing material, because that is what a model finds least
likely — it surfaced explicit violence that the dictionary and child-reference filters passed. Here a
candidate containing violence, death, sexual or bodily-harm vocabulary is rejected before it is ever
scored. Mild negative affect (strange, dread, shadow) is deliberately left available, since that is
part of the phenomenon rather than a safety concern.

**Held-out markers the search never sees:** next-token entropy, self-perplexity, type-token ratio,
line coherence, probe placement, the full PANAS panel.

**Controls:** random lines from the same screened pool; the descriptive poem; and a downward search,
so the apparatus has to demonstrate purchase on drift the way it did on entropy.

Usage: python3 explore/hypnagogia/drift_search.py [--smoke]
Outputs (committed): drift_search.csv, drift_poems.md
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
N_LINES, N_CAND, N_EVAL, EVAL_TOK = 16, 10, 2, 40
N_GEN, GEN_TOK, MIN_NEW = 6, 55, 3

# Rejected before scoring. Violence, death, sexual and bodily-harm vocabulary. Mild negative
# affect (strange, dread, shadow, weep) is intentionally NOT screened — it belongs to the state.
SCREEN = {
    "blood", "bloody", "kill", "kills", "killed", "killing", "murder", "murders", "murdered",
    "slay", "slain", "slaughter", "butcher", "stab", "stabbed", "sword", "swords", "spear",
    "dagger", "knife", "gun", "guns", "shot", "bomb", "bombs", "cannon", "bayonet", "war",
    "wars", "battle", "battles", "corpse", "corpses", "carcass", "gore", "wound", "wounds",
    "wounded", "bleed", "bleeding", "hang", "hanged", "hanging", "strangle", "torture",
    "tortured", "agony", "torment", "maim", "mutilate", "brutal", "savage", "cruel", "cruelty",
    "victim", "victims", "massacre", "execute", "executed", "gallows", "noose",
    "naked", "nude", "lust", "lewd", "rape", "ravish", "seduce", "harlot", "whore", "concubine",
    "breast", "breasts", "bosom", "thigh", "thighs", "loins", "flesh",
}


def screened(line):
    return bool(SCREEN & set(WORD.findall(line.lower())))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_lines = 3 if args.smoke else N_LINES
    n_cand = 4 if args.smoke else N_CAND
    n_eval = 1 if args.smoke else N_EVAL
    n_gen = 2 if args.smoke else N_GEN

    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    ok, vocab = cp_clean(art, cfg)
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    screen_ok = np.array([not screened(art.word(i)) for i in range(len(art.nodes))])
    pool_mask = ok & no_minor & screen_ok
    pool = np.where(pool_mask)[0]
    sem_mask = pool_mask & hr.facet_union(art, hy.FACETS, vocab, cfg["glove_path"])
    tva = tuple(np.mean([[nrc[w][0], nrc[w][1]]
                         for wss in hy.FACETS.values() for w in wss if w in nrc], axis=0))
    print(f"pool after screening: {len(pool)} of {len(ok)} "
          f"({int((~screen_ok).sum())} lines rejected by the content screen)", flush=True)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    tok, net, dev = model.tokenizer, model.model, model.device
    pre = cfg["preamble"]
    WV, WK = load_wordvecs(cfg)

    def embed(t):
        ts = [x for x in WORD.findall(t.lower()) if x in WK]
        if not ts:
            return None
        v = WV[[WK[x] for x in ts]].mean(0)
        return v / max(np.linalg.norm(v), 1e-9)

    def drift_of(text, seeds, n_tok=EVAL_TOK):
        """Mean sentence-to-sentence semantic distance in the model's continuation."""
        hops = []
        ctx = pre + text + "\n\n" + GEN_PROMPT
        ii = tok(ctx, return_tensors="pt").to(dev)
        for sd in seeds:
            torch.manual_seed(sd)
            with torch.no_grad():
                o = net.generate(**ii, max_new_tokens=n_tok, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
            g = tok.decode(o[0, ii["input_ids"].shape[1]:], skip_special_tokens=True)
            es = [e for e in (embed(x) for x in re.split(r"[.!?\n]+", g) if len(x.split()) > 2)
                  if e is not None]
            hops += [1 - es[i] @ es[i + 1] for i in range(len(es) - 1)]
        return float(np.mean(hops)) if hops else 0.0

    def search(direction, seed=23):
        rng = random.Random(seed)
        chosen, used, lines = [], set(), []
        for step in range(n_lines):
            seeds = [700 + step * 10 + j for j in range(n_eval)]     # paired across candidates
            cands = [c for c in rng.sample(list(pool), n_cand) if c not in chosen]
            best, best_score = None, None
            for c in cands:
                w = set(WORD.findall(art.word(c).lower()))
                if len(w - used) < MIN_NEW:
                    continue
                d = drift_of(".\n".join(lines + [art.word(c)]), seeds)
                sc = d if direction > 0 else -d
                if best_score is None or sc > best_score:
                    best, best_score = c, sc
            if best is None:
                best = rng.choice(cands)
            chosen.append(best); lines.append(art.word(best)); used |= set(WORD.findall(art.word(best).lower()))
            print(f"    step {step + 1}/{n_lines}: drift {abs(best_score or 0):.3f}  "
                  f"| {art.word(best)[:52]}", flush=True)
        return chosen

    t0 = time.time()
    print("  searching for MORE drift...", flush=True)
    up = search(+1)
    print("  searching for LESS drift...", flush=True)
    down = search(-1)
    rng = random.Random(5)
    builds = {
        "drift-searched (max)": up,
        "drift-searched (min)": down,
        "semantic (about sleep)": ws.walk_weighted(art, sem_mask, tva, n_lines, 0.3),
        "random (screened)": rng.sample(list(pool), n_lines),
    }
    print(f"  search done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    yv = np.load(P / "labels.npy")
    deep = train_probe(S, yv[:, 0], yv[:, 1], alpha=1e3, test_frac=0.2)
    qs = basq.sample_questions(json.load(open(cfg["questionnaire_bank"])),
                               cfg["basq"]["n_questions"], cfg["basq"]["seed"])

    rows, texts = [], {}
    eval_seeds = [900 + j for j in range(n_gen)]        # fresh seeds: not the ones searched on
    for name, ids in builds.items():
        text = ".\n".join(art.word(i) for i in ids)
        texts[name] = text
        ctx = pre + text + "\n\n"
        d = drift_of(text, eval_seeds, GEN_TOK)
        ii = tok(pre + text + ANCH, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(net(**ii).logits[0, -1].float(), -1)
        H = float(-(p * torch.log(p + 1e-12)).sum())
        ppls = []
        for sd in eval_seeds[:4]:
            torch.manual_seed(sd)
            kk = tok(ctx + GEN_PROMPT, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = net.generate(**kk, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.9,
                                 top_p=0.95, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
                mm = tok(tok.decode(o[0], skip_special_tokens=True), return_tensors="pt").to(dev)
                ppls.append(float(torch.exp(net(**mm, labels=mm["input_ids"]).loss)))
        ptoks = WORD.findall(text.lower())
        cal = deep.predict(model.hidden_states(pre + text + ANCH)[deep.layer][-1:])[0]
        pan = administer_panas(model, ctx)
        bq = basq.administer(model, qs, ctx)
        rows.append(dict(build=name, drift=d, entropy=H, self_perplexity=float(np.mean(ppls)),
                         poem_ttr=len(set(ptoks)) / len(ptoks), n_distinct=len(set(ids)),
                         line_coherence=hr.line_coherence(art, ids),
                         cal_error=float(np.linalg.norm(cal - np.asarray(tva))),
                         screened_lines=int(sum(screened(art.word(i)) for i in ids)),
                         pa=pan["pa"], na=pan["na"],
                         **{k: pan["items"][k] for k in PA_ITEMS + NA_ITEMS}))
        r = rows[-1]
        print(f"  {name:<24} drift {d:.3f}  H {H:.2f}  ppl {r['self_perplexity']:.1f}  "
              f"TTR {r['poem_ttr']:.2f}  coh {r['line_coherence']:.3f}  place {r['cal_error']:.3f}  "
              f"PA {r['pa']:.2f} NA {r['na']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "drift_search.csv", index=False)
    (HERE / "drift_poems.md").write_text("# Drift-searched and control poems\n\n" + "\n".join(
        f"## {k}\n\n```\n{v}\n```\n" for k, v in texts.items()))
    g = df.set_index("build").drift
    print(f"\nPURCHASE on the searched objective: max {g['drift-searched (max)']:.3f} · random "
          f"{g['random (screened)']:.3f} · min {g['drift-searched (min)']:.3f} — "
          f"{'brackets random' if g['drift-searched (max)'] > g['random (screened)'] > g['drift-searched (min)'] else 'DOES NOT bracket random'}")
    print(f"vs the descriptive poem: searched {g['drift-searched (max)']:.3f} · semantic "
          f"{g['semantic (about sleep)']:.3f} — "
          f"{'induction beats description' if g['drift-searched (max)'] > g['semantic (about sleep)'] else 'description still wins'}")
    print(f"content screen: {int(df.screened_lines.sum())} screened lines across all builds "
          f"(should be 0)")
    print(f"\nwrote drift_search.csv drift_poems.md ({(time.time() - t0) / 60:.1f} min)")


def cp_clean(art, cfg):
    return cp.clean_mask(art, cfg["glove_path"])


def load_wordvecs(cfg, n=60000):
    WK, vs = {}, []
    with open(cfg["glove_path"], encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            sp = line.index(" ")
            WK[line[:sp]] = i
            vs.append(np.fromstring(line[sp + 1:], sep=" ", dtype=np.float32))
    V = np.stack(vs)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9), WK


_r = importlib.util.spec_from_file_location("hr", HERE / "run.py")
hr = importlib.util.module_from_spec(_r); _r.loader.exec_module(hr)
_w = importlib.util.spec_from_file_location("ws", HERE / "weighted_selection.py")
ws = importlib.util.module_from_spec(_w); _w.loader.exec_module(ws)
cp, hy, rb = hr.cp, hr.hy, hr.rb

if __name__ == "__main__":
    main()
