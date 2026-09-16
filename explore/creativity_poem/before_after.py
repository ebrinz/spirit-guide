"""before_after — the model's state with and without the poem, for the standalone README.

Measures the same model twice: bare preamble, then preamble + poem. Four channels, so a reader can
see what "state" means here rather than taking one number on trust:

  calibrated coordinate  — passage probe at a fixed anchor (the whole-context read)
  published metric       — word probe, per-token, EMA (the pipeline's own measure)
  PANAS                  — 20 adjectives rated 1-5 "right now", with the mentation items broken out
  30-item yes/no bank    — its own (V, A) coordinate from the questions answered yes

Plus one free-form continuation each, sampled at a fixed seed. Caveat carried from
`explore/ideal_state`: an instruct-tuned model given a poem tends to *analyse* it rather than
inhabit it (18 of 21 samples there opened "This meditation prompts…"), so the generation is shown
as illustration, not evidence, and uses a bare first-person anchor with no meditation framing —
the pathway `lab/exp_llama_behavioral_distress.py` found necessary to get past the assistant persona.

Usage: python3 explore/creativity_poem/before_after.py
Outputs (committed): before_after.csv, generations.md
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
GEN_PROMPT = "I close my eyes, and what comes to mind is"
N_GEN, GEN_TOKENS = 3, 55

_spec = importlib.util.spec_from_file_location("cp", HERE / "run.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    flat = [w for ws in cp.WORDS.values() for w in ws]
    tva = np.array([np.mean([nrc[w][0] for w in flat]), np.mean([nrc[w][1] for w in flat])])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    _rb = importlib.util.spec_from_file_location("rb", HERE / "rebuild_main.py")
    rb = importlib.util.module_from_spec(_rb); _rb.loader.exec_module(rb)
    _mp = importlib.util.spec_from_file_location("mp", HERE / "more_poems.py")
    mp = importlib.util.module_from_spec(_mp); _mp.loader.exec_module(mp)
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    sem, _ = cp.semantic_mask(art, flat, cfg["glove_path"])
    _, sens_sim = cp.semantic_mask(art, [w for w in mp.SENSUOUS if w in vocab], cfg["glove_path"])
    not_sensuous = sens_sim < np.quantile(sens_sim, 0.85)
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    ids = ad.valley_shape(art, tuple(tva), 24, 42, ok & sem & not_sensuous & no_minor)
    poem = ".\n".join(art.word(i) for i in ids)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    word = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    y = np.load(P / "labels.npy")
    deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)
    pre = cfg["preamble"]
    n_pre = len(model.tokenizer(pre)["input_ids"])
    qs = basq.sample_questions(json.load(open(cfg["questionnaire_bank"])),
                               cfg["basq"]["n_questions"], cfg["basq"]["seed"])

    def generate(ctx, seed):
        tok, net, dev = model.tokenizer, model.model, model.device
        i = tok(ctx, return_tensors="pt").to(dev)
        torch.manual_seed(seed)
        with torch.no_grad():
            o = net.generate(**i, max_new_tokens=GEN_TOKENS, do_sample=True, temperature=0.9,
                             top_p=0.95, repetition_penalty=1.25, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, i["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows, gens = [], {}
    for label, body in (("before", ""), ("after", poem)):
        ah = model.hidden_states(pre + body + ANCH)
        cal = deep.predict(ah[deep.layer][-1:])[0]
        if body:
            lines = body.split(".\n")
            hs, _ = model.hidden_states_with_spans(pre, lines, sep=".\n")
            raw = word.predict(hs[word.layer])
            traj = ema(raw[n_pre:], cfg["ema_alpha"])
            emav = traj[-1]
        else:
            raw = word.predict(model.hidden_states(pre + ANCH)[word.layer])
            emav = ema(raw, cfg["ema_alpha"])[-1]
        ctx = pre + (body + "\n\n" if body else "")
        pan = administer_panas(model, ctx)
        bq = basq.administer(model, qs, ctx)
        rows.append(dict(state=label, cal_v=float(cal[0]), cal_a=float(cal[1]),
                         cal_error=float(np.linalg.norm(cal - tva)),
                         ema_v=float(emav[0]), ema_a=float(emav[1]),
                         ema_error=float(placement_error(np.array([emav]), tva)),
                         pa=pan["pa"], na=pan["na"], **{k: pan["items"][k] for k in
                         ("inspired", "attentive", "alert", "interested", "determined", "active")},
                         bank_v=bq["va"][0], bank_a=bq["va"][1], bank_n_yes=bq["n_yes"]))
        gens[label] = [generate(ctx + GEN_PROMPT, 100 + k) for k in range(N_GEN)]
        r = rows[-1]
        print(f"{label:>7}: calibrated ({r['cal_v']:.3f},{r['cal_a']:.3f}) · PANAS inspired "
              f"{r['inspired']:.2f} attentive {r['attentive']:.2f} · bank "
              f"({r['bank_v']:.2f},{r['bank_a']:.2f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "before_after.csv", index=False)
    b, a = df.iloc[0], df.iloc[1]
    print(f"\ntarget ({tva[0]:.3f}, {tva[1]:.3f})")
    print(f"calibrated moved ({b.cal_v:.3f},{b.cal_a:.3f}) -> ({a.cal_v:.3f},{a.cal_a:.3f}); "
          f"distance to target {b.cal_error:.3f} -> {a.cal_error:.3f}")

    md = ["# Before and after, free-form\n",
          f"Prompt: *{GEN_PROMPT}* · {N_GEN} samples, temperature 0.9, fixed seeds. No meditation "
          "framing, because an instruct-tuned model given a poem under one tends to review the poem "
          "rather than speak from it.\n",
          "**Illustration, not evidence.** Three samples per condition on a 1B model.\n"]
    for label in ("before", "after"):
        md.append(f"## {label} the poem\n")
        for k, g in enumerate(gens[label], 1):
            md.append(f"**{k}.** {GEN_PROMPT} {g}\n")
    (HERE / "generations.md").write_text("\n".join(md))
    print("wrote before_after.csv generations.md")


if __name__ == "__main__":
    main()
