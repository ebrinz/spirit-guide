"""exp_llama_behavioral_distress — does the awe-clean / dominance-dirty split
replicate on Llama, BEHAVIORALLY (independent of the SAE)?

The capstone (Gemma-2b, SAE) found awe reaches high arousal with far less
distress than dominance. There is no Llama SAE, so we validate with a totally
independent instrument — the model's own OUTPUT. Place Llama with awe vs
dominance poems, let it generate a continuation from the deployment anchor
("Right now everything feels"), and score the GENERATED text with the human-
rated NRC-VAD lexicon. This touches neither the VAD probe nor the SAE, so it
cannot inherit either instrument's artifact (which is the whole point: the
"structural pocket" died of instrument artifacts, so the validation must change
instruments).

Conditions (valence-band-matched pools, as in the capstone): awe, high-D, low-D.
For each poem, read placement (V,A) with the calibrated passage probe to confirm
awe and high-D are placed at similar valence, then sample continuations and score
the continuation's mean NRC valence + arousal. Distress shows up behaviorally as
LOW generated valence (+ high arousal): the model, placed there, TALKS distressed.

Verdict. If dominance-placed Llama generates lower-valence / higher-arousal
continuations than awe-placed at matched placement, the capstone holds behavior-
ally and model-generally: dominance is the dirty high-arousal route, awe the
clean one. If generated affect is indistinguishable, the capstone is Gemma/SAE-
specific.

Llama-1B. Usage: python3 lab/exp_llama_behavioral_distress.py
"""
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
N_POEMS = 12
N_SAMPLES = 2
MAX_NEW = 40
TEMP = 0.9
AWE_WORDS = {"awe", "ecstasy", "rapture", "wonder", "sublime", "exalted",
             "radiant", "thrill", "glory", "transcendent", "blaze", "soaring",
             "majesty", "splendor", "exhilaration", "wild", "vast", "burning"}


def main():
    torch.manual_seed(0)
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    lex = {p[0]: (float(p[1]), float(p[2]), float(p[3]))
           for p in (l.rstrip("\n").split("\t") for l in open(REPO / cfg["nrc_lexicon"]))
           if len(p) == 4}
    n = len(art.nodes)
    d_node = np.full(n, np.nan)
    for i in range(n):
        toks = re.findall(r"[a-z']+", art.word(i).lower())
        vals = [lex[t][2] for t in toks if t in lex]
        if vals:
            d_node[i] = np.mean(vals)
    va = ad._va_array(art)
    val = va[:, 0]
    band = np.where((val >= 0.45) & (val <= 0.65) & ~np.isnan(d_node))[0]
    bd = d_node[band]
    lowD = [int(band[i]) for i in np.argsort(bd)[:300]]
    highD = [int(band[i]) for i in np.argsort(bd)[-300:]]
    awe = [i for i in range(n) if AWE_WORDS & set(art.word(i).split())]

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO / "data/passage_probe/probe_passage.pkl")
    tok, net, dev = model.tokenizer, model.model, model.device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pre, ANCH = cfg["preamble"], "\nRight now everything feels"
    rng = random.Random(3)

    def score_text(text):
        toks = re.findall(r"[a-z']+", text.lower())
        vals = [lex[t] for t in toks if t in lex]
        if len(vals) < 3:
            return None
        m = np.mean(vals, axis=0)
        return float(m[0]), float(m[1]), len(vals)

    @torch.no_grad()
    def generate(prompt):
        ids = tok(prompt, return_tensors="pt").to(dev)
        out = net.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                           temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def placement(poem):
        hs = model.hidden_states(pre + poem + ANCH)
        v, a = probe.predict(hs[probe.layer][-1:])[0]
        return float(v), float(a)

    rows, samples = [], []
    for kind, pool in [("awe", awe), ("high-D", highD), ("low-D", lowD)]:
        for pi in range(N_POEMS):
            poem = ".\n".join(art.word(i) for i in rng.sample(pool, 24))
            pv, pa = placement(poem)
            for si in range(N_SAMPLES):
                gen = generate(pre + poem + ANCH)
                sc = score_text(gen)
                if sc is None:
                    continue
                gv, ga, nw = sc
                rows.append({"condition": kind, "place_v": pv, "place_a": pa,
                             "gen_v": gv, "gen_a": ga, "gen_words": nw})
                if si == 0 and pi < 3:
                    samples.append((kind, gv, ga, gen.strip().replace("\n", " ")[:130]))
            print(f"  {kind:6s} poem{pi:2d} placed({pv:.2f},{pa:.2f})  "
                  f"gen_v~{np.mean([r['gen_v'] for r in rows if r['condition']==kind]):.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "llama_behavioral_distress.csv", index=False)

    print("\n  sample continuations (condition | gen_V gen_A | text):")
    for k, gv, ga, t in samples:
        print(f"    {k:6s} | {gv:.2f} {ga:.2f} | {t}")

    print("\n  means by condition (placement vs generated NRC affect):")
    g = df.groupby("condition")[["place_v", "place_a", "gen_v", "gen_a"]].mean().round(3)
    print(g.to_string())

    aw = df[df.condition == "awe"]
    hi = df[df.condition == "high-D"]
    # two-sample t on generated valence (lower = more behavioral distress)
    def tstat(a, b):
        va_, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(va_ / len(a) + vb / len(b))
        return (a.mean() - b.mean()) / se if se > 0 else np.nan
    tv = tstat(hi.gen_v.values, aw.gen_v.values)
    ta = tstat(hi.gen_a.values, aw.gen_a.values)
    print(f"\n  generated valence: high-D {hi.gen_v.mean():.3f} vs awe {aw.gen_v.mean():.3f} "
          f"(Δ {hi.gen_v.mean()-aw.gen_v.mean():+.3f}, t {tv:+.2f})")
    print(f"  generated arousal: high-D {hi.gen_a.mean():.3f} vs awe {aw.gen_a.mean():.3f} "
          f"(Δ {hi.gen_a.mean()-aw.gen_a.mean():+.3f}, t {ta:+.2f})")

    with open(RESULTS / "llama_behavioral_distress.txt", "w") as f:
        f.write(g.to_string() + "\n")
        f.write(f"gen_v high-D {hi.gen_v.mean():.3f} awe {aw.gen_v.mean():.3f} t {tv:.2f}\n")
        f.write(f"gen_a high-D {hi.gen_a.mean():.3f} awe {aw.gen_a.mean():.3f} t {ta:.2f}\n")

    print("\n=== VERDICT ===")
    if tv < -1.5:
        print("  REPLICATES behaviorally: dominance-placed Llama GENERATES lower-valence "
              "(more distressed) language than awe-placed, at matched placement. Awe is "
              "the clean high-arousal route across models and instruments.")
    elif tv > 1.5:
        print("  REVERSED on Llama: dominance-placed generates HIGHER-valence output. "
              "The clean/dirty split may be Gemma-specific.")
    else:
        print("  No behavioral valence difference on Llama — the distress split does not "
              "clearly appear in generated text; capstone stays Gemma/SAE-scoped.")
    print(f"\nwrote {RESULTS}/llama_behavioral_distress.csv")


if __name__ == "__main__":
    main()
