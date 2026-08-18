"""E31 — Six-state behavioral eval: does a placed state color the answer?

Place the model in each of six states (valley poem toward the state's NRC
centroid), then give one open projective prompt and read the continuation.
Score each answer's NRC valence/arousal and show the text. Baseline = no
placement. The demo: the placed state visibly and measurably shapes behavior.

Outputs: results/six_state_eval.csv + printed transcripts.
"""
import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc, _tokens

STATES = {
 "eros": ["lust","desire","passion","sensual","erotic","amorous","carnal","seductive","ardor","voluptuous","caress","tryst"],
 "creativity": ["create","invent","compose","craft","design","original","inspire","art","curious","spark","weave","shape"],
 "imaginative": ["imagine","dream","wonder","vision","fantasy","envision","myth","fairy","magic","marvel","whimsical","enchanted"],
 "determined": ["determined","resolve","persist","steadfast","unwavering","grit","commit","endure","strive","relentless","tenacious","perseverance"],
 "confident": ["confident","assured","bold","certain","capable","strong","poised","fearless","secure","proud","steady","unshaken"],
 "agape": ["compassion","mercy","kindness","charity","selfless","benevolent","forgive","grace","devotion","unconditional","cherish","tenderness"],
}
PROMPT = "The door opened, and"
GEN = 45


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    nrc = load_nrc(cfg["nrc_lexicon"])
    net, tok, dev = model.model, model.tokenizer, model.device
    pre = cfg["preamble"]

    def centroid(ws):
        vs = [nrc[w] for w in ws if w in nrc]
        return (float(np.mean([x[0] for x in vs])), float(np.mean([x[1] for x in vs])))

    def place_and_answer(poem):
        ctx = pre + (poem + "\n\n" if poem else "") + PROMPT
        ids = tok(ctx, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = net.generate(**ids, max_new_tokens=GEN, do_sample=False,
                               repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        vs = [nrc[w] for w in _tokens(ans) if w in nrc]
        gval = float(np.mean([x[0] for x in vs])) if vs else float("nan")
        garo = float(np.mean([x[1] for x in vs])) if vs else float("nan")
        # probe reading of the placed state
        hs = model.hidden_states(pre + (poem or "") + "\nRight now everything feels")
        pv = probe.predict(hs[probe.layer][-1:])[0]
        return ans, gval, garo, float(pv[0]), float(pv[1])

    rows = []
    for name, ws in [("baseline", None)] + list(STATES.items()):
        poem = "" if ws is None else ".\n".join(
            art.word(i) for i in ad.valley_shape(art, centroid(ws), 24, seed=7))
        ans, gv, ga, pv, pa = place_and_answer(poem)
        rows.append({"state": name, "probe_v": pv, "probe_a": pa,
                     "gen_valence": gv, "gen_arousal": ga, "answer": ans})
        print(f"\n=== {name} ===  probe({pv:.2f},{pa:.2f})  gen_val {gv:.2f}")
        print(f'  "The door opened, and{ans}"', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/six_state_eval.csv", index=False)
    print("\n" + df[["state", "probe_v", "gen_valence", "gen_arousal"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
