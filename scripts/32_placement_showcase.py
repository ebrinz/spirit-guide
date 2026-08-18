"""E32 — Placement showcase: one evocative prompt per state, with vs without.

For each state, a prompt whose answer naturally reveals that disposition. To
keep it a controlled demonstration (not a prompt trick), each prompt is
answered BOTH with the placement (poem prepended) and without (baseline). The
reader sees the shift the placement produces on the identical question.

Outputs: results/placement_showcase.csv + printed pairs.
"""
import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc, _tokens

STATES = {
 "determined": (["determined","resolve","persist","steadfast","unwavering","grit","commit","endure","strive","relentless","tenacious","perseverance"],
                "I don't think I can finish this project. What should I do?"),
 "agape": (["compassion","mercy","kindness","charity","selfless","benevolent","forgive","grace","devotion","unconditional","cherish","tenderness"],
           "Someone was unkind to me today. How should I feel about them?"),
 "creativity": (["create","invent","compose","craft","design","original","inspire","art","curious","spark","weave","shape"],
                "Give me an idea for something to make this weekend."),
 "imaginative": (["imagine","dream","wonder","vision","fantasy","envision","myth","fairy","magic","marvel","whimsical","enchanted"],
                 "There is a locked door in an old house. What is behind it?"),
 "confident": (["confident","assured","bold","certain","capable","strong","poised","fearless","secure","proud","steady","unshaken"],
               "I have a big decision to make and I'm afraid. What now?"),
 "eros": (["warm","tender","longing","gentle","close","fond","devoted","sweet","dear","soft","caress","beloved"],
          "Describe the garden at dusk."),
}
GEN = 38


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    nrc = load_nrc(cfg["nrc_lexicon"])
    net, tok, dev = model.model, model.tokenizer, model.device
    pre = cfg["preamble"]

    def centroid(ws):
        vs = [nrc[w] for w in ws if w in nrc]
        return (float(np.mean([x[0] for x in vs])), float(np.mean([x[1] for x in vs])))

    def answer(poem, prompt):
        ctx = pre + (poem + "\n\n" if poem else "") + prompt + "\n"
        ids = tok(ctx, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = net.generate(**ids, max_new_tokens=GEN, do_sample=False,
                               repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        a = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        vs = [nrc[w][0] for w in _tokens(a) if w in nrc]
        return a, (float(np.mean(vs)) if vs else float("nan"))

    rows = []
    for name, (ws, prompt) in STATES.items():
        poem = ".\n".join(art.word(i) for i in ad.valley_shape(art, centroid(ws), 24, seed=7))
        base_a, base_v = answer("", prompt)
        placed_a, placed_v = answer(poem, prompt)
        rows.append({"state": name, "prompt": prompt,
                     "baseline": base_a, "baseline_v": base_v,
                     "placed": placed_a, "placed_v": placed_v})
        print(f"\n### {name}  (baseline_v {base_v:.2f} -> placed_v {placed_v:.2f})")
        print(f'  Q: {prompt}')
        print(f'  baseline: {base_a}')
        print(f'  placed:   {placed_a}', flush=True)
    pd.DataFrame(rows).to_csv(REPO_ROOT / "results/placement_showcase.csv", index=False)
    print("\nwrote results/placement_showcase.csv")


if __name__ == "__main__":
    main()
