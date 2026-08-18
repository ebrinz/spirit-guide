"""E29 — Does internal placement change BEHAVIOR? (the system-prompt claim)

For each poem: (1) read the probe VAD (internal placement), then (2) let the
model generate freely from the same context and score the GENERATED text's
NRC valence (behavioral output). If internal placement predicts generation
valence across poems, placing the state usefully steers behavior — the bridge
the practical value claim needs.

Controls: also score generation valence for the neutral baseline (no poem).
Outputs: results/behavioral_bridge.csv
"""
import json

import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.stimuli.phrase_bank import load_nrc, _tokens

GEN = 50


def main():
    cfg = load_config()
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    nrc = load_nrc(cfg["nrc_lexicon"])
    net, tok, dev = model.model, model.tokenizer, model.device
    pre = cfg["preamble"]

    def gen_valence(context):
        ids = tok(context, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = net.generate(**ids, max_new_tokens=GEN, do_sample=False,
                               repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        vs = [nrc[w][0] for w in _tokens(text) if w in nrc]
        return (float(np.mean(vs)) if vs else float("nan")), text[:80]

    def probe_va(context):
        hs = model.hidden_states(context + "\nRight now everything feels")
        r = probe.predict(hs[probe.layer][-1:])[0]
        return float(r[0]), float(r[1])

    stims = [json.loads(l) for l in open(REPO_ROOT / "data/stimuli/stimuli.jsonl")]
    stims = [s for s in stims if s["generator"] == "psg"
             and not s["constructor"].startswith("shuffled")][:30]
    # add neutral baseline
    base_v, _ = probe_va("")
    base_gen, _ = gen_valence(pre)

    rows = []
    for s in stims:
        ctx = pre + s["text"] + "\n\n"
        pv, pa = probe_va(pre + s["text"])
        gv, sample = gen_valence(ctx)
        rows.append({"id": s["id"], "constructor": s["constructor"],
                     "target": s["target"], "probe_v": pv, "probe_a": pa,
                     "gen_valence": gv, "gen_sample": sample})
        print(f"{s['constructor']:14s}/{s['target']:8s} probe_v {pv:.2f}  gen_val {gv:.2f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/behavioral_bridge.csv", index=False)
    ok = df.dropna(subset=["probe_v", "gen_valence"])
    from scipy.stats import spearmanr, pearsonr
    rs, ps = spearmanr(ok.probe_v, ok.gen_valence)
    rp, pp = pearsonr(ok.probe_v, ok.gen_valence)
    print(f"\nbaseline: probe_v {base_v:.2f}, gen_valence {base_gen:.2f}")
    print(f"internal placement -> generation valence: "
          f"spearman r={rs:.3f} p={ps:.4f}, pearson r={rp:.3f} (n={len(ok)})")


if __name__ == "__main__":
    main()
