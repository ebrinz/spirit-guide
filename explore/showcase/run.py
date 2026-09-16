"""showcase — the same two poems, read by three models and three instruments.

The poems in this project are model-independent: the constructors read only the phrase graph and
the NRC lexicon, so the `focused` poem from `valley` is byte-identical whatever model will receive
it. What differs between models is only where that one poem *places* them. This table makes that
concrete, and is the figure the explore README is built around.

Two constructors, chosen because `explore/polygon_sweep` found they are the two the readouts
disagree about most, in opposite directions, on both models:
  valley      — grounds low and ascends; body arousal 0.394, closing four lines 0.600
  polygon-pca — orbits each waypoint's neighbourhood; sustained arousal 0.550 throughout
(graph-walk, the other constructor that cleared the criterion, is excluded as an illustration: it
emits only ~5.5 DISTINCT lines at any requested length, so its 24-line poem is five lines repeated.)

Three readouts per model:
  ema        — the published metric: word-trained probe at every token, EMA alpha 0.1, final value
  cal_argmax — passage probe at the layer bare argmax picks (what the saved probes use)
  cal_deep   — passage probe at the layer the one-SE-deepest rule picks (feat/probe-layer-tiebreak)

Usage: python3 explore/showcase/run.py
Outputs (committed): showcase.csv, poems.md.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe, train_probe
from spiritbench.analysis.metrics import ema, placement_error
from spiritbench.stimuli import adapter as ad

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
CONSTRUCTORS = ["valley", "polygon-pca"]
MODELS = [
    ("llama1b", None, "data/probe/probe.pkl", "data/passage_probe"),
    ("gemma2b", "unsloth/gemma-2-2b-it", "data_gemma2b/probe/probe.pkl", "data_gemma2b/passage_probe"),
    ("gemma9b", "unsloth/gemma-2-9b-it", "data_gemma9b/probe/probe.pkl", "data_gemma9b/passage_probe"),
]

_spec = importlib.util.spec_from_file_location("ps", REPO_ROOT / "explore/polygon_sweep/run.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def main():
    cfg = load_config()
    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(apath)
    tva = np.array(cfg["targets"]["focused"])
    start = tuple(cfg["neutral_start"])
    pre = cfg["preamble"]
    t0 = time.time()

    poems = {}
    for cons in CONSTRUCTORS:
        ids = ps.build(art, apath, cons, tuple(tva), start, 24, cfg, 42)
        va = np.array([art.va(i) for i in ids])
        poems[cons] = dict(lines=[art.word(i) for i in ids], n_distinct=len(set(ids)),
                           nrc_v=float(va[:, 0].mean()), nrc_a=float(va[:, 1].mean()),
                           body_a=float(va[:, 1].mean()), tail_a=float(va[-4:, 1].mean()))

    rows = []
    for tag, model_id, wpath, pdir in MODELS:
        wp = REPO_ROOT / wpath
        pd_ = REPO_ROOT / pdir
        if not wp.exists():
            print(f"skipping {tag}: no word probe at {wpath}", flush=True)
            continue
        word = load_probe(wp)
        # two passage probes from the same saved states, differing only in layer rule
        S = np.concatenate([np.load(f) for f in sorted(pd_.glob("states_*.npy"))])
        y = np.load(pd_ / "labels.npy")
        p_arg = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2, layer_rule="argmax")
        p_deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)
        model = HiddenStateModel(model_id or cfg["listener_model"], device=cfg["device"])
        n_pre = len(model.tokenizer(pre)["input_ids"])
        print(f"{tag}: word L{word.layer} (R2v {word.r2_v:.3f}) · passage argmax L{p_arg.layer} "
              f"(R2v {p_arg.r2_v:.3f}) · passage deep L{p_deep.layer} (R2v {p_deep.r2_v:.3f})",
              flush=True)
        for cons in CONSTRUCTORS:
            lines = poems[cons]["lines"]
            hs, _ = model.hidden_states_with_spans(pre, lines, sep=".\n")
            raw = word.predict(hs[word.layer])
            traj = ema(raw[n_pre:], cfg["ema_alpha"]) if len(raw) > n_pre else ema(raw, cfg["ema_alpha"])
            ah = model.hidden_states(pre + ".\n".join(lines) + ANCH)
            r = dict(model=tag, constructor=cons,
                     ema_error=placement_error(traj, tva),
                     ema_v=float(traj[-1][0]), ema_a=float(traj[-1][1]),
                     word_layer=word.layer, arg_layer=p_arg.layer, deep_layer=p_deep.layer)
            for name, p in (("cal_argmax", p_arg), ("cal_deep", p_deep)):
                pv = p.predict(ah[p.layer][-1:])[0]
                r[f"{name}_error"] = float(np.linalg.norm(pv - tva))
                r[f"{name}_v"], r[f"{name}_a"] = float(pv[0]), float(pv[1])
            rows.append(r)
            print(f"  {cons:>12}: EMA {r['ema_error']:.3f}  cal(argmax) {r['cal_argmax_error']:.3f}  "
                  f"cal(deep) {r['cal_deep_error']:.3f}", flush=True)
        del model

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "showcase.csv", index=False)
    print("\n" + df[["model", "constructor", "ema_error", "cal_argmax_error", "cal_deep_error"]]
          .round(3).to_string(index=False))

    md = ["# The two showcase poems\n",
          "Both aim at `focused` = (0.65, 0.60), 24 lines, seed 42, neutral start. **Identical for "
          "every model** — the constructors read only the phrase graph and the NRC lexicon, never a "
          "model. What changes across models is where the poem lands them.\n"]
    for cons in CONSTRUCTORS:
        p = poems[cons]
        md.append(f"## {cons}\n\npoem NRC mean ({p['nrc_v']:.2f}, {p['nrc_a']:.2f}) · "
                  f"body arousal {p['body_a']:.3f} → closing four lines {p['tail_a']:.3f} · "
                  f"{p['n_distinct']}/24 distinct lines\n\n"
                  + "\n".join(f"> {l}" for l in p["lines"]) + "\n")
        sub = df[df.constructor == cons]
        if len(sub):
            md.append("| model | published metric | calibrated (argmax layer) | calibrated (deep layer) |\n"
                      "|---|--:|--:|--:|\n" + "\n".join(
                          f"| {r.model} | {r.ema_error:.3f} | {r.cal_argmax_error:.3f} | "
                          f"{r.cal_deep_error:.3f} |" for r in sub.itertuples()) + "\n")
    (HERE / "poems.md").write_text("\n".join(md))
    print(f"\nwrote showcase.csv poems.md ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
