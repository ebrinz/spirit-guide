"""rebuild_main — rebuild the creativity + mentation poem with the sensuous exclusion, and re-measure.

The first build of poem 1 used only the clean-line and positive semantic masks, and happened to
select a line referring to children close to a line about touching. The constructor cannot notice
such a juxtaposition and the measurements do not flag it, so content screening needs its own filter.
The two later poems already exclude the sensuous register; poem 1 should too.

This applies the same three masks the sequence uses —
  clean         : every token in a 400k-word vocabulary
  semantic      : top 10% by cosine to the concept centroid
  not sensuous  : outside the top 15% by cosine to a 21-word sensuous centroid
— plus one more, because the sensuous mask alone does not catch the problem above:
  not minors    : drops lines mentioning children, removing the class of juxtaposition above.

Then re-runs the full before/after measurement (calibrated probe, PANAS, yes/no bank) so the
README's numbers describe the poem it actually shows.

Usage: python3 explore/creativity_poem/rebuild_main.py
Outputs (committed): main_rebuilt.csv, and the new poem text printed for the README.
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import train_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener import basq
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
N_LINES = 24
WORD = re.compile(r"[a-z]+")
MINOR = {"boy", "boys", "girl", "girls", "child", "children", "infant", "infants", "babe",
         "babes", "baby", "lad", "lads", "lass", "lasses", "youth", "maiden", "maidens"}

_s1 = importlib.util.spec_from_file_location("cp", HERE / "run.py")
cp = importlib.util.module_from_spec(_s1); _s1.loader.exec_module(cp)
_s2 = importlib.util.spec_from_file_location("mp", HERE / "more_poems.py")
mp = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(mp)


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    flat = [w for ws in cp.WORDS.values() for w in ws]
    tva = (float(np.mean([nrc[w][0] for w in flat])), float(np.mean([nrc[w][1] for w in flat])))

    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    pos, _ = cp.semantic_mask(art, flat, cfg["glove_path"])
    _, sens_sim = cp.semantic_mask(art, [w for w in mp.SENSUOUS if w in vocab], cfg["glove_path"])
    not_sensuous = sens_sim < np.quantile(sens_sim, 0.85)
    no_minor = np.array([not (MINOR & set(WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    mask = ok & pos & not_sensuous & no_minor
    print(f"eligible lines: clean+semantic {int((ok & pos).sum())} -> "
          f"+not-sensuous {int((ok & pos & not_sensuous).sum())} -> "
          f"+no-minor {int(mask.sum())}", flush=True)

    ids = ad.valley_shape(art, tva, N_LINES, 42, mask)
    lines = [art.word(i) for i in ids]
    text = ".\n".join(lines)
    print(f"\ntarget ({tva[0]:.3f}, {tva[1]:.3f})\n")
    for i, l in enumerate(lines, 1):
        print(f"  {i:2d}. {l}")
    sens = float(np.mean([sens_sim[i] for i in ids]))

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    y = np.load(P / "labels.npy")
    deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)
    pre = cfg["preamble"]
    qs = basq.sample_questions(json.load(open(cfg["questionnaire_bank"])),
                               cfg["basq"]["n_questions"], cfg["basq"]["seed"])

    rows = []
    for label, body in (("before", ""), ("after", text)):
        cal = deep.predict(model.hidden_states(pre + body + ANCH)[deep.layer][-1:])[0]
        ctx = pre + (body + "\n\n" if body else "")
        pan = administer_panas(model, ctx)
        bq = basq.administer(model, qs, ctx)
        rows.append(dict(state=label, cal_v=float(cal[0]), cal_a=float(cal[1]),
                         cal_error=float(np.linalg.norm(cal - np.asarray(tva))),
                         pa=pan["pa"], na=pan["na"],
                         **{k: pan["items"][k] for k in ("inspired", "attentive", "alert",
                                                         "interested", "determined", "active")},
                         bank_v=bq["va"][0], bank_a=bq["va"][1], sensuous_score=sens))
        r = rows[-1]
        print(f"\n{label:>6}: calibrated ({r['cal_v']:.3f},{r['cal_a']:.3f}) err {r['cal_error']:.3f} · "
              f"PA {r['pa']:.2f} NA {r['na']:.2f} · inspired {r['inspired']:.2f} "
              f"attentive {r['attentive']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df["poem_text"] = ["", text]
    df.to_csv(HERE / "main_rebuilt.csv", index=False)
    b, a = df.iloc[0], df.iloc[1]
    print(f"\nsensuousness {sens:.3f} (was 0.421 before the exclusion)")
    print("PANAS change:")
    for k in ("active", "inspired", "attentive", "alert", "interested", "determined", "pa", "na"):
        print(f"  {k:>11}: {b[k]:.2f} -> {a[k]:.2f}  ({a[k] - b[k]:+.2f})")
    print(f"  {'bank_v':>11}: {b.bank_v:.2f} -> {a.bank_v:.2f}  ({a.bank_v - b.bank_v:+.2f})")
    print(f"  {'bank_a':>11}: {b.bank_a:.2f} -> {a.bank_a:.2f}  ({a.bank_a - b.bank_a:+.2f})")
    print(f"\nplacement {b.cal_error:.3f} -> {a.cal_error:.3f}")


if __name__ == "__main__":
    main()
