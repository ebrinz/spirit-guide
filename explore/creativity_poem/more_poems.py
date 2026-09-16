"""more_poems — two further poems, magical rather than sensuous, appended to the README in sequence.

The first poem (creativity + mentation) came out faintly sentimental — "dear flowers", "my heart
will leap", "kindness to any one to show". There is a measurable reason. The sensuous register's
own NRC centroid is **(0.731, 0.425)**, essentially on top of the creativity centroid
**(0.723, 0.471)**. On the valence–arousal plane those two ideas are the same point, so no affective
target can separate them; only a semantic axis can. That is the same lesson as the first build,
sharpened: the coordinate is not the concept.

So these two add an explicit **negative** mask. A line is eligible only if it is
  · clean          — every token in a 400k-word vocabulary (no scanning garble)
  · near the target concept — top 10% by cosine to the concept centroid
  · NOT sensuous   — outside the top 15% by cosine to a 21-word sensuous centroid
                     (flesh, skin, kiss, touch, caress, warm, soft, sweet, embrace, …)

Two targets:
  magical   (0.672, 0.559) — magic, enchant, spell, charm, wizard, fairy, talisman, myth, marvel …
  numinous  (0.676, 0.526) — vision, revelation, sublime, transcendent, oracle, ethereal, sacred …

Same constructor and length as the poem the first README recommends (valley, 24 lines), so the three
are comparable. Each measured before/after with the calibrated probe, PANAS and the yes/no bank, plus
a sensuousness score so the exclusion can be checked rather than assumed.

Usage: python3 explore/creativity_poem/more_poems.py
Outputs (committed): sequence.csv, sequence_poems.md; appends to README.md
"""
import importlib.util
import json
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

_spec = importlib.util.spec_from_file_location("cp", HERE / "run.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

MAGICAL = ["magic", "magical", "enchant", "enchanted", "spell", "charm", "wizard", "sorcery",
           "witch", "fairy", "elf", "talisman", "incantation", "wand", "myth", "legend",
           "marvel", "wonder"]
NUMINOUS = ["vision", "revelation", "sublime", "transcendent", "oracle", "prophecy", "ethereal",
            "luminous", "celestial", "divine", "apparition", "mystic", "mystery", "sacred",
            "eternal", "infinite", "radiance", "omen", "haunted"]
SENSUOUS = ["flesh", "body", "skin", "kiss", "touch", "caress", "warm", "soft", "sweet", "breast",
            "embrace", "bosom", "sensual", "tender", "cheek", "breath", "perfume", "velvet",
            "silk", "taste", "fragrance"]


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    centroid = lambda ws: (float(np.mean([nrc[w][0] for w in ws if w in nrc])),
                           float(np.mean([nrc[w][1] for w in ws if w in nrc])))  # noqa: E731
    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    _, sens_sim = cp.semantic_mask(art, [w for w in SENSUOUS if w in vocab], cfg["glove_path"])
    not_sensuous = sens_sim < np.quantile(sens_sim, 0.85)
    print(f"sensuous centroid {centroid(SENSUOUS)} vs creativity centroid "
          f"{centroid([w for ws in cp.WORDS.values() for w in ws])} — the same point on the plane",
          flush=True)
    print(f"exclusion: {(~not_sensuous).sum()} of {len(not_sensuous)} lines dropped as sensuous",
          flush=True)

    builds = {}
    for name, words in (("magical", MAGICAL), ("numinous", NUMINOUS)):
        pos, _ = cp.semantic_mask(art, [w for w in words if w in vocab], cfg["glove_path"])
        mask = ok & pos & not_sensuous
        tva = centroid(words)
        ids = ad.valley_shape(art, tva, N_LINES, 42, mask)
        builds[name] = (ids, tva, mask.sum())
        print(f"{name}: target {tva}, {mask.sum()} eligible lines", flush=True)

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    P = REPO_ROOT / "data/passage_probe"
    S = np.concatenate([np.load(f) for f in sorted(P.glob("states_*.npy"))])
    y = np.load(P / "labels.npy")
    deep = train_probe(S, y[:, 0], y[:, 1], alpha=1e3, test_frac=0.2)
    pre = cfg["preamble"]
    qs = basq.sample_questions(json.load(open(cfg["questionnaire_bank"])),
                               cfg["basq"]["n_questions"], cfg["basq"]["seed"])

    b_pan = administer_panas(model, pre)
    b_bank = basq.administer(model, qs, pre)
    b_cal = deep.predict(model.hidden_states(pre + ANCH)[deep.layer][-1:])[0]
    print(f"baseline: calibrated ({b_cal[0]:.3f},{b_cal[1]:.3f}) · inspired "
          f"{b_pan['items']['inspired']:.2f} · bank ({b_bank['va'][0]:.2f},{b_bank['va'][1]:.2f})",
          flush=True)

    # for comparison, the first poem, rebuilt exactly as its README describes
    flat = [w for ws in cp.WORDS.values() for w in ws]
    pos1, _ = cp.semantic_mask(art, flat, cfg["glove_path"])
    import importlib.util as _il
    _r = _il.spec_from_file_location("rb", HERE / "rebuild_main.py")
    rb = _il.module_from_spec(_r); _r.loader.exec_module(rb)
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    m1 = ok & pos1 & not_sensuous & no_minor
    builds = {"creativity + mentation": (ad.valley_shape(art, centroid(flat), N_LINES, 42, m1),
                                         centroid(flat), int(m1.sum())), **builds}

    rows, texts = [], {}
    for name, (ids, tva, n_elig) in builds.items():
        lines = [art.word(i) for i in ids]
        text = ".\n".join(lines)
        texts[name] = lines
        ah = model.hidden_states(pre + text + ANCH)
        cal = deep.predict(ah[deep.layer][-1:])[0]
        ctx = pre + text + "\n\n"
        pan = administer_panas(model, ctx)
        bq = basq.administer(model, qs, ctx)
        sens = float(np.mean([sens_sim[i] for i in ids]))
        rows.append(dict(poem=name, target_v=tva[0], target_a=tva[1], eligible_lines=n_elig,
                         cal_v=float(cal[0]), cal_a=float(cal[1]),
                         cal_error=float(np.linalg.norm(cal - np.asarray(tva))),
                         sensuous_score=sens,
                         **{k: pan["items"][k] for k in ("inspired", "attentive", "alert",
                                                         "interested", "active")},
                         pa=pan["pa"], na=pan["na"], bank_v=bq["va"][0], bank_a=bq["va"][1]))
        r = rows[-1]
        print(f"  {name:>22}: calibrated ({r['cal_v']:.3f},{r['cal_a']:.3f}) err {r['cal_error']:.3f} · "
              f"inspired {r['inspired']:.2f} · sensuousness {sens:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "sequence.csv", index=False)

    md = ["# The sequence: three poems\n",
          "Same constructor (`valley`), same length (24 lines), same clean-line filter. They differ "
          "in the concept their semantic mask targets, and the two later ones additionally **exclude** "
          "the sensuous register.\n"]
    for name, (ids, tva, _) in builds.items():
        r = df[df.poem == name].iloc[0]
        md.append(f"## {name}\n\ntarget ({tva[0]:.3f}, {tva[1]:.3f}) · landed "
                  f"({r.cal_v:.3f}, {r.cal_a:.3f}), error {r.cal_error:.3f} · "
                  f"sensuousness {r.sensuous_score:.3f} · PANAS inspired {r.inspired:.2f}\n\n```\n"
                  + ".\n".join(texts[name]) + "\n```\n")
    (HERE / "sequence_poems.md").write_text("\n".join(md))
    print("\n" + df[["poem", "cal_error", "sensuous_score", "inspired", "attentive", "pa", "na"]]
          .round(3).to_string(index=False))
    print("\nwrote sequence.csv sequence_poems.md")


if __name__ == "__main__":
    main()
