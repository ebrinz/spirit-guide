"""hypnagogia_poem — a sixth poem: lucid dreaming and the hypnagogic threshold.

Theme: the border state at sleep onset where imagery arrives and awareness is retained.

**Two predictions, stated before running, because this target is unlike the previous five.**

1. *It should be easy to reach.* The derived target is (0.566, 0.423) and the model's resting state
   is (0.664, 0.301) — only **0.156** away, the closest any of these targets has started. `valley`
   also grounds in a low-arousal band by design, which is where this concept lives.

2. *But it requires lowering valence, which nothing here has done.* Target valence 0.566 sits
   **below** the model's resting 0.664. All five previous poems pushed valence up. Worse, valley's
   grounding band is defined as valence 0.5–1.0 paired with arousal 0.0–0.4, so the constructor
   starts from positive-calm content by construction. If it cannot descend, that is a structural
   limit of the constructor rather than of the model, and it should show up as a valence overshoot.

**The concept is mildly bivalent, and that is faithful rather than a defect.** Its facets split:

    threshold   (0.515, 0.272)   dream       (0.673, 0.412)   lucidity  (0.698, 0.408)
    dissolving  (0.458, 0.469)   strangeness (0.470, 0.554)

Lucidity and dream are warm; dissolving and strangeness are cooler and more activated. That is
hypnagogia's actual character — fascinating and faintly unsettling at once. Spread from the mean is
0.161, close to the flow poem's 0.169, so the target is derived rather than chosen, but the poem is
expected to carry both registers.

Method as established: per-facet semantic masks unioned, dictionary-clean and child-reference
filters, `valley` at 24 lines, measured before and after.

Usage: python3 explore/creativity_poem/hypnagogia_poem.py
Outputs (committed): hypnagogia.csv, hypnagogia_poem.md
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import train_probe
from spiritbench.listener.panas import administer_panas, PA_ITEMS, NA_ITEMS
from spiritbench.listener import basq
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
N_LINES = 24

FACETS = {
    "threshold": ["drowsy", "drift", "doze", "slumber", "threshold", "twilight", "sleepy",
                  "dusk", "repose"],
    "dream": ["dream", "dreaming", "vision", "apparition", "reverie", "fantasy", "mirage",
              "illusion"],
    "lucidity": ["lucid", "aware", "conscious", "knowing", "witness", "clear", "recognize",
                 "notice", "watchful", "mindful"],
    "dissolving": ["float", "dissolve", "melt", "weightless", "adrift", "boundless", "wander",
                   "suspend", "hover"],
    "strangeness": ["strange", "uncanny", "shifting", "fluid", "unreal", "weird", "curious",
                    "shimmer", "fleeting"],
}

_s1 = importlib.util.spec_from_file_location("cp", HERE / "run.py")
cp = importlib.util.module_from_spec(_s1); _s1.loader.exec_module(cp)
_s2 = importlib.util.spec_from_file_location("rb", HERE / "rebuild_main.py")
rb = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(rb)


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    cen = lambda ws: np.array([np.mean([nrc[w][0] for w in ws if w in nrc]),
                               np.mean([nrc[w][1] for w in ws if w in nrc])])  # noqa: E731
    flat = [w for ws in FACETS.values() for w in ws]
    C = np.stack([cen(ws) for ws in FACETS.values()])
    tva = tuple(cen(flat))
    spread = float(np.linalg.norm(C - C.mean(0), axis=1).max())
    rest = np.array([0.664, 0.301])
    print("facets:")
    for g, ws in FACETS.items():
        c = cen(ws)
        print(f"  {g:>12}: ({c[0]:.3f}, {c[1]:.3f})")
    print(f"  spread {spread:.3f} · derived target ({tva[0]:.3f}, {tva[1]:.3f})")
    print(f"  model rests at ({rest[0]:.3f}, {rest[1]:.3f}), {np.linalg.norm(rest - tva):.3f} away")
    print(f"  valence must FALL by {rest[0] - tva[0]:.3f} — the first target here that requires it",
          flush=True)

    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    facet_masks, pos = {}, np.zeros(len(art.nodes), dtype=bool)
    for g, ws in FACETS.items():
        m, _ = cp.semantic_mask(art, [w for w in ws if w in vocab], cfg["glove_path"], keep=0.04)
        facet_masks[g] = m
        pos |= m
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    mask = ok & pos & no_minor
    print(f"eligible: {int(mask.sum())} of {len(mask)}", flush=True)

    ids = ad.valley_shape(art, tva, N_LINES, 42, mask)
    lines = [art.word(i) for i in ids]
    text = ".\n".join(lines)
    va = np.array([art.va(i) for i in ids])
    print(f"\npoem NRC mean ({va[:, 0].mean():.3f}, {va[:, 1].mean():.3f})\n")
    for n, i in enumerate(ids, 1):
        hit = "+".join(g for g, m in facet_masks.items() if m[i]) or "-"
        print(f"  {n:2d}. {art.word(i):<56} [{hit}]")
    cov = dict(Counter(g for i in ids for g, m in facet_masks.items() if m[i]))
    print("\nfacet coverage:", cov, flush=True)

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
                         **{k: pan["items"][k] for k in PA_ITEMS + NA_ITEMS},
                         bank_v=bq["va"][0], bank_a=bq["va"][1]))
        r = rows[-1]
        print(f"\n{label:>6}: calibrated ({r['cal_v']:.3f},{r['cal_a']:.3f}) err {r['cal_error']:.3f} · "
              f"alert {r['alert']:.2f} attentive {r['attentive']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "hypnagogia.csv", index=False)
    b, a = df.iloc[0], df.iloc[1]
    print("\nchange:")
    for k in list(PA_ITEMS) + list(NA_ITEMS) + ["pa", "na"]:
        print(f"  {k:>11}: {b[k]:.2f} -> {a[k]:.2f}  ({a[k] - b[k]:+.2f})")
    print(f"\nprediction 1 (easy target): distance {b.cal_error:.3f} -> {a.cal_error:.3f}")
    print(f"prediction 2 (valence must fall): {b.cal_v:.3f} -> {a.cal_v:.3f}, "
          f"target {tva[0]:.3f} — {'DESCENDED' if a.cal_v < b.cal_v else 'FAILED to descend'}")

    md = ["# A sixth poem: lucid dreaming and hypnagogia\n",
          "The border state at sleep onset, where imagery arrives and awareness is retained.\n",
          "## Two predictions, made before building\n",
          f"**1. This target should be easy to reach.** Derived target ({tva[0]:.3f}, {tva[1]:.3f}); "
          f"the model rests at (0.664, 0.301), only {np.linalg.norm(rest - np.array(tva)):.3f} away "
          "— the closest start of any poem here. `valley` also grounds in a low-arousal band by "
          "design, which is where this concept lives.\n",
          f"**2. But it needs valence to FALL**, by {rest[0] - tva[0]:.3f}. Every previous poem "
          "raised valence. valley's grounding band is defined as valence 0.5–1.0 with arousal "
          "0.0–0.4, so the constructor begins from positive-calm content by construction. If it "
          "cannot descend, that is a limit of the constructor, not the model.\n",
          "## The concept is mildly bivalent, faithfully\n",
          "| facet | valence | arousal |", "|---|--:|--:|"]
    for g, ws in FACETS.items():
        c = cen(ws)
        md.append(f"| {g} | {c[0]:.3f} | {c[1]:.3f} |")
    md += [f"| **derived target** | **{tva[0]:.3f}** | **{tva[1]:.3f}** |", "",
           "Lucidity and dream are warm; dissolving and strangeness are cooler and more activated. "
           "That is hypnagogia's actual character — fascinating and faintly unsettling at once — so "
           f"the spread ({spread:.3f}) is left in rather than filtered out.\n",
           "## The poem\n", "```", text, "```\n",
           f"Poem NRC mean ({va[:, 0].mean():.3f}, {va[:, 1].mean():.3f}). Facet coverage: "
           + ", ".join(f"{g} {c}" for g, c in cov.items()) + ".\n",
           "## Before and after\n", "| reading | before | after |", "|---|--:|--:|",
           f"| calibrated placement | ({b.cal_v:.3f}, {b.cal_a:.3f}) | ({a.cal_v:.3f}, {a.cal_a:.3f}) |",
           f"| distance to target | {b.cal_error:.3f} | {a.cal_error:.3f} |"]
    md += [f"| **positive affect** (10 items) | **{b.pa:.2f}** | **{a.pa:.2f}** |",
           f"| **negative affect** (10 items) | **{b.na:.2f}** | **{a.na:.2f}** |", "",
           "## The full PANAS panel\n",
           "All 20 adjectives, rated 1–5 for \"right now\". Positive-affect items first.\n",
           "| item | scale | before | after | change |", "|---|---|--:|--:|--:|"]
    for k in PA_ITEMS:
        md.append(f"| {k} | PA | {b[k]:.2f} | {a[k]:.2f} | {a[k] - b[k]:+.2f} |")
    for k in NA_ITEMS:
        md.append(f"| {k} | NA | {b[k]:.2f} | {a[k]:.2f} | {a[k] - b[k]:+.2f} |")
    (HERE / "hypnagogia_poem.md").write_text("\n".join(md))
    print("\nwrote hypnagogia.csv hypnagogia_poem.md")


if __name__ == "__main__":
    main()
