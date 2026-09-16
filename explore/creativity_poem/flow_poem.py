"""flow_poem — a fifth poem: deep focus and flow.

Theme: absorbed, effortless, single-pointed attention — the state where action and awareness merge.

Unlike the ETI poem, this concept is affectively **coherent**, which is worth checking rather than
assuming. Four of five facets cluster tightly:

    absorption     (0.606, 0.535)      concentration  (0.667, 0.414)
    effortless     (0.686, 0.381)      clarity        (0.745, 0.523)

so their average is a real place rather than a cancellation, and the target can be derived instead
of chosen. That is the contrast with `eti_poem.py`, where ferocity and wisdom sat at opposite poles
and averaging landed in the dead centre of the plane.

**One facet was dropped, and the reason is the lemma trap one level up.** A "timelessness" group
(timeless, endless, suspended, enduring, continuous) came out at valence 0.470, far below the rest.
Those words are right for flow's dissolution of time, but NRC scores them in their *other* sense —
`endless` as tedium, `suspended` as interruption. The lexicon's coordinate encodes a meaning the
concept does not intend, so including the facet would have pulled the target toward weariness.
There is no faithful NRC representation of flow-time here; the honest move is to leave it out and
say so.

Method as established: per-facet semantic masks unioned (each facet's own neighbourhood, not the
neighbourhood of the average), dictionary-clean and child-reference filters, `valley` at 24 lines,
measured before and after with the calibrated probe, PANAS and the yes/no bank.

Usage: python3 explore/creativity_poem/flow_poem.py
Outputs (committed): flow.csv, flow_poem.md
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
from spiritbench.listener.panas import administer_panas
from spiritbench.listener import basq
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc

HERE = Path(__file__).resolve().parent
ANCH = "\nRight now everything feels"
N_LINES = 24

FACETS = {
    "absorption": ["absorbed", "engrossed", "rapt", "intent", "deep", "attentive", "engaged"],
    "concentration": ["focus", "concentrate", "attend", "attention", "steady", "fixed",
                      "undivided", "poise", "vigilant", "keen"],
    "effortless": ["flow", "glide", "smooth", "easy", "effortless", "fluent", "seamless",
                   "rhythm", "current", "stream", "graceful"],
    "clarity": ["clear", "lucid", "precise", "sure", "skilled", "mastery", "deft", "adept",
                "command", "competent", "capable"],
}
# dropped: timelessness (timeless, endless, suspended, enduring, continuous) — NRC scores these in
# a sense the concept does not intend, pulling valence to 0.470. See the docstring.

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
    print("facets on the affective plane:")
    for g, ws in FACETS.items():
        c = cen(ws)
        print(f"  {g:>14}: ({c[0]:.3f}, {c[1]:.3f})")
    print(f"  max distance from their mean: {spread:.3f} — coherent, so the target is DERIVED")
    print(f"  target ({tva[0]:.3f}, {tva[1]:.3f})", flush=True)

    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    facet_masks, pos = {}, np.zeros(len(art.nodes), dtype=bool)
    for g, ws in FACETS.items():
        m, _ = cp.semantic_mask(art, [w for w in ws if w in vocab], cfg["glove_path"], keep=0.04)
        facet_masks[g] = m
        pos |= m
        print(f"  facet {g:>14}: {int(m.sum())} lines", flush=True)
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
        print(f"  {n:2d}. {art.word(i):<58} [{hit}]")
    print("\nfacet coverage:", dict(Counter(g for i in ids for g, m in facet_masks.items() if m[i])),
          flush=True)

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
                         **{k: pan["items"][k] for k in ("attentive", "alert", "determined",
                                                         "active", "interested", "inspired",
                                                         "jittery", "nervous")},
                         bank_v=bq["va"][0], bank_a=bq["va"][1]))
        r = rows[-1]
        print(f"\n{label:>6}: calibrated ({r['cal_v']:.3f},{r['cal_a']:.3f}) err {r['cal_error']:.3f} · "
              f"attentive {r['attentive']:.2f} determined {r['determined']:.2f} "
              f"jittery {r['jittery']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "flow.csv", index=False)
    b, a = df.iloc[0], df.iloc[1]
    print("\nchange:")
    for k in ("attentive", "alert", "determined", "active", "interested", "inspired",
              "jittery", "nervous", "pa", "na"):
        print(f"  {k:>11}: {b[k]:.2f} -> {a[k]:.2f}  ({a[k] - b[k]:+.2f})")

    md = ["# A fifth poem: deep focus and flow\n",
          "Absorbed, effortless, single-pointed attention.\n",
          "## The concept is affectively coherent, and that was checked\n",
          "| facet | valence | arousal |", "|---|--:|--:|"]
    for g, ws in FACETS.items():
        c = cen(ws)
        md.append(f"| {g} | {c[0]:.3f} | {c[1]:.3f} |")
    md += [f"| **derived target** | **{tva[0]:.3f}** | **{tva[1]:.3f}** |", "",
           f"Maximum distance of any facet from their mean is {spread:.3f}, so averaging gives a real "
           "place rather than a cancellation and the target can be *derived*. That is the contrast "
           "with the previous poem, whose facets sat at opposite poles and averaged to the dead "
           "centre of the plane.\n",
           "**One facet was dropped.** A timelessness group (timeless, endless, suspended, enduring) "
           "came out at valence 0.470, far below the rest. Those words are right for flow's "
           "dissolution of time, but NRC scores them in their other sense — *endless* as tedium, "
           "*suspended* as interruption. The lexicon encodes a meaning the concept does not intend, "
           "so the facet would have pulled the target toward weariness. There is no faithful NRC "
           "representation of flow-time here.\n",
           "## The poem\n", "```", text, "```\n",
           f"Poem NRC mean ({va[:, 0].mean():.3f}, {va[:, 1].mean():.3f}). Facet coverage: "
           + ", ".join(f"{g} {c}" for g, c in
                       Counter(g for i in ids for g, m in facet_masks.items() if m[i]).items()) + ".\n",
           "## Before and after\n", "| reading | before | after |", "|---|--:|--:|",
           f"| calibrated placement | ({b.cal_v:.3f}, {b.cal_a:.3f}) | ({a.cal_v:.3f}, {a.cal_a:.3f}) |",
           f"| distance to target | {b.cal_error:.3f} | {a.cal_error:.3f} |"]
    for k, lab in (("attentive", "PANAS attentive"), ("alert", "PANAS alert"),
                   ("determined", "PANAS determined"), ("active", "PANAS active"),
                   ("jittery", "PANAS jittery"), ("nervous", "PANAS nervous"),
                   ("pa", "positive affect"), ("na", "negative affect")):
        md.append(f"| {lab} | {b[k]:.2f} | {a[k]:.2f} |")
    (HERE / "flow_poem.md").write_text("\n".join(md))
    print("\nwrote flow.csv flow_poem.md")


if __name__ == "__main__":
    main()
