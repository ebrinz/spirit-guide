"""eti_poem — a fourth poem: contact, occulted wisdom, ferocity.

Theme: a mystical resource of non-human intelligence that presses toward revelation and stays
hidden — wise and ferocious at once.

**This one cannot be built the usual way, and the reason is the folder's central finding.** The
concept is bivalent. Its parts sit at opposite ends of the valence–arousal plane:

    ferocity            (0.275, 0.845)     wisdom              (0.762, 0.418)
    occulted/withheld   (0.412, 0.457)     begs-reveal         (0.587, 0.498)
    contact/otherness   (0.566, 0.378)

Averaged, the 68-word centroid is **(0.503, 0.528)** — the dead centre of the plane, which is
roughly where the model sits before reading anything. Targeting it would be asking the model to go
nowhere, and it would cancel exactly the tension the theme is made of.

So the two axes are set separately, which is what this folder concluded they are:

  SUBJECT  — a **per-facet** semantic mask: the union of the lines nearest each of the five groups
             separately, rather than the lines nearest the average of all 68 words. Averaging in
             meaning-space turned out to make the same mistake as averaging in affect-space one
             level up: the grand centroid sat in the devotional middle of this corpus, and the first
             build came back reverent but with no ferocity and no concealment. Taking each facet's
             own neighbourhood keeps lines that strongly express ONE facet, so the poem carries the
             tension instead of its average.
  FEELING  — an explicitly chosen affective target rather than an averaged one. The lab's finding
             (`lab/EXPERIMENT_LOG.md`, 2026-08-19) is that awe reaches high arousal with *less*
             distress than dominance does, so the feeling target is awe/dread: high arousal, still
             positive valence. That keeps ferocity without buying it through distress.

**A stratified variant was tried and abandoned.** Forcing an equal quota per facet did balance the
coverage (4 occulted lines instead of 2) but cost placement badly, 0.210 -> 0.382, and filled the
ferocity slots with that neighbourhood's mildest members ("blue skies and silver clouds and gentle
winds"). The cause is structural: `valley` ascends toward a high-valence target, so from any facet
whose own valence sits low it selects the gentlest available line. You cannot have both a
high-valence placement and genuinely fierce lines out of an ascending constructor — that would need
a constructor that does not sort by proximity to a single affective goal.

Content filters as before: dictionary-clean, no child references. The sensuous exclusion is not
applied — it is irrelevant to this theme, and sensuous verse is not itself a problem.

Usage: python3 explore/creativity_poem/eti_poem.py
Outputs (committed): eti.csv, eti_poem.md
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

SUBJECT = {
    "contact": ["alien", "stranger", "herald", "messenger", "distant", "star", "celestial",
                "cosmos", "void", "sky", "heavens", "afar", "remote", "unknown"],
    "occulted": ["hidden", "concealed", "veiled", "secret", "cipher", "riddle", "enigma", "occult",
                 "arcane", "obscure", "shadow", "mask", "silence", "unseen", "mystery"],
    "begs_reveal": ["reveal", "unveil", "disclose", "summon", "beckon", "call", "signal", "sign",
                    "omen", "awaken", "emerge", "dawn"],
    "wisdom": ["wisdom", "knowledge", "insight", "understanding", "sage", "oracle", "prophet",
               "lore", "learning", "truth", "counsel"],
    "ferocity": ["fierce", "ferocious", "terrible", "dread", "fearsome", "mighty", "wrath",
                 "blaze", "burning", "savage", "storm", "thunder", "fury", "roar", "devour"],
}
# the FEELING target, chosen not averaged: awe/dread — high arousal, still positive valence
FEELING = ["awe", "wonder", "sublime", "vast", "solemn", "reverence", "majesty", "tremendous",
           "immense", "grandeur", "radiant", "exalted"]

_s1 = importlib.util.spec_from_file_location("cp", HERE / "run.py")
cp = importlib.util.module_from_spec(_s1); _s1.loader.exec_module(cp)
_s2 = importlib.util.spec_from_file_location("rb", HERE / "rebuild_main.py")
rb = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(rb)


def main():
    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    cen = lambda ws: (float(np.mean([nrc[w][0] for w in ws if w in nrc])),
                      float(np.mean([nrc[w][1] for w in ws if w in nrc])))  # noqa: E731
    subj = [w for ws in SUBJECT.values() for w in ws]
    print("subject groups, on the affective plane:")
    for g, ws in SUBJECT.items():
        print(f"  {g:>12}: {cen(ws)}")
    print(f"  {'AVERAGED':>12}: {cen(subj)}  <- the middle; targeting this asks for no movement")
    tva = cen([w for w in FEELING if w in nrc])
    print(f"\nFEELING target (chosen, not averaged): {tva}", flush=True)

    ok, vocab = cp.clean_mask(art, cfg["glove_path"])
    # union of per-facet neighbourhoods, not the neighbourhood of the average
    facet_masks, pos = {}, np.zeros(len(art.nodes), dtype=bool)
    for g, ws in SUBJECT.items():
        m, _ = cp.semantic_mask(art, [w for w in ws if w in vocab], cfg["glove_path"], keep=0.03)
        facet_masks[g] = m
        pos |= m
        print(f"  facet {g:>12}: {int(m.sum())} lines", flush=True)
    no_minor = np.array([not (rb.MINOR & set(rb.WORD.findall(art.word(i).lower())))
                         for i in range(len(art.nodes))])
    mask = ok & pos & no_minor
    print(f"eligible lines: {int(mask.sum())} of {len(mask)}", flush=True)

    ids = ad.valley_shape(art, tva, N_LINES, 42, mask)
    lines = [art.word(i) for i in ids]
    text = ".\n".join(lines)
    va = np.array([art.va(i) for i in ids])
    facet_of = {}
    for i in ids:
        hit = [g for g, m in facet_masks.items() if m[i]]
        facet_of[i] = "+".join(hit) if hit else "-"
    print(f"\npoem NRC mean ({va[:, 0].mean():.3f}, {va[:, 1].mean():.3f})\n")
    for n, i in enumerate(ids, 1):
        print(f"  {n:2d}. {art.word(i):<58} [{facet_of[i]}]")
    from collections import Counter
    cnt = Counter(g for i in ids for g, m in facet_masks.items() if m[i])
    print("\nfacet coverage:", dict(cnt), flush=True)

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
                                                         "interested", "afraid", "nervous",
                                                         "strong", "active")},
                         bank_v=bq["va"][0], bank_a=bq["va"][1]))
        r = rows[-1]
        print(f"\n{label:>6}: calibrated ({r['cal_v']:.3f},{r['cal_a']:.3f}) err {r['cal_error']:.3f} · "
              f"PA {r['pa']:.2f} NA {r['na']:.2f} · alert {r['alert']:.2f} strong {r['strong']:.2f} "
              f"afraid {r['afraid']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "eti.csv", index=False)
    b, a = df.iloc[0], df.iloc[1]
    print("\nchange:")
    for k in ("alert", "active", "strong", "inspired", "attentive", "interested",
              "afraid", "nervous", "pa", "na"):
        print(f"  {k:>11}: {b[k]:.2f} -> {a[k]:.2f}  ({a[k] - b[k]:+.2f})")

    md = ["# A fourth poem: contact, occulted wisdom, ferocity\n",
          "A mystical resource of non-human intelligence that presses toward revelation and stays "
          "hidden — wise and ferocious at once.\n",
          "## Why this one needed the two axes set separately\n",
          "The concept is bivalent, and its parts sit at opposite ends of the affective plane:\n",
          "| group | valence | arousal |", "|---|--:|--:|"]
    for g, ws in SUBJECT.items():
        c = cen(ws)
        md.append(f"| {g.replace('_', ' ')} | {c[0]:.3f} | {c[1]:.3f} |")
    c = cen(subj)
    md += [f"| **all 68 averaged** | **{c[0]:.3f}** | **{c[1]:.3f}** |", "",
           "Ferocity and wisdom are near-opposites, so the average lands in the dead centre of the "
           "plane — approximately where the model already sits before reading anything. Targeting it "
           "would ask for no movement and would cancel the tension the theme is made of.\n",
           "So the subject is carried by a **semantic mask** over all 68 words (meaning-space does "
           "not cancel the way the affective plane does), while the **feeling** is an explicitly "
           f"chosen target — awe/dread at ({tva[0]:.3f}, {tva[1]:.3f}), high arousal but still "
           "positive valence. The lab's own result is that awe reaches high arousal with less "
           "distress than the alternatives, so ferocity is bought without buying distress.\n",
           "## The poem\n", "```", text, "```\n",
           f"Poem NRC mean ({va[:, 0].mean():.3f}, {va[:, 1].mean():.3f}). Eligible pool "
           f"{int(mask.sum())} lines after the dictionary, semantic and child-reference filters.\n",
           "## Before and after\n",
           "| reading | before | after |", "|---|--:|--:|",
           f"| calibrated placement | ({b.cal_v:.3f}, {b.cal_a:.3f}) | ({a.cal_v:.3f}, {a.cal_a:.3f}) |",
           f"| distance to target | {b.cal_error:.3f} | {a.cal_error:.3f} |"]
    for k, lab in (("alert", "PANAS alert"), ("active", "PANAS active"), ("strong", "PANAS strong"),
                   ("inspired", "PANAS inspired"), ("afraid", "PANAS afraid"),
                   ("nervous", "PANAS nervous"), ("pa", "positive affect"), ("na", "negative affect")):
        md.append(f"| {lab} | {b[k]:.2f} | {a[k]:.2f} |")
    (HERE / "eti_poem.md").write_text("\n".join(md))
    print("\nwrote eti.csv eti_poem.md")


if __name__ == "__main__":
    main()
