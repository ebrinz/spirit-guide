"""exp_dominance_corpus_slice — is the dominance-distress a property of the AXIS
or of THIS CORPUS?

The capstone found dominance content reaches high arousal with high distress
(27.7 vs awe 7.4). But the high-dominance pool is high-NRC-dominance *poetry
lines*, and poetry expresses dominance largely through war / conquest / force
imagery — which is distressing content regardless of the abstract "feeling in
control" that NRC dominance is supposed to measure. So the distress may be a
CORPUS confound (violent content), not an AXIS property (dominance per se).

This 🟢 slice is lexical only (saved node data, no model). It asks:
  1. Is the high-D pool enriched in violence/threat vocabulary vs low-D and awe?
  2. Across in-band nodes, does NRC dominance CORRELATE with violence-word count?
  3. Can we carve a clean "high-D but NON-violent" sub-pool that is still genuinely
     high-dominance (mastery/command/strength, not war)? How many lines, and do
     they still read as dominant?

If high-D is violence-saturated AND a clean non-violent high-D sub-pool exists,
the corpus-vs-axis question is worth a forward-pass follow-up (compare the SAE
distress of violent vs non-violent high-D content at matched V/A/D). This script
decides whether that slice is clean enough to be worth running.

Usage: python3 lab/exp_dominance_corpus_slice.py
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# tight martial/physical-violence set; deliberately EXCLUDES nature/awe-ambiguous
# words (wild, burning, fire, flame, storm, blaze) that overlap AWE_WORDS
VIOLENCE = {"war", "battle", "sword", "blood", "bloody", "kill", "killed", "death",
            "die", "died", "dead", "slay", "slain", "fight", "foe", "enemy",
            "conquer", "crush", "destroy", "ruin", "wrath", "strike", "smite",
            "wound", "spear", "gun", "cannon", "chain", "chains", "slave",
            "tyrant", "savage", "prey", "terror", "agony", "torment", "cruel",
            "warrior", "warriors", "soldier", "army", "arms", "steel", "weapon"}
POWER = {"master", "command", "throne", "crown", "king", "queen", "rule", "reign",
         "lord", "might", "mighty", "power", "strength", "strong", "noble",
         "proud", "triumph", "victory", "glory", "will", "control", "steady",
         "firm", "sure", "great", "high", "rise", "stand", "conquering"}
AWE_WORDS = {"awe", "ecstasy", "rapture", "wonder", "sublime", "exalted",
             "radiant", "thrill", "glory", "transcendent", "blaze", "soaring",
             "majesty", "splendor", "exhilaration", "wild", "vast", "burning"}


def main():
    from spiritbench.config import load_config
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    lex = {p[0]: (float(p[1]), float(p[2]), float(p[3]))
           for p in (l.rstrip("\n").split("\t") for l in open(REPO / cfg["nrc_lexicon"]))
           if len(p) == 4}
    n = len(art.nodes)
    d_node = np.full(n, np.nan)
    words_of = [None] * n
    for i in range(n):
        toks = re.findall(r"[a-z']+", art.word(i).lower())
        words_of[i] = toks
        vals = [lex[t][2] for t in toks if t in lex]
        if vals:
            d_node[i] = np.mean(vals)
    va = ad._va_array(art)
    val = va[:, 0]

    def vcount(i):
        return sum(1 for w in words_of[i] if w in VIOLENCE)

    def pcount(i):
        return sum(1 for w in words_of[i] if w in POWER)

    band = np.where((val >= 0.45) & (val <= 0.65) & ~np.isnan(d_node))[0]
    bd = d_node[band]
    highD = [int(band[i]) for i in np.argsort(bd)[-300:]]
    lowD = [int(band[i]) for i in np.argsort(bd)[:300]]
    awe = [i for i in range(n) if AWE_WORDS & set(art.word(i).split())]

    def summarize(name, pool):
        vc = np.array([vcount(i) for i in pool])
        pc = np.array([pcount(i) for i in pool])
        vrate = np.mean(vc > 0)
        print(f"  {name:16s} n={len(pool):4d}  meanD={np.nanmean(d_node[pool]):.3f}  "
              f"violence: {vrate*100:4.0f}% of lines, {vc.mean():.2f}/line   "
              f"power: {np.mean(pc>0)*100:4.0f}%, {pc.mean():.2f}/line")

    print("Pool composition (valence-matched band [0.45,0.65]):")
    summarize("high-D", highD)
    summarize("low-D", lowD)
    summarize("awe", awe)

    # correlation across the whole band: does dominance track violence content?
    vc_band = np.array([vcount(int(i)) for i in band])
    r = np.corrcoef(bd, vc_band)[0, 1]
    print(f"\nacross in-band nodes: corr(dominance, violence-word count) = {r:+.3f}")

    # OBJECTIVE: words that most distinguish high-D from low-D content (let the
    # corpus say what "high dominance" means here, no hand-picked list)
    def wordbag(pool):
        c = Counter()
        for i in pool:
            c.update(set(words_of[i]))          # per-line presence
        return c
    STOP = set("the a an and or of to in on at is was be for with as it his her "
               "he she they we you i my me him them their our that this by from "
               "not no so but all are were will would can may".split())
    hc, lc = wordbag(highD), wordbag(lowD)
    scores = []
    for w in set(hc) | set(lc):
        if w in STOP or len(w) < 3:
            continue
        hf, lf = (hc[w] + 1) / (len(highD) + 2), (lc[w] + 1) / (len(lowD) + 2)
        if hc[w] >= 4:
            scores.append((np.log(hf / lf), w, hc[w]))
    scores.sort(reverse=True)
    print("\nwords most distinctive of HIGH-D vs low-D (log-odds, count):")
    print("   " + ", ".join(f"{w}({c})" for _, w, c in scores[:22]))

    print("\nsample HIGH-D lines (top dominance in band):")
    for i in highD[-8:]:
        tag = "  [viol]" if vcount(i) else ""
        print(f"    D{d_node[i]:.2f}{tag}  {art.word(i)}")

    # carve a clean high-D, NON-violent sub-pool
    hd_clean = [i for i in highD if vcount(i) == 0]
    hd_viol = [i for i in highD if vcount(i) > 0]
    print(f"\nhigh-D split: {len(hd_viol)} violent, {len(hd_clean)} non-violent")
    print(f"  mean dominance:  violent {np.nanmean(d_node[hd_viol]):.3f}  "
          f"non-violent {np.nanmean(d_node[hd_clean]):.3f}  "
          f"(if close, the non-violent slice is genuinely high-D, not just weaker)")
    print("\nsample NON-VIOLENT high-D lines (the clean slice, if any):")
    for i in sorted(hd_clean, key=lambda k: -d_node[k])[:10]:
        print(f"    D{d_node[i]:.2f}  {art.word(i)}")

    print("\n=== READ ===")
    print("  In THIS corpus, high dominance = MARTIAL/HIERARCHICAL power (battle, king,")
    print("  command, war, warrior, supreme, conquer). Explicit gore is a minority (~20%,")
    print("  weak corr r~0.16), but even the non-violent high-D slice is conquest/command/")
    print("  rank themed — there is essentially NO 'serene dominance / calm mastery'.")
    print("  So axis vs corpus is NOT separable within this corpus: dominance is available")
    print("  only as power-struggle content, which is intrinsically tense. That is likely")
    print("  WHY dominance routes to distress (capstone). A 🟡 forward-pass test of gore vs")
    print("  non-gore high-D would isolate only the gore sub-component; a true axis test")
    print("  needs out-of-corpus calm-mastery content (breaks the phrase-graph substrate).")


if __name__ == "__main__":
    main()
