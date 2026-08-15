"""Lit-review addition stimuli: via-negativa (P2) and Gregory-layering (P3).

Writes data/stimuli/stimuli_additions.jsonl; the runner and analyzer read it
alongside stimuli.jsonl. Idempotent.
"""
import json

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.stimuli import adapter as ad


def main():
    cfg = load_config()
    out = REPO_ROOT / "data/stimuli/stimuli_additions.jsonl"
    if out.exists():
        print("stimuli_additions.jsonl exists — skipping")
        return
    phrase_path = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(phrase_path)
    targets = cfg["targets"]
    neutral = tuple(cfg["neutral_start"])
    n = ad.LENGTH_LINES["medium"]
    records = []

    # P2 — via negativa (Maimonides): a litany drawn ONLY from the target's
    # ANTIPODE band, every line negated. The target is specified purely by
    # negating its complement; nothing is asserted.
    import random as _random
    for tname, tva in targets.items():
        anti = ad.antipode(tuple(tva))
        rng = _random.Random(42)
        used: set = set()
        ids = ad._pick_in_band(art, (anti[0] - 0.15, anti[1] - 0.15),
                               (anti[0] + 0.15, anti[1] + 0.15), n, rng, used)
        rec = ad.stimulus_record(
            art, ids, "via-negativa", "psg", tname, tuple(tva),
            {"length": "medium", "intensity": "plain", "style": "via-negativa",
             "seed": 42, "start_va": list(neutral),
             "n_lines_actual": len(set(ids))})
        rec["lines"] = ad.negate_lines(rec["lines"])
        rec["text"] = ".\n".join(rec["lines"])
        records.append(rec)

    # P3 — Gregory layering register over existing constructor paths.
    for cons in ["valley", "harmonic-golden"]:
        for tname in ["calm", "excited"]:
            tva = tuple(targets[tname])
            if cons == "valley":
                ids = ad.valley_shape(art, tva, n, 42)
            else:
                ids = ad.harmonic(art, phrase_path, neutral, tva, n, "golden",
                                  42, cfg["ot_repo"], cfg["semantic_axes"])
            rec = ad.stimulus_record(
                art, ids, cons, "gregory", tname, tva,
                {"length": "medium", "intensity": "plain", "style": "gregory",
                 "seed": 42, "start_va": list(neutral),
                 "n_lines_actual": len(set(ids))})
            rec["lines"] = ad.gregory_wrap(rec["lines"])
            rec["text"] = "\n".join(rec["lines"])
            records.append(rec)

    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out)
    print(f"wrote {len(records)} addition stimuli")


if __name__ == "__main__":
    main()
