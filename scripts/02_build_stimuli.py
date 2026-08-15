"""Build all bench stimuli (core grid + sweeps + controls) into data/stimuli/stimuli.jsonl."""
import json
import os
from pathlib import Path

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.controls import shuffled, neutral_stimulus

CONSTRUCTORS = ["graph-walk", "valley", "harmonic-golden", "harmonic-prime",
                "harmonic-organic", "polygon-pca"]


def build_one(art, artifact_path, constructor, target_name, target_va, start_va,
              length, intensity, style, seed, cfg):
    n = ad.LENGTH_LINES[length]
    mask = ad.node_mask(art, "heightened") if intensity == "heightened" \
        else ad.node_mask(art, None)
    mask_active = intensity == "heightened"
    if style not in (None, "unfiltered"):
        mask &= ad.style_mask(art, style, cfg["semantic_axes"])
        mask_active = True
    if constructor == "graph-walk":
        ids = ad.graph_walk(art, start_va, target_va, n, seed, cfg["ot_repo"])
    elif constructor == "valley":
        ids = ad.valley_shape(art, target_va, n, seed, mask)
    elif constructor.startswith("harmonic-"):
        ids = ad.harmonic(art, artifact_path, start_va, target_va, n,
                          constructor.split("-")[1], seed, cfg["ot_repo"],
                          cfg["semantic_axes"])
    elif constructor == "polygon-pca":
        ids = ad.polygon_pca(art, start_va, target_va, n, seed)
    else:
        raise ValueError(constructor)
    if mask_active:
        ids = ad.apply_mask_to_path(art, ids, mask)
    return ids


def main():
    cfg = load_config()
    out = REPO_ROOT / "data/stimuli/stimuli.jsonl"
    if out.exists():
        print("stimuli.jsonl exists — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    word_art = ad.load_art(cfg["word_artifact"])
    phrase_path = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    phrase_art = ad.load_art(phrase_path)
    arts = {"psg": (phrase_art, phrase_path),
            "word-template": (word_art, cfg["word_artifact"])}
    targets = dict(cfg["targets"])
    runs = []
    # core: every constructor x generator x target, medium/plain/unfiltered
    for cons in CONSTRUCTORS:
        for gen in ["psg", "word-template"]:
            for tname, tva in list(targets.items()) + [("rescue", targets["calm"])]:
                if cons == "valley" and tname == "rescue":
                    continue  # valley ignores start_va: output is byte-identical to calm
                start = cfg["rescue_start"] if tname == "rescue" else cfg["neutral_start"]
                runs.append((cons, gen, tname, tuple(tva), tuple(start),
                             "medium", "plain", None, 42))
    # sweeps on valley + harmonic-golden, psg, calm + rescue
    for cons in ["valley", "harmonic-golden"]:
        for tname in ["calm", "rescue"]:
            if cons == "valley" and tname == "rescue":
                continue  # valley ignores start_va: output is byte-identical to calm
            tva = tuple(targets["calm"])
            start = cfg["rescue_start"] if tname == "rescue" else cfg["neutral_start"]
            for length in ["short", "long"]:
                runs.append((cons, "psg", tname, tva, tuple(start), length, "plain", None, 42))
            runs.append((cons, "psg", tname, tva, tuple(start), "medium", "heightened", None, 42))
            for style in ["imagist", "abstract"]:
                runs.append((cons, "psg", tname, tva, tuple(start), "medium", "plain", style, 42))
    records, failures = [], []
    for cons, gen, tname, tva, start, length, intensity, style, seed in runs:
        art, apath = arts[gen]
        try:
            ids = build_one(art, apath, cons, tname, tva, start, length,
                            intensity, style, seed, cfg)
        except Exception as e:
            failures.append((cons, gen, tname, length, intensity, style, repr(e)))
            continue
        rec = ad.stimulus_record(art, ids, cons, gen, tname, tva,
                                 {"length": length, "intensity": intensity,
                                  "style": style or "unfiltered", "seed": seed,
                                  "start_va": list(start),
                                  "n_lines_actual": len(set(ids))})
        if gen == "word-template":
            rec["lines"] = ad.template_wrap(rec["lines"], length, seed, cfg["ot_repo"])
            rec["text"] = "\n".join(rec["lines"])
        records.append(rec)
    # controls: shuffled for each core psg calm stimulus, one neutral
    for rec in [r for r in records if r["generator"] == "psg" and r["target"] == "calm"
                and r["params"]["length"] == "medium"
                and r["params"]["intensity"] == "plain"
                and r["params"]["style"] == "unfiltered"]:
        records.append(shuffled(rec, seed=99))
    records.append(neutral_stimulus("calm", targets["calm"]))
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    print(f"wrote {len(records)} stimuli; {len(failures)} failures")
    for fail in failures:
        print("FAILED:", fail)  # no silent caps


if __name__ == "__main__":
    main()
