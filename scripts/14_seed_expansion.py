"""C2/C7 seed expansion.

Part A (C2): ordered + shuffled pairs for the 6 core psg constructors at
calm, seeds 43/44 — phase-1 trajectory protocol -> per-pair order effect.
Part B (C7): the same new-seed calm stimuli under the phase-2b protocol
(post-induction alleviation vs the existing shared checkpoints).

Outputs: data/seed_expansion/*.json, results/seed_expansion.csv.
"""
import importlib.util
import json
import os
import random

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.analysis.metrics import placement_error
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.controls import shuffled as shuffle_ctrl

spec = importlib.util.spec_from_file_location(
    "runner", REPO_ROOT / "scripts/05_run_listener.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

CONSTRUCTORS = ["graph-walk", "valley", "harmonic-golden", "harmonic-prime",
                "harmonic-organic", "polygon-pca"]
SEEDS = [43, 44]


def build_stim(art, phrase_path, cons, seed, cfg):
    n = ad.LENGTH_LINES["medium"]
    tva = tuple(cfg["targets"]["calm"])
    neutral = tuple(cfg["neutral_start"])
    if cons == "graph-walk":
        ids = ad.graph_walk(art, neutral, tva, n, seed, cfg["ot_repo"])
    elif cons == "valley":
        ids = ad.valley_shape(art, tva, n, seed)
    elif cons.startswith("harmonic-"):
        ids = ad.harmonic(art, phrase_path, neutral, tva, n,
                          cons.split("-")[1], seed, cfg["ot_repo"],
                          cfg["semantic_axes"])
    else:
        ids = ad.polygon_pca(art, neutral, tva, n, seed)
    return ad.stimulus_record(art, ids, cons, "psg", "calm", tva,
                              {"length": "medium", "intensity": "plain",
                               "style": "unfiltered", "seed": seed})


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/seed_expansion"
    out_dir.mkdir(parents=True, exist_ok=True)
    phrase_path = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(phrase_path)
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    with open(cfg["questionnaire_bank"]) as f:
        bank = json.load(f)
    with open(REPO_ROOT / "data/phase2b/shared_checkpoints.json") as f:
        shared = json.load(f)
    pre_va = np.array(shared["pre"]["probe_va"])
    ind_va = np.array(shared["induced"]["probe_va"])
    induction = (REPO_ROOT / "data/phase2b/induction.txt").read_text()
    induced_ctx = cfg["preamble"] + induction + "\n\n"
    rows = []
    for cons in CONSTRUCTORS:
        for seed in SEEDS:
            stim = build_stim(art, phrase_path, cons, seed, cfg)
            for variant, st in (("ordered", stim),
                                ("shuffled", shuffle_ctrl(stim, seed))):
                out = out_dir / f"{cons}_s{seed}_{variant}.json"
                if out.exists():
                    with open(out) as f:
                        rec = json.load(f)
                else:
                    rec = runner.run_stimulus(model, probe, st, cfg["preamble"],
                                              cfg["ema_alpha"], bank, cfg["basq"])
                    # C7: alleviation read after the shared prose induction
                    hs = model.hidden_states(induced_ctx + st["text"]
                                             + "\n\nRight now everything feels")
                    post_va = probe.predict(hs[probe.layer][-1:])[0]
                    rec["rescue_post_va"] = [float(post_va[0]), float(post_va[1])]
                    tmp = out.with_suffix(".json.tmp")
                    with open(tmp, "w") as f:
                        json.dump(rec, f)
                    os.replace(tmp, out)
                traj = np.asarray(rec["traj"])
                allev = float(np.linalg.norm(ind_va - pre_va)
                              - np.linalg.norm(np.array(rec["rescue_post_va"]) - pre_va))
                rows.append({"constructor": cons, "seed": seed, "variant": variant,
                             "placement_error": placement_error(traj, stim["target_va"]),
                             "rescue_alleviation": allev})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/seed_expansion.csv", index=False)
    piv = df.pivot_table(index="constructor", columns="variant",
                         values="placement_error")
    piv["order_effect"] = piv["shuffled"] - piv["ordered"]
    print("\nC2 order effect (shuffled - ordered placement, +ve = order helps):")
    print(piv.round(3).to_string())
    print("\nC7 rescue alleviation by constructor (ordered only):")
    print(df[df.variant == "ordered"].groupby("constructor")
          .rescue_alleviation.mean().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
