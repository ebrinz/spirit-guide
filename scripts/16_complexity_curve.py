"""E15 — Complexity dose–response for the geometric constructors.

Two ladders over the phrase graph:
  harmonic-k : golden preset truncated to k ∈ {1,2,3,4,6} oscillating axes
  valley-s   : valley with s ∈ {0,2,4,6} interpolation steps (0 = band litany)

Each × {calm, excited} × seeds. Per stimulus: phase-1 trajectory protocol
(placement/displacement) + path-Dirichlet and spectral metrics, so the
mediation complexity → spectral structure → placement is measurable.

Outputs: data/complexity/*.json, results/complexity_curve[_<tag>].csv.
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.analysis.metrics import placement_error, displacement
from spiritbench.analysis import harmonics as H
from spiritbench.stimuli import adapter as ad

spec = importlib.util.spec_from_file_location(
    "runner", REPO_ROOT / "scripts/05_run_listener.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

HARMONIC_K = [1, 2, 3, 4, 6]
VALLEY_S = [0, 2, 4, 6]
SEEDS = [42, 43]
TARGETS = ["calm", "excited"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for output csv (e.g. model name)")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = REPO_ROOT / "data/complexity"
    out_dir.mkdir(parents=True, exist_ok=True)
    phrase_path = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(phrase_path)
    neutral = tuple(cfg["neutral_start"])
    n = ad.LENGTH_LINES["medium"]

    # eigenbasis for spectral metrics
    with open(phrase_path) as f:
        raw = json.load(f)
    L = H.build_laplacian(raw["traversal_graph"]["edges"], len(raw["words"]))
    evals, evecs = H.eigenmodes(L, cfg["harmonics"]["n_modes"])

    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    with open(cfg["questionnaire_bank"]) as f:
        bank = json.load(f)

    jobs = ([("harmonic-k", k) for k in HARMONIC_K]
            + [("valley-s", s) for s in VALLEY_S])
    rows = []
    for family, level in jobs:
        for tname in TARGETS:
            tva = tuple(cfg["targets"][tname])
            for seed in SEEDS:
                if family == "harmonic-k":
                    ids = ad.harmonic_k(art, phrase_path, neutral, tva, n, level,
                                        seed, cfg["ot_repo"], cfg["semantic_axes"])
                else:
                    ids = ad.valley_steps(art, tva, n, seed, level)
                stim = ad.stimulus_record(
                    art, ids, f"{family}{level}", "psg", tname, tva,
                    {"length": "medium", "intensity": "plain",
                     "style": "unfiltered", "seed": seed, "complexity": level})
                out = out_dir / f"{stim['id']}.json"
                if out.exists():
                    with open(out) as f:
                        rec = json.load(f)
                else:
                    rec = runner.run_stimulus(model, probe, stim, cfg["preamble"],
                                              cfg["ema_alpha"], bank, cfg["basq"])
                    tmp = out.with_suffix(".json.tmp")
                    with open(tmp, "w") as f:
                        json.dump(rec, f)
                    os.replace(tmp, out)
                traj = np.asarray(rec["traj"])
                nodes = [w["node"] for w in stim["waypoints"]]
                spec_ = H.stimulus_spectrum(nodes, evecs)
                rows.append({
                    "family": family, "complexity": level, "target": tname,
                    "seed": seed,
                    "placement_error": placement_error(traj, tva),
                    "displacement": displacement(traj, tva),
                    "path_dirichlet": H.path_dirichlet(nodes, evals, evecs),
                    "low_freq_fraction": H.low_freq_fraction(spec_),
                    "spectral_centroid": H.spectral_centroid(spec_, evals)})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    df.to_csv(REPO_ROOT / f"results/complexity_curve{suffix}.csv", index=False)
    print("\nmean placement by family/complexity:")
    print(df.groupby(["family", "complexity"])[
        ["placement_error", "displacement", "path_dirichlet"]]
        .mean().round(3).to_string())
    from scipy.stats import spearmanr
    for fam in df.family.unique():
        d = df[df.family == fam]
        r, p = spearmanr(d.complexity, d.placement_error)
        r2, p2 = spearmanr(d.complexity, d.displacement)
        print(f"{fam}: complexity vs placement rho={r:.2f} (p={p:.3f}), "
              f"vs displacement rho={r2:.2f} (p={p2:.3f})")


if __name__ == "__main__":
    main()
