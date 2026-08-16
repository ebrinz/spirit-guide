"""E17 — Polygon shape study: does angular step size behave as the smoothness
theory predicts?

Shapes (fixed radius, same linear VA track, only vertex count / traversal
order vary):
  octagon   n=8 skip=1  (45° steps — smoothest)
  pentagon  n=5 skip=1  (72° — the original polygon-pca)
  triangle  n=3 skip=1  (120°)
  pentagram n=5 skip=2  (144° — same vertices as pentagon, star order)
  octagram  n=8 skip=3  (135° — same vertices as octagon, star order)

Pre-registered prediction (from §7/§13): displacement ranks inversely with
angular step size / path Dirichlet energy; star polygons underperform their
perimeter twins at identical vertex sets.

Outputs: data/polygon_shapes/*.json, results/polygon_shapes[_tag].csv.
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

SHAPES = {
    "octagon": (8, 1, 45.0),
    "pentagon": (5, 1, 72.0),
    "triangle": (3, 1, 120.0),
    "octagram": (8, 3, 135.0),
    "pentagram": (5, 2, 144.0),
}
SEEDS = [42, 43]
TARGETS = ["calm", "excited"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = REPO_ROOT / "data/polygon_shapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    phrase_path = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(phrase_path)
    neutral = tuple(cfg["neutral_start"])
    n = ad.LENGTH_LINES["medium"]
    with open(phrase_path) as f:
        raw = json.load(f)
    L = H.build_laplacian(raw["traversal_graph"]["edges"], len(raw["words"]))
    evals, evecs = H.eigenmodes(L, cfg["harmonics"]["n_modes"])
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    with open(cfg["questionnaire_bank"]) as f:
        bank = json.load(f)
    rows = []
    for shape, (nv, skip, angle) in SHAPES.items():
        for tname in TARGETS:
            tva = tuple(cfg["targets"][tname])
            for seed in SEEDS:
                ids = ad.polygon_shape(art, neutral, tva, n, seed,
                                       n_vertices=nv, skip=skip)
                stim = ad.stimulus_record(
                    art, ids, f"poly-{shape}", "psg", tname, tva,
                    {"length": "medium", "intensity": "plain",
                     "style": "unfiltered", "seed": seed, "shape": shape,
                     "angle_deg": angle})
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
                rows.append({"shape": shape, "angle_deg": angle, "target": tname,
                             "seed": seed,
                             "placement_error": placement_error(traj, tva),
                             "displacement": displacement(traj, tva),
                             "path_dirichlet": H.path_dirichlet(nodes, evals, evecs)})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    df.to_csv(REPO_ROOT / f"results/polygon_shapes{suffix}.csv", index=False)
    print("\nby shape (angular order):")
    print(df.groupby(["angle_deg", "shape"])[
        ["placement_error", "displacement", "path_dirichlet"]]
        .mean().round(4).to_string())
    from scipy.stats import spearmanr
    r1, p1 = spearmanr(df.angle_deg, df.path_dirichlet)
    r2, p2 = spearmanr(df.angle_deg, df.displacement)
    r3, p3 = spearmanr(df.path_dirichlet, df.displacement)
    print(f"\nangle vs dirichlet: rho={r1:.2f} p={p1:.3f}")
    print(f"angle vs displacement: rho={r2:.2f} p={p2:.3f}")
    print(f"dirichlet vs displacement: rho={r3:.2f} p={p3:.3f}")


if __name__ == "__main__":
    main()
