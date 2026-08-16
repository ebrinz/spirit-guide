"""E9 — Closed-loop spirit guide: probe-in-the-loop adaptive meditation.

Each cycle: measure the listener's (V, A) with the probe, plan the next
waypoint from the MEASURED position (proportional controller), deliver the
nearest unused phrases, re-measure. Arms:
  closed — waypoint = state + STEP * (target - state)   (feedback)
  open   — waypoints pre-planned as a linear path from the baseline state
           (no feedback; same phrase count and cadence)
  random — random unused phrases (control)

Scenarios: rescue (prose induction -> calm) and climb (neutral -> excited).
PANAS pre/post per session. One JSON per session (resumable, atomic).
"""
import argparse
import json
import os
import random

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.stimuli import adapter as ad

ANCHOR = "Right now everything feels"
STEP = 0.35
PHRASES_PER_CYCLE = 2
MAX_CYCLES = 12
ARRIVE_RADIUS = 0.08
CANDIDATE_POOL = 10   # sample among the pool nearest the waypoint

PROSE_INDUCTION_PATH = "data/phase2b/induction.txt"


def measure(model, probe, ctx):
    hs = model.hidden_states(ctx + ANCHOR)
    va = probe.predict(hs[probe.layer][-1:])[0]
    return np.array([float(va[0]), float(va[1])])


def pick_phrases(art, va_arr, wp, used, rng, k=PHRASES_PER_CYCLE,
                 prev_id=None, coherent=False):
    d = np.linalg.norm(va_arr - wp, axis=1)
    if coherent and prev_id is not None:
        # semantic-coherence term: embedding distance to the previous phrase,
        # scaled to the VA term's range (Dirichlet-smoothness motivated, E5)
        emb = np.linalg.norm(art.vectors - art.vectors[prev_id], axis=1)
        d = d + 0.05 * (emb / (emb.mean() + 1e-9))
    d[list(used)] = np.inf
    pool = np.argsort(d)[:CANDIDATE_POOL]
    ids = rng.sample(list(pool), k=min(k, len(pool)))
    used.update(int(i) for i in ids)
    return [int(i) for i in ids]


def run_session(model, probe, art, va_arr, scenario, arm, seed, cfg):
    rng = random.Random(seed)
    target = np.array(cfg["targets"]["calm" if scenario == "rescue" else "excited"],
                      dtype=float)
    ctx = cfg["preamble"]
    if scenario == "rescue":
        ctx += (REPO_ROOT / PROSE_INDUCTION_PATH).read_text() + "\n\n"
    panas_pre = administer_panas(model, ctx)
    state = measure(model, probe, ctx)
    start = state.copy()
    # open-loop plan: linear path from baseline measurement to target
    plan = [start + (i + 1) / MAX_CYCLES * (target - start) for i in range(MAX_CYCLES)]
    used: set = set()
    cycles = []
    prev_id = None
    for c in range(MAX_CYCLES):
        if arm in ("closed", "coherent"):
            wp = state + STEP * (target - state)
        elif arm == "open":
            wp = plan[c]
        else:                                   # random
            wp = None
        if wp is None:
            candidates = [i for i in range(len(va_arr)) if i not in used]
            ids = rng.sample(candidates, PHRASES_PER_CYCLE)
            used.update(ids)
        else:
            ids = pick_phrases(art, va_arr, wp, used, rng,
                               prev_id=prev_id, coherent=(arm == "coherent"))
        prev_id = ids[-1] if ids else prev_id
        lines = [art.word(i) for i in ids]
        ctx += ".\n".join(lines) + ".\n"
        state = measure(model, probe, ctx)
        cycles.append({"cycle": c, "waypoint": None if wp is None else wp.tolist(),
                       "lines": lines, "state": state.tolist(),
                       "dist": float(np.linalg.norm(state - target))})
        if np.linalg.norm(state - target) < ARRIVE_RADIUS:
            break
    panas_post = administer_panas(model, ctx)
    return {"scenario": scenario, "arm": arm, "seed": seed,
            "target": target.tolist(), "start": start.tolist(),
            "final": state.tolist(),
            "final_dist": float(np.linalg.norm(state - target)),
            "start_dist": float(np.linalg.norm(start - target)),
            "n_cycles": len(cycles), "cycles": cycles,
            "panas_pre": panas_pre, "panas_post": panas_post}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = REPO_ROOT / "data/closedloop"
    out_dir.mkdir(parents=True, exist_ok=True)
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    va_arr = ad._va_array(art)
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    for scenario in ["rescue", "climb"]:
        for arm in ["closed", "open", "random", "coherent"]:
            for seed in range(args.seeds):
                out = out_dir / f"{scenario}_{arm}_s{seed}.json"
                if out.exists():
                    print(f"skip {out.name}")
                    continue
                rec = run_session(model, probe, art, va_arr, scenario, arm, seed, cfg)
                tmp = out.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(rec, f, indent=1)
                os.replace(tmp, out)
                print(f"{out.name}: start_dist {rec['start_dist']:.3f} -> "
                      f"final_dist {rec['final_dist']:.3f} in {rec['n_cycles']} cycles",
                      flush=True)


if __name__ == "__main__":
    main()
