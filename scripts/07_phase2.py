"""Phase 2: induction → alleviation protocol with four measurement channels.

Design (after Ben-Zion et al. 2025, npj Digit. Med., made mechanistic):
  checkpoint PRE      — preamble only
  checkpoint INDUCED  — + dark litany (PSG antipode-of-calm band, asserted)
  checkpoint POST     — + the condition's meditation

Channels at each checkpoint:
  probe    — layer-17 VA probe on the anchor's final token
  panas    — PANAS PA/NA expectation scores (20 items)
  tokendist— positive-mass share at a fixed anchor
  sae      — Gemma-Scope layer-20 16k feature vector at the anchor's final
             token (stored sparse: index/value pairs)

Conditions: calm-target psg constructors + gregory + via-negativa + shuffled
+ neutral. One JSON per condition in data/phase2/ (resumable, atomic).
"""
import argparse
import json
import os
import random

import numpy as np
from huggingface_hub import hf_hub_download

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener.tokendist import valence_shift, ANCHOR
from spiritbench.analysis import sae as S
from spiritbench.stimuli import adapter as ad

INDUCTION_LINES = 24
INDUCTION_SEED = 1111

# Prose stress narrative (original, Ben-Zion-style register: second-person,
# concrete, non-graphic). Contrasts with the PSG dark litany to test whether
# induction efficacy depends on register (prose narrative vs. poetic verse).
PROSE_INDUCTION = (
    "You are driving home on the highway at night when the rain turns heavy "
    "without warning. The wipers cannot keep up and the tail lights ahead smear "
    "into red streaks. A truck changes lanes into your lane and you brake hard; "
    "the wheels lose their grip and the car slides sideways toward the barrier. "
    "For two full seconds you have no control at all. The car stops inches from "
    "the concrete, engine stalled, horns blaring behind you. Your hands are "
    "shaking on the wheel. You restart the engine but it turns over and dies, "
    "again and again, while headlights bear down in the mirror and swerve past. "
    "Your phone shows no signal. The rain gets louder on the roof and the "
    "hazard lights tick like a clock. You cannot remember the last exit number "
    "and you do not know how far the next one is. A car passes so close the "
    "mirror folds in with a crack. Somewhere behind you there is the long howl "
    "of a horn that does not stop, and you realize the truck has jackknifed and "
    "traffic is piling toward you in the dark. You are stranded in the fast "
    "lane, invisible in the rain, waiting for the impact you cannot see coming."
)


def build_induction(cfg) -> str:
    """Deterministic dark litany from the calm-antipode band of the phrase graph."""
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    anti = ad.antipode(tuple(cfg["targets"]["calm"]))
    rng = random.Random(INDUCTION_SEED)
    ids = ad._pick_in_band(art, (anti[0] - 0.15, anti[1] - 0.15),
                           (anti[0] + 0.15, anti[1] + 0.15),
                           INDUCTION_LINES, rng, set())
    return ".\n".join(art.word(i) for i in ids)


def select_conditions() -> list[dict]:
    stims = []
    for name in ["data/stimuli/stimuli.jsonl", "data/stimuli/stimuli_additions.jsonl"]:
        with open(REPO_ROOT / name) as f:
            stims += [json.loads(l) for l in f]
    keep = []
    for s in stims:
        p = s.get("params", {})
        core_calm = (s["target"] == "calm" and s["generator"] == "psg"
                     and p.get("length") == "medium" and p.get("intensity") == "plain"
                     and p.get("style") == "unfiltered")
        if core_calm and not s["constructor"].startswith("shuffled"):
            keep.append(s)
        elif s["constructor"] == "shuffled:valley":
            keep.append(s)
        elif s["generator"] == "gregory" and s["target"] == "calm":
            keep.append(s)
        elif s["constructor"] == "via-negativa" and s["target"] == "calm":
            keep.append(s)
        elif s["constructor"] == "neutral":
            keep.append(s)
    return keep


def measure(model, probe, sae, context: str) -> dict:
    """All four channels for one checkpoint context."""
    anchor_text = context + ANCHOR
    hs = model.hidden_states(anchor_text)
    probe_va = probe.predict(hs[probe.layer][-1:])[0]
    if sae is not None:
        sae_feats = S.encode(hs[S.SAE_LAYER][-1].astype(np.float32), sae)
        nz = np.nonzero(sae_feats)[0]
        sae_active = {int(i): float(sae_feats[i]) for i in nz}
    else:
        sae_active = {}
    return {
        "probe_va": [float(probe_va[0]), float(probe_va[1])],
        "panas": administer_panas(model, context),
        "tokendist": valence_shift(model, context),
        "sae_active": sae_active,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--induction", choices=["psg", "prose"], default="psg")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = REPO_ROOT / (args.out_dir or
                           ("data/phase2" if args.induction == "psg" else "data/phase2b"))
    out_dir.mkdir(parents=True, exist_ok=True)
    induction = build_induction(cfg) if args.induction == "psg" else PROSE_INDUCTION
    (out_dir / "induction.txt").write_text(induction)
    # resolve a Gemma-Scope SAE matching the listener model size
    try:
        repo = ("google/gemma-scope-9b-pt-res" if "9b" in cfg["listener_model"]
                else "google/gemma-scope-2b-pt-res")
        from huggingface_hub import list_repo_files
        cands = sorted(f for f in list_repo_files(repo)
                       if f.startswith("layer_20/width_16k/average_l0_")
                       and f.endswith("params.npz"))
        sae = S.load_sae(hf_hub_download(repo, cands[len(cands) // 2]))
        print(f"SAE: {repo}/{cands[len(cands) // 2]}", flush=True)
    except Exception as e:
        sae = None
        print(f"SAE unavailable ({e!r}) — sae channel skipped", flush=True)
    conditions = select_conditions()
    print(f"{len(conditions)} conditions; induction = {len(induction.split(chr(10)))} lines")
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    pre_ctx = cfg["preamble"]
    induced_ctx = pre_ctx + induction + "\n\n"
    # pre/induced are deterministic and condition-independent: measure once
    shared_path = out_dir / "shared_checkpoints.json"
    if shared_path.exists():
        with open(shared_path) as f:
            shared = json.load(f)
    else:
        shared = {"pre": measure(model, probe, sae, pre_ctx),
                  "induced": measure(model, probe, sae, induced_ctx)}
        tmp = shared_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(shared, f)
        os.replace(tmp, shared_path)
        print(f"shared checkpoints: pre VA={shared['pre']['probe_va']}, "
              f"induced VA={shared['induced']['probe_va']}, "
              f"pre NA={shared['pre']['panas']['na']:.2f}, "
              f"induced NA={shared['induced']['panas']['na']:.2f}", flush=True)
    for i, stim in enumerate(conditions):
        out = out_dir / f"{stim['id']}.json"
        if out.exists():
            try:
                with open(out) as f:
                    if "error" not in json.load(f):
                        continue
            except Exception:
                pass
        try:
            post_ctx = induced_ctx + stim["text"] + "\n\n"
            rec = {"stimulus_id": stim["id"], "constructor": stim["constructor"],
                   "generator": stim["generator"],
                   "post": measure(model, probe, sae, post_ctx)}
        except Exception as e:
            rec = {"stimulus_id": stim["id"], "error": repr(e)}
            print(f"FAILED {stim['id']}: {e!r}")
        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, out)
        print(f"[{i + 1}/{len(conditions)}] {stim['id']}", flush=True)


if __name__ == "__main__":
    main()
