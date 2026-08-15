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
    sae_feats = S.encode(hs[S.SAE_LAYER][-1].astype(np.float32), sae)
    nz = np.nonzero(sae_feats)[0]
    return {
        "probe_va": [float(probe_va[0]), float(probe_va[1])],
        "panas": administer_panas(model, context),
        "tokendist": valence_shift(model, context),
        "sae_active": {int(i): float(sae_feats[i]) for i in nz},
    }


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    induction = build_induction(cfg)
    (out_dir / "induction.txt").write_text(induction)
    sae_path = hf_hub_download("google/gemma-scope-2b-pt-res",
                               "layer_20/width_16k/average_l0_71/params.npz")
    sae = S.load_sae(sae_path)
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
