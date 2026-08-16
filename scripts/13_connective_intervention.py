"""P3 / E12 — Causal test of the `and`-density covariate (C4).

Take core psg stimuli VERBATIM and create two minimal variants per stimulus:
  plain — original lines (as swept in phase 1)
  anded — every second line prefixed with "and " (raises and-initial density
          toward Wårvik's Bible-register level; nothing else changes)

Run both through the phase-1 trajectory protocol and compare placement.
If the §3.4 correlation is causal, anded > plain; if constructor-confounded
(Walkden's genre warning, Gregory result), no difference.
Outputs: data/connective/*.json, results/connective_intervention.csv.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.analysis.metrics import placement_error, displacement

spec = importlib.util.spec_from_file_location(
    "runner", REPO_ROOT / "scripts/05_run_listener.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def anded_lines(lines):
    return [("and " + l if i % 2 == 1 else l) for i, l in enumerate(lines)]


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/connective"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(REPO_ROOT / "data/stimuli/stimuli.jsonl") as f:
        stims = [json.loads(l) for l in f]
    core = [s for s in stims if s["generator"] == "psg"
            and not s["constructor"].startswith("shuffled")
            and s["target"] in ("calm", "excited")
            and s["params"].get("length") == "medium"
            and s["params"].get("intensity") == "plain"
            and s["params"].get("style") == "unfiltered"]
    print(f"{len(core)} base stimuli x 2 variants")
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    with open(cfg["questionnaire_bank"]) as f:
        bank = json.load(f)
    rows = []
    for s in core:
        for variant in ("plain", "anded"):
            out = out_dir / f"{s['id']}_{variant}.json"
            if out.exists():
                with open(out) as f:
                    rec = json.load(f)
            else:
                stim = dict(s)
                if variant == "anded":
                    stim["lines"] = anded_lines(s["lines"])
                    stim["text"] = ".\n".join(stim["lines"])
                rec = runner.run_stimulus(model, probe, stim, cfg["preamble"],
                                          cfg["ema_alpha"], bank, cfg["basq"])
                tmp = out.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(rec, f)
                os.replace(tmp, out)
            traj = np.asarray(rec["traj"])
            rows.append({"id": s["id"], "constructor": s["constructor"],
                         "target": s["target"], "variant": variant,
                         "placement_error": placement_error(traj, s["target_va"]),
                         "displacement": displacement(traj, s["target_va"])})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/connective_intervention.csv", index=False)
    piv = df.pivot_table(index=["constructor", "target"], columns="variant",
                         values="placement_error")
    piv["delta_anded_minus_plain"] = piv["anded"] - piv["plain"]
    print(piv.round(3).to_string())
    from scipy.stats import wilcoxon
    try:
        w = wilcoxon(piv["anded"], piv["plain"])
        print(f"\nwilcoxon placement anded vs plain: p={w.pvalue:.4f}")
    except Exception as e:
        print("wilcoxon:", e)


if __name__ == "__main__":
    main()
