"""E27 — Cross-model stimulus transfer.

The phase-1 sweep showed constructor RANKINGS replicate across models. This
tests the stronger claim: does a SPECIFIC poem that places well on one model
place well on another? Runs a fixed poem set under a given model's probe;
run for two models, then per-poem correlation = effect transfer (not just
ranking transfer).

Usage: python3 scripts/29_cross_model_transfer.py --model <id> --probe <pkl> --tag <t>
Outputs: results/transfer_<tag>.csv  (merge/correlate across tags after)
"""
import argparse
import json

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.analysis.metrics import placement_error, ema, per_line_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config()
    model = HiddenStateModel(args.model, device=cfg["device"])
    probe = load_probe(REPO_ROOT / args.probe)
    pre = cfg["preamble"]

    stims = [json.loads(l) for l in open(REPO_ROOT / "data/stimuli/stimuli.jsonl")]
    stims = [s for s in stims if s["generator"] == "psg"][:40]  # fixed poem set

    rows = []
    for s in stims:
        hs, spans = model.hidden_states_with_spans(pre, s["lines"])
        n_pre = len(model.tokenizer(pre)["input_ids"])
        raw = probe.predict(hs[probe.layer])
        traj = ema(raw[n_pre:], cfg["ema_alpha"]) if len(raw) > n_pre else ema(raw, cfg["ema_alpha"])
        rows.append({"id": s["id"], "constructor": s["constructor"],
                     "target": s["target"],
                     "placement_error": placement_error(traj, s["target_va"]),
                     "final_v": float(traj[-1][0]), "final_a": float(traj[-1][1])})
        print(f"[{args.tag}] {s['constructor']:14s}/{s['target']:8s} "
              f"place {rows[-1]['placement_error']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / f"results/transfer_{args.tag}.csv", index=False)
    print(f"wrote results/transfer_{args.tag}.csv ({len(df)} poems)")


if __name__ == "__main__":
    main()
