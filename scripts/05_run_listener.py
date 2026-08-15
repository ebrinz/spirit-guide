"""Run every stimulus through the listener; write one JSON per stimulus (resumable)."""
import json
from pathlib import Path

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.basq import administer, sample_questions
from spiritbench.analysis.metrics import ema, per_line_va


def run_stimulus(model, probe, stim, preamble, ema_alpha, bank, basq_cfg) -> dict:
    questions = sample_questions(bank, basq_cfg["n_questions"], basq_cfg["seed"])
    basq_pre = administer(model, questions, context=preamble)
    lines = stim["lines"] if stim["lines"] else [stim["text"]]
    hs, spans = model.hidden_states_with_spans(preamble, lines)
    hidden = hs[probe.layer]                      # [n_tokens, d]
    raw = probe.predict(hidden)                   # [n_tokens, 2]
    n_pre = len(model.tokenizer(preamble)["input_ids"])
    traj = ema(raw[n_pre:], ema_alpha) if len(raw) > n_pre else ema(raw, ema_alpha)
    spans0 = [(max(0, s - n_pre), max(1, e - n_pre)) for s, e in spans]
    line_vas = per_line_va(traj, spans0)
    post_ctx = preamble + stim["text"] + "\n\n"
    basq_post = administer(model, questions, context=post_ctx)
    return {"stimulus_id": stim["id"], "traj": traj.tolist(),
            "line_vas": line_vas.tolist(), "basq_pre": basq_pre,
            "basq_post": basq_post, "n_tokens": int(hs.shape[1])}


def main():
    cfg = load_config()
    runs_dir = REPO_ROOT / "data/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stims = []
    for name in ["data/stimuli/stimuli.jsonl", "data/renders/renders.jsonl"]:
        p = REPO_ROOT / name
        if p.exists():
            stims += [json.loads(l) for l in open(p)]
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    bank = json.load(open(cfg["questionnaire_bank"]))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    for i, stim in enumerate(stims):
        out = runs_dir / f"{stim['id']}.json"
        if out.exists():
            continue
        try:
            rec = run_stimulus(model, probe, stim, cfg["preamble"],
                               cfg["ema_alpha"], bank, cfg["basq"])
            if (stim.get("generator") == "psg" and stim.get("target") == "calm"
                    and stim.get("params", {}).get("length") == "medium"):
                rec["noframe"] = run_stimulus(model, probe, stim, "", cfg["ema_alpha"],
                                              bank, cfg["basq"])
        except Exception as e:
            rec = {"stimulus_id": stim["id"], "error": repr(e)}
            print(f"FAILED {stim['id']}: {e!r}")
        with open(out, "w") as f:
            json.dump(rec, f)
        print(f"[{i + 1}/{len(stims)}] {stim['id']}")


if __name__ == "__main__":
    main()
