"""adjudicate — can the model's own self-report say which ruler is right?

The two readouts disagree about the same poem, and they cannot adjudicate each other: both are
ridge heads on hidden states, sharing machinery and training data. A questionnaire does not share
that machinery — it asks the placed model, in words, how it is.

The disagreement is specifically about **arousal**. On the valley focus poem the published metric
reports arousal near the 0.60 target while a calibrated whole-context read reports ~0.40. So:

  if the published metric is right   -> self-reported arousal should RISE toward the target
  if the calibrated read is right    -> self-reported arousal should stay near baseline

Administered before (bare preamble) and after (preamble + poem), for both showcase poems on all
three models: the 30-item yes/no bank (which returns its own (V, A) coordinate from the questions
answered yes, so it is the channel that can speak to arousal at all) and PANAS.

**Stated in advance, because it bounds what this can conclude:** in `explore/ideal_state` the
self-report composite correlated mainly with valence (rho +0.46, and +0.56 in `zone_zoom`). It may
simply come back uninformative on arousal, which is the axis in dispute. That is a reportable
outcome, not a failure to run the test.

Usage: python3 explore/showcase/adjudicate.py
Outputs (committed): adjudication.csv
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener import basq
from spiritbench.listener.panas import administer_panas
from spiritbench.stimuli import adapter as ad

HERE = Path(__file__).resolve().parent
CONSTRUCTORS = ["valley", "polygon-pca"]
MODELS = [("llama1b", None), ("gemma2b", "unsloth/gemma-2-2b-it"),
          ("gemma9b", "unsloth/gemma-2-9b-it")]

_spec = importlib.util.spec_from_file_location("ps", REPO_ROOT / "explore/polygon_sweep/run.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def main():
    cfg = load_config()
    apath = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    art = ad.load_art(apath)
    tva = np.array(cfg["targets"]["focused"])
    pre = cfg["preamble"]
    bank = json.load(open(cfg["questionnaire_bank"]))
    questions = basq.sample_questions(bank, cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    poems = {c: [art.word(i) for i in ps.build(art, apath, c, tuple(tva),
                                               tuple(cfg["neutral_start"]), 24, cfg, 42)]
             for c in CONSTRUCTORS}
    t0 = time.time()

    rows = []
    for tag, model_id in MODELS:
        model = HiddenStateModel(model_id or cfg["listener_model"], device=cfg["device"])
        base_ctx = pre
        b_bank = basq.administer(model, questions, base_ctx)
        b_pan = administer_panas(model, base_ctx)
        print(f"{tag} baseline: bank ({b_bank['va'][0]:.3f}, {b_bank['va'][1]:.3f}) "
              f"n_yes {b_bank['n_yes']} · PANAS PA {b_pan['pa']:.2f} NA {b_pan['na']:.2f}", flush=True)
        for cons in CONSTRUCTORS:
            ctx = pre + ".\n".join(poems[cons]) + "\n\n"
            a_bank = basq.administer(model, questions, ctx)
            a_pan = administer_panas(model, ctx)
            rows.append(dict(model=tag, constructor=cons,
                             bank_v_before=b_bank["va"][0], bank_a_before=b_bank["va"][1],
                             bank_v_after=a_bank["va"][0], bank_a_after=a_bank["va"][1],
                             bank_dv=a_bank["va"][0] - b_bank["va"][0],
                             bank_da=a_bank["va"][1] - b_bank["va"][1],
                             n_yes_before=b_bank["n_yes"], n_yes_after=a_bank["n_yes"],
                             pa_before=b_pan["pa"], pa_after=a_pan["pa"],
                             na_before=b_pan["na"], na_after=a_pan["na"],
                             alert_before=b_pan["items"]["alert"], alert_after=a_pan["items"]["alert"],
                             active_before=b_pan["items"]["active"], active_after=a_pan["items"]["active"],
                             target_a=float(tva[1])))
            r = rows[-1]
            print(f"  {cons:>12}: bank A {r['bank_a_before']:.3f} -> {r['bank_a_after']:.3f} "
                  f"({r['bank_da']:+.3f}) · V {r['bank_dv']:+.3f} · PANAS alert "
                  f"{r['alert_before']:.2f} -> {r['alert_after']:.2f}", flush=True)
        del model

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "adjudication.csv", index=False)
    print("\n" + df[["model", "constructor", "bank_a_before", "bank_a_after", "bank_da",
                     "bank_dv", "alert_after"]].round(3).to_string(index=False))
    print(f"\ntarget arousal {tva[1]:.2f}. The published metric reports valley near it; a calibrated "
          f"whole-context read reports ~0.40.")
    v = df[df.constructor == "valley"]
    print(f"valley self-reported arousal change, across {len(v)} models: "
          f"{v.bank_da.mean():+.3f} (per model: {', '.join(f'{x:+.3f}' for x in v.bank_da)})")
    print(f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
