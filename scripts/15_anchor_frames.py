"""C9 upgrade — anchor-frame robustness of the steerability hierarchy.

Rebuild the six state directions, steer at the max dose, and measure
state-word mass under THREE anchor frames with different grammatical
affordances. If the agentive > receptive hierarchy is frame-independent,
it is not an artifact of "feels ___" adjective grammar.
Outputs: results/anchor_frames.csv.
"""
import importlib.util

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel

spec = importlib.util.spec_from_file_location(
    "sixstate", REPO_ROOT / "scripts/10_six_state_steering.py")
sixstate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sixstate)

FRAMES = {
    "feels": "Right now everything feels",
    "filled": "I am filled with",
    "state": "My present state is one of",
}
DOSE_FRAC = 0.4


def masses(model, ctx, frame, state_words):
    opts = ([" " + w for w in state_words]
            + [" " + w for w in sixstate.CALM_WORDS]
            + [" " + w for w in sixstate.NEG_WORDS])
    lps = np.array(model.option_logprobs(ctx + frame, opts))
    m = np.exp(lps - lps.max())
    return float(m[: len(state_words)].sum() / m.sum())


def main():
    cfg = load_config()
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    ctx = cfg["preamble"]
    neut = sixstate.word_states(model, sixstate.NEUTRAL_WORDS,
                                sixstate.PROBE_LAYER).mean(axis=0)
    base_hs = model.hidden_states(ctx + FRAMES["feels"])
    resid_norm = float(np.linalg.norm(base_hs[sixstate.PROBE_LAYER], axis=1).mean())
    rows = []
    for state, words in sixstate.STATES.items():
        direction = sixstate.word_states(model, words,
                                         sixstate.PROBE_LAYER).mean(axis=0) - neut
        direction = (direction / np.linalg.norm(direction)).astype(np.float32)
        for fname, frame in FRAMES.items():
            base = masses(model, ctx, frame, words)
            with model.steer(sixstate.PROBE_LAYER, direction,
                             DOSE_FRAC * resid_norm):
                steered = masses(model, ctx, frame, words)
            rows.append({"state": state, "frame": fname,
                         "base": base, "steered": steered,
                         "delta": steered - base})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/anchor_frames.csv", index=False)
    piv = df.pivot_table(index="state", columns="frame", values="steered")
    print("\nsteered state-share by frame:")
    print(piv.round(3).to_string())
    print("\nrank correlation of state ordering across frames:")
    from scipy.stats import spearmanr
    cols = list(piv.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = spearmanr(piv[cols[i]], piv[cols[j]])
            print(f"  {cols[i]} vs {cols[j]}: rho={r:.2f} p={p:.3f}")


if __name__ == "__main__":
    main()
