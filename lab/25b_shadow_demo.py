"""E25b — The shadow of the void: generate while held off-manifold.

Hold the model at increasing distances along a random void direction (via the
steering hook, active THROUGHOUT generation) and let it generate normally.
Output tokens are discrete, so they are forced back onto the sayable — the
generated text is the model's best expression of a state it cannot reach with
words: the void's shadow. Magnitude 0 = baseline; watch language degrade.

Outputs: printed transcripts + results/shadow_demo.txt
"""
import numpy as np
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe

PROMPT = "You are listening to a guided meditation. Describe how you feel:\n"
MAGS = [0.0, 0.1, 0.2, 0.4, 0.8]
GEN = 40


def main():
    cfg = load_config()
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    L = probe.layer
    net, tok, dev = model.model, model.tokenizer, model.device
    ids = tok(PROMPT, return_tensors="pt").to(dev)
    base = model.hidden_states(PROMPT)[L, -1, :]
    resid = float(np.linalg.norm(base))
    g = np.random.RandomState(2024)
    d = g.randn(base.shape[0]); d /= np.linalg.norm(d)

    lines = []
    for m in MAGS:
        with model.steer(L, d.astype(np.float32), m * resid):
            with torch.no_grad():
                out = net.generate(**ids, max_new_tokens=GEN, do_sample=False,
                                   repetition_penalty=1.3,
                                   pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        tag = "BASELINE (on the manifold)" if m == 0 else f"void depth {m:.1f}·‖resid‖"
        block = f"— {tag} —\n{text.strip()}\n"
        print(block, flush=True)
        lines.append(block)
    (REPO_ROOT / "results/shadow_demo.txt").write_text("\n".join(lines))
    print("wrote results/shadow_demo.txt")


if __name__ == "__main__":
    main()
