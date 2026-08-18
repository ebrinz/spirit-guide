"""E24 — Soft-prompt attack on the void floor: geometry or dictionary?

E23 proved discrete tokens cannot approach injection-minted states (exact
single-token floor > 1.0). Here we drop discreteness: optimize P continuous
embedding vectors by gradient descent to minimize distance to the target
state, model frozen. This is the strongest possible prompt-side attack —
unconstrained to real tokens.

  - break through the floor  -> "the sayable" was a DICTIONARY limit; the
    continuous embedding space reaches these states
  - still floors            -> the limit is GEOMETRIC even off the token
    simplex: the model's forward map cannot produce these states from any
    prefix, real or imagined

Controls per target:
  soft   — P free embedding vectors, Adam, many steps
  drift  — same P vectors, random-token-embedding init, NO optimization
  We also report how far the learned soft embeddings drift from the nearest
  real token embedding (are they "words" or off-manifold vectors?).

Targets: injection-minted (random dirs, valence, PC1) + text-created controls.
Outputs: results/soft_prompt_void.csv
"""
import random

import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.stimuli import adapter as ad

ANCH = "\nRight now everything feels"
INJECT_FRAC = 0.2
P = 8            # soft-prompt length (tokens' worth of free vectors)
STEPS = 400
LR = 0.05


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    L = probe.layer
    tok = model.tokenizer
    dev = model.device
    net = model.model
    embed = net.get_input_embeddings()          # [V, d]
    emb_w = embed.weight.detach()

    pre_ids = tok(cfg["preamble"], return_tensors="pt")["input_ids"][0].to(dev)
    anch_ids = tok(ANCH, add_special_tokens=False,
                   return_tensors="pt")["input_ids"][0].to(dev)
    pre_emb = embed(pre_ids).detach()            # [np, d]
    anch_emb = embed(anch_ids).detach()          # [na, d]
    d_model = pre_emb.shape[1]

    def state_from_embeds(mid_emb):
        """layer-L final-token state for [pre; mid; anchor] via inputs_embeds."""
        seq = torch.cat([pre_emb, mid_emb, anch_emb], dim=0).unsqueeze(0)
        out = net(inputs_embeds=seq, output_hidden_states=True)
        return out.hidden_states[L][0, -1]        # [d], grad-enabled

    @torch.no_grad()
    def state_ids(mid_ids):
        seq = torch.cat([pre_ids, mid_ids, anch_ids]).unsqueeze(0)
        return net(input_ids=seq, output_hidden_states=True
                   ).hidden_states[L][0, -1].float().cpu().numpy().astype(np.float64)

    base = state_ids(torch.empty(0, dtype=pre_ids.dtype, device=dev))
    resid_norm = float(np.linalg.norm(base))

    # PC / valence dirs for targets
    rng = random.Random(99)
    sample = rng.sample(range(len(art.nodes)), 200)
    A = np.stack([model.hidden_states(art.word(i))[L, -1, :].astype(np.float64)
                  for i in sample])
    Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)[2]
    gv = probe.ridge_v.coef_ / probe.scaler.scale_; gv /= np.linalg.norm(gv)

    targets = {}
    g = np.random.RandomState(777)
    for k in range(2):
        d = g.randn(base.shape[0]); d /= np.linalg.norm(d)
        with model.steer(L, d.astype(np.float32), INJECT_FRAC * resid_norm):
            targets[f"inject-rand{k}"] = state_ids(torch.empty(0, dtype=pre_ids.dtype, device=dev))
    with model.steer(L, gv.astype(np.float32), INJECT_FRAC * resid_norm):
        targets["inject-valence"] = state_ids(torch.empty(0, dtype=pre_ids.dtype, device=dev))
    with model.steer(L, Vt[0].astype(np.float32), INJECT_FRAC * resid_norm):
        targets["inject-pc1"] = state_ids(torch.empty(0, dtype=pre_ids.dtype, device=dev))
    tr = random.Random(4242)
    for k in range(2):
        ids = tok(".\n".join(art.word(i) for i in tr.sample(range(len(art.nodes)), 10)),
                  add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev)
        targets[f"text{k}"] = state_ids(ids)

    emb_mean = emb_w.mean(0)
    emb_std = emb_w.std(0).mean().item()

    rows = []
    for name, t_np in targets.items():
        t = torch.tensor(t_np, dtype=pre_emb.dtype, device=dev)
        d0 = float(np.linalg.norm(base - t_np))
        # init soft prompt at random real-token embeddings
        init_ids = torch.tensor(random.sample(range(tok.vocab_size), P), device=dev)
        soft = emb_w[init_ids].detach().clone().to(torch.float32)
        soft.requires_grad_(True)
        opt = torch.optim.Adam([soft], lr=LR)
        best = d0
        for step in range(STEPS):
            opt.zero_grad()
            st = state_from_embeds(soft.to(pre_emb.dtype))
            loss = torch.norm(st.float() - t.float())
            loss.backward()
            opt.step()
            best = min(best, float(loss.item()))
        # how far did the soft vectors drift from any real token?
        with torch.no_grad():
            sf = soft.detach()
            dists = torch.cdist(sf.float(), emb_w.float())     # [P, V]
            nearest = dists.min(dim=1).values.mean().item()
        rows.append({"target": name, "start_dist": d0,
                     "start_over_resid": d0 / resid_norm,
                     "soft_ratio": best / d0, "soft_closed": 1 - best / d0,
                     "emb_drift_over_std": nearest / emb_std})
        print(f"{name:16s} start {d0:6.1f}  soft best ratio {best/d0:.3f}  "
              f"closed {1-best/d0:+.1%}  emb-drift {nearest/emb_std:.1f}σ", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/soft_prompt_void.csv", index=False)
    df["kind"] = df.target.str.replace(r"\d+$", "", regex=True)
    print("\nmean by kind:")
    print(df.groupby("kind")[["soft_ratio", "soft_closed", "emb_drift_over_std"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
