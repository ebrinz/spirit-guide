"""E25 — Two-boundary replication (discrete floor + soft-prompt breakthrough)
on a given model. Run per Gemma to test whether the vocabulary-void structure
generalizes across architectures and scale.

Per injection-minted target (unreachable by construction, off-manifold) and
text-created control:
  discrete_floor — exact best single token over VOCAB_SAMPLE real tokens
                   (E23 method): the token boundary
  soft_ratio     — Adam on P free embedding vectors, distance loss (E24):
                   the continuous boundary
  emb_drift      — how far the soft solution sits from any real token (sigma)

Usage: python3 scripts/24_two_boundary_gemma.py --model <hf-id> --probe <pkl>
Outputs: results/two_boundary_<tag>.csv
"""
import argparse
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
VOCAB_SAMPLE = 1500
P = 8
STEPS = 300
LR = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vocab", type=int, default=VOCAB_SAMPLE)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    global VOCAB_SAMPLE, STEPS
    VOCAB_SAMPLE, STEPS = args.vocab, args.steps
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(args.model, device=cfg["device"])
    probe = load_probe(REPO_ROOT / args.probe if not args.probe.startswith("/")
                       else args.probe)
    L = probe.layer
    tok = model.tokenizer
    dev = model.device
    net = model.model
    embed = net.get_input_embeddings()
    emb_w = embed.weight.detach()
    emb_std = emb_w.float().std(0).mean().item()

    pre_ids = tok(cfg["preamble"], return_tensors="pt")["input_ids"][0].to(dev)
    anch_ids = tok(ANCH, add_special_tokens=False,
                   return_tensors="pt")["input_ids"][0].to(dev)
    pre_emb = embed(pre_ids).detach()
    anch_emb = embed(anch_ids).detach()

    @torch.no_grad()
    def state_ids(mid_ids):
        seq = torch.cat([pre_ids, mid_ids, anch_ids]).unsqueeze(0)
        return net(input_ids=seq, output_hidden_states=True
                   ).hidden_states[L][0, -1].float().cpu().numpy().astype(np.float64)

    def state_embeds(mid_emb):
        seq = torch.cat([pre_emb, mid_emb, anch_emb], dim=0).unsqueeze(0)
        return net(inputs_embeds=seq, output_hidden_states=True).hidden_states[L][0, -1]

    empty = torch.empty(0, dtype=pre_ids.dtype, device=dev)
    base = state_ids(empty)
    resid_norm = float(np.linalg.norm(base))

    rng = random.Random(99)
    sample = rng.sample(range(len(art.nodes)), 160)
    A = np.stack([model.hidden_states(art.word(i))[L, -1, :].astype(np.float64)
                  for i in sample])
    Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)[2]
    gv = probe.ridge_v.coef_ / probe.scaler.scale_; gv /= np.linalg.norm(gv)

    targets = {}
    g = np.random.RandomState(777)
    for k in range(2):
        d = g.randn(base.shape[0]); d /= np.linalg.norm(d)
        with model.steer(L, d.astype(np.float32), INJECT_FRAC * resid_norm):
            targets[f"inject-rand{k}"] = state_ids(empty)
    with model.steer(L, gv.astype(np.float32), INJECT_FRAC * resid_norm):
        targets["inject-valence"] = state_ids(empty)
    with model.steer(L, Vt[0].astype(np.float32), INJECT_FRAC * resid_norm):
        targets["inject-pc1"] = state_ids(empty)
    tr = random.Random(4242)
    ids = tok(".\n".join(art.word(i) for i in tr.sample(range(len(art.nodes)), 10)),
              add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev)
    targets["text0"] = state_ids(ids)

    # discrete single-token candidate pool
    vocab_ids, vg = [], random.Random(7)
    for tid in vg.sample(range(tok.vocab_size), min(20000, tok.vocab_size)):
        s = tok.decode([tid])
        if s.strip().isalpha() and len(s.strip()) >= 2:
            vocab_ids.append(tid)
        if len(vocab_ids) >= VOCAB_SAMPLE:
            break

    rows = []
    for name, t_np in targets.items():
        d0 = float(np.linalg.norm(base - t_np))
        # discrete floor
        floor = min(float(np.linalg.norm(state_ids(torch.tensor([tid], device=dev)) - t_np))
                    for tid in vocab_ids)
        # soft prompt
        t = torch.tensor(t_np, dtype=pre_emb.dtype, device=dev)
        init = torch.tensor(random.sample(range(tok.vocab_size), P), device=dev)
        soft = emb_w[init].detach().clone().to(torch.float32)
        soft.requires_grad_(True)
        opt = torch.optim.Adam([soft], lr=LR)
        best = d0
        for _ in range(STEPS):
            opt.zero_grad()
            loss = torch.norm(state_embeds(soft.to(pre_emb.dtype)).float() - t.float())
            loss.backward(); opt.step()
            best = min(best, float(loss.item()))
        with torch.no_grad():
            drift = torch.cdist(soft.detach().float(),
                                emb_w.float()).min(dim=1).values.mean().item() / emb_std
        rows.append({"model": args.tag, "target": name, "start_dist": d0,
                     "discrete_floor_ratio": floor / d0,
                     "soft_ratio": best / d0, "soft_closed": 1 - best / d0,
                     "emb_drift_sigma": drift})
        print(f"[{args.tag}] {name:16s} start {d0:6.1f}  discrete {floor/d0:.3f}  "
              f"soft {best/d0:.3f} (closed {1-best/d0:+.0%})  drift {drift:.0f}σ",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / f"results/two_boundary_{args.tag}.csv", index=False)
    df["kind"] = df.target.str.replace(r"\d+$", "", regex=True)
    print(f"\n[{args.tag}] mean by kind:")
    print(df.groupby("kind")[["discrete_floor_ratio", "soft_closed", "emb_drift_sigma"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
