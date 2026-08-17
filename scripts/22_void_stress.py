"""E23 — Stress the void floor with a much stronger optimizer.

E22 used greedy search over 300 phrases and got exactly 0% closure on
injection targets. Here we escalate to decide geometry-vs-search:

  A. VOCAB SWEEP — for each target, evaluate the single best next token over a
     large token sample (2000 real vocab tokens) at the anchor position. This
     is the exact argmin over one-token continuations: a hard lower bound on
     what any first token can do.
  B. BEAM SEARCH — width-8 beam over up to 6 appended tokens from the top-K
     per-step tokens (K=64), loss = distance to target. Far stronger than
     greedy phrase search.
  C. Report closure for both, plus the single-token floor, against the same
     injection targets and text-created positive controls as E22.

If beam + vocab sweep still floor at ~0% on injection targets while closing
on text targets, the void is geometric, not a search artifact.

Outputs: results/void_stress.csv
"""
import random

import numpy as np
import pandas as pd
import torch

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.stimuli import adapter as ad

ANCH_IDS_TEXT = "\nRight now everything feels"
INJECT_FRAC = 0.2
VOCAB_SAMPLE = 2000
BEAM_WIDTH = 8
BEAM_TOPK = 64
BEAM_DEPTH = 6


def main():
    cfg = load_config()
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    L = probe.layer
    tok = model.tokenizer
    dev = model.device
    pre_ids = tok(cfg["preamble"], return_tensors="pt")["input_ids"][0].to(dev)
    anch_ids = tok(ANCH_IDS_TEXT, add_special_tokens=False,
                   return_tensors="pt")["input_ids"][0].to(dev)

    @torch.no_grad()
    def state_of_ids(mid_ids):
        """anchor-final-token layer-L state for pre + mid + anchor."""
        ids = torch.cat([pre_ids, mid_ids, anch_ids]).unsqueeze(0)
        out = model.model(input_ids=ids, output_hidden_states=True)
        return out.hidden_states[L][0, -1].float().cpu().numpy().astype(np.float64)

    empty = torch.empty(0, dtype=pre_ids.dtype, device=dev)
    base = state_of_ids(empty)
    resid_norm = float(np.linalg.norm(base))

    # basis for PC targets
    rng = random.Random(99)
    sample = rng.sample(range(len(art.nodes)), 200)
    A = np.stack([model.hidden_states(art.word(i))[L, -1, :].astype(np.float64)
                  for i in sample])
    mu = A.mean(0)
    Vt = np.linalg.svd(A - mu, full_matrices=False)[2]
    gv = probe.ridge_v.coef_ / probe.scaler.scale_; gv /= np.linalg.norm(gv)

    # mint targets
    targets = {}
    g = np.random.RandomState(777)
    for k in range(2):
        d = g.randn(base.shape[0]); d /= np.linalg.norm(d)
        with model.steer(L, d.astype(np.float32), INJECT_FRAC * resid_norm):
            targets[f"inject-rand{k}"] = state_of_ids(empty)
    with model.steer(L, gv.astype(np.float32), INJECT_FRAC * resid_norm):
        targets["inject-valence"] = state_of_ids(empty)
    with model.steer(L, Vt[0].astype(np.float32), INJECT_FRAC * resid_norm):
        targets["inject-pc1"] = state_of_ids(empty)
    tr = random.Random(4242)
    for k in range(2):
        ids = tok(".\n".join(art.word(i) for i in tr.sample(range(len(art.nodes)), 10)),
                  add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev)
        targets[f"text{k}"] = state_of_ids(ids)

    # candidate vocab: sample of real, mostly-alphabetic tokens
    vocab_ids = []
    vg = random.Random(7)
    cand_pool = vg.sample(range(tok.vocab_size), min(20000, tok.vocab_size))
    for tid in cand_pool:
        s = tok.decode([tid])
        if s.strip().isalpha() and len(s.strip()) >= 2:
            vocab_ids.append(tid)
        if len(vocab_ids) >= VOCAB_SAMPLE:
            break
    vocab_ids = torch.tensor(vocab_ids, device=dev)

    def single_token_floor(t):
        best = np.inf
        for tid in vocab_ids.tolist():
            st = state_of_ids(torch.tensor([tid], device=dev))
            best = min(best, float(np.linalg.norm(st - t)))
        return best

    def beam(t, d0):
        beams = [(empty, d0)]
        best = d0
        for _ in range(BEAM_DEPTH):
            cand = []
            for seq, _ in beams:
                pick = vocab_ids[torch.randperm(len(vocab_ids), device=dev)[:BEAM_TOPK]]
                for tid in pick.tolist():
                    seq2 = torch.cat([seq, torch.tensor([tid], device=dev)])
                    dist = float(np.linalg.norm(state_of_ids(seq2) - t))
                    cand.append((seq2, dist))
            cand.sort(key=lambda x: x[1])
            beams = cand[:BEAM_WIDTH]
            best = min(best, beams[0][1])
        return best

    rows = []
    for name, t in targets.items():
        d0 = float(np.linalg.norm(base - t))
        floor1 = single_token_floor(t)
        bbest = beam(t, d0)
        rows.append({"target": name, "start_dist": d0,
                     "start_over_resid": d0 / resid_norm,
                     "single_tok_floor_ratio": floor1 / d0,
                     "beam_ratio": bbest / d0,
                     "beam_closed": 1 - bbest / d0})
        print(f"{name:16s} start {d0:6.1f}  1-tok floor {floor1/d0:.3f}  "
              f"beam {bbest/d0:.3f}  closed {1-bbest/d0:+.1%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results/void_stress.csv", index=False)
    df["kind"] = df.target.str.replace(r"\d+$", "", regex=True)
    print("\nmean by kind:")
    print(df.groupby("kind")[["single_tok_floor_ratio", "beam_ratio", "beam_closed"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
