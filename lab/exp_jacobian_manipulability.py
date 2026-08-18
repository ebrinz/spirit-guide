"""exp_jacobian_manipulability — voids as kinematic singularities of the
language->state map.

Control-theory framing (joint/task space + Jacobian):
  joint space  = the control input: P soft-prompt embedding vectors we can vary
  task space   = the 2-D VAD readout (probe)
  Jacobian J   = d(VAD) / d(soft embeddings), the local map from control to task
A void/pocket should be a SINGULARITY of J: a region where the achievable-VAD
"manipulability ellipsoid" collapses — you can push the input but VAD can't move
in some direction. The collapsed (forbidden) direction should, at high arousal,
point along +valence (confirming the arousal/valence inverse coupling structurally).

The probe is linear (standardize + ridge) so VAD is a differentiable linear
function of the hidden state; autograd gives J via two backward passes per cell.
J is 2 x (P*d); its 2x2 Gram G = J J^T has eigenvalues σ1^2 ≥ σ2^2 and
eigenvectors = principal reachable directions in (V,A).

Metrics per grid cell (base = a valley poem placing near the cell):
  residual        — how far the poem placed from target (the empirical pocket signal)
  manipulability  — sqrt(σ1 σ2): area of the reachable-VAD ellipse per unit input
  min_control     — σ2: worst-case controllability (small => singular => pocket)
  forbidden_deg   — angle in (V,A) of the least-reachable direction (eigvec of σ2)

Verdict: (1) does low manipulability predict the empirical pockets (residual)?
(2) in the high-arousal band, does the forbidden direction align with +valence?
If yes, voids are Jacobian singularities and the coupling is a structural
null-space of language control.

Llama-1B (light backprop). Usage: python3 lab/exp_jacobian_manipulability.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
GRID = 5
V_RANGE = (0.30, 0.80)
A_RANGE = (0.25, 0.80)
P = 4        # soft-prompt length (the control handles)
N = 24
ANCH = "\nRight now everything feels"


def main():
    from spiritbench.config import load_config
    from spiritbench.listener.model import HiddenStateModel
    from spiritbench.listener.probe import load_probe
    from spiritbench.stimuli import adapter as ad

    cfg = load_config()
    art = ad.load_art(str(REPO / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO / "data/probe/probe.pkl")
    L = probe.layer
    net, tok, dev, embed = model.model, model.tokenizer, model.device, \
        model.model.get_input_embeddings()
    emb_mean = embed.weight.detach().mean(0)

    # probe as a torch linear map: VAD = W ((h - mean)/scale) + b
    scale = torch.tensor(probe.scaler.scale_, dtype=torch.float32, device=dev)
    mean = torch.tensor(probe.scaler.mean_, dtype=torch.float32, device=dev)
    wv = torch.tensor(probe.ridge_v.coef_, dtype=torch.float32, device=dev) / scale
    wa = torch.tensor(probe.ridge_a.coef_, dtype=torch.float32, device=dev) / scale

    anch_emb = embed(tok(ANCH, add_special_tokens=False,
                         return_tensors="pt")["input_ids"][0].to(dev)).detach()
    pre_emb = embed(tok(cfg["preamble"], return_tensors="pt")["input_ids"][0].to(dev)).detach()

    @torch.no_grad()
    def place(poem):
        hs = model.hidden_states(cfg["preamble"] + poem + ANCH)
        v, a = probe.predict(hs[L][-1:])[0]
        return float(v), float(a)

    def jacobian(poem_emb):
        """G (2x2) and its eig, for soft tokens at emb_mean, base = poem_emb."""
        soft = emb_mean.repeat(P, 1).clone().to(torch.float32).requires_grad_(True)
        seq = torch.cat([pre_emb, poem_emb, soft.to(pre_emb.dtype), anch_emb], 0).unsqueeze(0)
        h = net(inputs_embeds=seq, output_hidden_states=True).hidden_states[L][0, -1].float()
        gv = torch.autograd.grad((h * wv).sum(), soft, retain_graph=True)[0].flatten()
        ga = torch.autograd.grad((h * wa).sum(), soft)[0].flatten()
        G = torch.tensor([[float(gv @ gv), float(gv @ ga)],
                          [float(gv @ ga), float(ga @ ga)]])
        evals, evecs = torch.linalg.eigh(G)         # ascending
        return evals.numpy(), evecs.numpy()

    vs = np.linspace(*V_RANGE, GRID)
    as_ = np.linspace(*A_RANGE, GRID)
    rows = []
    for v in vs:
        for a in as_:
            ids = ad.valley_shape(art, (float(v), float(a)), N, seed=7)
            poem = ".\n".join(art.word(i) for i in ids)
            pv, pa = place(poem)
            resid = float(np.hypot(pv - v, pa - a))
            poem_emb = embed(tok(poem, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0].to(dev)).detach()
            evals, evecs = jacobian(poem_emb)
            s2, s1 = np.sqrt(max(evals[0], 0)), np.sqrt(max(evals[1], 0))
            forbidden = evecs[:, 0]                  # eigvec of the SMALL eigenvalue
            forbidden_deg = float(np.degrees(np.arctan2(forbidden[1], forbidden[0])))
            rows.append({"target_v": float(v), "target_a": float(a), "residual": resid,
                         "manipulability": float(s1 * s2), "min_control": float(s2),
                         "max_control": float(s1),
                         "forbidden_deg": forbidden_deg})
            print(f"  ({v:.2f},{a:.2f}) resid {resid:.3f} manip {s1*s2:.2e} "
                  f"min_ctrl {s2:.2e} forbid {forbidden_deg:+.0f}°", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "jacobian_manipulability.csv", index=False)

    from scipy.stats import spearmanr
    r1, p1 = spearmanr(df.min_control, df.residual)
    r2, p2 = spearmanr(df.manipulability, df.residual)
    print(f"\nmin-controllability vs residual: spearman r={r1:.3f} p={p1:.4f}  "
          "(negative => singular regions ARE the pockets)")
    print(f"manipulability vs residual:      spearman r={r2:.3f} p={p2:.4f}")
    hi = df[df.target_a >= 0.65]
    print(f"\nhigh-arousal band forbidden-direction angles (0°=+V, 90°=+A): "
          f"{hi.forbidden_deg.round(0).tolist()}")
    print(f"  |projection on +valence axis| mean: "
          f"{np.abs(np.cos(np.radians(hi.forbidden_deg))).mean():.2f}  "
          "(near 1 => the forbidden direction is +valence)")
    print(f"wrote {RESULTS}/jacobian_manipulability.csv")


if __name__ == "__main__":
    main()
