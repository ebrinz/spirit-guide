"""E8 — Six-state steering: causal injection into six target states, and route discernment.

States: eros, creativity, imaginative, determined, confident, agape.

Design (measurement-only; no open-ended generation):
  1. Per state: direction = difference of mean layer-17 states between the
     state's word set and one shared neutral set (probe's carrier machinery).
  2. Inject alpha * direction at layer 17 (fractions of the natural residual
     norm) and read: probe VA (manipulation check), PANAS, token-mass over
     {state words, calm words, negative words}, layer-20 SAE features.
  3. Text route per state: a litany of the phrase graph's nearest Gutenberg
     lines to the state's GloVe centroid. Compare activation delta + SAE
     top-feature overlap with the injected route = "how it got there".

Outputs: data/steering/steering.json, per-state litany texts,
results/steering_dose_response.csv.
"""
import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.analysis import sae as S
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_glove

STATES = {
    "eros": ["desire", "passion", "kiss", "caress", "embrace", "tender",
             "longing", "intimate", "sensual", "beloved", "romance", "adore"],
    "creativity": ["create", "invent", "compose", "craft", "design", "original",
                   "inspire", "art", "curious", "spark", "weave", "shape"],
    "imaginative": ["imagine", "dream", "wonder", "vision", "fantasy", "envision",
                    "myth", "fairy", "magic", "marvel", "whimsical", "enchanted"],
    "determined": ["determined", "resolve", "persist", "steadfast", "unwavering",
                   "grit", "commit", "endure", "strive", "relentless",
                   "tenacious", "perseverance"],
    "confident": ["confident", "assured", "bold", "certain", "capable", "strong",
                  "poised", "fearless", "secure", "proud", "steady", "unshaken"],
    "agape": ["compassion", "mercy", "kindness", "charity", "selfless",
              "benevolent", "forgive", "grace", "devotion", "unconditional",
              "cherish", "tenderness"],
}
NEUTRAL_WORDS = ["table", "window", "street", "paper", "walk", "morning",
                 "door", "water", "stone", "room", "letter", "field"]
CALM_WORDS = ["calm", "peaceful", "serene", "quiet", "gentle", "still"]
NEG_WORDS = ["afraid", "terrible", "hopeless", "angry", "cold", "broken"]

CARRIER = "The word is {w}"
ANCHOR = "Right now everything feels"
ALPHA_FRACS = [0.0, 0.05, 0.1, 0.2, 0.4]
COMPARISON_FRAC = 0.2
PROBE_LAYER = 17
PANAS_KEEP = ["excited", "inspired", "determined", "proud", "strong", "nervous"]


def word_states(model, words, layer):
    return np.stack([model.hidden_states(CARRIER.format(w=w))[layer, -1, :]
                     for w in words])


def token_masses(model, context, state_words):
    opts = ([" " + w for w in state_words] + [" " + w for w in CALM_WORDS]
            + [" " + w for w in NEG_WORDS])
    lps = np.array(model.option_logprobs(context + ANCHOR, opts))
    m = np.exp(lps - lps.max())
    n1, n2 = len(state_words), len(CALM_WORDS)
    total = m.sum()
    return {"state_share": float(m[:n1].sum() / total),
            "calm_share": float(m[n1:n1 + n2].sum() / total),
            "neg_share": float(m[n1 + n2:].sum() / total)}


def channel_read(model, probe, sae, context, state_words):
    hs = model.hidden_states(context + ANCHOR)
    va = probe.predict(hs[probe.layer][-1:])[0]
    feats = S.encode(hs[S.SAE_LAYER][-1].astype(np.float32), sae)
    return {"probe_va": [float(va[0]), float(va[1])],
            "panas": administer_panas(model, context),
            "masses": token_masses(model, context, state_words),
            "resid20": hs[S.SAE_LAYER][-1].astype(np.float32),
            "sae": feats}


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    sae = S.load_sae(hf_hub_download("google/gemma-scope-2b-pt-res",
                                     "layer_20/width_16k/average_l0_71/params.npz"))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    all_state_words = {w for ws in STATES.values() for w in ws}
    glove = load_glove(cfg["glove_path"], all_state_words)

    ctx = cfg["preamble"]
    neut = word_states(model, NEUTRAL_WORDS, PROBE_LAYER).mean(axis=0)
    base_hs = model.hidden_states(ctx + ANCHOR)
    resid_norm = float(np.linalg.norm(base_hs[PROBE_LAYER], axis=1).mean())
    print(f"layer-{PROBE_LAYER} mean residual norm = {resid_norm:.1f}", flush=True)

    rows, summaries = [], {}
    for state, words in STATES.items():
        print(f"\n=== {state} ===", flush=True)
        direction = word_states(model, words, PROBE_LAYER).mean(axis=0) - neut
        direction = (direction / np.linalg.norm(direction)).astype(np.float32)
        reads = {}
        for frac in ALPHA_FRACS:
            if frac == 0.0 and "base" in summaries:
                reads[frac] = summaries["base"]      # baseline shared across states
            elif frac == 0.0:
                reads[frac] = channel_read(model, probe, sae, ctx, words)
                summaries["base"] = reads[frac]
            else:
                with model.steer(PROBE_LAYER, direction, frac * resid_norm):
                    reads[frac] = channel_read(model, probe, sae, ctx, words)
            r = reads[frac]
            row = {"state": state, "alpha_frac": frac,
                   "probe_v": r["probe_va"][0], "probe_a": r["probe_va"][1],
                   **r["masses"], "panas_pa": r["panas"]["pa"],
                   "panas_na": r["panas"]["na"],
                   **{k: r["panas"]["items"][k] for k in PANAS_KEEP},
                   "n_sae_active": int((r["sae"] > 0).sum())}
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v)
                   for k, v in row.items()}, flush=True)

        # text route
        evec = np.mean([glove[w] for w in words if w in glove], axis=0)
        sims = (art.vectors @ evec) / (np.linalg.norm(art.vectors, axis=1)
                                       * np.linalg.norm(evec) + 1e-9)
        litany = ".\n".join(art.word(int(i)) for i in np.argsort(-sims)[:12])
        (out_dir / f"litany_{state}.txt").write_text(litany)
        text_read = channel_read(model, probe, sae, ctx + litany + "\n\n", words)

        base, inj = reads[0.0], reads[COMPARISON_FRAC]
        d_inj = inj["resid20"] - base["resid20"]
        d_txt = text_read["resid20"] - base["resid20"]
        cos = float(np.dot(d_inj, d_txt)
                    / (np.linalg.norm(d_inj) * np.linalg.norm(d_txt) + 1e-9))
        top_inj = set(np.argsort(-(inj["sae"] - base["sae"]))[:20].tolist())
        top_txt = set(np.argsort(-(text_read["sae"] - base["sae"]))[:20].tolist())
        summaries[state] = {
            "text_route": {"probe_va": text_read["probe_va"],
                           "masses": text_read["masses"],
                           "panas_pa": text_read["panas"]["pa"],
                           "panas_na": text_read["panas"]["na"]},
            "route_comparison": {
                "cosine_layer20_delta": cos,
                "sae_top20_jaccard": len(top_inj & top_txt) / len(top_inj | top_txt),
                "shared_features": sorted(top_inj & top_txt)},
            "litany_first_lines": litany.split("\n")[:3],
        }
        print(f"{state}: text-route cos={cos:.3f} "
              f"jaccard={summaries[state]['route_comparison']['sae_top20_jaccard']:.2f}",
              flush=True)

    pd.DataFrame(rows).to_csv(REPO_ROOT / "results/steering_dose_response.csv",
                              index=False)
    summaries.pop("base", None)
    with open(out_dir / "steering.json", "w") as f:
        json.dump(summaries, f, indent=2, default=float)
    print("\nwrote results/steering_dose_response.csv and data/steering/steering.json")


if __name__ == "__main__":
    main()
