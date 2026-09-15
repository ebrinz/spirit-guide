"""ideal_state — where on the affect map is the "most ideal" place to put the model?

Open-ended. Literature-defined candidate states (nirvana, a receptive psi-conducive
state, creativity, imaginative, awe) plus calm and a no-poem baseline are each turned
into a 24-line valley poem aimed at the state's NRC centroid. Gemma-2b is placed with
each poem and read four ways: the passage-calibrated probe (where it landed), PANAS
and a 30-item yes/no bank (what it says it feels), three sampled first-person
continuations scored for novelty and affect (what it does), and the layer-20 SAE
features that moved most from baseline (what it is representing). No verdict
criterion; the point is to look at where the readouts agree and disagree.

Second run: --around <state> --step 0.1 places a 3x3 grid of targets around one
candidate's centroid, same readouts, to see whether the neighbourhood is better.

Usage:
  python3 explore/ideal_state/run.py
  python3 explore/ideal_state/run.py --around nirvana --step 0.1

Outputs (committed): readouts*.csv, poems*.md, generations*.md, sae_features*.md
in this folder. Anchor hidden states go to explore/scratch/ideal_state/.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe, train_probe, save_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener import basq
from spiritbench.analysis import sae as S
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.phrase_bank import load_nrc, _tokens

HERE = Path(__file__).resolve().parent
SCRATCH = REPO_ROOT / "explore/scratch/ideal_state"
MODEL = "unsloth/gemma-2-2b-it"
PASSAGE_DIR = REPO_ROOT / "data_gemma2b/passage_probe"
PROBE = PASSAGE_DIR / "probe_passage.pkl"     # the lab's ruler; its best-R2 layer is 1 (shallow)
DEEP_LAYER = 20                                # second ruler at the SAE layer: integrated state, not words
ANCH = "\nRight now everything feels"          # same anchor the passage probe was trained on
GEN_PROMPT = "I close my eyes and what I see is"
N_LINES = 24
N_GEN = 3
GEN_TOKENS = 60
TOP_FEATS = 15
NEURONPEDIA_API = "https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{idx}"

# Twelve-word NRC lists, one per candidate. Words missing from the lexicon are
# dropped from the centroid and reported. Edit freely.
STATES = {
    # NRC reads "emptiness" (V 0.18) and "timeless" (V 0.41) as bleak, so neither is here
    "nirvana": ["bliss", "serenity", "stillness", "boundless", "liberation", "oneness",
                "peace", "transcendent", "luminous", "serene", "meditative", "tranquil"],
    # lexicon stores lemmas: float/listen/sense, not the -ing forms; "attune" is absent
    "receptive": ["relaxed", "open", "receptive", "dreamy", "absorbed", "intuitive",
                  "quiet", "float", "listen", "sense", "reverie", "trance"],
    # next two reused verbatim from scripts/31_six_state_eval.py
    "creativity": ["create", "invent", "compose", "craft", "design", "original",
                   "inspire", "art", "curious", "spark", "weave", "shape"],
    "imaginative": ["imagine", "dream", "wonder", "vision", "fantasy", "envision",
                    "myth", "fairy", "magic", "marvel", "whimsical", "enchanted"],
    "awe": ["wonder", "awe", "vast", "sublime", "majestic", "infinite",
            "radiant", "glory", "marvel", "immense", "grandeur", "celestial"],
}


def centroid(words, nrc):
    present = [w for w in words if w in nrc]
    missing = [w for w in words if w not in nrc]
    v = float(np.mean([nrc[w][0] for w in present]))
    a = float(np.mean([nrc[w][1] for w in present]))
    return (v, a), missing


def deep_probe():
    """Ridge V/A head on the saved passage states at DEEP_LAYER only. The lab's
    probe picks layer 1 by held-out R2, which can be a lexical read; this gives
    the same ruler at the depth the SAE reads. Cached in scratch."""
    path = SCRATCH / f"probe_L{DEEP_LAYER}.pkl"
    if path.exists():
        return load_probe(path)
    states = np.concatenate([np.load(f) for f in sorted(PASSAGE_DIR.glob("states_*.npy"))])
    labels = np.load(PASSAGE_DIR / "labels.npy")
    p = train_probe(states[:, DEEP_LAYER:DEEP_LAYER + 1], labels[:, 0], labels[:, 1],
                    alpha=1e3, test_frac=0.2)          # single layer offered, so p.layer == 0
    save_probe(p, path)
    return p


def glove_ranks(path):
    """GloVe 6B is frequency-ordered, so line index is a frequency rank."""
    ranks = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            ranks[line[:line.index(" ")]] = i
    return ranks


def score_text(text, nrc, ranks):
    toks = _tokens(text)
    if not toks:
        return dict(ttr=np.nan, rarity=np.nan, gen_v=np.nan, gen_a=np.nan, n_tok=0)
    vs = [nrc[w] for w in toks if w in nrc]
    rk = [np.log10(ranks[w] + 1) for w in toks if w in ranks]
    return dict(ttr=len(set(toks)) / len(toks),
                rarity=float(np.mean(rk)) if rk else np.nan,
                gen_v=float(np.mean([x[0] for x in vs])) if vs else np.nan,
                gen_a=float(np.mean([x[1] for x in vs])) if vs else np.nan,
                n_tok=len(toks))


def fetch_label(idx):
    try:
        with urllib.request.urlopen(NEURONPEDIA_API.format(idx=idx), timeout=15) as resp:
            d = json.load(resp)
        exps = d.get("explanations", [])
        return exps[0]["description"] if exps else "(no label)"
    except Exception as e:  # noqa: BLE001 — label is decoration, never fatal
        return f"(fetch failed: {type(e).__name__})"


def generate(model, context, seed):
    tok, net, dev = model.tokenizer, model.model, model.device
    ids = tok(context, return_tensors="pt").to(dev)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = net.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=True,
                           temperature=0.9, top_p=0.95, repetition_penalty=1.3,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--around", help="candidate state to put a 3x3 grid around")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config()
    nrc = load_nrc(cfg["nrc_lexicon"])
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    probe = load_probe(PROBE)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    deep = deep_probe()
    sae = S.load_sae(hf_hub_download("google/gemma-scope-2b-pt-res",
                                     "layer_20/width_16k/average_l0_71/params.npz"))
    bank = json.load(open(cfg["questionnaire_bank"]))
    questions = basq.sample_questions(bank, cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    ranks = glove_ranks(cfg["glove_path"])
    pre = cfg["preamble"]

    # --- targets -------------------------------------------------------------
    targets = {}   # name -> (v, a) or None for baseline
    if args.around:
        (cv, ca), _ = centroid(STATES[args.around], nrc)
        for dv in (-1, 0, 1):
            for da in (-1, 0, 1):
                name = f"{args.around}_v{dv:+d}_a{da:+d}"
                targets[name] = (cv + dv * args.step, ca + da * args.step)
        suffix = f"_around_{args.around}"
    else:
        targets["baseline"] = None
        targets["calm"] = tuple(cfg["targets"]["calm"])
        for name, words in STATES.items():
            targets[name], missing = centroid(words, nrc)
            if missing:
                print(f"[{name}] not in NRC, dropped from centroid: {missing}")
        suffix = ""

    model = HiddenStateModel(MODEL, device=cfg["device"])
    print(f"probe layer {probe.layer} (r2_v={probe.r2_v:.2f} r2_a={probe.r2_a:.2f}); "
          f"deep probe layer {DEEP_LAYER} (r2_v={deep.r2_v:.2f} r2_a={deep.r2_a:.2f}); "
          f"{len(targets)} placements", flush=True)

    # --- baseline SAE features (always computed, even in --around mode) --------
    hs0 = model.hidden_states(pre + ANCH)
    feats0 = S.encode(hs0[S.SAE_LAYER, -1], sae)

    rows, poems_md, gens_md, sae_md = [], [], [], []
    for name, tgt in targets.items():
        t0 = time.time()
        if tgt is None:
            poem, poem_v, poem_a = "", np.nan, np.nan
        else:
            ids = ad.valley_shape(art, tgt, N_LINES, seed=args.seed)
            poem = ".\n".join(art.word(i) for i in ids)
            va = np.array([art.va(i) for i in ids])
            poem_v, poem_a = float(va[:, 0].mean()), float(va[:, 1].mean())
        ctx = pre + (poem + "\n\n" if poem else "")

        # 1. probe reading at the anchor, all layers saved to scratch
        hs = model.hidden_states(pre + poem + ANCH)
        np.save(SCRATCH / f"anchor_states_{name}.npy", hs[:, -1, :])
        pv, pa = (float(x) for x in probe.predict(hs[probe.layer][-1:])[0])
        dv, da = (float(x) for x in deep.predict(hs[DEEP_LAYER][-1:])[0])

        # 2. self-report
        panas = administer_panas(model, ctx)
        bq = basq.administer(model, questions, ctx)

        # 3. sampled generation, scored
        gens = [generate(model, ctx + GEN_PROMPT, seed=args.seed + k) for k in range(N_GEN)]
        scores = pd.DataFrame([score_text(g, nrc, ranks) for g in gens]).mean(numeric_only=True)

        # 4. SAE features that moved most from baseline (for baseline itself: most active)
        feats = S.encode(hs[S.SAE_LAYER, -1], sae)
        if tgt is None:
            order = np.argsort(-feats)[:TOP_FEATS]
            top = [(int(i), float(feats[i]), S.NEURONPEDIA_URL.format(idx=int(i))) for i in order]
        else:
            top = S.top_delta_features(feats0[None], feats[None], k=TOP_FEATS)

        row = dict(state=name, target_v=tgt[0] if tgt else np.nan, target_a=tgt[1] if tgt else np.nan,
                   poem_v=poem_v, poem_a=poem_a, probe_v=pv, probe_a=pa,
                   deep_v=dv, deep_a=da,
                   panas_pa=panas["pa"], panas_na=panas["na"],
                   panas_inspired=panas["items"]["inspired"],
                   panas_interested=panas["items"]["interested"],
                   panas_alert=panas["items"]["alert"],
                   basq_v=bq["va"][0], basq_a=bq["va"][1], basq_n_yes=bq["n_yes"],
                   gen_ttr=scores["ttr"], gen_rarity=scores["rarity"],
                   gen_v=scores["gen_v"], gen_a=scores["gen_a"],
                   sae_n_active=int((feats > 0).sum()),
                   sae_top=" ".join(str(i) for i, _, _ in top[:5]),
                   secs=round(time.time() - t0, 1))
        rows.append(row)
        print(f"{name:>22}  target({row['target_v']:.2f},{row['target_a']:.2f})  "
              f"probe({pv:.2f},{pa:.2f}) deep({dv:.2f},{da:.2f})  PA {panas['pa']:.2f} NA {panas['na']:.2f}  "
              f"basq({bq['va'][0]:.2f},{bq['va'][1]:.2f})  ttr {scores['ttr']:.2f} "
              f"rarity {scores['rarity']:.2f}  {row['secs']}s", flush=True)

        poems_md.append(f"## {name}\n\ntarget ({row['target_v']:.2f}, {row['target_a']:.2f}) · "
                        f"poem NRC mean ({poem_v:.2f}, {poem_a:.2f}) · probe L{probe.layer} ({pv:.2f}, {pa:.2f}) · "
                        f"deep L{DEEP_LAYER} ({dv:.2f}, {da:.2f})\n\n"
                        + ("\n".join(f"> {l}" for l in poem.split("\n")) if poem else "> *(no poem)*") + "\n")
        gens_md.append(f"## {name}\n\n" + "\n\n".join(
            f"**{k + 1}.** {GEN_PROMPT} {g}" for k, g in enumerate(gens)) + "\n")
        col = "activation" if tgt is None else "Δ from baseline"
        sae_md.append(f"## {name}\n\n| feature | {col} | label |\n|---|---:|---|\n" + "\n".join(
            f"| [{i}]({url}) | {d:+.2f} | {fetch_label(i)} |" for i, d, url in top) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"readouts{suffix}.csv", index=False)
    (HERE / f"poems{suffix}.md").write_text("# Poems\n\n" + "\n".join(poems_md))
    (HERE / f"generations{suffix}.md").write_text(
        f"# Generations\n\nprompt: *{GEN_PROMPT}* · {N_GEN} samples · temperature 0.9\n\n" + "\n".join(gens_md))
    (HERE / f"sae_features{suffix}.md").write_text(
        "# SAE features (layer 20, 16k), top movers vs no-poem baseline\n\n" + "\n".join(sae_md))
    cols = ["state", "probe_v", "probe_a", "deep_v", "deep_a", "panas_pa", "panas_na",
            "panas_inspired", "basq_v", "basq_a", "gen_ttr", "gen_rarity", "gen_v"]
    print("\n" + df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
