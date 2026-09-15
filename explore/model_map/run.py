"""model_map — pick target states from the model's own geometry, not from emotion words.

Open-ended. Four clouds of layer-20 anchor states, all read through the identical pathway
(preamble + text + anchor): the 1200 poem passages already on disk, 1200 English Wikipedia
paragraphs and 1200 contemplative-prose paragraphs from the channel-x corpus (both trimmed
to the poems' length range), and 100 random-token strings. PCA to 32 components, whitened,
fitted on all four. Targets come from the map: gaps (circumcenters of the largest Delaunay
simplices of the poem cloud in the first 6 whitened components, lifted to 32-d), hubs
(k-means centroids of the poem cloud), unpoemed points (prose states farthest from any poem
state, one arm per prose cloud), and random points inside the poem hull as the control.
Each target is landed blindly with the lab's E19 greedy phrase search (objective: whitened
distance to the target), with the E19 drift control and held-out-anchor transfer check, then
read by the self-report battery from explore/ideal_state (probes, PANAS, yes/no bank). The
six language-arm poems from ideal_state join the comparison with their existing scores.

Granularity is measured up front (nearest-neighbour grain, local spacing per target,
intrinsic dimension per cloud) so a later zoom into one zone can be cast at the right scale.

Usage:
  python3 explore/model_map/run.py            # full run, ~1.5 h, checkpointed and resumable
  python3 explore/model_map/run.py --smoke    # tiny end-to-end pass into a separate scratch dir

Outputs (committed): granularity.csv, targets.csv, landings.csv, poems.md, sae_features.md,
map.png. Scratch: explore/scratch/model_map/ (clouds, pca, per-target landings and states).
"""
import argparse
import json
import os
import pickle
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener import basq
from spiritbench.analysis import sae as S
from spiritbench.stimuli import adapter as ad

HERE = Path(__file__).resolve().parent
PASSAGE_DIR = REPO_ROOT / "data_gemma2b/passage_probe"
IDEAL = REPO_ROOT / "explore/ideal_state"
IDEAL_SCRATCH = REPO_ROOT / "explore/scratch/ideal_state"
CHANNELX = Path(os.environ.get("CHANNELX_DATA", REPO_ROOT.parent / "channel-x/data"))
MODEL = "unsloth/gemma-2-2b-it"
LAYER = 20                                    # map layer = SAE layer = deep-probe layer
ANCH = "\nRight now everything feels"        # train anchor (passage probe pathway)
HELD_ANCH = "\nMy present state is one of"   # E19 held-out anchor
N_PCS = 32
N_TRI = 6                                     # Delaunay dimension (feasible to 6; 8 takes minutes)
NEURONPEDIA_API = "https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{idx}"
LANGUAGE_STATES = ["calm", "nirvana", "receptive", "creativity", "imaginative", "awe"]

FULL = dict(n_prose=1200, n_noise=100, gaps=16, hubs=12, unpoemed=12, rand=12,
            steps=8, cands=30, pool=400, sae_top=3)
SMOKE = dict(n_prose=40, n_noise=10, gaps=2, hubs=2, unpoemed=2, rand=2,
             steps=2, cands=5, pool=60, sae_top=1)


# ----------------------------------------------------------------------------- texts
def passage_word_counts(art, n=1200, seed=13):
    """Word counts of the passages the probe was built on (same generator, same seed)."""
    rng = random.Random(seed)
    va = ad._va_array(art)
    out = []
    for k in range(n):
        length = rng.randint(4, 16)
        if k % 5 == 0:
            ids = rng.sample(range(len(va)), length)
        else:
            c = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
            pool = list(np.argsort(np.linalg.norm(va - c, axis=1))[:250])
            ids = rng.sample(pool, min(length, len(pool)))
        out.append(sum(len(art.word(int(i)).split()) for i in ids))
    return out


def prose_texts(jsonl, n, lengths, rng, skip=()):
    rows = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d["text"].strip()
            if d.get("word_count", 0) < 60 or any(s in t for s in skip):
                continue
            rows.append(t)
    picks = rng.sample(rows, n)
    return [" ".join(t.split()[:rng.choice(lengths)]) for t in picks]


def noise_texts(tok, n, lengths, rng):
    vocab = tok.vocab_size
    out = []
    for _ in range(n):
        ids = [rng.randrange(1000, vocab) for _ in range(int(rng.choice(lengths) * 1.3))]
        out.append(tok.decode(ids, skip_special_tokens=True))
    return out


# ----------------------------------------------------------------------------- map
def whiten(pca, X):
    return pca.transform(X) / np.sqrt(pca.explained_variance_)


def two_nn_dimension(Z):
    """Facco et al. 2017 TwoNN intrinsic-dimension estimate."""
    d, _ = NearestNeighbors(n_neighbors=3).fit(Z).kneighbors(Z)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = np.sort(mu)[: int(0.9 * len(mu))]
    F = np.arange(1, len(mu) + 1) / len(Z)
    x, y = np.log(mu), -np.log(1 - F)
    return float((x @ y) / (x @ x))


def nn_stats(Z):
    d, _ = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
    nn = d[:, 1]
    rms = float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()))
    return dict(n=len(Z), nn_median=float(np.median(nn)), nn_p10=float(np.percentile(nn, 10)),
                nn_p90=float(np.percentile(nn, 90)), rms_radius=rms,
                grain=float(np.median(nn) / rms), intrinsic_dim=two_nn_dimension(Z))


def circumcenters(P):
    """P: [m, d+1, d] simplex vertices -> centers [m, d], radii [m], |det| [m]."""
    A = 2 * (P[:, 1:, :] - P[:, :1, :])
    b = (P[:, 1:, :] ** 2).sum(-1) - (P[:, :1, :] ** 2).sum(-1)
    det = np.abs(np.linalg.det(A))
    ok = det > 1e-9
    c = np.full((len(P), P.shape[2]), np.nan)
    c[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    r = np.linalg.norm(c - P[:, 0, :], axis=1)
    return c, r, det


def greedy_separated(cands, scores, k, min_sep):
    """Take the top-k by score with pairwise separation >= min_sep."""
    out = []
    for i in np.argsort(-scores):
        if all(np.linalg.norm(cands[i] - cands[j]) >= min_sep for j in out):
            out.append(i)
        if len(out) == k:
            break
    return out


def make_targets(Zp, Zw, Zc, cfg_n, rng):
    """Gaps / hubs / unpoemed / random, all as 32-d whitened coordinates."""
    grain32 = nn_stats(Zp)["nn_median"]
    Z6 = Zp[:, :N_TRI]
    grain6 = nn_stats(Z6)["nn_median"]
    tri = Delaunay(Z6)
    P = Z6[tri.simplices]
    cent, rad, det = circumcenters(P)
    inside = np.zeros(len(cent), bool)
    ok = ~np.isnan(cent).any(1)
    inside[ok] = tri.find_simplex(cent[ok]) >= 0
    T = []
    # gaps: largest circumradius among centers inside the hull, well separated
    cand = np.where(inside)[0]
    pick = greedy_separated(cent[cand], rad[cand], cfg_n["gaps"], 2 * grain6)
    for j, i in enumerate(cand[pick]):
        lift = np.concatenate([cent[i], Zp[tri.simplices[i], N_TRI:].mean(0)])
        T.append(dict(id=f"gap_{j:02d}", arm="gap", z=lift, circumradius=float(rad[i] / grain6)))
    # hubs: k-means centroids in 32-d
    km = KMeans(n_clusters=cfg_n["hubs"], random_state=0, n_init=10).fit(Zp)
    for j, c in enumerate(km.cluster_centers_):
        T.append(dict(id=f"hub_{j:02d}", arm="hub", z=c, circumradius=np.nan))
    # unpoemed: prose states farthest from any poem state
    nnp = NearestNeighbors(n_neighbors=1).fit(Zp)
    for arm, Zx in (("unpoemed_wiki", Zw), ("unpoemed_contemplative", Zc)):
        d = nnp.kneighbors(Zx)[0][:, 0]
        for j, i in enumerate(greedy_separated(Zx, d, cfg_n["unpoemed"], 2 * grain32)):
            T.append(dict(id=f"{arm}_{j:02d}", arm=arm, z=Zx[i], circumradius=np.nan))
    # random inside the poem hull: simplex chosen by volume, Dirichlet barycentric weights
    w = det * inside
    w = w / w.sum()
    for j in range(cfg_n["rand"]):
        s = rng.choice(len(w), p=w)
        bary = rng.dirichlet(np.ones(N_TRI + 1))
        T.append(dict(id=f"random_{j:02d}", arm="random", z=bary @ Zp[tri.simplices[s]],
                      circumradius=np.nan))
    # local granularity per target
    d10 = NearestNeighbors(n_neighbors=10).fit(Zp)
    for t in T:
        d, _ = d10.kneighbors(t["z"][None])
        t["local_spacing"] = float(d[0, -1] / grain32)
        t["n_within_grain"] = int((d[0] <= grain32).sum())
    return T, tri, grain32


# ----------------------------------------------------------------------------- landing
def fetch_label(idx):
    try:
        with urllib.request.urlopen(NEURONPEDIA_API.format(idx=idx), timeout=15) as resp:
            d = json.load(resp)
        exps = d.get("explanations", [])
        return exps[0]["description"] if exps else "(no label)"
    except Exception as e:  # noqa: BLE001
        return f"(fetch failed: {type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    N = SMOKE if args.smoke else FULL
    tag = "_smoke" if args.smoke else ""
    scratch = REPO_ROOT / f"explore/scratch/model_map{tag}"
    for sub in ("clouds", "landings", "battery"):
        (scratch / sub).mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    pre = cfg["preamble"]
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(MODEL, device=cfg["device"])
    tok = model.tokenizer
    rng = random.Random(2026)

    def state(text):
        """All-layer last-token state [n_layers, d] for preamble + text."""
        return model.hidden_states(pre + text)[:, -1, :].astype(np.float32)

    # --- 1. clouds -----------------------------------------------------------
    t0 = time.time()
    poems = np.concatenate([np.load(f) for f in sorted(PASSAGE_DIR.glob("states_*.npy"))])[:, LAYER]
    lengths = passage_word_counts(art)

    def collect(name, texts):
        path = scratch / "clouds" / f"{name}.npy"
        if path.exists():
            return np.load(path)
        (scratch / "clouds" / f"{name}_texts.json").write_text(json.dumps(texts))
        X = np.stack([state(t + ANCH)[LAYER] for t in texts])
        np.save(path, X)
        print(f"cloud {name}: {len(X)} states in {time.time() - t0:.0f}s", flush=True)
        return X

    wiki = collect("wiki", prose_texts(CHANNELX / "wikipedia/chunks_wikipedia_body.jsonl",
                                        N["n_prose"], lengths, rng))
    contemplative = collect("contemplative", prose_texts(
        CHANNELX / "control/chunks_control_body.jsonl", N["n_prose"], lengths, rng,
        skip=("Transcriber", "Gutenberg", "Project", "eBook")))
    noise = collect("noise", noise_texts(tok, N["n_noise"], lengths, rng))

    # --- 2. map --------------------------------------------------------------
    ppath = scratch / "pca.pkl"
    if ppath.exists():
        pca = pickle.load(open(ppath, "rb"))
    else:
        pca = PCA(n_components=N_PCS, random_state=0).fit(np.concatenate([poems, wiki, contemplative, noise]))
        pickle.dump(pca, open(ppath, "wb"))
    Z = {k: whiten(pca, v) for k, v in dict(poems=poems, wiki=wiki, contemplative=contemplative, noise=noise).items()}
    print(f"PCA-{N_PCS} keeps {pca.explained_variance_ratio_.sum():.2f} of variance", flush=True)

    gran = pd.DataFrame([dict(cloud=k, **nn_stats(v)) for k, v in Z.items()])
    gran.to_csv(HERE / f"granularity{tag}.csv", index=False)
    print(gran.round(3).to_string(index=False), flush=True)

    # --- 3. targets ------------------------------------------------------------
    T, tri, grain32 = make_targets(Z["poems"], Z["wiki"], Z["contemplative"], N, np.random.default_rng(2026))
    np.save(scratch / "targets_z.npy", np.stack([t["z"] for t in T]))
    print(f"{len(T)} targets; poem grain {grain32:.3f} (whitened units)", flush=True)

    # --- 4. landing (E19 search, point objective) ------------------------------
    pool = [art.word(i) for i in random.Random(99).sample(range(len(art.nodes)), N["pool"])]
    base_tr, base_hd = state(ANCH), state(HELD_ANCH)
    z_base_tr, z_base_hd = whiten(pca, base_tr[LAYER][None])[0], whiten(pca, base_hd[LAYER][None])[0]
    sae = S.load_sae(hf_hub_download("google/gemma-scope-2b-pt-res",
                                     "layer_20/width_16k/average_l0_71/params.npz"))
    feats0 = S.encode(base_tr[LAYER], sae)

    def zproj(text, anch):
        hs = state(text + anch)
        return whiten(pca, hs[LAYER][None])[0], hs

    for idx, t in enumerate(T):
        jpath = scratch / "landings" / f"{t['id']}.json"
        if jpath.exists():
            continue
        t1 = time.time()
        tz = t["z"]
        dist = lambda z: float(np.linalg.norm(z - tz))  # noqa: E731
        d_before, d_hd_before = dist(z_base_tr), dist(z_base_hd)
        wr = random.Random(500 + idx)
        ctx, best_d, trail = "", d_before, []
        for _ in range(N["steps"]):
            cands = wr.sample(pool, N["cands"])
            scored = [(dist(zproj(ctx + c + ". ", ANCH)[0]), c) for c in cands]
            best_d, best_c = min(scored, key=lambda s: s[0])
            ctx += best_c + ". "
            trail.append(best_d)
        z_final, hs_final = zproj(ctx, ANCH)
        d_after = dist(z_final)
        d_hd_after = dist(zproj(ctx, HELD_ANCH)[0])
        drifts = []
        for r in range(3):
            dr = random.Random(900 + 10 * idx + r)
            dctx = "".join(c + ". " for c in dr.sample(pool, N["steps"]))
            drifts.append(1 - dist(zproj(dctx, ANCH)[0]) / d_before)
        rec = dict(id=t["id"], arm=t["arm"], ctx=ctx, d_before=d_before, d_after=d_after,
                   closing=1 - d_after / d_before, residual_grain=d_after / grain32,
                   transfer_closing=1 - d_hd_after / d_hd_before, drift_closing=float(np.mean(drifts)),
                   trail=trail, secs=round(time.time() - t1, 1))
        np.save(scratch / "landings" / f"{t['id']}_state.npy", hs_final)
        jpath.write_text(json.dumps(rec))
        print(f"[{idx + 1}/{len(T)}] {t['id']:>26}  closing {rec['closing']:.2f} "
              f"(drift {rec['drift_closing']:.2f}, held-out {rec['transfer_closing']:.2f})  "
              f"residual {rec['residual_grain']:.1f} grain  {rec['secs']}s", flush=True)

    # --- 5. battery ---------------------------------------------------------------
    probe = load_probe(PASSAGE_DIR / "probe_passage.pkl")
    deep = load_probe(IDEAL_SCRATCH / "probe_L20.pkl")
    bank = json.load(open(cfg["questionnaire_bank"]))
    questions = basq.sample_questions(bank, cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    for t in T:
        bpath = scratch / "battery" / f"{t['id']}.json"
        if bpath.exists():
            continue
        rec = json.loads((scratch / "landings" / f"{t['id']}.json").read_text())
        hs = np.load(scratch / "landings" / f"{t['id']}_state.npy")
        ctx_text = pre + rec["ctx"].strip() + "\n\n"
        pv, pa = (float(x) for x in probe.predict(hs[probe.layer][None])[0])
        dv, da = (float(x) for x in deep.predict(hs[LAYER][None])[0])
        pan = administer_panas(model, ctx_text)
        bq = basq.administer(model, questions, ctx_text)
        bpath.write_text(json.dumps(dict(
            probe_v=pv, probe_a=pa, deep_v=dv, deep_a=da, panas_pa=pan["pa"], panas_na=pan["na"],
            panas_inspired=pan["items"]["inspired"], basq_v=bq["va"][0], basq_a=bq["va"][1])))
        print(f"battery {t['id']:>26}  deep({dv:.2f},{da:.2f})  PA {pan['pa']:.2f} NA {pan['na']:.2f} "
              f"inspired {pan['items']['inspired']:.2f}", flush=True)

    # --- 6. assemble: landings + language arm -------------------------------------
    rows = []
    for t in T:
        rec = json.loads((scratch / "landings" / f"{t['id']}.json").read_text())
        bat = json.loads((scratch / "battery" / f"{t['id']}.json").read_text())
        rows.append({**{k: rec[k] for k in ("id", "arm", "closing", "transfer_closing", "drift_closing",
                                            "residual_grain", "d_before")},
                     "local_spacing": t["local_spacing"], "n_within_grain": t["n_within_grain"],
                     "circumradius": t["circumradius"], **bat, "text": rec["ctx"].strip()})
    lang_z = {}
    ideal = pd.read_csv(IDEAL / "readouts.csv").set_index("state")
    for name in LANGUAGE_STATES:
        hs = np.load(IDEAL_SCRATCH / f"anchor_states_{name}.npy")
        lang_z[name] = whiten(pca, hs[LAYER][None])[0]
        r = ideal.loc[name]
        rows.append(dict(id=f"language_{name}", arm="language", closing=np.nan, transfer_closing=np.nan,
                         drift_closing=np.nan, residual_grain=np.nan, d_before=np.nan,
                         local_spacing=np.nan, n_within_grain=np.nan, circumradius=np.nan,
                         **{k: float(r[k]) for k in ("probe_v", "probe_a", "deep_v", "deep_a", "panas_pa",
                                                     "panas_na", "panas_inspired", "basq_v", "basq_a")},
                         text=f"(valley poem, see explore/ideal_state/poems.md#{name})"))
    df = pd.DataFrame(rows)
    z = lambda s: (s - s.mean()) / s.std(ddof=0)  # noqa: E731
    df["score"] = (z(df.panas_inspired) + z(df.panas_pa - df.panas_na)) / 2
    LZ = np.stack(list(lang_z.values()))
    LA = ideal.loc[LANGUAGE_STATES, ["deep_v", "deep_a"]].to_numpy()
    zs = {t["id"]: whiten(pca, np.load(scratch / "landings" / f"{t['id']}_state.npy")[LAYER][None])[0] for t in T}
    df["map_novelty_grain"] = [np.linalg.norm(LZ - zs[i], axis=1).min() / grain32 if i in zs else np.nan for i in df.id]
    df["affect_novelty"] = [np.linalg.norm(LA - np.array([v, a]), axis=1).min() for v, a in zip(df.deep_v, df.deep_a)]
    df.drop(columns="text").to_csv(HERE / f"landings{tag}.csv", index=False)
    pd.DataFrame([{k: v for k, v in t.items() if k != "z"} for t in T]).to_csv(HERE / f"targets{tag}.csv", index=False)

    arm = df.groupby("arm").agg(n=("id", "size"), closing=("closing", "mean"), drift=("drift_closing", "mean"),
                                held_out=("transfer_closing", "mean"), score=("score", "mean"),
                                inspired=("panas_inspired", "mean"), na=("panas_na", "mean"),
                                affect_novelty=("affect_novelty", "mean"), map_novelty=("map_novelty_grain", "mean"))
    print("\n" + arm.round(3).to_string(), flush=True)

    # --- 7. poems.md + SAE for the top landings --------------------------------------
    md = ["# Landed found-poems\n", "Each is the E19 greedy search's phrase sequence for one map target, "
          "read at the same anchor as every other state here.\n"]
    for _, r in df[df.arm != "language"].sort_values("score", ascending=False).iterrows():
        md.append(f"## {r.id}  · score {r.score:+.2f} · closing {r.closing:.2f} (drift {r.drift_closing:.2f}, "
                  f"held-out {r.transfer_closing:.2f}) · deep ({r.deep_v:.2f}, {r.deep_a:.2f}) · "
                  f"PA {r.panas_pa:.2f} NA {r.panas_na:.2f} inspired {r.panas_inspired:.2f}\n\n"
                  + "\n".join(f"> {p.strip()}" for p in r.text.split(". ") if p.strip()) + "\n")
    (HERE / f"poems{tag}.md").write_text("\n".join(md))
    sae_md = ["# SAE features (layer 20, 16k): top movers vs the bare-preamble baseline, best landings by score\n"]
    for _, r in df[df.arm != "language"].sort_values("score", ascending=False).head(N["sae_top"]).iterrows():
        feats = S.encode(np.load(scratch / "landings" / f"{r.id}_state.npy")[LAYER], sae)
        top = S.top_delta_features(feats0[None], feats[None], k=15)
        sae_md.append(f"## {r.id} (score {r.score:+.2f})\n\n| feature | Δ | label |\n|---|---:|---|\n"
                      + "\n".join(f"| [{i}]({u}) | {d:+.1f} | {fetch_label(i)} |" for i, d, u in top) + "\n")
    (HERE / f"sae_features{tag}.md").write_text("\n".join(sae_md))

    # --- 8. map figure --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    tri2 = Delaunay(Z["poems"][:, :2])
    ax.triplot(Z["poems"][:, 0], Z["poems"][:, 1], tri2.simplices, color="#e2e1dc", lw=0.4, zorder=1)
    for name, color, lab in (("noise", "#b8b7b0", "random tokens"), ("wiki", "#eb6834", "Wikipedia prose"),
                             ("contemplative", "#1baf7a", "contemplative prose"), ("poems", "#2a78d6", "poem passages")):
        ax.scatter(Z[name][:, 0], Z[name][:, 1], s=6, color=color, alpha=0.55, lw=0, label=lab, zorder=2)
    marks = dict(gap="x", hub="o", unpoemed_wiki="^", unpoemed_contemplative="v", random="+", language="*")
    TZ = {t["id"]: t["z"] for t in T}
    for a, m in marks.items():
        pts = [lang_z[n] for n in LANGUAGE_STATES] if a == "language" else [TZ[t["id"]] for t in T if t["arm"] == a]
        if pts:
            P2 = np.stack(pts)
            ax.scatter(P2[:, 0], P2[:, 1], marker=m, s=46, color="#0b0b0b", lw=1.1,
                       facecolors="none" if m in "o^v" else "#0b0b0b", label=f"target: {a}", zorder=3)
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title("Layer-20 state clouds, Delaunay of the poem cloud, and map targets", color="#0b0b0b", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(HERE / f"map{tag}.png", facecolor=fig.get_facecolor())
    print(f"\nwrote landings{tag}.csv targets{tag}.csv granularity{tag}.csv poems{tag}.md "
          f"sae_features{tag}.md map{tag}.png  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
