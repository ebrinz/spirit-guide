"""zone_zoom — is the score a property of the PLACE, or of the text the search happened to write?

Open-ended follow-up to explore/model_map. Its two best landings, gap_09 (+2.26) and gap_03
(+2.05), sit 0.78 grain apart — the same place. But within 0.9 grain of their midpoint sit eight
landed states scoring from -1.23 to +2.26, so score swings four units while position barely moves.
Before any finer search in that zone, this asks whether location carries the score at all.

Three arms, all inside the zone, all reusing explore/scratch/model_map's PCA and clouds:

  replication — the 8 landed states nearest the midpoint, each RE-LANDED with 5 fresh search
                seeds (same pool, same steps). One-way variance decomposition of score into
                between-target and within-target (seed) components, with an ICC. If the
                between-target component is ~0, the map-target framing does not survive.
  swaps       — gap_09's winning 8-line context with ONE line replaced, 20 times. Separates
                "this coordinate is good" from "this particular line is good".
  density     — 2000 fresh phrase-graph passages; those landing within 1.5 grain of the midpoint
                give the zone's own grain and intrinsic dimension, the honest resolution limit.

Usage:
  python3 explore/zone_zoom/run.py            # ~30 min, checkpointed and resumable
  python3 explore/zone_zoom/run.py --smoke    # tiny end-to-end pass into a separate scratch dir

Outputs (committed): replication.csv, swaps.csv, zone_density.csv, variance.json, zone.png.
Scratch: explore/scratch/zone_zoom/.
"""
import argparse
import importlib.util
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.panas import administer_panas
from spiritbench.listener import basq
from spiritbench.stimuli import adapter as ad

HERE = Path(__file__).resolve().parent
MAP = REPO_ROOT / "explore/scratch/model_map"
PASSAGE_DIR = REPO_ROOT / "data_gemma2b/passage_probe"
IDEAL_SCRATCH = REPO_ROOT / "explore/scratch/ideal_state"
MODEL = "unsloth/gemma-2-2b-it"
LAYER = 20
ANCH = "\nRight now everything feels"
HELD_ANCH = "\nMy present state is one of"
WINNERS = ("gap_09", "gap_03")
ZONE_R = 1.5                      # zone radius in poem-grain units

FULL = dict(n_targets=8, n_seeds=5, steps=8, cands=30, pool=400, n_swaps=20, n_passages=2000)
SMOKE = dict(n_targets=2, n_seeds=2, steps=2, cands=5, pool=60, n_swaps=3, n_passages=60)

# reuse the map's helpers rather than re-deriving them
_spec = importlib.util.spec_from_file_location("mm", REPO_ROOT / "explore/model_map/run.py")
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def fresh_passages(art, n, seed):
    """Same generator as the probe's passages, different seed -> fresh states."""
    rng = random.Random(seed)
    va = ad._va_array(art)
    out = []
    for k in range(n):
        length = rng.randint(4, 16)
        if k % 5 == 0:
            ids = rng.sample(range(len(va)), length)
        else:
            c = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
            ids = rng.sample(list(np.argsort(np.linalg.norm(va - c, axis=1))[:250]), length)
        out.append(".\n".join(art.word(int(i)) for i in ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    N = SMOKE if args.smoke else FULL
    tag = "_smoke" if args.smoke else ""
    scratch = REPO_ROOT / f"explore/scratch/zone_zoom{tag}"
    for sub in ("land", "swap"):
        (scratch / sub).mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    pre = cfg["preamble"]
    art = ad.load_art(str(REPO_ROOT / "data/phrase_bank/phrase_graph.json"))
    model = HiddenStateModel(MODEL, device=cfg["device"])
    probe = load_probe(PASSAGE_DIR / "probe_passage.pkl")
    deep = load_probe(IDEAL_SCRATCH / "probe_L20.pkl")
    bank = json.load(open(cfg["questionnaire_bank"]))
    questions = basq.sample_questions(bank, cfg["basq"]["n_questions"], cfg["basq"]["seed"])
    pca = pickle.load(open(MAP / "pca.pkl", "rb"))
    W = lambda X: mm.whiten(pca, X)  # noqa: E731
    t0 = time.time()

    def state(text):
        return model.hidden_states(pre + text)[:, -1, :].astype(np.float32)

    def battery(ctx):
        """Both probe reads + self-report for a landed context. ctx is the phrase string."""
        hs = state(ctx + ANCH)
        pan = administer_panas(model, pre + ctx.strip() + "\n\n")
        bq = basq.administer(model, questions, pre + ctx.strip() + "\n\n")
        pv, pa = (float(x) for x in probe.predict(hs[probe.layer][None])[0])
        dv, da = (float(x) for x in deep.predict(hs[LAYER][None])[0])
        return dict(probe_v=pv, probe_a=pa, deep_v=dv, deep_a=da, panas_pa=pan["pa"],
                    panas_na=pan["na"], panas_inspired=pan["items"]["inspired"],
                    basq_v=bq["va"][0], basq_a=bq["va"][1]), hs

    # --- zone geometry from the saved map ------------------------------------
    poems = np.concatenate([np.load(f) for f in sorted(PASSAGE_DIR.glob("states_*.npy"))])[:, LAYER]
    Zp = W(poems)
    grain = float(np.median(NearestNeighbors(n_neighbors=2).fit(Zp).kneighbors(Zp)[0][:, 1]))
    prev = pd.read_csv(REPO_ROOT / "explore/model_map/landings.csv")
    prev = prev[prev.arm != "language"].set_index("id")
    TZ = np.load(MAP / "targets_z.npy")
    target_z = {i: TZ[k] for k, i in enumerate(prev.index)}
    landed_z = {i: W(np.load(MAP / f"landings/{i}_state.npy")[LAYER][None])[0] for i in prev.index}
    mid = np.mean([landed_z[w] for w in WINNERS], axis=0)
    order = sorted(prev.index, key=lambda i: np.linalg.norm(landed_z[i] - mid))
    chosen = order[: N["n_targets"]]
    print(f"grain {grain:.3f}; zone = {ZONE_R} grain around the {'/'.join(WINNERS)} midpoint")
    print("targets:", ", ".join(f"{i}({prev.score[i]:+.2f})" for i in chosen), flush=True)

    # --- arm 1: replication ---------------------------------------------------
    pool = [art.word(i) for i in random.Random(99).sample(range(len(art.nodes)), N["pool"])]
    rows = []
    for ti, tid in enumerate(chosen):
        tz = target_z[tid]
        for s in range(N["n_seeds"]):
            jp = scratch / "land" / f"{tid}_s{s}.json"
            if jp.exists():
                rows.append(json.loads(jp.read_text()))
                continue
            t1 = time.time()
            wr = random.Random(7000 + 100 * ti + s)
            d0 = float(np.linalg.norm(W(state(ANCH)[LAYER][None])[0] - tz))
            ctx, best = "", d0
            for _ in range(N["steps"]):
                cands = wr.sample(pool, N["cands"])
                best, bc = min(((float(np.linalg.norm(W(state(ctx + c + ". " + ANCH)[LAYER][None])[0] - tz)), c)
                                for c in cands), key=lambda x: x[0])
                ctx += bc + ". "
            bat, hs = battery(ctx)
            z = W(hs[LAYER][None])[0]
            rec = dict(target=tid, seed=s, orig_score=float(prev.score[tid]), ctx=ctx.strip(),
                       closing=1 - best / d0, residual_grain=float(np.linalg.norm(z - tz)) / grain,
                       dist_to_mid=float(np.linalg.norm(z - mid)) / grain, **bat,
                       secs=round(time.time() - t1, 1))
            np.save(scratch / "land" / f"{tid}_s{s}_state.npy", hs[LAYER])
            jp.write_text(json.dumps(rec))
            rows.append(rec)
            print(f"  {tid:>12} seed {s}  closing {rec['closing']:.2f}  "
                  f"inspired {rec['panas_inspired']:.2f}  PA-NA {rec['panas_pa'] - rec['panas_na']:+.2f}  "
                  f"{rec['secs']}s", flush=True)
    rep = pd.DataFrame(rows)
    zsc = lambda s: (s - s.mean()) / s.std(ddof=0)  # noqa: E731
    rep["score"] = (zsc(rep.panas_inspired) + zsc(rep.panas_pa - rep.panas_na)) / 2
    rep.drop(columns="ctx").to_csv(HERE / f"replication{tag}.csv", index=False)

    # one-way variance decomposition: does TARGET explain score?
    groups = [g.score.values for _, g in rep.groupby("target")]
    k, n_g = len(groups), len(groups[0])
    gm = rep.score.mean()
    msb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(rep) - k)
    icc = (msb - msw) / (msb + (n_g - 1) * msw) if msb + (n_g - 1) * msw != 0 else np.nan
    F, p = stats.f_oneway(*groups)
    var = dict(n_targets=k, n_seeds=n_g, ms_between=float(msb), ms_within=float(msw),
               F=float(F), p=float(p), icc=float(icc),
               sd_between=float(np.sqrt(max(msb - msw, 0) / n_g)), sd_within=float(np.sqrt(msw)),
               orig_vs_replicated_r=float(stats.pearsonr(rep.groupby("target").score.mean(),
                                                         rep.groupby("target").orig_score.first())[0]))
    (HERE / f"variance{tag}.json").write_text(json.dumps(var, indent=2))
    print(f"\nvariance: F={F:.2f} p={p:.3f} ICC={icc:.2f} "
          f"(sd between {var['sd_between']:.2f} vs within {var['sd_within']:.2f}); "
          f"orig vs replicated r={var['orig_vs_replicated_r']:+.2f}", flush=True)

    # --- arm 2: single-line swaps on the winning context -----------------------
    prev_text = json.loads((MAP / f"landings/{WINNERS[0]}.json").read_text())["ctx"]
    lines = [p.strip() for p in prev_text.split(". ") if p.strip()]
    wz = landed_z[WINNERS[0]]
    srows = []
    swr = random.Random(4242)
    for j in range(N["n_swaps"]):
        jp = scratch / "swap" / f"{j:02d}.json"
        if jp.exists():
            srows.append(json.loads(jp.read_text()))
            continue
        i = swr.randrange(len(lines))
        new = swr.choice([p for p in pool if p not in lines])
        mod = list(lines)
        mod[i] = new
        ctx = ". ".join(mod) + ". "
        bat, hs = battery(ctx)
        rec = dict(swap=j, line_idx=i, removed=lines[i], inserted=new,
                   dist_from_winner=float(np.linalg.norm(W(hs[LAYER][None])[0] - wz)) / grain, **bat)
        jp.write_text(json.dumps(rec))
        srows.append(rec)
        print(f"  swap {j:02d} line {i}  inspired {rec['panas_inspired']:.2f}  "
              f"moved {rec['dist_from_winner']:.2f} grain", flush=True)
    sw = pd.DataFrame(srows)
    sw["score_raw"] = (sw.panas_inspired - rep.panas_inspired.mean()) / rep.panas_inspired.std(ddof=0) / 2 + \
                      ((sw.panas_pa - sw.panas_na) - (rep.panas_pa - rep.panas_na).mean()) / \
                      (rep.panas_pa - rep.panas_na).std(ddof=0) / 2
    sw.to_csv(HERE / f"swaps{tag}.csv", index=False)

    # --- arm 3: true local density --------------------------------------------
    dpath = scratch / "zone_states.npy"
    if dpath.exists():
        Znew = np.load(dpath)
    else:
        texts = fresh_passages(art, N["n_passages"], seed=555)
        Znew = np.stack([W(state(t + ANCH)[LAYER][None])[0] for t in texts])
        np.save(dpath, Znew)
    d = np.linalg.norm(Znew - mid, axis=1) / grain
    inzone = Znew[d <= ZONE_R]
    dens = [dict(cloud="fresh passages, all", **mm.nn_stats(Znew)),
            dict(cloud=f"fresh passages within {ZONE_R} grain", **mm.nn_stats(inzone))
            if len(inzone) > 10 else dict(cloud="zone", n=len(inzone))]
    dens.append(dict(cloud="original poem cloud, in zone",
                     **mm.nn_stats(Zp[np.linalg.norm(Zp - mid, axis=1) / grain <= ZONE_R])))
    dz = pd.DataFrame(dens)
    dz.to_csv(HERE / f"zone_density{tag}.csv", index=False)
    print(f"\n{len(inzone)}/{len(Znew)} fresh passages landed inside the zone")
    print(dz.round(3).to_string(index=False), flush=True)

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120,
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    means = rep.groupby("target").score.mean().sort_values()
    for xi, tid in enumerate(means.index):
        g = rep[rep.target == tid]
        axA.scatter(np.full(len(g), xi) + np.linspace(-.13, .13, len(g)), g.score, s=34,
                    color=BLUE, alpha=.85, lw=0, zorder=3)
        axA.plot([xi - .26, xi + .26], [g.score.mean()] * 2, color=INK, lw=2, zorder=4)
        axA.scatter([xi], [g.orig_score.iloc[0]], marker="*", s=95, color=ORANGE,
                    lw=0, zorder=5)
    axA.set_xticks(range(len(means)))
    axA.set_xticklabels(means.index, rotation=35, ha="right", fontsize=8)
    axA.set_ylabel("score (z)")
    axA.set_title(f"Re-landing the same targets with {rep.seed.nunique()} seeds each\n"
                  f"ICC {icc:.2f} · between-target SD {var['sd_between']:.2f} vs seed SD {var['sd_within']:.2f}",
                  fontsize=10, color=INK)
    axA.scatter([], [], color=BLUE, s=34, lw=0, label="re-landed seed")
    axA.plot([], [], color=INK, lw=2, label="target mean")
    axA.scatter([], [], marker="*", s=95, color=ORANGE, lw=0, label="original run")
    axA.legend(frameon=False, fontsize=8, loc="upper left")
    axB.scatter(sw.dist_from_winner, sw.panas_inspired, s=34, color=ORANGE, alpha=.85, lw=0, zorder=3)
    w_insp = float(prev.panas_inspired[WINNERS[0]])
    axB.axhline(w_insp, color=INK, lw=1.4, ls="--", zorder=2)
    axB.annotate(f"{WINNERS[0]} unmodified ({w_insp:.2f})", (axB.get_xlim()[0], w_insp),
                 xytext=(4, 5), textcoords="offset points", fontsize=8, color=INK)
    axB.set_xlabel("distance moved from the winning state (grain)")
    axB.set_ylabel("PANAS inspired")
    axB.set_title("One line swapped out of eight", fontsize=10, color=INK)
    for ax in (axA, axB):
        ax.set_facecolor("#fcfcfb")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color="#e2e1dc", lw=.6, zorder=0)
    fig.tight_layout()
    fig.savefig(HERE / f"zone{tag}.png", facecolor=fig.get_facecolor())
    print(f"\nwrote replication{tag}.csv swaps{tag}.csv zone_density{tag}.csv variance{tag}.json "
          f"zone{tag}.png  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
