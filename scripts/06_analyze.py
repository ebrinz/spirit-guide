"""Assemble metrics.csv, figures, leaderboard, harmonic predictiveness."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.analysis import metrics as M
from spiritbench.analysis import harmonics as H
from spiritbench.analysis import figures as F
from spiritbench.analysis import covariates as C

COVARIATE_COLS = ["nv_ratio", "and_initial_per_1000", "then_per_1000",
                  "subordinator_per_1000", "noun_shen", "noun_apen", "cv_line_len"]

EXCITED = (0.80, 0.85)


def assemble_metrics(stims, runs, harmonic_ctx) -> pd.DataFrame:
    rows = []
    for s in stims:
        r = runs.get(s["id"])
        if r is None or "error" in r:
            continue
        traj = np.asarray(r["traj"])
        wps = np.asarray([[w["v"], w["a"]] for w in s["waypoints"]]) \
            if s["waypoints"] else np.empty((0, 2))
        lf, sc, pd_ = float("nan"), float("nan"), float("nan")
        if harmonic_ctx is not None and s["waypoints"] and \
                s["generator"] in harmonic_ctx:
            vals, vecs = harmonic_ctx[s["generator"]]
            nodes = [w["node"] for w in s["waypoints"]]
            spec = H.stimulus_spectrum(nodes, vecs)
            lf, sc = H.low_freq_fraction(spec), H.spectral_centroid(spec, vals)
            pd_ = H.path_dirichlet(nodes, vals, vecs)
        basq_disp = (np.linalg.norm(np.asarray(r["basq_pre"]["va"]) - s["target_va"])
                     - np.linalg.norm(np.asarray(r["basq_post"]["va"]) - s["target_va"]))
        rows.append({
            "id": s["id"], "constructor": s["constructor"], "generator": s["generator"],
            "target": s["target"], "length": s["params"].get("length"),
            "intensity": s["params"].get("intensity"), "style": s["params"].get("style"),
            "placement_error": M.placement_error(traj, s["target_va"]),
            "displacement": M.displacement(traj, s["target_va"]),
            "adherence": M.adherence(np.asarray(r["line_vas"]), wps),
            "stability": M.stability(traj),
            "basq_displacement": float(basq_disp),
            "low_freq_fraction": lf, "spectral_centroid": sc,
            "path_dirichlet": pd_,
            "mismatch_placement_error": M.placement_error(traj, EXCITED)
            if s["target"] in ("calm", "rescue") else float("nan"),
            **C.covariates(s["text"], s["lines"] or [s["text"]]),
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    fig_dir = REPO_ROOT / "data/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stims = []
    for name in ["data/stimuli/stimuli.jsonl", "data/stimuli/stimuli_additions.jsonl",
                 "data/renders/renders.jsonl"]:
        p = REPO_ROOT / name
        if p.exists():
            with open(p) as f:
                stims += [json.loads(l) for l in f]
    runs = {}
    for p in (REPO_ROOT / "data/runs").glob("*.json"):
        with open(p) as f:
            rec = json.load(f)
        runs[rec["stimulus_id"]] = rec
    # harmonic context per generator's artifact
    harmonic_ctx = {}
    for gen, apath in [("psg", REPO_ROOT / "data/phrase_bank/phrase_graph.json"),
                       ("word-template", Path(cfg["word_artifact"]))]:
        apath = Path(apath)
        if not apath.exists():
            continue
        try:
            with open(apath) as f:
                art = json.load(f)
            n_nodes = len(art["words"])
            if gen == "word-template" and n_nodes > 100_000:
                print(f"SKIPPING harmonic context for {gen}: {n_nodes} nodes > 100k cap")
                continue
            L = H.build_laplacian(art["traversal_graph"]["edges"], n_nodes)
            harmonic_ctx[gen] = H.eigenmodes(L, cfg["harmonics"]["n_modes"])
        except Exception as e:
            print(f"FAILED harmonic context for {gen}: {e!r}")
    df = assemble_metrics(stims, runs, harmonic_ctx)
    if df.empty:
        print("no scored stimuli — nothing to analyze")
        return
    df.sort_values("placement_error").to_csv(fig_dir / "leaderboard.csv", index=False)
    # figures
    for target in df["target"].unique():
        sub = [s for s in stims if s["target"] == target and s["id"] in runs
               and "error" not in runs[s["id"]]]
        trajs = {f"{s['constructor']}/{s['generator']}/{s['params'].get('length')}":
                 runs[s["id"]]["traj"] for s in sub[:12]}
        if sub:
            F.circumplex_plot(trajs, sub[0]["target_va"],
                              fig_dir / f"trajectories_{target}.png")
    F.scatter(df["displacement"], df["basq_displacement"],
              "probe displacement", "BASQ displacement", fig_dir / "probe_vs_basq.png")
    psg = df[df["generator"] == "psg"]
    r1 = F.scatter(psg["low_freq_fraction"], psg["placement_error"],
                   "low-freq fraction", "placement error", fig_dir / "harm_vs_placement.png")
    r2 = F.scatter(psg["low_freq_fraction"], psg["displacement"],
                   "low-freq fraction", "displacement", fig_dir / "harm_vs_displacement.png")
    from scipy.stats import spearmanr as _sp
    pd_ok = psg.dropna(subset=["path_dirichlet"])
    rd1, pd1 = (_sp(pd_ok["path_dirichlet"], pd_ok["placement_error"])
                if len(pd_ok) > 4 else (float("nan"),) * 2)
    rd2, pd2 = (_sp(pd_ok["path_dirichlet"], pd_ok["displacement"])
                if len(pd_ok) > 4 else (float("nan"),) * 2)
    # order-sensitivity check: shuffled stimuli should have HIGHER dirichlet
    shuf = df[df["constructor"].str.startswith("shuffled:")]
    shuf_note = ""
    if len(shuf):
        pairs = []
        for _, srow in shuf.iterrows():
            base_cons = srow["constructor"].split(":", 1)[1]
            base = df[(df["constructor"] == base_cons) & (df["generator"] == "psg")
                      & (df["target"] == srow["target"])]
            if len(base) and not np.isnan(srow["path_dirichlet"]):
                pairs.append((base["path_dirichlet"].iloc[0], srow["path_dirichlet"]))
        if pairs:
            n_higher = sum(b > a for a, b in pairs)
            shuf_note = (f"shuffled dirichlet higher than source in "
                         f"{n_higher}/{len(pairs)} pairs\n")
    (fig_dir / "harmonic_predictiveness.txt").write_text(
        f"low_freq_fraction vs placement_error: spearman r={r1}\n"
        f"low_freq_fraction vs displacement: spearman r={r2}\n"
        f"path_dirichlet vs placement_error: spearman r={rd1:.3f} p={pd1:.4f}\n"
        f"path_dirichlet vs displacement: spearman r={rd2:.3f} p={pd2:.4f}\n"
        + shuf_note)
    # P1 — register-covariate predictiveness (the Bisconti §6.5 decomposition)
    from scipy.stats import spearmanr
    cov_lines = ["covariate, spearman_r_vs_placement_error, p, "
                 "spearman_r_vs_displacement, p, n"]
    for col in COVARIATE_COLS:
        x = df[col].to_numpy(dtype=float)
        row = [col]
        for ycol in ["placement_error", "displacement"]:
            y = df[ycol].to_numpy(dtype=float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() > 4:
                r, p = spearmanr(x[ok], y[ok])
                row += [f"{r:.3f}", f"{p:.4f}"]
            else:
                row += ["nan", "nan"]
        row.append(str(int((~np.isnan(x)).sum())))
        cov_lines.append(", ".join(row))
    (fig_dir / "covariate_predictiveness.csv").write_text("\n".join(cov_lines) + "\n")
    print("\n".join(cov_lines))
    print(df.groupby(["constructor", "generator"])["placement_error"].mean()
          .sort_values().to_string())
    print(f"\nwrote {fig_dir}/leaderboard.csv and figures")


if __name__ == "__main__":
    main()
