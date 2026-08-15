import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def circumplex_plot(trajs: dict, target_va, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, traj in trajs.items():
        traj = np.asarray(traj)
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, label=label)
        ax.plot(traj[-1, 0], traj[-1, 1], "o", ms=4)
    ax.plot(*target_va, "r*", ms=16, label="target")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("valence"); ax.set_ylabel("arousal")
    ax.legend(fontsize=7)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scatter(x, y, xlabel, ylabel, out_path) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    r, p = spearmanr(x[ok], y[ok]) if ok.sum() > 2 else (float("nan"), float("nan"))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x[ok], y[ok], s=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"spearman r={r:.2f} p={p:.3f}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return r
