import numpy as np


def ema(traj: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(traj, dtype=float)
    out[0] = traj[0]
    for i in range(1, len(traj)):
        out[i] = alpha * traj[i] + (1 - alpha) * out[i - 1]
    return out


def placement_error(traj, target_va) -> float:
    return float(np.linalg.norm(traj[-1] - np.asarray(target_va)))


def displacement(traj, target_va) -> float:
    t = np.asarray(target_va)
    return float(np.linalg.norm(traj[0] - t) - np.linalg.norm(traj[-1] - t))


def stability(traj) -> float:
    tail = traj[len(traj) * 2 // 3:]
    return float(np.mean(np.std(tail, axis=0)))


def per_line_va(traj, spans) -> np.ndarray:
    return np.array([traj[s:e].mean(axis=0) for s, e in spans])


def adherence(line_vas, waypoint_vas) -> float:
    if len(waypoint_vas) == 0 or len(line_vas) == 0:
        return float("nan")
    n = min(len(line_vas), len(waypoint_vas))
    return float(np.mean(np.linalg.norm(line_vas[:n] - waypoint_vas[:n], axis=1)))
