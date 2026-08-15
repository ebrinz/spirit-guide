import numpy as np
from spiritbench.analysis.metrics import (
    ema, placement_error, displacement, stability, adherence, per_line_va)


def test_ema_smooths():
    traj = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
    sm = ema(traj, alpha=0.5)
    assert sm.shape == traj.shape
    assert np.all(np.abs(np.diff(sm[:, 0])) <= np.abs(np.diff(traj[:, 0])) + 1e-12)


def test_placement_and_displacement():
    traj = np.array([[0.5, 0.5], [0.6, 0.4], [0.75, 0.2]])
    assert placement_error(traj, (0.75, 0.2)) < 1e-9
    assert displacement(traj, (0.75, 0.2)) > 0.3


def test_stability_low_when_settled():
    settled = np.vstack([np.random.RandomState(0).randn(30, 2) * 0.001 + 0.7])
    wild = np.random.RandomState(0).randn(30, 2) * 0.3
    assert stability(settled) < stability(wild)


def test_adherence_and_per_line():
    traj = np.array([[0.1, 0.1]] * 4 + [[0.9, 0.9]] * 4)
    lines = per_line_va(traj, [(0, 4), (4, 8)])
    assert np.allclose(lines, [[0.1, 0.1], [0.9, 0.9]])
    wps = np.array([[0.1, 0.1], [0.9, 0.9]])
    assert adherence(lines, wps) < 1e-9
    assert np.isnan(adherence(lines, np.empty((0, 2))))
