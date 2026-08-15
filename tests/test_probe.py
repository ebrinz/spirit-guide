import numpy as np
from spiritbench.listener.probe import train_probe, save_probe, load_probe


def _synthetic(n=400, d=16, layers=3, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    v, a = rng.rand(n), rng.rand(n)
    states = rng.randn(n, layers, d) * 0.1
    # layer 1 linearly encodes (v, a); other layers are noise
    w_v, w_a = rng.randn(d), rng.randn(d)
    states[:, 1, :] += np.outer(v, w_v) + np.outer(a, w_a)
    states[:, 1, :] += noise * rng.randn(n, d)
    return states, v, a


def test_probe_recovers_signal_and_layer(tmp_path):
    states, v, a = _synthetic()
    probe = train_probe(states, v, a, alpha=1.0, test_frac=0.2)
    assert probe.layer == 1
    assert probe.r2_v > 0.8 and probe.r2_a > 0.8
    preds = probe.predict(states[:5, 1, :])
    assert preds.shape == (5, 2)
    save_probe(probe, tmp_path / "p.pkl")
    p2 = load_probe(tmp_path / "p.pkl")
    assert p2.layer == 1


def test_probe_fails_on_noise():
    rng = np.random.RandomState(1)
    states = rng.randn(300, 3, 16)
    probe = train_probe(states, rng.rand(300), rng.rand(300), alpha=1.0, test_frac=0.2)
    assert probe.r2_v < 0.3
