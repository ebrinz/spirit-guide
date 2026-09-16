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


def _tied_layers(n=400, d=16, layers=6, noise=0.05, seed=3):
    """Signal at a shallow AND a deep layer, equally strong: a tie the old argmax broke
    arbitrarily. Deeper layers are the more integrated read, so the tie-break must take it."""
    rng = np.random.RandomState(seed)
    v, a = rng.rand(n), rng.rand(n)
    states = rng.randn(n, layers, d) * 0.1
    w_v, w_a = rng.randn(d), rng.randn(d)
    for L in (1, layers - 1):
        states[:, L, :] += np.outer(v, w_v) + np.outer(a, w_a)
        states[:, L, :] += noise * rng.randn(n, d)
    return states, v, a


def test_one_se_rule_prefers_the_deeper_of_two_tied_layers():
    states, v, a = _tied_layers()
    deep = train_probe(states, v, a, alpha=1.0, test_frac=0.25)
    assert deep.layer == states.shape[1] - 1, f"expected deepest tied layer, got {deep.layer}"
    assert deep.layer_rule == "one_se_deepest"
    assert deep.r2_se is not None and deep.r2_se >= 0
    assert deep.r2_v > 0.8
    # the two signal layers really are tied within the SE the rule used
    assert abs(deep.layer_r2_v[1] - deep.layer_r2_v[states.shape[1] - 1]) <= deep.r2_se


def test_argmax_rule_reproduces_the_old_behaviour():
    states, v, a = _tied_layers()
    old = train_probe(states, v, a, alpha=1.0, test_frac=0.25, layer_rule="argmax")
    assert old.layer == int(np.argmax(old.layer_r2_v))
    assert old.layer_rule == "argmax" and old.r2_se is None


def test_tie_break_does_not_override_a_genuinely_better_layer():
    """One clear signal layer and no tie: both rules must pick it, not a deeper noise layer."""
    states, v, a = _synthetic(layers=5)
    for rule in ("one_se_deepest", "argmax"):
        p = train_probe(states, v, a, alpha=1.0, test_frac=0.2, layer_rule=rule)
        assert p.layer == 1, f"{rule} picked layer {p.layer}"


def test_unknown_layer_rule_rejected():
    states, v, a = _synthetic()
    try:
        train_probe(states, v, a, alpha=1.0, test_frac=0.2, layer_rule="deepest")
    except ValueError as e:
        assert "layer_rule" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_probes_pickled_before_the_new_fields_still_load(tmp_path):
    """Saved probes predate layer_r2_v/layer_rule/r2_se; loading must not break."""
    states, v, a = _synthetic()
    p = train_probe(states, v, a, alpha=1.0, test_frac=0.2)
    for name in ("layer_r2_v", "layer_rule", "r2_se"):
        del p.__dict__[name]          # simulate an old pickle
    save_probe(p, tmp_path / "old.pkl")
    loaded = load_probe(tmp_path / "old.pkl")
    assert loaded.predict(states[:3, loaded.layer, :]).shape == (3, 2)
    assert getattr(loaded, "layer_rule", None) is None
