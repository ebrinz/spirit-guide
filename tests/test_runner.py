import importlib.util
import numpy as np
from pathlib import Path
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import Probe
from sklearn.linear_model import Ridge

spec = importlib.util.spec_from_file_location(
    "runner", Path(__file__).parents[1] / "scripts/05_run_listener.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

BANK = [{"id": f"q{i}", "text": f"Does word{i} fit?", "v": 0.5, "a": 0.5} for i in range(5)]


def _tiny_probe(d, n_layers):
    rng = np.random.RandomState(0)
    X = rng.randn(50, d)
    rv = Ridge().fit(X, rng.rand(50))
    ra = Ridge().fit(X, rng.rand(50))
    return Probe(layer=n_layers - 1, ridge_v=rv, ridge_a=ra, r2_v=0.9, r2_a=0.9)


def test_run_stimulus_shapes():
    model = HiddenStateModel("sshleifer/tiny-gpt2", device="cpu")
    d = model.hidden_states("x").shape[2]
    probe = _tiny_probe(d, model.n_layers)
    stim = {"id": "s1", "lines": ["calm river", "bright joy"],
            "text": "calm river.\nbright joy",
            "waypoints": [{"node": 0, "v": 0.7, "a": 0.3}, {"node": 1, "v": 0.8, "a": 0.6}]}
    rec = runner.run_stimulus(model, probe, stim, preamble="Listen:\n",
                              ema_alpha=0.2, bank=BANK,
                              basq_cfg={"n_questions": 3, "seed": 1})
    assert len(rec["line_vas"]) == 2
    assert len(rec["traj"]) >= 2
    assert "va" in rec["basq_pre"] and "va" in rec["basq_post"]
