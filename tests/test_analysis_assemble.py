import importlib.util
import numpy as np
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).parents[1] / "scripts/06_analyze.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def _stim(sid, target="calm", target_va=(0.75, 0.2)):
    return {"id": sid, "constructor": "valley", "generator": "psg", "target": target,
            "target_va": list(target_va),
            "params": {"length": "medium", "intensity": "plain", "style": "unfiltered"},
            "waypoints": [{"node": 0, "v": 0.6, "a": 0.4}, {"node": 1, "v": 0.75, "a": 0.2}],
            "lines": ["a", "b"], "text": "a.\nb"}


def _run(sid):
    return {"stimulus_id": sid,
            "traj": [[0.5, 0.5], [0.6, 0.4], [0.74, 0.21]],
            "line_vas": [[0.6, 0.4], [0.74, 0.21]],
            "basq_pre": {"va": [0.5, 0.5]}, "basq_post": {"va": [0.7, 0.25]},
            "n_tokens": 10}


def test_assemble_metrics_columns():
    df = analyze.assemble_metrics([_stim("s1")], {"s1": _run("s1")}, harmonic_ctx=None)
    row = df.iloc[0]
    assert row["placement_error"] < 0.05
    assert row["displacement"] > 0.3
    assert row["basq_displacement"] > 0.2
    assert np.isnan(row["low_freq_fraction"])
    assert not np.isnan(row["mismatch_placement_error"])  # calm stim scored vs excited
