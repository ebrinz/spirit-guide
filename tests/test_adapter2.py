import json
import numpy as np
import pytest
from spiritbench.config import load_config
from spiritbench.stimuli.adapter import (
    load_art, harmonic, polygon_pca, style_mask, template_wrap, LENGTH_LINES)

CFG = load_config()


def _axes_file(tmp_path):
    # concreteness axis along embedding dim 2 of the toy vectors
    p = tmp_path / "axes.json"
    p.write_text(json.dumps({"concreteness": {"positive": ["calm"], "negative": ["dread"]}}))
    return str(p)


def test_polygon_pca_returns_n_nodes(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = polygon_pca(art, (0.5, 0.5), (0.75, 0.2), n_lines=6, seed=2)
    assert len(ids) == 6
    assert all(0 <= i < 12 for i in ids)


def test_style_mask_partitions(toy_artifact_dir, tmp_path):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    imag = style_mask(art, "imagist", _axes_file(tmp_path))
    abst = style_mask(art, "abstract", _axes_file(tmp_path))
    assert imag.sum() > 0 and abst.sum() > 0
    assert not np.any(imag & abst)


def test_template_wrap_lengths():
    words = ["calm", "river", "stone", "light", "joy", "mist", "reed", "moon"]
    out = template_wrap(words, "short", seed=1, ot_repo=CFG["ot_repo"])
    assert len(out) >= 1 and all(isinstance(s, str) for s in out)
    assert LENGTH_LINES["medium"] == 24


@pytest.mark.skip(reason="OT needs 300d; covered by smoke run")
def test_harmonic_runs_on_artifact(toy_artifact_dir, tmp_path):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = harmonic(art, str(toy_artifact_dir / "toy.json"), (0.5, 0.5), (0.75, 0.2),
                   n_lines=5, preset="golden", seed=1, ot_repo=CFG["ot_repo"],
                   semantic_axes_path=_axes_file(tmp_path))
    assert 1 <= len(ids) <= 5
