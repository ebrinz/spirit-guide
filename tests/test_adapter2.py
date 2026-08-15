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


def _axes_file_pos_neg(tmp_path):
    # same axis, but using the real OT config/semantic_axes.json key spelling
    p = tmp_path / "axes_posneg.json"
    p.write_text(json.dumps({"concreteness": {"pos": ["calm"], "neg": ["dread"]}}))
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


def test_style_mask_partitions_pos_neg_keys(toy_artifact_dir, tmp_path):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    imag = style_mask(art, "imagist", _axes_file_pos_neg(tmp_path))
    abst = style_mask(art, "abstract", _axes_file_pos_neg(tmp_path))
    assert imag.sum() > 0 and abst.sum() > 0
    assert not np.any(imag & abst)


def test_template_wrap_lengths():
    words = ["calm", "river", "stone", "light", "joy", "mist", "reed", "moon"]
    out = template_wrap(words, "short", seed=1, ot_repo=CFG["ot_repo"])
    assert len(out) >= 1 and all(isinstance(s, str) for s in out)
    assert LENGTH_LINES["medium"] == 24


def test_template_wrap_preserves_first_pair_order():
    # OT's builders shuffle their word pool internally; template_wrap must not, so
    # the first waypoints stay legible as the first line.
    words = ["calm", "river", "stone", "light", "joy", "mist", "reed", "moon"]
    out = template_wrap(words, "short", seed=1, ot_repo=CFG["ot_repo"])
    assert out[0] == "calm. river."


def test_template_wrap_long_does_not_raise():
    words = ["calm", "river", "stone", "light", "joy", "mist", "reed", "moon"]
    out = template_wrap(words, "long", seed=1, ot_repo=CFG["ot_repo"])
    assert len(out) >= 1 and all(isinstance(s, str) for s in out)


@pytest.mark.skip(reason="OT load_semantic_axes rejects toy axes/vocab; covered by smoke run")
def test_harmonic_runs_on_artifact(toy_artifact_dir, tmp_path):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = harmonic(art, str(toy_artifact_dir / "toy.json"), (0.5, 0.5), (0.75, 0.2),
                   n_lines=5, preset="golden", seed=1, ot_repo=CFG["ot_repo"],
                   semantic_axes_path=_axes_file(tmp_path))
    assert 1 <= len(ids) <= 5


def _phrase_like_artifact(tmp_path):
    # Multi-word "phrase" nodes: none of build_va_axes'/load_semantic_axes' single-word
    # anchors ("joy", "calm", "dread", ...) can appear in this vocabulary, so
    # load_harmonic_inputs degenerates to a nan/0-d axis -- same failure mode as the
    # real phrase artifact built from Gutenberg lines.
    phrases = [
        ("soft warm light drifting slow", 0.55, 0.3), ("cold dark night falls fast", 0.2, 0.6),
        ("bright open sky calls gently", 0.8, 0.4), ("heavy grey clouds press low", 0.3, 0.4),
        ("quiet still water holds fast", 0.6, 0.15), ("wild bright fire leaps high", 0.75, 0.85),
    ]
    vecs = np.stack([_toy_vec(v, a, i) for i, (_, v, a) in enumerate(phrases)])
    nodes = {str(i): {"word": w, "neighbors": [], "vad": {"v": v, "a": a, "source": "nrc_mean"}}
             for i, (w, v, a) in enumerate(phrases)}
    edges = [{"from": i, "to": i + 1, "distance": 0.1} for i in range(len(phrases) - 1)]
    artifact = {"metadata": {"n_words": len(phrases), "k_neighbors": 1,
                             "artifact_type": "word_graph", "vectors_file": "phrase_vectors.npy"},
                "words": nodes, "traversal_graph": {"edges": edges}}
    np.save(tmp_path / "phrase_vectors.npy", vecs)
    p = tmp_path / "phrase.json"
    p.write_text(json.dumps(artifact))
    return p


def _word_axis_artifact(tmp_path):
    # Small single-word "word artifact" covering every anchor word build_va_axes/
    # load_semantic_axes needs (at least one per pos/neg list), same 4-d space as
    # the phrase-like artifact above.
    words = [("joy", 0.9, 0.7), ("sad", 0.1, 0.3), ("excited", 0.7, 0.9),
             ("calm", 0.75, 0.2), ("dread", 0.1, 0.9)]
    vecs = np.stack([_toy_vec(v, a, i) for i, (_, v, a) in enumerate(words)])
    nodes = {str(i): {"word": w, "neighbors": [], "vad": {"v": v, "a": a, "source": "direct"}}
             for i, (w, v, a) in enumerate(words)}
    artifact = {"metadata": {"n_words": len(words), "k_neighbors": 1,
                             "artifact_type": "word_graph", "vectors_file": "word_axis_vectors.npy"},
                "words": nodes, "traversal_graph": {"edges": []}}
    np.save(tmp_path / "word_axis_vectors.npy", vecs)
    p = tmp_path / "word_axis.json"
    p.write_text(json.dumps(artifact))
    return p


def _toy_vec(v, a, i):
    rng = np.random.RandomState(100 + i)
    return np.array([v, a, 0.1 * rng.rand(), 0.1 * rng.rand()], dtype=np.float32)


def test_style_mask_falls_back_to_word_artifact_axes(tmp_path, monkeypatch):
    # The phrase-like artifact's nodes are multi-word lines, so "calm"/"dread"
    # (the toy concreteness anchors) aren't in its id_of and the direct path can't
    # build a direction — style_mask must fall back to the word artifact's own
    # vocabulary (same 4-d toy embedding space here) instead of raising.
    phrase_path = _phrase_like_artifact(tmp_path)
    word_artifact_path = _word_axis_artifact(tmp_path)
    monkeypatch.setattr("spiritbench.config.load_config",
                        lambda *a, **k: {"word_artifact": str(word_artifact_path)})
    art = load_art(str(phrase_path))
    assert "calm" not in art.id_of and "dread" not in art.id_of
    imag = style_mask(art, "imagist", _axes_file(tmp_path))
    abst = style_mask(art, "abstract", _axes_file(tmp_path))
    assert imag.sum() > 0 and abst.sum() > 0
    assert not np.any(imag & abst)


def test_harmonic_phrase_artifact_falls_back_to_word_artifact_axes(tmp_path, monkeypatch):
    phrase_path = _phrase_like_artifact(tmp_path)
    word_artifact_path = _word_axis_artifact(tmp_path)
    monkeypatch.setattr("spiritbench.config.load_config",
                        lambda *a, **k: {"word_artifact": str(word_artifact_path)})
    art = load_art(str(phrase_path))
    ids = harmonic(art, str(phrase_path), (0.5, 0.5), (0.75, 0.2), n_lines=5,
                   preset="golden", seed=1, ot_repo=CFG["ot_repo"],
                   semantic_axes_path=_axes_file_pos_neg(tmp_path))
    assert 1 <= len(ids) <= 5
    assert all(0 <= i < 6 for i in ids)
    # Regression: a mis-signed get_vocab_mask monkeypatch used to suppress every
    # candidate, forcing every intermediate step's argmax to collapse onto node 0
    # regardless of the actual point on the path (the final step is pinned to the
    # target word either way, so it alone wouldn't catch this).
    assert len(set(ids[:-1])) > 1
