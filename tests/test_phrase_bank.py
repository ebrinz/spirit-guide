import json
import numpy as np
from spiritbench.stimuli.phrase_bank import (
    load_nrc, filter_lines, line_vad, line_vector, build_phrase_artifact, NEGATORS)


def _glove():
    rng = np.random.RandomState(0)
    return {w: rng.rand(4).astype(np.float32)
            for w in ["calm", "river", "stone", "light", "joy", "dread", "the", "of"]}


def test_load_nrc(toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    assert nrc["calm"] == (0.75, 0.2)


def test_filter_drops_short_long_negated_lowcov(toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    lines = [
        "calm river stone",                # keep
        "calm",                            # too short
        "w " * 11,                         # too long
        "do not fear the calm river",      # negator
        "xyzzy qwfp zxcv",                 # zero NRC coverage
        "Calm River Stone!",               # keep (case/punct normalized)
    ]
    kept = filter_lines(lines, nrc)
    assert kept == ["calm river stone", "calm river stone"]
    assert "not" in NEGATORS


def test_line_vad_is_nrc_mean(toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    v, a = line_vad("calm river", nrc)
    assert abs(v - (0.75 + 0.65) / 2) < 1e-6
    assert abs(a - (0.2 + 0.35) / 2) < 1e-6


def test_build_phrase_artifact_matches_ot_schema(tmp_path, toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    glove = _glove()
    lines = ["calm river stone", "joy light river", "dread stone light",
             "calm joy light", "river stone light", "calm light joy"]
    jpath, npath = build_phrase_artifact(lines, glove, nrc, k=2, out_dir=tmp_path)
    art = json.load(open(jpath))
    assert art["metadata"]["artifact_type"] == "word_graph"
    node0 = art["words"]["0"]
    assert node0["word"] == "calm river stone"
    assert "vad" in node0 and "neighbors" in node0
    assert len(art["traversal_graph"]["edges"]) > 0
    vecs = np.load(npath)
    assert vecs.shape == (6, 4)
