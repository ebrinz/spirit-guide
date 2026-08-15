import numpy as np
from spiritbench.config import load_config
from spiritbench.stimuli.adapter import (
    load_art, node_mask, nearest_node_to_va, graph_walk, valley_shape, stimulus_record)

CFG = load_config()


def test_load_art(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    assert art.vectors.shape[0] == 12
    assert art.va(art.id_of["calm"]) == (0.75, 0.2)


def test_node_mask_intensity(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    plain = node_mask(art, "plain")
    heightened = node_mask(art, "heightened")
    assert plain[art.id_of["plain"]] and not plain[art.id_of["dread"]]
    assert heightened[art.id_of["dread"]]


def test_graph_walk_reaches_target(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    path = graph_walk(art, (0.5, 0.5), (0.75, 0.2), n_lines=6, seed=1,
                      ot_repo=CFG["ot_repo"])
    assert 1 <= len(path) <= 6
    v, a = art.va(path[-1])
    assert abs(v - 0.75) < 0.2 and abs(a - 0.2) < 0.2


def test_valley_shape_descends_then_targets(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = valley_shape(art, (0.85, 0.7), n_lines=8, seed=3)
    assert len(ids) == 8
    aro = [art.va(i)[1] for i in ids]
    assert min(aro[:3]) < 0.5          # grounding phase is low-arousal
    assert abs(art.va(ids[-1])[0] - 0.85) < 0.25  # ends near target valence


def test_stimulus_record_schema(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    rec = stimulus_record(art, [6, 7], "valley", "psg", "calm", (0.75, 0.2),
                          {"length": "short"})
    assert rec["lines"] == ["calm", "serene"]
    assert rec["waypoints"][0] == {"node": 6, "v": 0.75, "a": 0.2}
    assert rec["text"] == "calm.\nserene"
    assert rec["id"].startswith("valley-psg-calm-")
