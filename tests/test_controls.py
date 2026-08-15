import numpy as np

from spiritbench.stimuli.adapter import load_art, stimulus_record, apply_mask_to_path
from spiritbench.stimuli.controls import shuffled, neutral_stimulus


def test_shuffled_permutes_but_preserves_multiset(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    stim = stimulus_record(art, [0, 3, 6, 9, 2, 5], "valley", "psg", "calm",
                           (0.75, 0.2), {})
    ctrl = shuffled(stim, seed=1)
    assert sorted(ctrl["lines"]) == sorted(stim["lines"])
    assert ctrl["lines"] != stim["lines"]
    assert ctrl["constructor"] == "shuffled:valley"
    assert ctrl["id"] != stim["id"]
    assert [w["node"] for w in ctrl["waypoints"]] != [w["node"] for w in stim["waypoints"]]


def test_neutral_stimulus():
    n = neutral_stimulus("calm", (0.75, 0.2))
    assert n["constructor"] == "neutral"
    assert len(n["text"]) > 200
    assert n["waypoints"] == []


def test_apply_mask_to_path_replaces_unmasked_with_nearest_masked(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = [0, 1, 2, 3]
    mask = np.ones(len(art.nodes), dtype=bool)
    mask[1] = False  # node 1 is in the path but masked out
    out = apply_mask_to_path(art, ids, mask)
    assert len(out) == len(ids)
    assert all(mask[i] for i in out)
    # unmasked ids stay put
    assert out[0] == 0
    assert out[2] == 2
    assert out[3] == 3
    # node 1 got replaced by the nearest masked node by embedding distance
    mask_indices = np.where(mask)[0]
    expected = int(mask_indices[np.argmin(
        np.linalg.norm(art.vectors[mask_indices] - art.vectors[1], axis=1))])
    assert out[1] == expected
