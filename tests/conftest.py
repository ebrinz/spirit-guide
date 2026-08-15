import json
import numpy as np
import pytest

# 12 nodes on a VA grid with 4-d embeddings; embedding dims 0/1 encode (v, a)
# so cosine neighbors correlate with VA neighbors — realistic enough for constructors.
TOY_WORDS = [
    ("dread", 0.1, 0.9), ("panic", 0.15, 0.85), ("gloom", 0.2, 0.3),
    ("plain", 0.5, 0.5), ("mild", 0.55, 0.45), ("steady", 0.6, 0.4),
    ("calm", 0.75, 0.2), ("serene", 0.8, 0.15), ("rest", 0.7, 0.25),
    ("joy", 0.85, 0.7), ("thrill", 0.8, 0.85), ("bliss", 0.9, 0.6),
]


def _vec(v, a, i):
    rng = np.random.RandomState(i)
    return np.array([v, a, 0.1 * rng.rand(), 0.1 * rng.rand()], dtype=np.float32)


@pytest.fixture
def toy_artifact_dir(tmp_path):
    vecs = np.stack([_vec(v, a, i) for i, (_, v, a) in enumerate(TOY_WORDS)])
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = unit @ unit.T
    words, nodes, edges, seen = {}, {}, [], set()
    for i, (w, v, a) in enumerate(TOY_WORDS):
        order = np.argsort(-sims[i])
        nbr_ids = [int(j) for j in order if j != i][:3]
        nodes[str(i)] = {"word": w, "neighbors": [TOY_WORDS[j][0] for j in nbr_ids],
                         "vad": {"v": v, "a": a, "source": "direct"}}
        for j in nbr_ids:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append({"from": i, "to": j, "distance": float(1 - sims[i, j])})
    artifact = {"metadata": {"n_words": len(TOY_WORDS), "k_neighbors": 3,
                             "artifact_type": "word_graph", "vectors_file": "toy_vectors.npy"},
                "words": nodes, "traversal_graph": {"edges": edges}}
    np.save(tmp_path / "toy_vectors.npy", vecs)
    with open(tmp_path / "toy.json", "w") as f:
        json.dump(artifact, f)
    return tmp_path


@pytest.fixture
def toy_nrc_file(tmp_path):
    p = tmp_path / "nrc.txt"
    rows = [(w, v, a, 0.5) for (w, v, a) in TOY_WORDS] + [
        ("river", 0.65, 0.35, 0.5), ("stone", 0.5, 0.3, 0.5), ("light", 0.7, 0.5, 0.5)]
    p.write_text("\n".join(f"{w}\t{v}\t{a}\t{d}" for w, v, a, d in rows))
    return p
