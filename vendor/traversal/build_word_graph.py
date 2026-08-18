# eeg/build_word_graph.py
"""
One-time offline build: GloVe 6B.300d → word-level k-NN graph artifact.

Run:
    python eeg/build_word_graph.py \
        --glove data/glove/glove.6B.300d.txt \
        --output-dir artifacts/

Estimated runtime: ~15 minutes on CPU for full 49k-word graph.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

_WORD_RE = re.compile(r'^[a-z]+$')   # lowercase alpha only — matches NRC VAD lexicon format


def load_glove(path: str) -> tuple[list[str], np.ndarray]:
    """
    Load and filter GloVe text file.

    Keeps only lowercase-alpha words with valid vectors.
    Returns (words, unit-normalized float32 vectors).
    """
    words: list[str] = []
    vecs: list[np.ndarray] = []
    expected_dim: int | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if not _WORD_RE.match(word):
                continue
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if len(vec) < 2:
                continue
            if expected_dim is None:
                expected_dim = len(vec)
            elif len(vec) != expected_dim:
                continue
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                continue
            words.append(word)
            vecs.append(vec / norm)
    return words, np.stack(vecs) if vecs else np.empty((0, expected_dim or 0), dtype=np.float32)


def build_graph(
    words: list[str], vectors: np.ndarray, k: int = 10
) -> tuple[dict, list[dict]]:
    """
    Build k-NN graph using cosine distance on unit vectors.

    Returns:
        word_nodes: {str(id): {"word": str, "neighbors": [str, ...]}}
        edges: [{"from": int, "to": int, "distance": float}, ...]  (undirected, deduplicated)
    """
    effective_k = min(k, len(words) - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine", algorithm="brute")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    word_nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[int, int]] = set()

    for i, word in enumerate(words):
        neighbor_indices = indices[i, 1:]      # skip rank-0 (self)
        neighbor_dists = distances[i, 1:]
        word_nodes[str(i)] = {
            "word": word,
            "neighbors": [words[int(j)] for j in neighbor_indices],
        }
        for j_idx, dist in zip(neighbor_indices, neighbor_dists):
            j_int = int(j_idx)
            key = (min(i, j_int), max(i, j_int))
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"from": i, "to": j_int, "distance": float(dist)})

    return word_nodes, edges


def save_artifact(
    words: list[str],
    word_nodes: dict,
    edges: list[dict],
    vectors: np.ndarray,
    output_dir: str,
    k: int = 10,
) -> tuple[str, str]:
    """
    Save artifact JSON and vectors .npy.

    Returns (json_path, npy_path).
    """
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    npy_filename = f"word_graph_{ts}_vectors.npy"
    npy_path = str(Path(output_dir) / npy_filename)
    np.save(npy_path, vectors.astype(np.float32))

    artifact = {
        "metadata": {
            "glove_version": "6B.300d",
            "n_words": len(words),
            "k_neighbors": k,
            "artifact_type": "word_graph",
            "vectors_file": npy_filename,
            "timestamp": now.isoformat(),
        },
        "words": word_nodes,
        "traversal_graph": {"edges": edges},
    }
    json_path = str(Path(output_dir) / f"word_graph_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f)

    return json_path, npy_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build word-level GloVe k-NN graph")
    parser.add_argument("--glove", required=True, help="Path to glove.6B.300d.txt")
    parser.add_argument("--output-dir", default="artifacts/", help="Output directory")
    args = parser.parse_args()

    print("[1/4] Loading GloVe...")
    words, vectors = load_glove(args.glove)
    print(f"      {len(words)} words loaded")

    print("[2/4] Building k-NN graph (k=10)...")
    word_nodes, edges = build_graph(words, vectors)
    print(f"      {len(edges)} edges")

    print("[3/4] Saving artifact...")
    json_path, npy_path = save_artifact(words, word_nodes, edges, vectors, args.output_dir)

    print(f"[4/4] Done.\n  JSON → {json_path}\n  NPY  → {npy_path}")


if __name__ == "__main__":
    main()
