# eeg/enrich_artifact.py
"""
Phase A enrichment: adds NRC VAD (valence, arousal) coordinates to each
word node in the word-graph traversal artifact.

Run:
    python eeg/enrich_artifact.py \
        --artifact artifacts/word_graph_2026-04-09.json \
        --nrc-vad  data/nrc_vad/NRC-VAD-Lexicon.txt \
        --output   artifacts/word_graph_2026-04-09_enriched.json
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_nrc_vad(path: str, with_dominance: bool = False):
    """Load NRC VAD Lexicon (v1 or v2). Returns {word: (v,a)} or, when
    with_dominance=True, {word: (v,a,d)}. Values rescaled to [0,1]."""
    vad: dict[str, tuple] = {}
    detected_range = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            word, v_str, a_str = parts[0], parts[1], parts[2]
            d_str = parts[3] if len(parts) > 3 else None
            if " " in word:
                continue
            try:
                v, a = float(v_str), float(a_str)
                d = float(d_str) if (with_dominance and d_str is not None) else 0.5
            except ValueError:
                continue
            if detected_range is None:
                detected_range = "bipolar" if v < 0 or a < 0 else None
            if detected_range == "bipolar" or v < 0 or a < 0:
                detected_range = "bipolar"
                v = (v + 1.0) / 2.0
                a = (a + 1.0) / 2.0
                d = (d + 1.0) / 2.0
            vad[word] = (v, a, d) if with_dominance else (v, a)
    return vad


def enrich_words(artifact: dict, vad_map: dict[str, tuple[float, float]]) -> dict:
    """
    Return a deep copy of the word-graph artifact with a 'vad' field added to every word node.

    Direct match: word_node["word"] found in vad_map.
    Interpolated: weighted mean of neighbour nodes using 1/distance as weight.
    Multiple passes until all nodes are assigned.
    Fallback: global mean of all assigned nodes.
    """
    result = copy.deepcopy(artifact)
    words_dict = result["words"]
    edges = result["traversal_graph"]["edges"]

    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in edges:
        adj[edge["from"]].append((edge["to"], edge["distance"]))
        adj[edge["to"]].append((edge["from"], edge["distance"]))

    # Pass 1: direct matches
    for sid_str, node in words_dict.items():
        word = node["word"]
        if word in vad_map:
            v, a = vad_map[word]
            node["vad"] = {"v": v, "a": a, "source": "direct"}

    # Subsequent passes: interpolate from any already-assigned neighbours
    for _ in range(20):
        unassigned = [s for s, n in words_dict.items() if "vad" not in n]
        if not unassigned:
            break
        for sid_str in unassigned:
            sid = int(sid_str)
            neighbours = adj.get(sid, [])
            assigned = [
                (nid, dist)
                for nid, dist in neighbours
                if "vad" in words_dict[str(nid)]
            ]
            if not assigned:
                continue
            total_weight = sum(1.0 / max(dist, 1e-9) for _, dist in assigned)
            v = sum(words_dict[str(nid)]["vad"]["v"] * (1.0 / max(dist, 1e-9))
                    for nid, dist in assigned) / total_weight
            a = sum(words_dict[str(nid)]["vad"]["a"] * (1.0 / max(dist, 1e-9))
                    for nid, dist in assigned) / total_weight
            words_dict[sid_str]["vad"] = {"v": float(v), "a": float(a), "source": "interpolated"}

    # Fallback: global mean
    assigned_all = [n["vad"] for n in words_dict.values() if "vad" in n]
    if not assigned_all:
        raise ValueError(
            "enrich_words: no word nodes matched the VAD lexicon. "
            "Ensure at least one word appears in the VAD lexicon."
        )
    fallback_v = sum(s["v"] for s in assigned_all) / len(assigned_all)
    fallback_a = sum(s["a"] for s in assigned_all) / len(assigned_all)
    for node in words_dict.values():
        if "vad" not in node:
            node["vad"] = {"v": fallback_v, "a": fallback_a, "source": "fallback"}

    return result


def save_enriched_artifact(artifact: dict, output_path: str) -> None:
    """Write enriched artifact to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved enriched artifact → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich word-graph artifact with NRC VAD coordinates")
    parser.add_argument("--artifact", required=True, help="Path to word-graph artifact JSON")
    parser.add_argument("--nrc-vad", required=True, help="Path to NRC-VAD-Lexicon.txt")
    parser.add_argument("--output", required=True, help="Output path for enriched artifact JSON")
    parser.add_argument("--method", choices=["ridge", "knn"], default="ridge",
                        help="OOV VAD method: ridge regression (default) or legacy knn interpolation")
    args = parser.parse_args()

    print("[1/3] Loading artifact...")
    with open(args.artifact) as f:
        artifact = json.load(f)
    n_words = len(artifact["words"])
    print(f"      {n_words} word nodes")

    print("[2/3] Loading NRC VAD lexicon...")
    if args.method == "ridge":
        vad_map = load_nrc_vad(args.nrc_vad, with_dominance=True)
        print(f"      {len(vad_map)} words loaded")
        print("[3/3] Enriching word nodes (ridge)...")
        art_dir = Path(args.artifact).parent
        vectors = np.load(art_dir / artifact["metadata"]["vectors_file"]).astype(np.float32)
        word_index = {n["word"]: int(sid) for sid, n in artifact["words"].items()}
        enriched = enrich_words_ridge(artifact, vad_map, vectors, word_index)
    else:
        vad_map = load_nrc_vad(args.nrc_vad)
        print(f"      {len(vad_map)} words loaded")
        print("[3/3] Enriching word nodes (knn)...")
        enriched = enrich_words(artifact, vad_map)

    words_dict = enriched["words"]
    from collections import Counter
    counts = Counter(n["vad"]["source"] for n in words_dict.values())
    print(f"      {dict(counts)}")

    save_enriched_artifact(enriched, args.output)


def fit_vad_ridge(vad3, vectors, word_index, alpha: float = 1.0):
    """Fit per-dimension ridge regression predicting (V,A,D) from vectors.
    Returns weight matrix W of shape (dim+1, 3); last row is bias."""
    rows, targets = [], []
    for w, vad in vad3.items():
        idx = word_index.get(w)
        if idx is None:
            continue
        rows.append(vectors[idx])
        targets.append(vad)
    X = np.asarray(rows, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    n, dim = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    reg = alpha * np.eye(dim + 1)
    reg[-1, -1] = 0.0  # do not regularize the bias term
    W = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Y)
    return W


def predict_vad(vec, W):
    """Predict (v,a,d) for one vector, clipped to [0,1]."""
    aug = np.concatenate([np.asarray(vec, dtype=np.float64), [1.0]])
    out = aug @ W
    return tuple(float(np.clip(x, 0.0, 1.0)) for x in out)


def enrich_words_ridge(artifact, vad3, vectors, word_index):
    """Deep-copied artifact with (v,a,d) on every node. Lexicon hits keep
    their values (source='direct'); others use ridge (source='regression')."""
    result = copy.deepcopy(artifact)
    W = fit_vad_ridge(vad3, vectors, word_index)
    global_mean = predict_vad(np.asarray(vectors, dtype=np.float64).mean(axis=0), W)
    for node in result["words"].values():
        w = node["word"]
        if w in vad3:
            v, a, d = vad3[w]
            node["vad"] = {"v": v, "a": a, "d": d, "source": "direct"}
        else:
            idx = word_index.get(w)
            if idx is not None:
                v, a, d = predict_vad(vectors[idx], W)
            else:
                v, a, d = global_mean
            node["vad"] = {"v": v, "a": a, "d": d, "source": "regression"}
    return result


if __name__ == "__main__":
    main()
