import json
import re
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

NEGATORS = frozenset("no not never nothing none nor neither cannot don't won't can't".split())
_WORD_RE = re.compile(r"[a-z]+")


def load_nrc(path) -> dict:
    nrc = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            try:
                nrc[parts[0].lower()] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue  # header row
    return nrc


def _tokens(line: str) -> list[str]:
    return _WORD_RE.findall(line.lower())


def filter_lines(lines, nrc, min_words=3, max_words=10, min_coverage=0.5) -> list[str]:
    kept = []
    for line in lines:
        toks = _tokens(line)
        if not (min_words <= len(toks) <= max_words):
            continue
        if any(t in NEGATORS for t in toks):
            continue
        cov = sum(t in nrc for t in toks) / len(toks)
        if cov < min_coverage:
            continue
        kept.append(" ".join(toks))
    return kept


def line_vad(line, nrc):
    vals = [nrc[t] for t in _tokens(line) if t in nrc]
    if not vals:
        return None
    return (float(np.mean([v for v, _ in vals])), float(np.mean([a for _, a in vals])))


def line_vector(line, glove):
    vecs = [glove[t] for t in _tokens(line) if t in glove]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def build_phrase_artifact(lines, glove, nrc, k, out_dir, name="phrase_graph"):
    rows, vecs = [], []
    for line in lines:
        vec, vad = line_vector(line, glove), line_vad(line, nrc)
        if vec is not None and vad is not None:
            rows.append((line, vad))
            vecs.append(vec)
    vectors = np.stack(vecs).astype(np.float32)
    unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    eff_k = min(k, len(rows) - 1)
    nn = NearestNeighbors(n_neighbors=eff_k + 1, metric="cosine", algorithm="brute").fit(unit)
    dists, idxs = nn.kneighbors(unit)
    nodes, edges, seen = {}, [], set()
    for i, (line, (v, a)) in enumerate(rows):
        nbrs = [int(j) for j in idxs[i, 1:]]
        nodes[str(i)] = {"word": line, "neighbors": [rows[j][0] for j in nbrs],
                         "vad": {"v": v, "a": a, "source": "nrc_mean"}}
        for j, d in zip(nbrs, dists[i, 1:]):
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append({"from": i, "to": int(j), "distance": float(d)})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_name = f"{name}_vectors.npy"
    np.save(out_dir / npy_name, vectors)
    artifact = {"metadata": {"n_words": len(rows), "k_neighbors": eff_k,
                             "artifact_type": "word_graph", "vectors_file": npy_name,
                             "source": "gutenberg-poetry-v001"},
                "words": nodes, "traversal_graph": {"edges": edges}}
    jpath = out_dir / f"{name}.json"
    with open(jpath, "w") as f:
        json.dump(artifact, f)
    return str(jpath), str(out_dir / npy_name)


def load_glove(path, vocab: set[str]) -> dict:
    """Load only vectors for words in vocab (memory-friendly)."""
    glove = {}
    for line in open(path, encoding="utf-8"):
        w, _, rest = line.partition(" ")
        if w in vocab:
            glove[w] = np.fromiter(map(float, rest.split()), dtype=np.float32)
    return glove
