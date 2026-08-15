"""The ONLY module allowed to import from ../ontological-traversal (via sys.path)."""
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _ot(ot_repo: str):
    if ot_repo not in sys.path:
        sys.path.insert(0, ot_repo)


@dataclass
class Art:
    nodes: dict
    vectors: np.ndarray
    id_of: dict

    def va(self, node_id: int):
        vad = self.nodes[str(node_id)]["vad"]
        return (vad["v"], vad["a"])

    def word(self, node_id: int) -> str:
        return self.nodes[str(node_id)]["word"]

    @property
    def edges(self):
        return self._edges


def load_art(artifact_path: str) -> Art:
    with open(artifact_path) as f:
        raw = json.load(f)
    vectors = np.load(Path(artifact_path).parent / raw["metadata"]["vectors_file"])
    nodes = raw["words"]
    art = Art(nodes=nodes, vectors=vectors,
              id_of={n["word"]: int(i) for i, n in nodes.items()})
    art._edges = raw["traversal_graph"]["edges"]
    return art


def _va_array(art: Art) -> np.ndarray:
    n = len(art.nodes)
    out = np.zeros((n, 2))
    for i in range(n):
        out[i] = art.va(i)
    return out


def node_mask(art: Art, intensity: str | None) -> np.ndarray:
    va = _va_array(art)
    r = np.linalg.norm(va - 0.5, axis=1)
    if intensity == "plain":
        return r <= 0.25
    if intensity == "heightened":
        return r >= 0.2
    return np.ones(len(va), dtype=bool)


def nearest_node_to_va(art: Art, va, mask=None) -> int:
    d = np.linalg.norm(_va_array(art) - np.asarray(va), axis=1)
    if mask is not None:
        d = np.where(mask, d, np.inf)
    return int(np.argmin(d))


def graph_walk(art: Art, start_va, target_va, n_lines, seed, ot_repo) -> list[int]:
    _ot(ot_repo)
    from eeg.path_planner import build_adjacency, find_path
    start = nearest_node_to_va(art, start_va)
    target = nearest_node_to_va(art, target_va)
    adjacency = build_adjacency(art.edges)
    path = find_path(start, target, art.nodes, adjacency, tuple(target_va))
    if len(path) > n_lines:  # resample evenly, always keeping endpoints
        idx = np.linspace(0, len(path) - 1, n_lines).round().astype(int)
        path = [path[i] for i in idx]
    return path


def _pick_in_band(art, va_lo, va_hi, k, rng, used, mask=None):
    """k nodes whose (v, a) falls in the band, nearest band-center first."""
    va = _va_array(art)
    lo, hi = np.asarray(va_lo), np.asarray(va_hi)
    ok = np.all((va >= lo) & (va <= hi), axis=1)
    if mask is not None:
        ok &= mask
    ids = [i for i in np.where(ok)[0] if i not in used]
    center = (lo + hi) / 2
    ids.sort(key=lambda i: np.linalg.norm(va[i] - center))
    picked = ids[:k]
    if len(picked) < k and ids:
        picked += list(rng.choices(ids, k=k - len(picked)))
    used.update(picked)
    return picked


def valley_shape(art: Art, target_va, n_lines, seed, mask=None) -> list[int]:
    rng = random.Random(seed)
    tv, ta = target_va
    used: set = set()
    n1 = max(1, n_lines // 3)
    n3 = max(1, n_lines // 4)
    n2 = n_lines - n1 - n3
    ids = _pick_in_band(art, (0.5, 0.0), (1.0, 0.4), n1, rng, used, mask)  # ground
    ground_a = 0.25
    for j in range(n2):  # ascend arousal (or descend) toward target in 3 sub-bands
        frac = (j + 1) / (n2 + 1)
        a_mid = ground_a + frac * (ta - ground_a)
        v_mid = 0.6 + frac * (tv - 0.6)
        ids += _pick_in_band(art, (v_mid - 0.15, a_mid - 0.15),
                             (v_mid + 0.15, a_mid + 0.15), 1, rng, used, mask)
    ids += _pick_in_band(art, (tv - 0.15, ta - 0.15), (tv + 0.15, ta + 0.15),
                         n3, rng, used, mask)
    return ids[:n_lines]


def stimulus_record(art: Art, node_ids, constructor, generator, target_name,
                    target_va, params) -> dict:
    lines = [art.word(i) for i in node_ids]
    text = ".\n".join(lines)
    sid = f"{constructor}-{generator}-{target_name}-" + \
        hashlib.sha1(text.encode()).hexdigest()[:8]
    return {"id": sid, "constructor": constructor, "generator": generator,
            "params": params, "target": target_name, "target_va": list(target_va),
            "waypoints": [{"node": int(i), "v": art.va(i)[0], "a": art.va(i)[1]}
                          for i in node_ids],
            "lines": lines, "text": text}
