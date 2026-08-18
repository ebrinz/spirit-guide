"""The ONLY module that imports the vendored constructor code (`vendor/eeg/`).

The three constructor modules (path_planner, harmonic_path, sentence_builder)
are vendored from the ontological-traversal project so this repo is
self-contained. `_ot()` puts the vendor dir on sys.path so `import eeg.X`
resolves to the local copy; if an external ot_repo is passed and the vendor
is absent, it falls back to that. The `ot_repo` argument is retained for
signature compatibility and is ignored when the vendor is present."""
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_VENDOR = str(Path(__file__).resolve().parents[3] / "vendor")


def _ot(ot_repo: str | None = None):
    """Make `import eeg.*` resolve to the vendored constructor modules."""
    if (Path(_VENDOR) / "eeg").is_dir():
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
    elif ot_repo and ot_repo not in sys.path:
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
    if len(path) != n_lines:  # resample (stretch or shrink) to n_lines, keeping endpoints
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


def apply_mask_to_path(art: Art, ids, mask) -> list[int]:
    mask_indices = np.where(mask)[0]
    out = []
    for i in ids:
        if mask[i]:
            out.append(i)
        else:
            d = np.linalg.norm(art.vectors[mask_indices] - art.vectors[i], axis=1)
            out.append(int(mask_indices[np.argmin(d)]))
    return out


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


LENGTH_LINES = {"short": 8, "medium": 24, "long": 56}

# Memoized word-artifact Art, keyed by path. The word artifact JSON+vectors can be
# multiple GB, so both harmonic()'s and style_mask()'s "anchor words absent from this
# artifact" fallback share one load per process instead of re-reading it every call.
_word_art_cache: dict[str, Art] = {}


def _load_word_art(word_artifact_path: str) -> Art:
    if word_artifact_path not in _word_art_cache:
        _word_art_cache[word_artifact_path] = load_art(word_artifact_path)
    return _word_art_cache[word_artifact_path]


def style_mask(art: Art, style, axes_path) -> np.ndarray:
    if style is None:
        return np.ones(len(art.nodes), dtype=bool)
    with open(axes_path) as f:
        axes = json.load(f)
    ax = axes["concreteness"]
    pos_words = ax.get("positive", ax.get("pos"))
    neg_words = ax.get("negative", ax.get("neg"))
    pos = np.mean([art.vectors[art.id_of[w]] for w in pos_words if w in art.id_of], axis=0) \
        if any(w in art.id_of for w in pos_words) else None
    neg = np.mean([art.vectors[art.id_of[w]] for w in neg_words if w in art.id_of], axis=0) \
        if any(w in art.id_of for w in neg_words) else None
    if pos is None or neg is None:
        # Axis words absent (e.g. phrase artifact: entries are multi-word lines, so
        # single-word anchors like "concrete"/"abstract" never match). Same fallback
        # pattern as harmonic(): build the axis from the word artifact's own vocabulary
        # instead (same 300-d GloVe space as art.vectors), then keep projecting art's
        # own vectors.
        from spiritbench.config import load_config
        word_art = _load_word_art(load_config()["word_artifact"])
        pos = np.mean([word_art.vectors[word_art.id_of[w]]
                       for w in pos_words if w in word_art.id_of], axis=0)
        neg = np.mean([word_art.vectors[word_art.id_of[w]]
                       for w in neg_words if w in word_art.id_of], axis=0)
    direction = pos - neg
    proj = art.vectors @ direction
    if style == "imagist":
        return proj >= np.quantile(proj, 0.6)
    if style == "abstract":
        return proj <= np.quantile(proj, 0.4)
    raise ValueError(style)


def polygon_pca(art: Art, start_va, target_va, n_lines, seed) -> list[int]:
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(seed)
    ids = []
    focus = nearest_node_to_va(art, start_va)
    for step in range(n_lines):
        frac = step / max(1, n_lines - 1)
        wp_va = (1 - frac) * np.asarray(start_va) + frac * np.asarray(target_va)
        focus = nearest_node_to_va(art, wp_va)
        fvec = art.vectors[focus]
        d = np.linalg.norm(art.vectors - fvec, axis=1)
        nn = np.argsort(d)[1:51]
        comps = PCA(n_components=2).fit(art.vectors[nn]).components_
        theta = np.deg2rad(15 * step) + 2 * np.pi * (step % 5) / 5
        radius = 0.15 * np.linalg.norm(fvec)
        probe_vec = fvec + radius * (np.cos(theta) * comps[0] + np.sin(theta) * comps[1])
        pick = int(np.argmin(np.linalg.norm(art.vectors - probe_vec, axis=1)))
        ids.append(pick)
    return ids


def harmonic(art: Art, artifact_path, start_va, target_va, n_lines, preset, seed,
             ot_repo, semantic_axes_path) -> list[int]:
    _ot(ot_repo)
    import eeg.harmonic_path as hp
    hp.get_vocab_mask = lambda id_to_word: np.zeros(len(id_to_word), dtype=bool)
    vectors, word_index, id_to_word, v_axis, a_axis, semantic_axes = \
        hp.load_harmonic_inputs(artifact_path, axes_path=semantic_axes_path,
                                vocab_cap=len(art.nodes))
    if v_axis.ndim == 0 or a_axis.ndim == 0 or np.isnan(v_axis).any() or np.isnan(a_axis).any():
        # Phrase artifact: VA/semantic axis anchor words (e.g. "good"/"joy") aren't
        # in its vocabulary (entries are multi-word lines), so build_va_axes silently
        # returns a degenerate (nan or 0-d) axis. Phrase vectors are mean-GloVe
        # vectors in the same 300-d space as the word artifact, so build the axes
        # from the word artifact's own vocabulary instead (memoized: see
        # _load_word_art), and keep traversing the phrase artifact's vectors/index
        # (already loaded above).
        from spiritbench.config import load_config
        word_art = _load_word_art(load_config()["word_artifact"])
        v_axis, a_axis = hp.build_va_axes(word_art.vectors, word_art.id_of)
        semantic_axes = hp.load_semantic_axes(semantic_axes_path, word_art.vectors, word_art.id_of)
    start_word = art.word(nearest_node_to_va(art, start_va))
    target_word = art.word(nearest_node_to_va(art, target_va))
    path = hp.plan_harmonic_waypoints(
        vectors, word_index, id_to_word, semantic_axes, v_axis, a_axis,
        start_word, target_word, steps=n_lines, preset=preset, seed=seed)
    return [art.id_of[s["focus"]] for s in path if s["focus"] in art.id_of][:n_lines]


def template_wrap(lines, length, seed, ot_repo) -> list[str]:
    # OT's build_sentences/build_short_sentences/build_long_sentences shuffle their
    # word pool internally, destroying waypoint order. Reimplement inline: pair (or
    # triple) words in ORDER and round-robin the same template lists OT uses, so the
    # meditation's line order tracks the waypoint order.
    _ot(ot_repo)
    from eeg.sentence_builder import TEMPLATES, SHORT_TEMPLATES, LONG_TEMPLATES
    templates, n_slots = {"short": (SHORT_TEMPLATES, 2), "medium": (TEMPLATES, 2),
                          "long": (LONG_TEMPLATES, 3)}[length]
    n = max(1, len(lines) // 2)
    pool = list(lines)
    while len(pool) < n_slots * n:  # pad by cycling the original list
        pool += lines
    keys = ["a", "b", "c"][:n_slots]
    sentences = []
    for i in range(n):
        values = {k: pool[n_slots * i + j] for j, k in enumerate(keys)}
        template = templates[i % len(templates)]
        sentences.append(template.format(**values))
    return sentences


# ── Lit-review additions (P2 via negativa, P3 Gregory layering) ──────────────

NEGATION_FRAMES = ["not {p}", "nor {p}"]


def antipode(target_va) -> tuple[float, float]:
    """The reflection of a VA target through the center of the circumplex."""
    return (1.0 - target_va[0], 1.0 - target_va[1])


def negate_lines(lines) -> list[str]:
    """Via negativa (Maimonides, Guide I.58-59): specify the target only by
    negating its complement — every antipode phrase is negated, nothing asserted."""
    return [NEGATION_FRAMES[i % len(NEGATION_FRAMES)].format(p=l)
            for i, l in enumerate(lines)]


def gregory_wrap(lines) -> list[str]:
    """Gregory (1992) semantic layering: opening statement, an and-layered
    parallel series (Wårvik's Bible-register and-initial density), closing
    statement — repeated in blocks of 5 along the waypoint path."""
    out, block = [], 5
    for i in range(0, len(lines), block):
        chunk = lines[i:i + block]
        out.append(f"there is {chunk[0]}")
        for line in chunk[1:-1]:
            out.append(f"and {line}")
        if len(chunk) > 1:
            out.append(f"this is {chunk[-1]}")
    return out


# ── E15 complexity ladders ───────────────────────────────────────────────────

def valley_steps(art: Art, target_va, n_lines, seed, n_steps) -> list[int]:
    """Valley with parameterized phase resolution. n_steps=0 -> pure
    target-band litany; n_steps=k -> ground band + k interpolation bands +
    target band (valley_shape's structure generalized)."""
    import random as _random
    rng = _random.Random(seed)
    tv, ta = target_va
    used: set = set()
    if n_steps == 0:
        return _pick_in_band(art, (tv - 0.15, ta - 0.15), (tv + 0.15, ta + 0.15),
                             n_lines, rng, used)
    n1 = max(1, n_lines // (n_steps + 2))
    n3 = max(1, n_lines // 4)
    n2 = n_lines - n1 - n3
    ids = _pick_in_band(art, (0.5, 0.0), (1.0, 0.4), n1, rng, used)   # ground
    ground_a, per = 0.25, max(1, n2 // n_steps)
    for j in range(n_steps):
        frac = (j + 1) / (n_steps + 1)
        a_mid = ground_a + frac * (ta - ground_a)
        v_mid = 0.6 + frac * (tv - 0.6)
        ids += _pick_in_band(art, (v_mid - 0.15, a_mid - 0.15),
                             (v_mid + 0.15, a_mid + 0.15), per, rng, used)
    ids += _pick_in_band(art, (tv - 0.15, ta - 0.15), (tv + 0.15, ta + 0.15),
                         max(1, n_lines - len(ids)), rng, used)
    # sparse-graph fill: nearest-to-target nodes (reuse allowed as last resort)
    if len(ids) < n_lines:
        d = np.linalg.norm(_va_array(art) - np.asarray(target_va), axis=1)
        order = [int(i) for i in np.argsort(d)]
        fresh = [i for i in order if i not in used]
        pool = fresh + order
        ids += pool[: n_lines - len(ids)]
    return ids[:n_lines]


def harmonic_k(art: Art, artifact_path, start_va, target_va, n_lines, k, seed,
               ot_repo, semantic_axes_path) -> list[int]:
    """Harmonic traversal with only the first k components of the golden
    preset active (harmonic-richness ladder)."""
    _ot(ot_repo)
    import eeg.harmonic_path as hp
    name = f"golden_k{k}"
    if name not in hp.PRESETS:
        g = hp.PRESETS["golden"]
        hp.PRESETS[name] = {**g, "harmonics": g["harmonics"][:k]}
    return harmonic(art, artifact_path, start_va, target_va, n_lines, name,
                    seed, ot_repo, semantic_axes_path)


def polygon_shape(art: Art, start_va, target_va, n_lines, seed,
                  n_vertices=5, skip=1, radius_frac=0.15) -> list[int]:
    """polygon_pca generalized: an n-gon whose vertices are visited in
    skip-order (skip=1 -> perimeter; skip coprime to n -> star polygon, e.g.
    {5/2} pentagram, {8/3} octagram). Fixed vertex set, variable traversal
    order — isolates angular step size."""
    from sklearn.decomposition import PCA
    ids = []
    for step in range(n_lines):
        frac = step / max(1, n_lines - 1)
        wp_va = (1 - frac) * np.asarray(start_va) + frac * np.asarray(target_va)
        focus = nearest_node_to_va(art, wp_va)
        fvec = art.vectors[focus]
        d = np.linalg.norm(art.vectors - fvec, axis=1)
        nn = np.argsort(d)[1:51]
        comps = PCA(n_components=2).fit(art.vectors[nn]).components_
        vertex = (step * skip) % n_vertices
        theta = np.deg2rad(15 * step) + 2 * np.pi * vertex / n_vertices
        radius = radius_frac * np.linalg.norm(fvec)
        probe_vec = fvec + radius * (np.cos(theta) * comps[0]
                                     + np.sin(theta) * comps[1])
        ids.append(int(np.argmin(np.linalg.norm(art.vectors - probe_vec, axis=1))))
    return ids
