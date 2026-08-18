# eeg/harmonic_path.py
"""Harmonic A→B path generation in GloVe embedding space.

Extracted from scripts/harmonic_traversal.py so the steering engine and the
CLI share one implementation. A harmonic focus word maps directly to its
artifact["words"] node id via word_index, so the SteeringEngine can consume
the waypoint sequence with no translation.
"""
import json
import math
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent   # repo root for default config paths

# ── Frequency presets ────────────────────────────────────────────────────────

PRESETS = {
    # Golden-ratio spaced frequencies — maximally incommensurate
    "golden": {
        "base_freq": 2.0,
        "breath_freq": 3.0,
        "harmonics": [
            # (freq_ratio, amplitude, phase)
            (1.0,           1.0,  0.0),              # valence
            (1.618,         0.8,  math.pi / 3),      # arousal (φ)
            (2.618,         0.6,  math.pi / 5),      # agency (φ²)
            (4.236,         0.4,  math.pi / 7),      # concreteness (φ³)
            (6.854,         0.25, math.pi / 11),      # sociality
            (11.09,         0.15, math.pi / 13),      # temporal
        ],
    },
    # Prime-spaced: each frequency is prime → no common factors
    "prime": {
        "base_freq": 1.5,
        "breath_freq": 2.5,
        "harmonics": [
            (2.0,  1.0,  0.0),
            (3.0,  0.7,  math.pi / 4),
            (5.0,  0.5,  math.pi / 6),
            (7.0,  0.35, math.pi / 8),
            (11.0, 0.2,  math.pi / 10),
            (13.0, 0.12, math.pi / 12),
        ],
    },
    # Organic: frequencies based on natural ratios (octave, fifth, etc.)
    "organic": {
        "base_freq": 1.0,
        "breath_freq": 2.0,
        "harmonics": [
            (1.0,   1.0,  0.0),
            (1.5,   0.8,  math.pi / 4),
            (2.0,   0.55, math.pi / 3),
            (2.5,   0.35, math.pi / 5),
            (3.0,   0.2,  2 * math.pi / 5),
            (4.0,   0.1,  math.pi / 7),
        ],
    },
    # Slow drift: low frequencies, large amplitudes — wide sweeping arcs
    "drift": {
        "base_freq": 0.5,
        "breath_freq": 1.0,
        "harmonics": [
            (1.0,          1.0,  0.0),
            (math.sqrt(2), 0.9,  math.pi / 6),
            (math.sqrt(3), 0.7,  math.pi / 3),
            (math.sqrt(5), 0.5,  math.pi / 2),
        ],
    },
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(artifact_path: str, vectors_path: str, vocab_cap: int = 30000):
    with open(artifact_path) as f:
        artifact = json.load(f)
    vectors = np.load(vectors_path).astype(np.float32)
    word_index: dict[str, int] = {}
    id_to_word: list[str] = [""] * len(vectors)
    for sid, node in artifact["words"].items():
        idx = int(sid)
        if idx >= vocab_cap:
            continue
        word_index[node["word"]] = idx
        id_to_word[idx] = node["word"]
    return vectors, word_index, id_to_word


# ── Semantic axis construction ───────────────────────────────────────────────

def load_semantic_axes(axes_path: str, vectors, word_index) -> list[tuple[str, np.ndarray]]:
    with open(axes_path, encoding="utf-8") as f:
        axes_config = json.load(f)

    result = []
    for name, data in axes_config.items():
        pos_vecs = [vectors[word_index[w]] for w in data["pos"] if w in word_index]
        neg_vecs = [vectors[word_index[w]] for w in data["neg"] if w in word_index]
        if not pos_vecs or not neg_vecs:
            continue
        axis = np.mean(pos_vecs, axis=0) - np.mean(neg_vecs, axis=0)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-9:
            result.append((name, (axis / norm).astype(np.float32)))
    return result


# ── VA estimation ───────────────────────────────────────────────────────────

def build_va_axes(vectors, word_index):
    def axis(pos_words, neg_words):
        pos = np.mean([vectors[word_index[w]] for w in pos_words if w in word_index], axis=0)
        neg = np.mean([vectors[word_index[w]] for w in neg_words if w in word_index], axis=0)
        a = pos - neg
        return a / np.linalg.norm(a)

    v_axis = axis(
        ["happy", "joy", "love", "pleasant", "delightful"],
        ["sad", "misery", "hate", "unpleasant", "awful"],
    )
    a_axis = axis(
        ["excited", "aroused", "energetic", "alert", "intense"],
        ["calm", "relaxed", "sleepy", "bored", "passive"],
    )
    return v_axis, a_axis


def estimate_va(vec, v_axis, a_axis):
    v_raw = float(vec @ v_axis)
    a_raw = float(vec @ a_axis)
    v = max(0.0, min(1.0, (v_raw + 0.3) / 0.6))
    a = max(0.0, min(1.0, (a_raw + 0.3) / 0.6))
    return v, a


# ── Nearest word lookup ──────────────────────────────────────────────────────

# Cache the empty-slot mask per id_to_word object. Keyed by id(), but guarded
# against a recycled address (after GC) returning a stale mask. Lists can't be
# weakly referenced, so we keep a strong reference to the cached object and
# compare with `is`: while the object is held it cannot be GC'd and its address
# cannot be reused, so identity (not just length) reliably drives the cache.
_VOCAB_MASK_CACHE: dict[int, tuple] = {}

def get_vocab_mask(id_to_word):
    """Boolean mask, True where a vocab slot is empty/invalid (so callers can
    suppress those rows). Cached per id_to_word object identity."""
    key = id(id_to_word)
    entry = _VOCAB_MASK_CACHE.get(key)
    if entry is not None:
        obj, mask = entry
        if obj is id_to_word and len(mask) == len(id_to_word):
            return mask
    mask = np.array([w == "" for w in id_to_word], dtype=bool)
    _VOCAB_MASK_CACHE[key] = (id_to_word, mask)
    return mask


def nearest_word(vectors, vec, word_index, id_to_word, exclude=None):
    sims = vectors @ vec
    sims[get_vocab_mask(id_to_word)] = -2.0
    if exclude:
        for w in exclude:
            if w in word_index:
                sims[word_index[w]] = -2.0
    return id_to_word[int(np.argmax(sims))]


# ── Polygon orbit ────────────────────────────────────────────────────────────

def polygon_orbit(center_vec, vectors, word_index, id_to_word, rng,
                  polygon_n, orbit_radius, focus_word):
    dim = len(center_vec)
    raw1 = rng.standard_normal(dim).astype(np.float32)
    raw1 -= float(raw1 @ center_vec) * center_vec
    u = raw1 / np.linalg.norm(raw1)
    raw2 = rng.standard_normal(dim).astype(np.float32)
    raw2 -= float(raw2 @ center_vec) * center_vec + float(raw2 @ u) * u
    v = raw2 / np.linalg.norm(raw2)

    words = []
    seen = {focus_word}
    for i in range(polygon_n):
        angle = 2.0 * math.pi * i / polygon_n
        vert = center_vec + orbit_radius * (math.cos(angle) * u + math.sin(angle) * v)
        norm = float(np.linalg.norm(vert))
        if norm > 1e-9:
            vert /= norm
        w = nearest_word(vectors, vert, word_index, id_to_word, exclude=seen)
        words.append(w)
        seen.add(w)
    return words


# ── Path generation ──────────────────────────────────────────────────────────

def orthogonalize_axes(semantic_axes, baseline_dir):
    """
    Project each semantic axis orthogonal to the baseline direction.
    Returns list of (name, orthogonal_unit_vector).
    Drops axes that become degenerate after projection.
    """
    result = []
    for name, axis_vec in semantic_axes:
        proj = axis_vec - float(axis_vec @ baseline_dir) * baseline_dir
        norm = float(np.linalg.norm(proj))
        if norm > 0.1:  # keep axes with meaningful perpendicular component
            result.append((name, (proj / norm).astype(np.float32)))
    return result


def slerp(v0, v1, t):
    """Spherical linear interpolation between unit vectors."""
    dot = float(np.clip(v0 @ v1, -1.0, 1.0))
    omega = math.acos(dot)
    if abs(omega) < 1e-6:
        return ((1 - t) * v0 + t * v1).astype(np.float32)
    s = math.sin(omega)
    return (math.sin((1 - t) * omega) / s * v0 + math.sin(t * omega) / s * v1).astype(np.float32)


def build_waypoints(start_vec, target_vec, n_bows, bow, rng):
    """
    Construct n_bows waypoints arranged in polygonal symmetry around the
    A-B axis, plus the final target. The baseline visits each waypoint
    in sequence before closing on B.

    For n_bows=1: A → W₁ → B  (single arc, original behavior)
    For n_bows=3: A → W₁ → W₂ → W₃ → B  (triangular detour)
    For n_bows=5: A → W₁ → W₂ → W₃ → W₄ → W₅ → B  (pentagonal)

    Each waypoint sits at a different angular position around the A-B axis,
    placed at varying interpolation depths along the journey so the path
    makes genuine progress toward B while visiting each lobe.
    """
    dim = len(start_vec)
    cos_dist = 1.0 - float(np.clip(start_vec @ target_vec, -1.0, 1.0))
    bow_magnitude = bow * cos_dist * 3.0

    # Build an orthonormal basis perpendicular to the A-B axis
    baseline_dir = target_vec - start_vec
    bl_norm = float(np.linalg.norm(baseline_dir))
    if bl_norm > 1e-9:
        baseline_dir = (baseline_dir / bl_norm).astype(np.float32)
    else:
        baseline_dir = start_vec.copy()

    # Two perpendicular basis vectors for the polygon plane
    raw1 = rng.standard_normal(dim).astype(np.float32)
    raw1 -= float(raw1 @ baseline_dir) * baseline_dir
    raw1 -= float(raw1 @ start_vec) * start_vec
    norm1 = float(np.linalg.norm(raw1))
    if norm1 < 1e-9:
        raw1 = rng.standard_normal(dim).astype(np.float32)
        raw1 -= float(raw1 @ baseline_dir) * baseline_dir
        norm1 = float(np.linalg.norm(raw1))
    perp_u = raw1 / norm1

    raw2 = rng.standard_normal(dim).astype(np.float32)
    raw2 -= float(raw2 @ baseline_dir) * baseline_dir
    raw2 -= float(raw2 @ start_vec) * start_vec
    raw2 -= float(raw2 @ perp_u) * perp_u
    norm2 = float(np.linalg.norm(raw2))
    if norm2 < 1e-9:
        raw2 = rng.standard_normal(dim).astype(np.float32)
        raw2 -= float(raw2 @ baseline_dir) * baseline_dir
        raw2 -= float(raw2 @ perp_u) * perp_u
        norm2 = float(np.linalg.norm(raw2))
    perp_v = raw2 / norm2

    waypoints = []
    for k in range(n_bows):
        # Interpolation depth: waypoints progress from near-start to near-target
        # but stay away from the direct geodesic
        depth = 0.15 + 0.7 * (k + 0.5) / n_bows

        # Angular position around the A-B axis
        angle = 2.0 * math.pi * k / n_bows

        # Base position: place waypoints along the start vector's neighborhood,
        # only gradually mixing in the target direction. The perpendicular
        # displacement is what creates the scenic route — the base stays
        # anchored near the start's region of space.
        #
        # depth_along_target controls how much of the target vector to mix in.
        # For early waypoints this is very small; even for the last waypoint
        # it's modest, leaving the final slerp segment to do the actual approach.
        depth_along_target = 0.15 * depth  # barely drift toward target
        base = ((1 - depth_along_target) * start_vec +
                depth_along_target * target_vec)
        base_norm = float(np.linalg.norm(base))
        if base_norm > 1e-9:
            base = base / base_norm

        # Strong perpendicular displacement — this IS the journey
        displacement = bow_magnitude * (math.cos(angle) * perp_u +
                                         math.sin(angle) * perp_v)
        wp = base + displacement
        wp_norm = float(np.linalg.norm(wp))
        if wp_norm > 1e-9:
            wp = wp / wp_norm
        waypoints.append(wp.astype(np.float32))

    return waypoints


def harmonic_path(
    start_vec, target_vec, steps, semantic_axes, preset, width, breath,
    bow, n_bows, rng,
):
    """
    Generate A→B traversal through n_bows polygonally-arranged waypoints
    with harmonic perturbations.

    The baseline follows a multi-segment slerp:
      A → W₁ → W₂ → ... → Wₙ → B

    Waypoints are arranged in a polygon around the A-B axis, each pulling
    the path through a different region of embedding space. Harmonic
    perturbations along semantic axes add texture between waypoints.
    """
    harmonics = preset["harmonics"]
    base_freq = preset["base_freq"]
    breath_freq = preset["breath_freq"]

    # Build waypoint sequence: [start, W₁, ..., Wₙ, target]
    waypoints = build_waypoints(start_vec, target_vec, n_bows, bow, rng)
    nodes = [start_vec] + waypoints + [target_vec]
    n_segments = len(nodes) - 1

    # Baseline direction for axis orthogonalization
    baseline_dir = target_vec - start_vec
    bl_norm = float(np.linalg.norm(baseline_dir))
    if bl_norm > 1e-9:
        baseline_dir = (baseline_dir / bl_norm).astype(np.float32)
    else:
        baseline_dir = start_vec.copy()

    # Project semantic axes perpendicular to baseline
    perp_axes = orthogonalize_axes(semantic_axes, baseline_dir)
    n_active = min(len(harmonics), len(perp_axes))
    active = [(perp_axes[k][0], perp_axes[k][1], harmonics[k])
              for k in range(n_active)]

    points = []
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 1.0

        # Map t to the correct segment
        seg_pos = t * n_segments
        seg_idx = min(int(seg_pos), n_segments - 1)
        seg_t = seg_pos - seg_idx

        baseline = slerp(nodes[seg_idx], nodes[seg_idx + 1], seg_t)
        b_norm = float(np.linalg.norm(baseline))
        if b_norm > 1e-9:
            baseline = baseline / b_norm

        # Perturbation amplitude: peaks mid-journey, breathes for texture
        mid_peak = math.sin(math.pi * t)
        breathing = 1.0 + breath * math.sin(2.0 * math.pi * breath_freq * t)
        amp = width * mid_peak * breathing
        amp = max(0.0, amp)

        # Sum harmonic oscillations along perpendicular semantic axes
        perturbation = np.zeros(len(start_vec), dtype=np.float32)
        active_components = {}
        for axis_name, axis_vec, (freq_ratio, a_ratio, phase) in active:
            theta = 2.0 * math.pi * freq_ratio * base_freq * t + phase
            component = a_ratio * math.sin(theta)
            perturbation += component * axis_vec
            active_components[axis_name] = round(component, 3)

        p_norm = float(np.linalg.norm(perturbation))
        if p_norm > 1e-9:
            perturbation /= p_norm

        point = baseline + amp * perturbation
        norm = float(np.linalg.norm(point))
        if norm > 1e-9:
            point /= norm

        points.append((point, t, amp, active_components))

    return points


# ── Main traversal ──────────────────────────────────────────────────────────

def harmonic_traversal(
    vectors, word_index, id_to_word, v_axis, a_axis, semantic_axes,
    start_word, target_word, steps, preset_name="golden", width=0.3,
    breath=0.4, bow=0.5, n_bows=1, polygon_n=0, orbit_radius=0.15, seed=42,
):
    rng = np.random.default_rng(seed)
    start_vec = vectors[word_index[start_word]]
    target_vec = vectors[word_index[target_word]]
    preset = PRESETS[preset_name]

    points = harmonic_path(
        start_vec, target_vec, steps, semantic_axes, preset, width, breath,
        bow, n_bows, rng,
    )

    path = []
    used_focus = {target_word}  # reserve target for final step
    for point_vec, t, amp, components in points[:-1]:
        focus = nearest_word(vectors, point_vec, word_index, id_to_word, exclude=used_focus)
        used_focus.add(focus)
        focus_vec = vectors[word_index[focus]]
        va = estimate_va(focus_vec, v_axis, a_axis)

        orbit = []
        if polygon_n > 0:
            orbit = polygon_orbit(
                focus_vec, vectors, word_index, id_to_word, rng,
                polygon_n, orbit_radius, focus,
            )

        # Dominant semantic axis at this step
        if components:
            dominant = max(components, key=lambda k: abs(components[k]))
            dominant_val = components[dominant]
        else:
            dominant = "-"
            dominant_val = 0.0

        path.append({
            "step": len(path),
            "focus": focus,
            "focus_v": round(va[0], 3),
            "focus_a": round(va[1], 3),
            "t": round(t, 4),
            "amp": round(amp, 4),
            "dominant_axis": dominant,
            "dominant_val": round(dominant_val, 3),
            "components": components,
            "orbit": orbit,
        })

    # Final step: the target word itself
    target_va = estimate_va(target_vec, v_axis, a_axis)
    path.append({
        "step": len(path),
        "focus": target_word,
        "focus_v": round(target_va[0], 3),
        "focus_a": round(target_va[1], 3),
        "t": 1.0,
        "amp": 0.0,
        "dominant_axis": "-",
        "dominant_val": 0.0,
        "components": {},
        "orbit": [],
    })

    return path


def plan_harmonic_waypoints(
    vectors, word_index, id_to_word, semantic_axes, v_axis, a_axis,
    start_word, target_word, *,
    steps=25, preset="golden", width=0.3, breath=0.4,
    bow=0.5, n_bows=1, polygon_n=0, orbit_radius=0.15, seed=42,
):
    """Return the ordered harmonic path (list of step dicts with keys
    focus, focus_v, focus_a, t, amp, dominant_axis, dominant_val,
    components, orbit). path[-1].focus is pinned to target_word; path[0].focus
    is the nearest word to the start vector (normally start_word)."""
    return harmonic_traversal(
        vectors, word_index, id_to_word, v_axis, a_axis, semantic_axes,
        start_word, target_word, steps,
        preset_name=preset, width=width, breath=breath,
        bow=bow, n_bows=n_bows, polygon_n=polygon_n,
        orbit_radius=orbit_radius, seed=seed,
    )


def load_harmonic_inputs(
    artifact_path: str,
    axes_path: str = "config/semantic_axes.json",
    vocab_cap: int = 30000,
):
    """Load vectors + indices + axes for harmonic planning from an artifact.

    Resolves the vectors .npy from artifact["metadata"]["vectors_file"]
    relative to the artifact's directory. Returns
    (vectors, word_index, id_to_word, v_axis, a_axis, semantic_axes).
    """
    with open(artifact_path) as f:
        artifact = json.load(f)
    vectors_file = artifact["metadata"]["vectors_file"]
    vectors_path = str(Path(artifact_path).parent / vectors_file)
    vectors, word_index, id_to_word = load_data(artifact_path, vectors_path, vocab_cap)
    if not Path(axes_path).is_absolute():
        axes_path = str(_REPO_ROOT / axes_path)
    v_axis, a_axis = build_va_axes(vectors, word_index)
    semantic_axes = load_semantic_axes(axes_path, vectors, word_index)
    return vectors, word_index, id_to_word, v_axis, a_axis, semantic_axes
