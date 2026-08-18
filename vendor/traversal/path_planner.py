# eeg/path_planner.py
"""
Dijkstra-based circuitous path planner through the traversal graph.

Cost favours:
  1. Semantic coherence (small graph edge distance)
  2. Gradual movement toward target in (valence, arousal) space

The combination produces paths that move through semantically connected
territory while drifting toward the target affective state.
"""
import heapq
from collections import defaultdict


def build_adjacency(edges: list[dict]) -> dict[int, list[tuple[int, float]]]:
    """
    Build bidirectional adjacency list from traversal graph edges.

    Returns {section_id: [(neighbour_id, semantic_distance), ...]}.
    """
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in edges:
        adj[edge["from"]].append((edge["to"], float(edge["distance"])))
        adj[edge["to"]].append((edge["from"], float(edge["distance"])))
    return dict(adj)


def va_distance(va1: tuple[float, float], va2: tuple[float, float]) -> float:
    """Euclidean distance in (valence, arousal) space."""
    return ((va1[0] - va2[0]) ** 2 + (va1[1] - va2[1]) ** 2) ** 0.5


def _edge_cost(
    from_va: tuple[float, float],
    to_va: tuple[float, float],
    target_va: tuple[float, float],
    semantic_distance: float,
) -> float:
    """
    Cost of traversing one edge.

    cost = semantic_distance × (1 − valence_progress)

    va_progress is the fraction of current VA-space distance to target
    that this step eliminates. Clamped to [0, 1].
    """
    d_from = va_distance(from_va, target_va)
    if d_from < 1e-9:
        return semantic_distance * 0.0  # already at target
    d_to = va_distance(to_va, target_va)
    progress = max(0.0, min(1.0, (d_from - d_to) / d_from))
    return semantic_distance * (1.0 - progress)


def find_path(
    start_id: int,
    target_id: int,
    sections: dict[str, dict],
    adjacency: dict[int, list[tuple[int, float]]],
    target_va: tuple[float, float],
) -> list[int]:
    """
    Find a path from start_id to target_id using valence-progress cost.

    Args:
        start_id: starting section ID
        target_id: destination section ID
        sections: artifact["words"] — dict keyed by str section ID
        adjacency: output of build_adjacency()
        target_va: (valence, arousal) of the target affective state

    Returns:
        Ordered list of section IDs from start to target (inclusive).
        Returns [start_id] if start_id == target_id.

    Raises:
        ValueError: if no path exists between start and target.
    """

    def get_va(sid: int) -> tuple[float, float]:
        key = str(sid)
        if key not in sections:
            raise ValueError(
                f"find_path: section id {sid} not found in sections dict"
            )
        vad = sections[key]["vad"]
        return (vad["v"], vad["a"])

    dist: dict[int, float] = {start_id: 0.0}
    prev: dict[int, int | None] = {start_id: None}
    pq: list[tuple[float, int]] = [(0.0, start_id)]

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist.get(u, float("inf")):
            continue
        if u == target_id:
            break
        u_va = get_va(u)
        for v, sem_dist in adjacency.get(u, []):
            v_va = get_va(v)
            edge_c = _edge_cost(u_va, v_va, target_va, sem_dist)
            new_cost = cost + edge_c
            if new_cost < dist.get(v, float("inf")):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    # Reconstruct path
    if target_id not in prev:
        raise ValueError(f"No path found from section {start_id} to section {target_id}")

    path: list[int] = []
    node: int | None = target_id
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path
