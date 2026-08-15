import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def build_laplacian(edges, n_nodes) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for e in edges:
        w = max(0.0, 1.0 - e["distance"])
        rows += [e["from"], e["to"]]
        cols += [e["to"], e["from"]]
        vals += [w, w]
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    d = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D = sparse.diags(d_inv_sqrt)
    return sparse.identity(n_nodes, format="csr") - D @ W @ D


def eigenmodes(L, k):
    # Smallest eigenpairs of L via shift-invert hang/OOM at 50k-317k nodes.
    # Reformulate: L = I - N, so the smallest eigenpairs of L are the largest
    # (algebraically) eigenpairs of N = I - L, which eigsh handles without a
    # factorization.
    k = min(k, L.shape[0] - 2)
    N = sparse.identity(L.shape[0], format="csr") - L
    vals_n, vecs = eigsh(N, k=k, which="LA")
    vals = 1.0 - vals_n
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def stimulus_spectrum(node_ids, eigvecs) -> np.ndarray:
    comps = eigvecs[np.asarray(node_ids, dtype=int)] ** 2   # [n_stim_nodes, k]
    spec = comps.mean(axis=0)
    return spec / spec.sum()


def low_freq_fraction(spectrum, frac=0.2) -> float:
    k = max(1, int(len(spectrum) * frac))
    return float(spectrum[:k].sum())


def spectral_centroid(spectrum, eigvals) -> float:
    return float(np.dot(spectrum, eigvals))


def path_dirichlet(node_ids, eigvals, eigvecs) -> float:
    """Order-SENSITIVE spectral roughness of a waypoint sequence: the mean
    lambda-weighted squared eigenmode step, i.e. the Dirichlet energy of the
    path in the (truncated) harmonic basis. Graph-adjacent steps are cheap,
    teleports expensive; shuffling a smooth path raises it. NaN for paths
    shorter than 2."""
    ids = np.asarray(node_ids, dtype=int)
    if len(ids) < 2:
        return float("nan")
    coords = eigvecs[ids]                      # [T, k]
    steps = np.diff(coords, axis=0)            # [T-1, k]
    return float(np.mean((steps ** 2) @ eigvals))
