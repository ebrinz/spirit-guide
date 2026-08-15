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
    k = min(k, L.shape[0] - 2)
    vals, vecs = eigsh(L, k=k, which="SM")
    return vals, vecs


def stimulus_spectrum(node_ids, eigvecs) -> np.ndarray:
    comps = eigvecs[np.asarray(node_ids, dtype=int)] ** 2   # [n_stim_nodes, k]
    spec = comps.mean(axis=0)
    return spec / spec.sum()


def low_freq_fraction(spectrum, frac=0.2) -> float:
    k = max(1, int(len(spectrum) * frac))
    return float(spectrum[:k].sum())


def spectral_centroid(spectrum, eigvals) -> float:
    return float(np.dot(spectrum, eigvals))
