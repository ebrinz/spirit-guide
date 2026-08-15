"""Gemma-Scope JumpReLU SAE encoding (layer-20 residual, 16k width).

params.npz keys: W_enc [d, f], b_enc [f], W_dec [f, d], b_dec [d],
threshold [f]. Encode: pre = x @ W_enc + b_enc; feats = pre * (pre > threshold).

The SAE is trained on gemma-2-2b (PT) residuals and applied here to the IT
model's activations — a standard transfer with a known caveat (noted in the
report).
"""
import numpy as np

SAE_LAYER = 20  # residual-stream hidden_states index 20 (post-block-19)

NEURONPEDIA_URL = "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/{idx}"


def load_sae(path) -> dict:
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in ("W_enc", "b_enc", "threshold")}


def encode(x: np.ndarray, sae: dict) -> np.ndarray:
    """x: [d] or [n, d] residual vector(s) -> sparse feature activations."""
    pre = x @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def top_delta_features(feats_a: np.ndarray, feats_b: np.ndarray, k: int = 20):
    """Features with the largest mean activation change from a to b across
    stimuli. feats_*: [n_stimuli, n_features]. Returns list of
    (feature_idx, mean_delta, neuronpedia_url), largest |delta| first."""
    delta = feats_b.mean(axis=0) - feats_a.mean(axis=0)
    order = np.argsort(-np.abs(delta))[:k]
    return [(int(i), float(delta[i]), NEURONPEDIA_URL.format(idx=int(i)))
            for i in order]
