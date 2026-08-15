import numpy as np
from spiritbench.analysis.harmonics import (
    build_laplacian, eigenmodes, stimulus_spectrum, low_freq_fraction, spectral_centroid)


def _path_graph(n=30):
    return [{"from": i, "to": i + 1, "distance": 0.0} for i in range(n - 1)]


def test_laplacian_properties():
    L = build_laplacian(_path_graph(), 30)
    assert L.shape == (30, 30)
    assert np.allclose(L.toarray(), L.toarray().T)
    vals, vecs = eigenmodes(L, 10)
    assert vals[0] < 1e-8            # constant mode
    assert np.all(np.diff(vals) >= -1e-10)
    assert vecs.shape == (30, 10)


def test_contiguous_walk_is_lower_frequency_than_scattered():
    L = build_laplacian(_path_graph(), 30)
    vals, vecs = eigenmodes(L, 10)
    contiguous = list(range(10, 16))
    scattered = [0, 29]
    sc_c = spectral_centroid(stimulus_spectrum(contiguous, vecs), vals)
    sc_s = spectral_centroid(stimulus_spectrum(scattered, vecs), vals)
    assert sc_c < sc_s
    spec = stimulus_spectrum(contiguous, vecs)
    assert abs(spec.sum() - 1) < 1e-9
    assert 0 <= low_freq_fraction(spec) <= 1
