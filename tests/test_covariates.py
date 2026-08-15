import numpy as np
from spiritbench.analysis.covariates import (
    nv_ratio, line_initial_density, word_density, shannon_entropy,
    approx_entropy, cv_line_length, covariates)


def test_nv_ratio_all_nouns_is_one():
    assert nv_ratio("river stone moonbeam") == 1.0


def test_line_initial_and_density():
    lines = ["and the river rests", "the stone waits", "and silence falls"]
    # 2 and-initial lines over 10 words -> 200 per 1000
    d = line_initial_density(lines, "and")
    assert abs(d - 200.0) < 1e-9


def test_then_density():
    assert abs(word_density("then the river then", {"then"}) - 500.0) < 1e-9


def test_shannon_entropy_constant_is_zero():
    assert shannon_entropy(np.array([3.0, 3.0, 3.0])) == 0.0


def test_approx_entropy_regular_lower_than_random():
    regular = np.array([1, 2] * 20, dtype=float)
    rng = np.random.RandomState(0)
    noisy = rng.rand(40)
    assert approx_entropy(regular) < approx_entropy(noisy)


def test_approx_entropy_short_series_nan():
    assert np.isnan(approx_entropy(np.array([1.0, 2.0])))


def test_cv_line_length():
    assert cv_line_length(["one two", "one two"]) == 0.0


def test_covariates_keys():
    text = "the quiet river rests. and the stone waits then sleeps. " * 10
    lines = ["the quiet river rests", "and the stone waits then sleeps"] * 10
    cov = covariates(text, lines)
    assert set(cov) == {"nv_ratio", "and_initial_per_1000", "then_per_1000",
                        "subordinator_per_1000", "noun_shen", "noun_apen",
                        "cv_line_len"}
    assert not np.isnan(cov["nv_ratio"])
