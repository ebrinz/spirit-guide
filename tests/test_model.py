import numpy as np
import pytest
from spiritbench.listener.model import HiddenStateModel

TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def model():
    return HiddenStateModel(TINY, device="cpu")


def test_hidden_states_shape(model):
    hs = model.hidden_states("calm river stone")
    assert hs.ndim == 3
    assert hs.shape[0] == model.n_layers
    assert hs.dtype == np.float32


def test_spans_cover_lines(model):
    hs, spans = model.hidden_states_with_spans("Listen:\n", ["calm river", "bright joy"])
    assert len(spans) == 2
    for s, e in spans:
        assert 0 <= s < e <= hs.shape[1]
    assert spans[0][1] <= spans[1][0]


def test_yes_no_logprobs(model):
    y, n = model.yes_no_logprobs("Answer yes or no: is water wet? Answer:")
    assert np.isfinite(y) and np.isfinite(n)
