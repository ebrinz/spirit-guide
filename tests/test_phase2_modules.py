import numpy as np
from spiritbench.listener.panas import administer_panas, score_item, PA_ITEMS, NA_ITEMS
from spiritbench.listener.tokendist import valence_shift, POS_WORDS, NEG_WORDS
from spiritbench.analysis.sae import encode, top_delta_features


class FakeModel:
    """option_logprobs stub: favors '5' for NA items when 'dread' in context,
    else favors '1'; for word options favors positive words."""
    def option_logprobs(self, prompt, options):
        if options == ["1", "2", "3", "4", "5"]:
            hi = any(w in prompt for w in ("distressed", "scared", "afraid")) \
                and "dread" in prompt
            return [0.0, -1, -2, -3, -4][::-1] if hi else [0.0, -1, -2, -3, -4]
        # word options: first half (positive) high
        n = len(options)
        return [0.0] * (n // 2) + [-3.0] * (n - n // 2)


def test_score_item_expectation_range():
    s = score_item(FakeModel(), "", "interested")
    assert 1.0 <= s <= 5.0
    assert s < 2.5  # favors "1"


def test_panas_na_rises_under_dread_context():
    m = FakeModel()
    base = administer_panas(m, "")
    dread = administer_panas(m, "dread everywhere. ")
    assert dread["na"] > base["na"]
    assert set(base["items"]) == set(PA_ITEMS + NA_ITEMS)


def test_valence_shift_pos_share():
    v = valence_shift(FakeModel(), "")
    assert 0.5 < v["pos_share"] <= 1.0


def test_sae_encode_jumprelu():
    d, f = 4, 6
    rng = np.random.RandomState(0)
    sae = {"W_enc": rng.randn(d, f).astype(np.float32),
           "b_enc": np.zeros(f, dtype=np.float32),
           "threshold": np.full(f, 0.5, dtype=np.float32)}
    x = rng.randn(3, d).astype(np.float32)
    feats = encode(x, sae)
    pre = x @ sae["W_enc"]
    assert feats.shape == (3, f)
    assert np.all(feats[pre <= 0.5] == 0)          # below threshold -> zero
    assert np.allclose(feats[pre > 0.5], pre[pre > 0.5])


def test_top_delta_features_ranks_by_change():
    a = np.zeros((5, 10)); b = np.zeros((5, 10))
    b[:, 3] = 2.0; b[:, 7] = -1.0
    top = top_delta_features(a, b, k=2)
    assert top[0][0] == 3 and abs(top[0][1] - 2.0) < 1e-9
    assert top[1][0] == 7
    assert "neuronpedia.org" in top[0][2]


def test_steer_shifts_hidden_states_and_removes_cleanly():
    import numpy as np
    from spiritbench.listener.model import HiddenStateModel
    m = HiddenStateModel("sshleifer/tiny-gpt2", device="cpu")
    d = m.hidden_states("calm river").shape[2]
    direction = np.ones(d, dtype=np.float32)
    base = m.hidden_states("calm river")
    with m.steer(1, direction, alpha=5.0):
        steered = m.hidden_states("calm river")
    after = m.hidden_states("calm river")
    # layer 1 shifted by ~alpha per dim; layer 0 (embeddings) untouched
    assert np.allclose(steered[0], base[0])
    assert np.abs(steered[1] - base[1]).mean() > 1.0
    # hook removed: back to baseline
    assert np.allclose(after[1], base[1])
