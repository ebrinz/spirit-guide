from spiritbench.listener.basq import sample_questions, administer

BANK = [{"id": f"q{i}", "text": f"Does word{i} match how you feel?", "v": i / 10, "a": 0.5}
        for i in range(10)]


class FakeModel:
    """Says yes to even-indexed questions."""
    def __init__(self):
        self.calls = 0

    def yes_no_logprobs(self, prompt):
        self.calls += 1
        return (0.0, -1.0) if self.calls % 2 == 1 else (-1.0, 0.0)


def test_sample_deterministic():
    a = sample_questions(BANK, 5, seed=1)
    b = sample_questions(BANK, 5, seed=1)
    assert [q["id"] for q in a] == [q["id"] for q in b]
    assert len(a) == 5


def test_administer_scores_yes_mean():
    res = administer(FakeModel(), BANK[:4])
    assert res["n_yes"] == 2
    assert abs(res["va"][0] - (0.0 + 0.2) / 2) < 1e-9


def test_administer_no_yes_gives_center():
    class NoModel:
        def yes_no_logprobs(self, prompt):
            return (-1.0, 0.0)
    res = administer(NoModel(), BANK[:3])
    assert res["va"] == [0.5, 0.5]
