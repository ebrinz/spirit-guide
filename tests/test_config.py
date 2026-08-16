from spiritbench.config import load_config


def test_load_config_resolves_paths():
    cfg = load_config()
    assert cfg["listener_model"] == "unsloth/gemma-2-9b-it"
    assert cfg["ot_repo"].startswith("/")
    assert cfg["targets"]["calm"] == [0.75, 0.20]
    assert cfg["phrase_bank"]["max_lines"] == 50000
