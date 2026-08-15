from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/bench.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    # resolve path-like fields relative to repo root
    for key in ("ot_repo", "word_artifact", "nrc_lexicon", "questionnaire_bank",
                "semantic_axes", "glove_path"):
        cfg[key] = str((REPO_ROOT / cfg[key]).resolve())
    return cfg
