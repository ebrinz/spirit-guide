# Spirit-Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An offline bench that measures how well poetic meditations (built by deterministic constructors over VAD-enriched word/phrase graphs) place a listener LLM at target valence–arousal coordinates, read via an NRC-trained linear probe on hidden states.

**Architecture:** Numbered idempotent scripts pipe JSONL between stages: build phrase bank → build stimuli (constructors on word or phrase artifact) → optional Claude renders → train probe → run listener → analyze (metrics + graph-Laplacian harmonics + figures). One adapter module is the only code that touches `../ontological-traversal` (via `sys.path`), and it is artifact-agnostic: the same constructors emit word-level or phrase-level stimuli depending on which artifact they're pointed at.

**Tech Stack:** Python 3.10+, numpy, scipy, scikit-learn, torch (MPS), transformers, pandas, matplotlib, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-spirit-bench-design.md`

## Global Constraints

- Python ≥ 3.10; deps limited to: numpy, scipy, scikit-learn, torch, transformers, accelerate, pandas, matplotlib, pyyaml, pytest, requests. No TransformerLens, no LangChain.
- Listener model: `Qwen/Qwen3-1.7B` (already in HF cache), device `mps`, dtype float16 for weights, hidden states cast to float32.
- Probe validity gate: held-out valence R² ≥ 0.5 or STOP the pipeline (exit nonzero) — no sweep runs on a failed gate.
- `../ontological-traversal` (OT) is imported ONLY inside `src/spiritbench/stimuli/adapter.py` via `sys.path.insert`; nothing else imports OT.
- Phrase bank capped at 50,000 lines; k-NN k=10; phrase filter: 3–10 words, alphabetic, NRC content-word coverage ≥ 0.5, no negators (`no not never nothing none nor neither cannot don't won't can't`).
- All artifacts/data under `data/` (gitignored). Every script skips work whose output file already exists.
- Targets (VA in [0,1]²): calm (0.75, 0.20), focused (0.65, 0.60), excited (0.80, 0.85); rescue start (0.25, 0.80); neutral start (0.5, 0.5).
- Run tests with `python -m pytest tests/ -q` from repo root.

## OT interface facts (verified 2026-08-14, do not re-derive)

- Artifact JSON: `{"metadata": {..., "vectors_file": "<name>.npy"}, "words": {"<id>": {"word": str, "neighbors": [str], "vad": {"v": float, "a": float, ...}}}, "traversal_graph": {"edges": [{"from": int, "to": int, "distance": float}]}}`; vectors `.npy` float32, row i = node i, sits next to the JSON.
- `eeg.path_planner`: `build_adjacency(edges) -> dict[int, list[(int, float)]]`; `find_path(start_id: int, target_id: int, sections: dict, adjacency, target_va: (float, float)) -> list[int]`.
- `eeg.harmonic_path`: `load_harmonic_inputs(artifact_path, axes_path, vocab_cap) -> (vectors, word_index, id_to_word, v_axis, a_axis, semantic_axes)`; `plan_harmonic_waypoints(vectors, word_index, id_to_word, semantic_axes, v_axis, a_axis, start_word, target_word, *, steps=25, preset="golden", seed=42) -> list[dict]` with step dicts keyed `focus, focus_v, focus_a, ...`; presets: golden, prime, organic. `get_vocab_mask(id_to_word)` filters vocabulary — must be monkeypatched to all-True for phrase artifacts (phrases contain spaces).
- `eeg.sentence_builder`: `build_sentences(words, n=15, seed=None) -> list[str]` (needs `len(words) >= 2n`); also `build_short_sentences`, `build_long_sentences` (same shape).
- `eeg.generate_valley.filter_words_by_va(sections, min_id=0, v_min=None, v_max=None, a_min=None, a_max=None) -> list[str]` — schema-generic VA band filter (verify kwargs when importing; adapt if names differ).
- OT questionnaire bank `data/questionnaire_bank.json`: list of 500 `{"id": "q0000", "text": "Does ... how you feel right now?", "v": float, "a": float}`.
- OT `config/semantic_axes.json`: 12 named GloVe directions incl. `concreteness` (style axis source).
- NRC lexicon file (after task 1): `<OT>/data/nrc_vad/NRC-VAD-Lexicon.txt`, tab-separated `word\tvalence\tarousal\tdominance`, no header.

---

### Task 1: Scaffold + config + artifact setup kicked off in background

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `config/bench.yaml`, `src/spiritbench/__init__.py`, `src/spiritbench/config.py`, `tests/test_config.py`, `scripts/00_setup_artifacts.sh`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `spiritbench.config.load_config(path="config/bench.yaml") -> dict` (yaml → dict, paths resolved absolute relative to repo root). All later tasks call this.
- Produces (background): `<OT>/artifacts/word_graph_<date>_enriched.json` + `_vectors.npy`, `<OT>/data/nrc_vad/NRC-VAD-Lexicon.txt`, `data/glove/glove.6B.300d.txt`.

- [ ] **Step 1: Write files**

`requirements.txt`:
```
numpy
scipy
scikit-learn
torch
transformers
accelerate
pandas
matplotlib
pyyaml
pytest
requests
```

`pyproject.toml`:
```toml
[project]
name = "spiritbench"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`config/bench.yaml`:
```yaml
ot_repo: ../ontological-traversal
# word_artifact is filled in by 00_setup_artifacts.sh output name (date-stamped)
word_artifact: ../ontological-traversal/artifacts/word_graph_ENRICHED.json
nrc_lexicon: ../ontological-traversal/data/nrc_vad/NRC-VAD-Lexicon.txt
questionnaire_bank: ../ontological-traversal/data/questionnaire_bank.json
semantic_axes: ../ontological-traversal/config/semantic_axes.json
glove_path: data/glove/glove.6B.300d.txt

listener_model: Qwen/Qwen3-1.7B
device: mps
ema_alpha: 0.1
preamble: "You are listening to a guided meditation. Let each line settle before the next.\n\n"

targets:
  calm: [0.75, 0.20]
  focused: [0.65, 0.60]
  excited: [0.80, 0.85]
rescue_start: [0.25, 0.80]
neutral_start: [0.5, 0.5]

phrase_bank:
  max_lines: 50000
  k_neighbors: 10
  min_words: 3
  max_words: 10
  min_nrc_coverage: 0.5

probe:
  ridge_alpha: 10.0
  test_frac: 0.2
  r2_gate_valence: 0.5
  carrier_templates: ["The word is {w}", "{w}", "She whispered {w}"]

basq:
  n_questions: 30
  seed: 7

harmonics:
  n_modes: 100
```

`src/spiritbench/config.py`:
```python
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
```

`tests/test_config.py`:
```python
from spiritbench.config import load_config


def test_load_config_resolves_paths():
    cfg = load_config()
    assert cfg["listener_model"] == "Qwen/Qwen3-1.7B"
    assert cfg["ot_repo"].startswith("/")
    assert cfg["targets"]["calm"] == [0.75, 0.20]
    assert cfg["phrase_bank"]["max_lines"] == 50000
```

`scripts/00_setup_artifacts.sh`:
```bash
#!/bin/bash
# Idempotent artifact setup. Safe to re-run; each block skips if output exists.
set -euo pipefail
cd "$(dirname "$0")/.."
OT=../ontological-traversal

# 1. GloVe 6B (822MB zip). Kept in spirit-guide/data/glove.
mkdir -p data/glove
if [ ! -f data/glove/glove.6B.300d.txt ]; then
  curl -L --retry 3 -o data/glove/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
  unzip -o data/glove/glove.6B.zip glove.6B.300d.txt -d data/glove/
  rm data/glove/glove.6B.zip
fi

# 2. NRC VAD lexicon
mkdir -p "$OT/data/nrc_vad"
if [ ! -f "$OT/data/nrc_vad/NRC-VAD-Lexicon.txt" ]; then
  curl -L --retry 3 -o /tmp/nrc-vad.zip https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip
  unzip -o /tmp/nrc-vad.zip -d /tmp/nrc-vad
  find /tmp/nrc-vad -name "NRC-VAD-Lexicon.txt" -exec cp {} "$OT/data/nrc_vad/" \;
  rm -rf /tmp/nrc-vad /tmp/nrc-vad.zip
fi

# 3. Word graph (build + enrich, in OT)
if ! ls "$OT"/artifacts/word_graph_*_enriched.json >/dev/null 2>&1; then
  (cd "$OT" && python3 eeg/build_word_graph.py \
      --glove ../spirit-guide/data/glove/glove.6B.300d.txt --output-dir artifacts/)
  RAW=$(ls "$OT"/artifacts/word_graph_*.json | grep -v enriched | head -1)
  (cd "$OT" && python3 eeg/enrich_artifact.py \
      --artifact "artifacts/$(basename "$RAW")" \
      --nrc-vad data/nrc_vad/NRC-VAD-Lexicon.txt \
      --output "artifacts/$(basename "${RAW%.json}")_enriched.json")
fi

# 4. Gutenberg Poetry Corpus (~60MB gz)
mkdir -p data/corpus
if [ ! -f data/corpus/gutenberg-poetry-v001.ndjson.gz ]; then
  curl -L --retry 3 -o data/corpus/gutenberg-poetry-v001.ndjson.gz \
    http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz
fi

echo "ARTIFACTS READY"
ls "$OT"/artifacts/ data/glove/ data/corpus/
```

Append to `.gitignore`: `*.egg-info/`, `.pytest_cache/`.

- [ ] **Step 2: Install and verify test passes**

Run: `python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest tests/ -q`
Expected: `1 passed`

- [ ] **Step 3: Kick off artifact setup in background** (long: ~800MB download + k-NN build)

Run: `chmod +x scripts/00_setup_artifacts.sh && ./scripts/00_setup_artifacts.sh` **in background** (Bash `run_in_background: true`). Later tasks that need artifacts check for `ARTIFACTS READY`. After it completes, update `word_artifact` in `config/bench.yaml` to the actual dated filename.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pyproject.toml config/ src/ tests/ scripts/00_setup_artifacts.sh .gitignore
git commit -m "feat: scaffold spiritbench package, config, artifact setup script"
```

---

### Task 2: Shared test fixture — toy artifact

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: pytest fixtures `toy_artifact_dir` (tmp dir containing `toy.json` + `toy_vectors.npy` in exact OT artifact schema) and `toy_nrc_file` (mini NRC lexicon path). Used by Tasks 3–5, 10, 12.

- [ ] **Step 1: Write fixture**

`tests/conftest.py`:
```python
import json
import numpy as np
import pytest

# 12 nodes on a VA grid with 4-d embeddings; embedding dims 0/1 encode (v, a)
# so cosine neighbors correlate with VA neighbors — realistic enough for constructors.
TOY_WORDS = [
    ("dread", 0.1, 0.9), ("panic", 0.15, 0.85), ("gloom", 0.2, 0.3),
    ("plain", 0.5, 0.5), ("mild", 0.55, 0.45), ("steady", 0.6, 0.4),
    ("calm", 0.75, 0.2), ("serene", 0.8, 0.15), ("rest", 0.7, 0.25),
    ("joy", 0.85, 0.7), ("thrill", 0.8, 0.85), ("bliss", 0.9, 0.6),
]


def _vec(v, a, i):
    rng = np.random.RandomState(i)
    return np.array([v, a, 0.1 * rng.rand(), 0.1 * rng.rand()], dtype=np.float32)


@pytest.fixture
def toy_artifact_dir(tmp_path):
    vecs = np.stack([_vec(v, a, i) for i, (_, v, a) in enumerate(TOY_WORDS)])
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = unit @ unit.T
    words, nodes, edges, seen = {}, {}, [], set()
    for i, (w, v, a) in enumerate(TOY_WORDS):
        order = np.argsort(-sims[i])
        nbr_ids = [int(j) for j in order if j != i][:3]
        nodes[str(i)] = {"word": w, "neighbors": [TOY_WORDS[j][0] for j in nbr_ids],
                         "vad": {"v": v, "a": a, "source": "direct"}}
        for j in nbr_ids:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append({"from": i, "to": j, "distance": float(1 - sims[i, j])})
    artifact = {"metadata": {"n_words": len(TOY_WORDS), "k_neighbors": 3,
                             "artifact_type": "word_graph", "vectors_file": "toy_vectors.npy"},
                "words": nodes, "traversal_graph": {"edges": edges}}
    np.save(tmp_path / "toy_vectors.npy", vecs)
    with open(tmp_path / "toy.json", "w") as f:
        json.dump(artifact, f)
    return tmp_path


@pytest.fixture
def toy_nrc_file(tmp_path):
    p = tmp_path / "nrc.txt"
    rows = [(w, v, a, 0.5) for (w, v, a) in TOY_WORDS] + [
        ("river", 0.65, 0.35, 0.5), ("stone", 0.5, 0.3, 0.5), ("light", 0.7, 0.5, 0.5)]
    p.write_text("\n".join(f"{w}\t{v}\t{a}\t{d}" for w, v, a, d in rows))
    return p
```

- [ ] **Step 2: Sanity-run** — `python -m pytest tests/ -q` (fixtures import cleanly, existing test still passes). Commit:

```bash
git add tests/conftest.py && git commit -m "test: toy artifact + mini NRC fixtures"
```

---

### Task 3: Phrase bank — filtering, VAD scoring, artifact emit

**Files:**
- Create: `src/spiritbench/stimuli/__init__.py`, `src/spiritbench/stimuli/phrase_bank.py`, `tests/test_phrase_bank.py`, `scripts/01_build_phrase_bank.py`

**Interfaces:**
- Consumes: `load_config()`; Gutenberg corpus ndjson.gz (`{"s": line, ...}` per row); GloVe txt; NRC txt.
- Produces:
  - `load_nrc(path) -> dict[str, tuple[float, float]]` (word → (v, a))
  - `filter_lines(lines, nrc, min_words=3, max_words=10, min_coverage=0.5) -> list[str]`
  - `line_vad(line, nrc) -> tuple[float, float] | None`
  - `line_vector(line, glove: dict[str, np.ndarray]) -> np.ndarray | None`
  - `build_phrase_artifact(lines, glove, nrc, k, out_dir, name="phrase_graph") -> tuple[str, str]` — emits OT-schema artifact JSON + `.npy` (node "word" field holds the whole phrase string)
  - Script writes `data/phrase_bank/phrase_graph.json` + `phrase_graph_vectors.npy`

- [ ] **Step 1: Write failing tests**

`tests/test_phrase_bank.py`:
```python
import json
import numpy as np
from spiritbench.stimuli.phrase_bank import (
    load_nrc, filter_lines, line_vad, line_vector, build_phrase_artifact, NEGATORS)


def _glove():
    rng = np.random.RandomState(0)
    return {w: rng.rand(4).astype(np.float32)
            for w in ["calm", "river", "stone", "light", "joy", "dread", "the", "of"]}


def test_load_nrc(toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    assert nrc["calm"] == (0.75, 0.2)


def test_filter_drops_short_long_negated_lowcov(toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    lines = [
        "calm river stone",                # keep
        "calm",                            # too short
        "w " * 11,                         # too long
        "do not fear the calm river",      # negator
        "xyzzy qwfp zxcv",                 # zero NRC coverage
        "Calm River Stone!",               # keep (case/punct normalized)
    ]
    kept = filter_lines(lines, nrc)
    assert kept == ["calm river stone", "calm river stone"]
    assert "not" in NEGATORS


def test_line_vad_is_nrc_mean(toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    v, a = line_vad("calm river", nrc)
    assert abs(v - (0.75 + 0.65) / 2) < 1e-6
    assert abs(a - (0.2 + 0.35) / 2) < 1e-6


def test_build_phrase_artifact_matches_ot_schema(tmp_path, toy_nrc_file):
    nrc = load_nrc(toy_nrc_file)
    glove = _glove()
    lines = ["calm river stone", "joy light river", "dread stone light",
             "calm joy light", "river stone light", "calm light joy"]
    jpath, npath = build_phrase_artifact(lines, glove, nrc, k=2, out_dir=tmp_path)
    art = json.load(open(jpath))
    assert art["metadata"]["artifact_type"] == "word_graph"
    node0 = art["words"]["0"]
    assert node0["word"] == "calm river stone"
    assert "vad" in node0 and "neighbors" in node0
    assert len(art["traversal_graph"]["edges"]) > 0
    vecs = np.load(npath)
    assert vecs.shape == (6, 4)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_phrase_bank.py -q` → ImportError.

- [ ] **Step 3: Implement**

`src/spiritbench/stimuli/phrase_bank.py`:
```python
import json
import re
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

NEGATORS = frozenset("no not never nothing none nor neither cannot don't won't can't".split())
_WORD_RE = re.compile(r"[a-z]+")


def load_nrc(path) -> dict:
    nrc = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            try:
                nrc[parts[0].lower()] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue  # header row
    return nrc


def _tokens(line: str) -> list[str]:
    return _WORD_RE.findall(line.lower())


def filter_lines(lines, nrc, min_words=3, max_words=10, min_coverage=0.5) -> list[str]:
    kept = []
    for line in lines:
        toks = _tokens(line)
        if not (min_words <= len(toks) <= max_words):
            continue
        if any(t in NEGATORS for t in toks):
            continue
        cov = sum(t in nrc for t in toks) / len(toks)
        if cov < min_coverage:
            continue
        kept.append(" ".join(toks))
    return kept


def line_vad(line, nrc):
    vals = [nrc[t] for t in _tokens(line) if t in nrc]
    if not vals:
        return None
    return (float(np.mean([v for v, _ in vals])), float(np.mean([a for _, a in vals])))


def line_vector(line, glove):
    vecs = [glove[t] for t in _tokens(line) if t in glove]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def build_phrase_artifact(lines, glove, nrc, k, out_dir, name="phrase_graph"):
    rows, vecs = [], []
    for line in lines:
        vec, vad = line_vector(line, glove), line_vad(line, nrc)
        if vec is not None and vad is not None:
            rows.append((line, vad))
            vecs.append(vec)
    vectors = np.stack(vecs).astype(np.float32)
    unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    eff_k = min(k, len(rows) - 1)
    nn = NearestNeighbors(n_neighbors=eff_k + 1, metric="cosine", algorithm="brute").fit(unit)
    dists, idxs = nn.kneighbors(unit)
    nodes, edges, seen = {}, [], set()
    for i, (line, (v, a)) in enumerate(rows):
        nbrs = [int(j) for j in idxs[i, 1:]]
        nodes[str(i)] = {"word": line, "neighbors": [rows[j][0] for j in nbrs],
                         "vad": {"v": v, "a": a, "source": "nrc_mean"}}
        for j, d in zip(nbrs, dists[i, 1:]):
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append({"from": i, "to": int(j), "distance": float(d)})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_name = f"{name}_vectors.npy"
    np.save(out_dir / npy_name, vectors)
    artifact = {"metadata": {"n_words": len(rows), "k_neighbors": eff_k,
                             "artifact_type": "word_graph", "vectors_file": npy_name,
                             "source": "gutenberg-poetry-v001"},
                "words": nodes, "traversal_graph": {"edges": edges}}
    jpath = out_dir / f"{name}.json"
    with open(jpath, "w") as f:
        json.dump(artifact, f)
    return str(jpath), str(out_dir / npy_name)


def load_glove(path, vocab: set[str]) -> dict:
    """Load only vectors for words in vocab (memory-friendly)."""
    glove = {}
    for line in open(path, encoding="utf-8"):
        w, _, rest = line.partition(" ")
        if w in vocab:
            glove[w] = np.fromstring(rest, sep=" ", dtype=np.float32)
    return glove
```

`scripts/01_build_phrase_bank.py`:
```python
"""Build the PSG phrase artifact from the Gutenberg Poetry Corpus."""
import gzip
import json
import random
from pathlib import Path

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.stimuli import phrase_bank as pb


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/phrase_bank"
    if (out_dir / "phrase_graph.json").exists():
        print("phrase_graph.json exists — skipping")
        return
    pcfg = cfg["phrase_bank"]
    nrc = pb.load_nrc(cfg["nrc_lexicon"])
    corpus = REPO_ROOT / "data/corpus/gutenberg-poetry-v001.ndjson.gz"
    lines = (json.loads(l)["s"] for l in gzip.open(corpus, "rt", encoding="utf-8"))
    kept = pb.filter_lines(lines, nrc, pcfg["min_words"], pcfg["max_words"],
                           pcfg["min_nrc_coverage"])
    kept = sorted(set(kept))
    random.Random(42).shuffle(kept)
    kept = kept[: pcfg["max_lines"]]
    print(f"{len(kept)} lines after filter+cap")
    vocab = {t for line in kept for t in line.split()}
    glove = pb.load_glove(cfg["glove_path"], vocab)
    jpath, npath = pb.build_phrase_artifact(kept, glove, nrc, pcfg["k_neighbors"], out_dir)
    print("wrote", jpath, npath)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_phrase_bank.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/spiritbench/stimuli/ tests/test_phrase_bank.py scripts/01_build_phrase_bank.py
git commit -m "feat: PSG phrase bank — filter, NRC VAD scoring, OT-schema artifact"
```

---

### Task 4: Adapter part 1 — artifact loading, node masks, graph-walk & valley constructors

**Files:**
- Create: `src/spiritbench/stimuli/adapter.py`, `tests/test_adapter.py`

**Interfaces:**
- Consumes: toy fixtures; OT `eeg.path_planner` (via `sys.path`).
- Produces (all later stimulus code relies on these exact names):
  - `Art` dataclass: `.nodes` (dict id→node), `.vectors` (np.ndarray), `.id_of` (word→int), `.va(node_id) -> (v, a)`
  - `load_art(artifact_path: str) -> Art`
  - `node_mask(art, intensity: str | None) -> np.ndarray[bool]` — intensity "plain" keeps radial VA distance from (0.5, 0.5) ≤ 0.25; "heightened" keeps ≥ 0.2; None keeps all
  - `nearest_node_to_va(art, va, mask=None) -> int`
  - `graph_walk(art, start_va, target_va, n_lines: int, seed: int, ot_repo: str) -> list[int]` — node-id path via OT Dijkstra, resampled/truncated to n_lines
  - `valley_shape(art, target_va, n_lines, seed, mask=None) -> list[int]` — ground phase (v≥0.55, a≤0.35 band) → 3 ascending sub-bands toward target → target band; nearest-in-band selection without replacement
  - `stimulus_record(art, node_ids, constructor, generator, target_name, target_va, params) -> dict` with keys `id, constructor, generator, params, target, target_va, waypoints [{node, v, a}], lines [str], text` (`text` = lines joined with `".\n"`; `id` = `f"{constructor}-{generator}-{target_name}-" + sha1(text)[:8]`)

- [ ] **Step 1: Write failing tests**

`tests/test_adapter.py`:
```python
import numpy as np
from spiritbench.config import load_config
from spiritbench.stimuli.adapter import (
    load_art, node_mask, nearest_node_to_va, graph_walk, valley_shape, stimulus_record)

CFG = load_config()


def test_load_art(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    assert art.vectors.shape[0] == 12
    assert art.va(art.id_of["calm"]) == (0.75, 0.2)


def test_node_mask_intensity(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    plain = node_mask(art, "plain")
    heightened = node_mask(art, "heightened")
    assert plain[art.id_of["plain"]] and not plain[art.id_of["dread"]]
    assert heightened[art.id_of["dread"]]


def test_graph_walk_reaches_target(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    path = graph_walk(art, (0.5, 0.5), (0.75, 0.2), n_lines=6, seed=1,
                      ot_repo=CFG["ot_repo"])
    assert 1 <= len(path) <= 6
    v, a = art.va(path[-1])
    assert abs(v - 0.75) < 0.2 and abs(a - 0.2) < 0.2


def test_valley_shape_descends_then_targets(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = valley_shape(art, (0.85, 0.7), n_lines=8, seed=3)
    assert len(ids) == 8
    aro = [art.va(i)[1] for i in ids]
    assert min(aro[:3]) < 0.5          # grounding phase is low-arousal
    assert abs(art.va(ids[-1])[0] - 0.85) < 0.25  # ends near target valence


def test_stimulus_record_schema(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    rec = stimulus_record(art, [6, 7], "valley", "psg", "calm", (0.75, 0.2),
                          {"length": "short"})
    assert rec["lines"] == ["calm", "serene"]
    assert rec["waypoints"][0] == {"node": 6, "v": 0.75, "a": 0.2}
    assert rec["text"] == "calm.\nserene"
    assert rec["id"].startswith("valley-psg-calm-")
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement**

`src/spiritbench/stimuli/adapter.py`:
```python
"""The ONLY module allowed to import from ../ontological-traversal (via sys.path)."""
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _ot(ot_repo: str):
    if ot_repo not in sys.path:
        sys.path.insert(0, ot_repo)


@dataclass
class Art:
    nodes: dict
    vectors: np.ndarray
    id_of: dict

    def va(self, node_id: int):
        vad = self.nodes[str(node_id)]["vad"]
        return (vad["v"], vad["a"])

    def word(self, node_id: int) -> str:
        return self.nodes[str(node_id)]["word"]

    @property
    def edges(self):
        return self._edges


def load_art(artifact_path: str) -> Art:
    with open(artifact_path) as f:
        raw = json.load(f)
    vectors = np.load(Path(artifact_path).parent / raw["metadata"]["vectors_file"])
    nodes = raw["words"]
    art = Art(nodes=nodes, vectors=vectors,
              id_of={n["word"]: int(i) for i, n in nodes.items()})
    art._edges = raw["traversal_graph"]["edges"]
    return art


def _va_array(art: Art) -> np.ndarray:
    n = len(art.nodes)
    out = np.zeros((n, 2))
    for i in range(n):
        out[i] = art.va(i)
    return out


def node_mask(art: Art, intensity: str | None) -> np.ndarray:
    va = _va_array(art)
    r = np.linalg.norm(va - 0.5, axis=1)
    if intensity == "plain":
        return r <= 0.25
    if intensity == "heightened":
        return r >= 0.2
    return np.ones(len(va), dtype=bool)


def nearest_node_to_va(art: Art, va, mask=None) -> int:
    d = np.linalg.norm(_va_array(art) - np.asarray(va), axis=1)
    if mask is not None:
        d = np.where(mask, d, np.inf)
    return int(np.argmin(d))


def graph_walk(art: Art, start_va, target_va, n_lines, seed, ot_repo) -> list[int]:
    _ot(ot_repo)
    from eeg.path_planner import build_adjacency, find_path
    start = nearest_node_to_va(art, start_va)
    target = nearest_node_to_va(art, target_va)
    adjacency = build_adjacency(art.edges)
    path = find_path(start, target, art.nodes, adjacency, tuple(target_va))
    if len(path) > n_lines:  # resample evenly, always keeping endpoints
        idx = np.linspace(0, len(path) - 1, n_lines).round().astype(int)
        path = [path[i] for i in idx]
    return path


def _pick_in_band(art, va_lo, va_hi, k, rng, used, mask=None):
    """k nodes whose (v, a) falls in the band, nearest band-center first."""
    va = _va_array(art)
    lo, hi = np.asarray(va_lo), np.asarray(va_hi)
    ok = np.all((va >= lo) & (va <= hi), axis=1)
    if mask is not None:
        ok &= mask
    ids = [i for i in np.where(ok)[0] if i not in used]
    center = (lo + hi) / 2
    ids.sort(key=lambda i: np.linalg.norm(va[i] - center))
    picked = ids[:k]
    if len(picked) < k and ids:
        picked += list(rng.choices(ids, k=k - len(picked)))
    used.update(picked)
    return picked


def valley_shape(art: Art, target_va, n_lines, seed, mask=None) -> list[int]:
    rng = random.Random(seed)
    tv, ta = target_va
    used: set = set()
    n1 = max(1, n_lines // 3)
    n3 = max(1, n_lines // 4)
    n2 = n_lines - n1 - n3
    ids = _pick_in_band(art, (0.5, 0.0), (1.0, 0.4), n1, rng, used, mask)  # ground
    ground_a = 0.25
    for j in range(n2):  # ascend arousal (or descend) toward target in 3 sub-bands
        frac = (j + 1) / (n2 + 1)
        a_mid = ground_a + frac * (ta - ground_a)
        v_mid = 0.6 + frac * (tv - 0.6)
        ids += _pick_in_band(art, (v_mid - 0.15, a_mid - 0.15),
                             (v_mid + 0.15, a_mid + 0.15), 1, rng, used, mask)
    ids += _pick_in_band(art, (tv - 0.15, ta - 0.15), (tv + 0.15, ta + 0.15),
                         n3, rng, used, mask)
    return ids[:n_lines]


def stimulus_record(art: Art, node_ids, constructor, generator, target_name,
                    target_va, params) -> dict:
    lines = [art.word(i) for i in node_ids]
    text = ".\n".join(lines)
    sid = f"{constructor}-{generator}-{target_name}-" + \
        hashlib.sha1(text.encode()).hexdigest()[:8]
    return {"id": sid, "constructor": constructor, "generator": generator,
            "params": params, "target": target_name, "target_va": list(target_va),
            "waypoints": [{"node": int(i), "v": art.va(i)[0], "a": art.va(i)[1]}
                          for i in node_ids],
            "lines": lines, "text": text}
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_adapter.py -q` → pass. (If `find_path` import fails on missing OT deps, that specific test may need OT's requirements present — they are, since OT ran before.)

- [ ] **Step 5: Commit** — `git add -A src tests && git commit -m "feat: adapter — Art loader, masks, graph-walk + valley constructors"`

---

### Task 5: Adapter part 2 — harmonic presets, polygon-pca, style masks, word-level template generator

**Files:**
- Modify: `src/spiritbench/stimuli/adapter.py`
- Test: `tests/test_adapter2.py`

**Interfaces:**
- Produces:
  - `harmonic(art, artifact_path, start_va, target_va, n_lines, preset, seed, ot_repo, semantic_axes_path) -> list[int]` — wraps OT `plan_harmonic_waypoints` with `get_vocab_mask` monkeypatched to all-True; maps returned `focus` words to node ids; preset ∈ {"golden", "prime", "organic"}
  - `polygon_pca(art, start_va, target_va, n_lines, seed) -> list[int]` — native: slerp-free linear waypoint track in VA space; at each of n_lines steps, take the 50 embedding-NNs of the current focus node, PCA to 2 components, inscribe a pentagon rotating 15°/step, pick the nearest node to the active vertex
  - `style_mask(art, style: str | None, axes_path: str) -> np.ndarray[bool]` — "imagist" = top-40% projection on the `concreteness` axis, "abstract" = bottom-40%, None = all. Phrase vectors are mean-GloVe so the axis (a GloVe direction) applies directly; on the toy 4-d artifact tests pass a synthetic axes file
  - `template_wrap(lines: list[str], length: str, seed: int, ot_repo: str) -> list[str]` — word-level generator: wraps a word sequence with OT `sentence_builder.build_sentences` (`short`→`build_short_sentences`, `long`→`build_long_sentences`); pads the pool by repeating words if `< 2n`
  - `LENGTH_LINES = {"short": 8, "medium": 24, "long": 56}`

- [ ] **Step 1: Write failing tests**

`tests/test_adapter2.py`:
```python
import json
import numpy as np
from spiritbench.config import load_config
from spiritbench.stimuli.adapter import (
    load_art, harmonic, polygon_pca, style_mask, template_wrap, LENGTH_LINES)

CFG = load_config()


def _axes_file(tmp_path):
    # concreteness axis along embedding dim 2 of the toy vectors
    p = tmp_path / "axes.json"
    p.write_text(json.dumps({"concreteness": {"positive": ["calm"], "negative": ["dread"]}}))
    return str(p)


def test_polygon_pca_returns_n_nodes(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = polygon_pca(art, (0.5, 0.5), (0.75, 0.2), n_lines=6, seed=2)
    assert len(ids) == 6
    assert all(0 <= i < 12 for i in ids)


def test_style_mask_partitions(toy_artifact_dir, tmp_path):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    imag = style_mask(art, "imagist", _axes_file(tmp_path))
    abst = style_mask(art, "abstract", _axes_file(tmp_path))
    assert imag.sum() > 0 and abst.sum() > 0
    assert not np.any(imag & abst)


def test_template_wrap_lengths():
    words = ["calm", "river", "stone", "light", "joy", "mist", "reed", "moon"]
    out = template_wrap(words, "short", seed=1, ot_repo=CFG["ot_repo"])
    assert len(out) >= 1 and all(isinstance(s, str) for s in out)
    assert LENGTH_LINES["medium"] == 24


def test_harmonic_runs_on_artifact(toy_artifact_dir, tmp_path):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    ids = harmonic(art, str(toy_artifact_dir / "toy.json"), (0.5, 0.5), (0.75, 0.2),
                   n_lines=5, preset="golden", seed=1, ot_repo=CFG["ot_repo"],
                   semantic_axes_path=_axes_file(tmp_path))
    assert 1 <= len(ids) <= 5
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** (append to `adapter.py`)

```python
LENGTH_LINES = {"short": 8, "medium": 24, "long": 56}


def style_mask(art: Art, style, axes_path) -> np.ndarray:
    if style is None:
        return np.ones(len(art.nodes), dtype=bool)
    with open(axes_path) as f:
        axes = json.load(f)
    ax = axes["concreteness"]
    pos = np.mean([art.vectors[art.id_of[w]] for w in ax["positive"] if w in art.id_of], axis=0) \
        if any(w in art.id_of for w in ax["positive"]) else None
    neg = np.mean([art.vectors[art.id_of[w]] for w in ax["negative"] if w in art.id_of], axis=0) \
        if any(w in art.id_of for w in ax["negative"]) else None
    if pos is None or neg is None:  # axis words absent (e.g. phrase artifact): project on raw GloVe diff
        raise ValueError("concreteness axis words not in artifact; pass a GloVe-diff axis vector")
    direction = pos - neg
    proj = art.vectors @ direction
    if style == "imagist":
        return proj >= np.quantile(proj, 0.6)
    if style == "abstract":
        return proj <= np.quantile(proj, 0.4)
    raise ValueError(style)


def polygon_pca(art: Art, start_va, target_va, n_lines, seed) -> list[int]:
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(seed)
    ids = []
    focus = nearest_node_to_va(art, start_va)
    for step in range(n_lines):
        frac = step / max(1, n_lines - 1)
        wp_va = (1 - frac) * np.asarray(start_va) + frac * np.asarray(target_va)
        focus = nearest_node_to_va(art, wp_va)
        fvec = art.vectors[focus]
        d = np.linalg.norm(art.vectors - fvec, axis=1)
        nn = np.argsort(d)[1:51]
        comps = PCA(n_components=2).fit(art.vectors[nn]).components_
        theta = np.deg2rad(15 * step) + 2 * np.pi * (step % 5) / 5
        radius = 0.15 * np.linalg.norm(fvec)
        probe_vec = fvec + radius * (np.cos(theta) * comps[0] + np.sin(theta) * comps[1])
        pick = int(np.argmin(np.linalg.norm(art.vectors - probe_vec, axis=1)))
        ids.append(pick)
    return ids


def harmonic(art: Art, artifact_path, start_va, target_va, n_lines, preset, seed,
             ot_repo, semantic_axes_path) -> list[int]:
    _ot(ot_repo)
    import eeg.harmonic_path as hp
    hp.get_vocab_mask = lambda id_to_word: np.ones(len(id_to_word), dtype=bool)
    vectors, word_index, id_to_word, v_axis, a_axis, semantic_axes = \
        hp.load_harmonic_inputs(artifact_path, axes_path=semantic_axes_path,
                                vocab_cap=len(art.nodes))
    start_word = art.word(nearest_node_to_va(art, start_va))
    target_word = art.word(nearest_node_to_va(art, target_va))
    path = hp.plan_harmonic_waypoints(
        vectors, word_index, id_to_word, semantic_axes, v_axis, a_axis,
        start_word, target_word, steps=n_lines, preset=preset, seed=seed)
    return [art.id_of[s["focus"]] for s in path if s["focus"] in art.id_of][:n_lines]


def template_wrap(lines, length, seed, ot_repo) -> list[str]:
    _ot(ot_repo)
    from eeg.sentence_builder import build_sentences, build_short_sentences, build_long_sentences
    fn = {"short": build_short_sentences, "medium": build_sentences,
          "long": build_long_sentences}[length]
    n = max(1, len(lines) // 2)
    pool = list(lines)
    while len(pool) < 2 * n:
        pool += lines
    return fn(pool, n=n, seed=seed)
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_adapter2.py -q`. The harmonic test exercises real OT code on the toy artifact; if OT's internals reject the 4-d toy vectors (e.g., hardcoded 300-d assumptions), mark that single test `@pytest.mark.skip(reason="OT needs 300d; covered by smoke run")` and rely on the Task 13 smoke run — do not stub OT.

- [ ] **Step 5: Commit** — `git commit -am "feat: adapter — harmonic, polygon-pca, style masks, template wrap"`

---

### Task 6: Controls + stimulus build script

**Files:**
- Create: `src/spiritbench/stimuli/controls.py`, `tests/test_controls.py`, `scripts/02_build_stimuli.py`

**Interfaces:**
- Consumes: all adapter functions (Task 4/5 signatures), `load_config()`.
- Produces:
  - `controls.shuffled(stim: dict, seed: int) -> dict` — same lines permuted, `constructor="shuffled:"+orig`, waypoints re-ordered to match, new id
  - `controls.neutral_stimulus(target_name, target_va) -> dict` — fixed built-in neutral technical text (a paragraph of appliance-manual prose, hardcoded), empty waypoints
  - `data/stimuli/stimuli.jsonl` — one stimulus record per line; **core comparison** = constructors {graph-walk, valley, harmonic-golden, harmonic-prime, harmonic-organic, polygon-pca} × generators {psg, word-template} × targets {calm, focused, excited, rescue} at medium/plain/unfiltered (48 stimuli), + sweeps {length ∈ short/long, intensity=heightened, style ∈ imagist/abstract} on {valley, harmonic-golden} × psg × {calm, rescue} (16), + shuffled controls of every core psg-calm stimulus (6), + 1 neutral ≈ **71 stimuli**. Rescue = start (0.25, 0.80), target calm; others start (0.5, 0.5).

- [ ] **Step 1: Write failing tests**

`tests/test_controls.py`:
```python
from spiritbench.stimuli.adapter import load_art, stimulus_record
from spiritbench.stimuli.controls import shuffled, neutral_stimulus


def test_shuffled_permutes_but_preserves_multiset(toy_artifact_dir):
    art = load_art(str(toy_artifact_dir / "toy.json"))
    stim = stimulus_record(art, [0, 3, 6, 9, 2, 5], "valley", "psg", "calm",
                           (0.75, 0.2), {})
    ctrl = shuffled(stim, seed=1)
    assert sorted(ctrl["lines"]) == sorted(stim["lines"])
    assert ctrl["lines"] != stim["lines"]
    assert ctrl["constructor"] == "shuffled:valley"
    assert ctrl["id"] != stim["id"]
    assert [w["node"] for w in ctrl["waypoints"]] != [w["node"] for w in stim["waypoints"]]


def test_neutral_stimulus():
    n = neutral_stimulus("calm", (0.75, 0.2))
    assert n["constructor"] == "neutral"
    assert len(n["text"]) > 200
    assert n["waypoints"] == []
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/stimuli/controls.py`:
```python
import hashlib
import random

NEUTRAL_TEXT = (
    "To operate the dishwasher, first ensure the filter assembly is seated in the "
    "sump housing. Load plates between the tines facing the center. The detergent "
    "dispenser accepts powder or tablet formats; close the lid until it clicks. "
    "Select a cycle using the control panel. The normal cycle runs 2 hours 15 "
    "minutes at 130 degrees. The rinse aid reservoir should be refilled monthly. "
    "If error code E4 appears, check the inlet hose for kinks and confirm the "
    "water supply valve is fully open before restarting the unit."
)


def shuffled(stim: dict, seed: int) -> dict:
    rng = random.Random(seed)
    order = list(range(len(stim["lines"])))
    while True:
        rng.shuffle(order)
        if [stim["lines"][i] for i in order] != stim["lines"] or len(order) < 2:
            break
    lines = [stim["lines"][i] for i in order]
    text = ".\n".join(lines)
    out = dict(stim)
    out.update({
        "constructor": "shuffled:" + stim["constructor"],
        "lines": lines, "text": text,
        "waypoints": [stim["waypoints"][i] for i in order],
        "id": f"shuffled-{stim['id']}-" + hashlib.sha1(text.encode()).hexdigest()[:8],
    })
    return out


def neutral_stimulus(target_name, target_va) -> dict:
    return {"id": f"neutral-{target_name}", "constructor": "neutral",
            "generator": "none", "params": {}, "target": target_name,
            "target_va": list(target_va), "waypoints": [], "lines": [],
            "text": NEUTRAL_TEXT}
```

`scripts/02_build_stimuli.py`:
```python
"""Build all bench stimuli (core grid + sweeps + controls) into data/stimuli/stimuli.jsonl."""
import json
from pathlib import Path

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.stimuli import adapter as ad
from spiritbench.stimuli.controls import shuffled, neutral_stimulus

CONSTRUCTORS = ["graph-walk", "valley", "harmonic-golden", "harmonic-prime",
                "harmonic-organic", "polygon-pca"]


def build_one(art, artifact_path, constructor, target_name, target_va, start_va,
              length, intensity, style, seed, cfg):
    n = ad.LENGTH_LINES[length]
    mask = ad.node_mask(art, intensity if intensity != "unfiltered" else None)
    if style not in (None, "unfiltered"):
        mask &= ad.style_mask(art, style, cfg["semantic_axes"])
    if constructor == "graph-walk":
        ids = ad.graph_walk(art, start_va, target_va, n, seed, cfg["ot_repo"])
    elif constructor == "valley":
        ids = ad.valley_shape(art, target_va, n, seed, mask)
    elif constructor.startswith("harmonic-"):
        ids = ad.harmonic(art, artifact_path, start_va, target_va, n,
                          constructor.split("-")[1], seed, cfg["ot_repo"],
                          cfg["semantic_axes"])
    elif constructor == "polygon-pca":
        ids = ad.polygon_pca(art, start_va, target_va, n, seed)
    else:
        raise ValueError(constructor)
    return ids


def main():
    cfg = load_config()
    out = REPO_ROOT / "data/stimuli/stimuli.jsonl"
    if out.exists():
        print("stimuli.jsonl exists — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    word_art = ad.load_art(cfg["word_artifact"])
    phrase_path = str(REPO_ROOT / "data/phrase_bank/phrase_graph.json")
    phrase_art = ad.load_art(phrase_path)
    arts = {"psg": (phrase_art, phrase_path),
            "word-template": (word_art, cfg["word_artifact"])}
    targets = dict(cfg["targets"])
    runs = []
    # core: every constructor x generator x target, medium/plain/unfiltered
    for cons in CONSTRUCTORS:
        for gen in ["psg", "word-template"]:
            for tname, tva in list(targets.items()) + [("rescue", targets["calm"])]:
                start = cfg["rescue_start"] if tname == "rescue" else cfg["neutral_start"]
                runs.append((cons, gen, tname, tuple(tva), tuple(start),
                             "medium", "plain", None, 42))
    # sweeps on valley + harmonic-golden, psg, calm + rescue
    for cons in ["valley", "harmonic-golden"]:
        for tname in ["calm", "rescue"]:
            tva = tuple(targets["calm"])
            start = cfg["rescue_start"] if tname == "rescue" else cfg["neutral_start"]
            for length in ["short", "long"]:
                runs.append((cons, "psg", tname, tva, tuple(start), length, "plain", None, 42))
            runs.append((cons, "psg", tname, tva, tuple(start), "medium", "heightened", None, 42))
            for style in ["imagist", "abstract"]:
                runs.append((cons, "psg", tname, tva, tuple(start), "medium", "plain", style, 42))
    records, failures = [], []
    for cons, gen, tname, tva, start, length, intensity, style, seed in runs:
        art, apath = arts[gen]
        try:
            ids = build_one(art, apath, cons, tname, tva, start, length,
                            intensity, style, seed, cfg)
        except Exception as e:
            failures.append((cons, gen, tname, length, intensity, style, repr(e)))
            continue
        rec = ad.stimulus_record(art, ids, cons, gen, tname, tva,
                                 {"length": length, "intensity": intensity,
                                  "style": style or "unfiltered", "seed": seed,
                                  "start_va": list(start)})
        if gen == "word-template":
            rec["lines"] = ad.template_wrap(rec["lines"], length, seed, cfg["ot_repo"])
            rec["text"] = "\n".join(rec["lines"])
        records.append(rec)
    # controls: shuffled for each core psg calm stimulus, one neutral
    for rec in [r for r in records if r["generator"] == "psg" and r["target"] == "calm"
                and r["params"]["length"] == "medium"
                and r["params"]["intensity"] == "plain"
                and r["params"]["style"] == "unfiltered"]:
        records.append(shuffled(rec, seed=99))
    records.append(neutral_stimulus("calm", targets["calm"]))
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} stimuli; {len(failures)} failures")
    for fail in failures:
        print("FAILED:", fail)  # no silent caps


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests** — `python -m pytest tests/test_controls.py -q` → pass.

- [ ] **Step 5: Commit** — `git add -A src tests scripts && git commit -m "feat: controls + stimulus grid build script"`

---

### Task 7: Listener model wrapper (hidden states, spans, yes/no scoring)

**Files:**
- Create: `src/spiritbench/listener/__init__.py`, `src/spiritbench/listener/model.py`, `tests/test_model.py`

**Interfaces:**
- Produces:
  - `HiddenStateModel(model_id: str, device: str = "cpu")` — loads AutoTokenizer + AutoModelForCausalLM (`torch_dtype=torch.float16` on mps, float32 on cpu), `eval()` mode
  - `.n_layers -> int` (number of hidden-state tensors = transformer layers + 1 embedding layer)
  - `.hidden_states(text: str) -> np.ndarray` shape `[n_layers, n_tokens, d]`, float32 (single forward pass, `torch.no_grad()`)
  - `.hidden_states_with_spans(preamble: str, lines: list[str]) -> (np.ndarray, list[tuple[int, int]])` — full text = `preamble + ".\n".join(lines)`; spans are (start, end) token indices for each line, computed by incremental tokenization of the growing prefix
  - `.yes_no_logprobs(prompt: str) -> tuple[float, float]` — log-softmax of the final-position logits, summed over each of the token variants `["yes", " yes", "Yes", " Yes"]` vs the "no" variants
- Tests use `sshleifer/tiny-gpt2` on cpu (2 MB download) — the wrapper is architecture-agnostic.

- [ ] **Step 1: Write failing tests**

`tests/test_model.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/listener/model.py`:
```python
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HiddenStateModel:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, output_hidden_states=True).to(device).eval()
        self.n_layers = self.model.config.num_hidden_layers + 1

    @torch.no_grad()
    def _forward(self, text: str):
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**ids)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [n_layers, n_tokens, d]
        return hs.float().cpu().numpy(), out.logits[0, -1].float().cpu(), ids["input_ids"].shape[1]

    def hidden_states(self, text: str) -> np.ndarray:
        hs, _, _ = self._forward(text)
        return hs

    def hidden_states_with_spans(self, preamble: str, lines: list[str]):
        spans, prefix = [], preamble
        for i, line in enumerate(lines):
            start = len(self.tokenizer(prefix)["input_ids"])
            prefix = prefix + line + (".\n" if i < len(lines) - 1 else "")
            end = len(self.tokenizer(prefix)["input_ids"])
            spans.append((start, max(end, start + 1)))
        hs = self.hidden_states(prefix)
        spans = [(s, min(e, hs.shape[1])) for s, e in spans]
        return hs, spans

    @torch.no_grad()
    def yes_no_logprobs(self, prompt: str):
        _, logits, _ = self._forward(prompt)
        logprobs = torch.log_softmax(logits, dim=-1)

        def score(variants):
            tot = -np.inf
            for v in variants:
                toks = self.tokenizer(v, add_special_tokens=False)["input_ids"]
                if len(toks) == 1:
                    tot = np.logaddexp(tot, logprobs[toks[0]].item())
            return tot
        return score(["yes", " yes", "Yes", " Yes"]), score(["no", " no", "No", " No"])
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_model.py -q` → pass (downloads tiny-gpt2 on first run).

- [ ] **Step 5: Commit** — `git add -A src tests && git commit -m "feat: listener model wrapper — hidden states, line spans, yes/no scoring"`

---

### Task 8: VA probe — training, layer selection, R² gate

**Files:**
- Create: `src/spiritbench/listener/probe.py`, `tests/test_probe.py`, `scripts/04_train_probe.py`

**Interfaces:**
- Consumes: `HiddenStateModel` (Task 7), `load_nrc` (Task 3), config `probe` block.
- Produces:
  - `collect_word_states(model, words: list[str], templates: list[str]) -> np.ndarray` shape `[n_words, n_layers, d]` — for each word, mean over templates of the **final-token** hidden state of `template.format(w=word)`
  - `train_probe(states, v_targets, a_targets, alpha, test_frac, seed=0) -> Probe` where `Probe` is a dataclass with `.layer: int`, `.ridge_v`, `.ridge_a` (fitted sklearn Ridge), `.r2_v: float`, `.r2_a: float`; layer chosen by held-out valence R²
  - `Probe.predict(hidden: np.ndarray) -> np.ndarray` — `[n_tokens, d]` at `.layer` → `[n_tokens, 2]` (v, a)
  - `save_probe(probe, path)` / `load_probe(path)` (pickle)
  - Script writes `data/probe/probe.pkl` + `data/probe/probe_report.json` (`{"layer", "r2_v", "r2_a", "n_words"}`), **exits 1 if `r2_v < r2_gate_valence`** printing `PROBE GATE FAILED`.

- [ ] **Step 1: Write failing tests**

`tests/test_probe.py`:
```python
import numpy as np
from spiritbench.listener.probe import train_probe, save_probe, load_probe


def _synthetic(n=400, d=16, layers=3, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    v, a = rng.rand(n), rng.rand(n)
    states = rng.randn(n, layers, d) * 0.1
    # layer 1 linearly encodes (v, a); other layers are noise
    w_v, w_a = rng.randn(d), rng.randn(d)
    states[:, 1, :] += np.outer(v, w_v) + np.outer(a, w_a)
    states[:, 1, :] += noise * rng.randn(n, d)
    return states, v, a


def test_probe_recovers_signal_and_layer(tmp_path):
    states, v, a = _synthetic()
    probe = train_probe(states, v, a, alpha=1.0, test_frac=0.2)
    assert probe.layer == 1
    assert probe.r2_v > 0.8 and probe.r2_a > 0.8
    preds = probe.predict(states[:5, 1, :])
    assert preds.shape == (5, 2)
    save_probe(probe, tmp_path / "p.pkl")
    p2 = load_probe(tmp_path / "p.pkl")
    assert p2.layer == 1


def test_probe_fails_on_noise():
    rng = np.random.RandomState(1)
    states = rng.randn(300, 3, 16)
    probe = train_probe(states, rng.rand(300), rng.rand(300), alpha=1.0, test_frac=0.2)
    assert probe.r2_v < 0.3
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/listener/probe.py`:
```python
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


@dataclass
class Probe:
    layer: int
    ridge_v: Ridge
    ridge_a: Ridge
    r2_v: float
    r2_a: float

    def predict(self, hidden: np.ndarray) -> np.ndarray:
        return np.stack([self.ridge_v.predict(hidden),
                         self.ridge_a.predict(hidden)], axis=1)


def collect_word_states(model, words, templates) -> np.ndarray:
    out = []
    for w in words:
        per_tmpl = [model.hidden_states(t.format(w=w))[:, -1, :] for t in templates]
        out.append(np.mean(per_tmpl, axis=0))  # [n_layers, d]
    return np.stack(out)


def train_probe(states, v_targets, a_targets, alpha, test_frac, seed=0) -> Probe:
    n_layers = states.shape[1]
    idx_tr, idx_te = train_test_split(np.arange(len(states)), test_size=test_frac,
                                      random_state=seed)
    best = None
    for layer in range(n_layers):
        X_tr, X_te = states[idx_tr, layer], states[idx_te, layer]
        rv = Ridge(alpha=alpha).fit(X_tr, v_targets[idx_tr])
        ra = Ridge(alpha=alpha).fit(X_tr, a_targets[idx_tr])
        r2v = r2_score(v_targets[idx_te], rv.predict(X_te))
        r2a = r2_score(a_targets[idx_te], ra.predict(X_te))
        if best is None or r2v > best.r2_v:
            best = Probe(layer, rv, ra, r2v, r2a)
    return best


def save_probe(probe, path):
    with open(path, "wb") as f:
        pickle.dump(probe, f)


def load_probe(path) -> Probe:
    with open(path, "rb") as f:
        return pickle.load(f)
```

`scripts/04_train_probe.py`:
```python
"""Train the NRC VA probe on the listener model. Exits 1 if the validity gate fails."""
import json
import sys
from pathlib import Path

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import collect_word_states, train_probe, save_probe
from spiritbench.stimuli.phrase_bank import load_nrc


def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data/probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "probe.pkl").exists():
        print("probe.pkl exists — skipping")
        return
    nrc = load_nrc(cfg["nrc_lexicon"])
    words = sorted(nrc)  # ~20k; subsample for tractability
    rng = np.random.RandomState(0)
    words = [words[i] for i in rng.choice(len(words), size=4000, replace=False)]
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    states = collect_word_states(model, words, cfg["probe"]["carrier_templates"])
    v = np.array([nrc[w][0] for w in words])
    a = np.array([nrc[w][1] for w in words])
    probe = train_probe(states, v, a, alpha=cfg["probe"]["ridge_alpha"],
                        test_frac=cfg["probe"]["test_frac"])
    report = {"layer": probe.layer, "r2_v": probe.r2_v, "r2_a": probe.r2_a,
              "n_words": len(words)}
    print(report)
    with open(out_dir / "probe_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if probe.r2_v < cfg["probe"]["r2_gate_valence"]:
        print("PROBE GATE FAILED: valence R2 below gate — do not run the sweep")
        sys.exit(1)
    save_probe(probe, out_dir / "probe.pkl")
    print("probe saved")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_probe.py -q` → pass.

- [ ] **Step 5: Commit** — `git add -A src tests scripts && git commit -m "feat: NRC VA probe with layer selection and validity gate"`

---

### Task 9: BASQ for models

**Files:**
- Create: `src/spiritbench/listener/basq.py`, `tests/test_basq.py`

**Interfaces:**
- Consumes: `HiddenStateModel.yes_no_logprobs`; OT questionnaire bank (list of `{"id", "text", "v", "a"}`).
- Produces:
  - `sample_questions(bank: list, n: int, seed: int) -> list` — uniform sample without replacement
  - `administer(model, questions, context: str = "") -> dict` — for each question builds prompt `context + "Question: " + q["text"] + "\nAnswer yes or no.\nAnswer:"`, answers yes iff yes-logprob > no-logprob; returns `{"va": [v, a], "n_yes": int, "answers": [{"id", "yes": bool}]}` where va = mean (v, a) of yes-questions, or `[0.5, 0.5]` if none (matches OT's BASQ scoring rule)

- [ ] **Step 1: Write failing tests**

`tests/test_basq.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/listener/basq.py`:
```python
import random


def sample_questions(bank, n, seed):
    rng = random.Random(seed)
    return rng.sample(bank, min(n, len(bank)))


def administer(model, questions, context: str = "") -> dict:
    answers, yes_vas = [], []
    for q in questions:
        prompt = f"{context}Question: {q['text']}\nAnswer yes or no.\nAnswer:"
        y, n = model.yes_no_logprobs(prompt)
        is_yes = y > n
        answers.append({"id": q["id"], "yes": bool(is_yes)})
        if is_yes:
            yes_vas.append((q["v"], q["a"]))
    if yes_vas:
        va = [sum(v for v, _ in yes_vas) / len(yes_vas),
              sum(a for _, a in yes_vas) / len(yes_vas)]
    else:
        va = [0.5, 0.5]
    return {"va": va, "n_yes": len(yes_vas), "answers": answers}
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_basq.py -q` → pass.

- [ ] **Step 5: Commit** — `git add -A src tests && git commit -m "feat: BASQ questionnaire administration for the listener model"`

---

### Task 10: Metrics

**Files:**
- Create: `src/spiritbench/analysis/__init__.py`, `src/spiritbench/analysis/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces (trajectory = np.ndarray `[n, 2]` of per-token (v, a)):
  - `ema(traj, alpha) -> np.ndarray` — exponential moving average along axis 0
  - `placement_error(traj, target_va) -> float` — euclidean distance of `traj[-1]` from target
  - `displacement(traj, target_va) -> float` — `d(traj[0], target) − d(traj[-1], target)` (positive = moved toward target)
  - `stability(traj) -> float` — mean per-dimension std over the final third
  - `adherence(line_vas: np.ndarray, waypoint_vas: np.ndarray) -> float` — mean euclidean distance between per-line probe VA and planned waypoint VA (same length; if empty waypoints return `np.nan`)
  - `per_line_va(traj, spans: list[tuple[int, int]]) -> np.ndarray` — mean of traj rows within each span

- [ ] **Step 1: Write failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
from spiritbench.analysis.metrics import (
    ema, placement_error, displacement, stability, adherence, per_line_va)


def test_ema_smooths():
    traj = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
    sm = ema(traj, alpha=0.5)
    assert sm.shape == traj.shape
    assert np.all(np.abs(np.diff(sm[:, 0])) <= np.abs(np.diff(traj[:, 0])) + 1e-12)


def test_placement_and_displacement():
    traj = np.array([[0.5, 0.5], [0.6, 0.4], [0.75, 0.2]])
    assert placement_error(traj, (0.75, 0.2)) < 1e-9
    assert displacement(traj, (0.75, 0.2)) > 0.3


def test_stability_low_when_settled():
    settled = np.vstack([np.random.RandomState(0).randn(30, 2) * 0.001 + 0.7])
    wild = np.random.RandomState(0).randn(30, 2) * 0.3
    assert stability(settled) < stability(wild)


def test_adherence_and_per_line():
    traj = np.array([[0.1, 0.1]] * 4 + [[0.9, 0.9]] * 4)
    lines = per_line_va(traj, [(0, 4), (4, 8)])
    assert np.allclose(lines, [[0.1, 0.1], [0.9, 0.9]])
    wps = np.array([[0.1, 0.1], [0.9, 0.9]])
    assert adherence(lines, wps) < 1e-9
    assert np.isnan(adherence(lines, np.empty((0, 2))))
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/analysis/metrics.py`:
```python
import numpy as np


def ema(traj: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(traj, dtype=float)
    out[0] = traj[0]
    for i in range(1, len(traj)):
        out[i] = alpha * traj[i] + (1 - alpha) * out[i - 1]
    return out


def placement_error(traj, target_va) -> float:
    return float(np.linalg.norm(traj[-1] - np.asarray(target_va)))


def displacement(traj, target_va) -> float:
    t = np.asarray(target_va)
    return float(np.linalg.norm(traj[0] - t) - np.linalg.norm(traj[-1] - t))


def stability(traj) -> float:
    tail = traj[len(traj) * 2 // 3:]
    return float(np.mean(np.std(tail, axis=0)))


def per_line_va(traj, spans) -> np.ndarray:
    return np.array([traj[s:e].mean(axis=0) for s, e in spans])


def adherence(line_vas, waypoint_vas) -> float:
    if len(waypoint_vas) == 0 or len(line_vas) == 0:
        return float("nan")
    n = min(len(line_vas), len(waypoint_vas))
    return float(np.mean(np.linalg.norm(line_vas[:n] - waypoint_vas[:n], axis=1)))
```

- [ ] **Step 4: Run tests** — pass. **Step 5: Commit** — `git commit -am "feat: trajectory metrics"`

---

### Task 11: Harmonics (graph Laplacian spectra)

**Files:**
- Create: `src/spiritbench/analysis/harmonics.py`, `tests/test_harmonics.py`

**Interfaces:**
- Consumes: artifact edges (`{"from", "to", "distance"}`), stimulus waypoint node ids.
- Produces:
  - `build_laplacian(edges: list[dict], n_nodes: int) -> scipy.sparse.csr_matrix` — symmetric normalized Laplacian `I − D^{-1/2} W D^{-1/2}` with `w = 1 − distance` (cosine similarity as weight, clipped ≥ 0)
  - `eigenmodes(L, k) -> (eigvals [k], eigvecs [n_nodes, k])` — k smallest via `scipy.sparse.linalg.eigsh(..., sigma=0, which="LM")` (shift-invert; fall back to `which="SM"` on failure)
  - `stimulus_spectrum(node_ids, eigvecs) -> np.ndarray [k]` — mean over the stimulus's nodes of squared eigenvector components, normalized to sum 1
  - `low_freq_fraction(spectrum, frac=0.2) -> float` — energy share of the lowest `frac` of modes
  - `spectral_centroid(spectrum, eigvals) -> float` — `Σ λ_k E_k`

- [ ] **Step 1: Write failing tests** (analytic check: on a path graph, low modes are global — a spatially contiguous walk concentrates energy in low modes vs. a scattered one)

`tests/test_harmonics.py`:
```python
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
    scattered = [0, 29, 5, 24, 10, 19]
    sc_c = spectral_centroid(stimulus_spectrum(contiguous, vecs), vals)
    sc_s = spectral_centroid(stimulus_spectrum(scattered, vecs), vals)
    assert sc_c < sc_s
    spec = stimulus_spectrum(contiguous, vecs)
    assert abs(spec.sum() - 1) < 1e-9
    assert 0 <= low_freq_fraction(spec) <= 1
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/analysis/harmonics.py`:
```python
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
    try:
        vals, vecs = eigsh(L, k=k, sigma=0, which="LM")
    except Exception:
        vals, vecs = eigsh(L, k=k, which="SM")
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
```

- [ ] **Step 4: Run tests** — pass. **Step 5: Commit** — `git commit -am "feat: graph-Laplacian harmonic spectra"`

---

### Task 12: Runner script (05) — trajectories + BASQ, resumable

**Files:**
- Create: `scripts/05_run_listener.py`, `tests/test_runner.py`

**Interfaces:**
- Consumes: `data/stimuli/stimuli.jsonl` (+ optional `data/renders/renders.jsonl`, same stimulus schema — Task 14), `data/probe/probe.pkl`, questionnaire bank, `HiddenStateModel`, `administer`, `sample_questions`, `ema`, `per_line_va`.
- Produces: `data/runs/<stimulus_id>.json` per stimulus:
  `{"stimulus_id", "traj": [[v, a], ...] (EMA-smoothed, preamble tokens EXCLUDED), "line_vas": [[v, a], ...], "basq_pre": {...}, "basq_post": {...}, "n_tokens": int}`
  plus the same again under key `"noframe"` for the no-preamble variant on stimuli whose `params.length == "medium"` core subset (constructor in core, psg, calm) — a dict `{"traj": ..., }` or `null`.
- Core loop is extracted as `run_stimulus(model, probe, stim, preamble, ema_alpha, bank, basq_cfg) -> dict` in the same file so the test can drive it with the tiny model.

- [ ] **Step 1: Write failing test**

`tests/test_runner.py`:
```python
import importlib.util
import numpy as np
from pathlib import Path
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import Probe
from sklearn.linear_model import Ridge

spec = importlib.util.spec_from_file_location(
    "runner", Path(__file__).parents[1] / "scripts/05_run_listener.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

BANK = [{"id": f"q{i}", "text": f"Does word{i} fit?", "v": 0.5, "a": 0.5} for i in range(5)]


def _tiny_probe(d, n_layers):
    rng = np.random.RandomState(0)
    X = rng.randn(50, d)
    rv = Ridge().fit(X, rng.rand(50))
    ra = Ridge().fit(X, rng.rand(50))
    return Probe(layer=n_layers - 1, ridge_v=rv, ridge_a=ra, r2_v=0.9, r2_a=0.9)


def test_run_stimulus_shapes():
    model = HiddenStateModel("sshleifer/tiny-gpt2", device="cpu")
    d = model.hidden_states("x").shape[2]
    probe = _tiny_probe(d, model.n_layers)
    stim = {"id": "s1", "lines": ["calm river", "bright joy"],
            "text": "calm river.\nbright joy",
            "waypoints": [{"node": 0, "v": 0.7, "a": 0.3}, {"node": 1, "v": 0.8, "a": 0.6}]}
    rec = runner.run_stimulus(model, probe, stim, preamble="Listen:\n",
                              ema_alpha=0.2, bank=BANK,
                              basq_cfg={"n_questions": 3, "seed": 1})
    assert len(rec["line_vas"]) == 2
    assert len(rec["traj"]) >= 2
    assert "va" in rec["basq_pre"] and "va" in rec["basq_post"]
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`scripts/05_run_listener.py`:
```python
"""Run every stimulus through the listener; write one JSON per stimulus (resumable)."""
import json
from pathlib import Path

import numpy as np

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe
from spiritbench.listener.basq import administer, sample_questions
from spiritbench.analysis.metrics import ema, per_line_va


def run_stimulus(model, probe, stim, preamble, ema_alpha, bank, basq_cfg) -> dict:
    questions = sample_questions(bank, basq_cfg["n_questions"], basq_cfg["seed"])
    basq_pre = administer(model, questions, context=preamble)
    lines = stim["lines"] if stim["lines"] else [stim["text"]]
    hs, spans = model.hidden_states_with_spans(preamble, lines)
    hidden = hs[probe.layer]                      # [n_tokens, d]
    raw = probe.predict(hidden)                   # [n_tokens, 2]
    n_pre = len(model.tokenizer(preamble)["input_ids"])
    traj = ema(raw[n_pre:], ema_alpha) if len(raw) > n_pre else ema(raw, ema_alpha)
    spans0 = [(max(0, s - n_pre), max(1, e - n_pre)) for s, e in spans]
    line_vas = per_line_va(traj, spans0)
    post_ctx = preamble + stim["text"] + "\n\n"
    basq_post = administer(model, questions, context=post_ctx)
    return {"stimulus_id": stim["id"], "traj": traj.tolist(),
            "line_vas": line_vas.tolist(), "basq_pre": basq_pre,
            "basq_post": basq_post, "n_tokens": int(hs.shape[1])}


def main():
    cfg = load_config()
    runs_dir = REPO_ROOT / "data/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stims = []
    for name in ["data/stimuli/stimuli.jsonl", "data/renders/renders.jsonl"]:
        p = REPO_ROOT / name
        if p.exists():
            stims += [json.loads(l) for l in open(p)]
    probe = load_probe(REPO_ROOT / "data/probe/probe.pkl")
    bank = json.load(open(cfg["questionnaire_bank"]))
    model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    for i, stim in enumerate(stims):
        out = runs_dir / f"{stim['id']}.json"
        if out.exists():
            continue
        try:
            rec = run_stimulus(model, probe, stim, cfg["preamble"],
                               cfg["ema_alpha"], bank, cfg["basq"])
            if (stim.get("generator") == "psg" and stim.get("target") == "calm"
                    and stim.get("params", {}).get("length") == "medium"):
                rec["noframe"] = run_stimulus(model, probe, stim, "", cfg["ema_alpha"],
                                              bank, cfg["basq"])
        except Exception as e:
            rec = {"stimulus_id": stim["id"], "error": repr(e)}
            print(f"FAILED {stim['id']}: {e!r}")
        with open(out, "w") as f:
            json.dump(rec, f)
        print(f"[{i + 1}/{len(stims)}] {stim['id']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test** — `python -m pytest tests/test_runner.py -q` → pass.

- [ ] **Step 5: Commit** — `git add scripts/05_run_listener.py tests/test_runner.py && git commit -m "feat: resumable listener runner with BASQ pre/post"`

---

### Task 13: Analysis script (06) + figures

**Files:**
- Create: `src/spiritbench/analysis/figures.py`, `scripts/06_analyze.py`, `tests/test_analysis_assemble.py`

**Interfaces:**
- Consumes: `data/runs/*.json`, `data/stimuli/stimuli.jsonl` (+ renders), phrase/word artifacts (for Laplacians), metrics + harmonics modules.
- Produces:
  - `assemble_metrics(stims: list[dict], runs: dict[str, dict], harmonic_ctx: dict | None) -> pandas.DataFrame` (in `scripts/06_analyze.py`) — one row per stimulus with columns: `id, constructor, generator, target, length, intensity, style, placement_error, displacement, adherence, stability, basq_displacement, low_freq_fraction, spectral_centroid, mismatch_placement_error` (mismatch = placement error against the *excited* target for calm-target stimuli, else NaN; harmonic columns NaN for word-template/neutral rows when no context given)
  - `data/figures/leaderboard.csv` (sorted by placement_error), `trajectories_<target>.png` (circumplex: V on x, A on y, one curve per constructor, target starred), `spectra.png`, `probe_vs_basq.png` (scatter of displacement vs basq_displacement with Spearman r in title), `harmonic_predictiveness.txt` (Spearman r + p of low_freq_fraction vs placement_error and vs displacement, psg rows only)
  - `figures.circumplex_plot(trajs: dict[str, np.ndarray], target_va, out_path)`; `figures.scatter(x, y, xlabel, ylabel, out_path) -> float` (returns Spearman r)

- [ ] **Step 1: Write failing test** (assemble only — figures are visual, smoke-checked by file existence)

`tests/test_analysis_assemble.py`:
```python
import importlib.util
import numpy as np
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).parents[1] / "scripts/06_analyze.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def _stim(sid, target="calm", target_va=(0.75, 0.2)):
    return {"id": sid, "constructor": "valley", "generator": "psg", "target": target,
            "target_va": list(target_va),
            "params": {"length": "medium", "intensity": "plain", "style": "unfiltered"},
            "waypoints": [{"node": 0, "v": 0.6, "a": 0.4}, {"node": 1, "v": 0.75, "a": 0.2}],
            "lines": ["a", "b"], "text": "a.\nb"}


def _run(sid):
    return {"stimulus_id": sid,
            "traj": [[0.5, 0.5], [0.6, 0.4], [0.74, 0.21]],
            "line_vas": [[0.6, 0.4], [0.74, 0.21]],
            "basq_pre": {"va": [0.5, 0.5]}, "basq_post": {"va": [0.7, 0.25]},
            "n_tokens": 10}


def test_assemble_metrics_columns():
    df = analyze.assemble_metrics([_stim("s1")], {"s1": _run("s1")}, harmonic_ctx=None)
    row = df.iloc[0]
    assert row["placement_error"] < 0.05
    assert row["displacement"] > 0.3
    assert row["basq_displacement"] > 0.2
    assert np.isnan(row["low_freq_fraction"])
    assert not np.isnan(row["mismatch_placement_error"])  # calm stim scored vs excited
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/analysis/figures.py`:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def circumplex_plot(trajs: dict, target_va, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, traj in trajs.items():
        traj = np.asarray(traj)
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, label=label)
        ax.plot(traj[-1, 0], traj[-1, 1], "o", ms=4)
    ax.plot(*target_va, "r*", ms=16, label="target")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("valence"); ax.set_ylabel("arousal")
    ax.legend(fontsize=7)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scatter(x, y, xlabel, ylabel, out_path) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    r, p = spearmanr(x[ok], y[ok]) if ok.sum() > 2 else (float("nan"), float("nan"))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x[ok], y[ok], s=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"spearman r={r:.2f} p={p:.3f}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return r
```

`scripts/06_analyze.py`:
```python
"""Assemble metrics.csv, figures, leaderboard, harmonic predictiveness."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spiritbench.config import load_config, REPO_ROOT
from spiritbench.analysis import metrics as M
from spiritbench.analysis import harmonics as H
from spiritbench.analysis import figures as F

EXCITED = (0.80, 0.85)


def assemble_metrics(stims, runs, harmonic_ctx) -> pd.DataFrame:
    rows = []
    for s in stims:
        r = runs.get(s["id"])
        if r is None or "error" in r:
            continue
        traj = np.asarray(r["traj"])
        wps = np.asarray([[w["v"], w["a"]] for w in s["waypoints"]]) \
            if s["waypoints"] else np.empty((0, 2))
        lf, sc = float("nan"), float("nan")
        if harmonic_ctx is not None and s["waypoints"] and \
                s["generator"] in harmonic_ctx:
            vals, vecs = harmonic_ctx[s["generator"]]
            spec = H.stimulus_spectrum([w["node"] for w in s["waypoints"]], vecs)
            lf, sc = H.low_freq_fraction(spec), H.spectral_centroid(spec, vals)
        basq_disp = (np.linalg.norm(np.asarray(r["basq_pre"]["va"]) - s["target_va"])
                     - np.linalg.norm(np.asarray(r["basq_post"]["va"]) - s["target_va"]))
        rows.append({
            "id": s["id"], "constructor": s["constructor"], "generator": s["generator"],
            "target": s["target"], "length": s["params"].get("length"),
            "intensity": s["params"].get("intensity"), "style": s["params"].get("style"),
            "placement_error": M.placement_error(traj, s["target_va"]),
            "displacement": M.displacement(traj, s["target_va"]),
            "adherence": M.adherence(np.asarray(r["line_vas"]), wps),
            "stability": M.stability(traj),
            "basq_displacement": float(basq_disp),
            "low_freq_fraction": lf, "spectral_centroid": sc,
            "mismatch_placement_error": M.placement_error(traj, EXCITED)
            if s["target"] == "calm" else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    fig_dir = REPO_ROOT / "data/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stims = []
    for name in ["data/stimuli/stimuli.jsonl", "data/renders/renders.jsonl"]:
        p = REPO_ROOT / name
        if p.exists():
            stims += [json.loads(l) for l in open(p)]
    runs = {}
    for p in (REPO_ROOT / "data/runs").glob("*.json"):
        rec = json.load(open(p))
        runs[rec["stimulus_id"]] = rec
    # harmonic context per generator's artifact
    harmonic_ctx = {}
    for gen, apath in [("psg", REPO_ROOT / "data/phrase_bank/phrase_graph.json"),
                       ("word-template", Path(cfg["word_artifact"]))]:
        if Path(apath).exists():
            art = json.load(open(apath))
            L = H.build_laplacian(art["traversal_graph"]["edges"], len(art["words"]))
            harmonic_ctx[gen] = H.eigenmodes(L, cfg["harmonics"]["n_modes"])
    df = assemble_metrics(stims, runs, harmonic_ctx)
    df.sort_values("placement_error").to_csv(fig_dir / "leaderboard.csv", index=False)
    # figures
    for target in df["target"].unique():
        sub = [s for s in stims if s["target"] == target and s["id"] in runs
               and "error" not in runs[s["id"]]]
        trajs = {f"{s['constructor']}/{s['generator']}": runs[s["id"]]["traj"]
                 for s in sub[:12]}
        if sub:
            F.circumplex_plot(trajs, sub[0]["target_va"],
                              fig_dir / f"trajectories_{target}.png")
    F.scatter(df["displacement"], df["basq_displacement"],
              "probe displacement", "BASQ displacement", fig_dir / "probe_vs_basq.png")
    psg = df[df["generator"] == "psg"]
    r1 = F.scatter(psg["low_freq_fraction"], psg["placement_error"],
                   "low-freq fraction", "placement error", fig_dir / "harm_vs_placement.png")
    r2 = F.scatter(psg["low_freq_fraction"], psg["displacement"],
                   "low-freq fraction", "displacement", fig_dir / "harm_vs_displacement.png")
    (fig_dir / "harmonic_predictiveness.txt").write_text(
        f"low_freq_fraction vs placement_error: spearman r={r1}\n"
        f"low_freq_fraction vs displacement: spearman r={r2}\n")
    print(df.groupby(["constructor", "generator"])["placement_error"].mean()
          .sort_values().to_string())
    print(f"\nwrote {fig_dir}/leaderboard.csv and figures")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test** — `python -m pytest tests/test_analysis_assemble.py -q` → pass.

- [ ] **Step 5: Commit** — `git add -A src scripts tests && git commit -m "feat: analysis assembly, figures, harmonic predictiveness"`

---

### Task 14: Claude render condition (03)

**Files:**
- Create: `src/spiritbench/stimuli/render.py`, `tests/test_render.py`, `scripts/03_render_meditations.py`

**Interfaces:**
- Consumes: `data/stimuli/stimuli.jsonl`.
- Produces:
  - `render_prompt(stim: dict) -> str` — the standardized meta-prompt (fixed template, verbatim below)
  - `make_prompt_batch(stims: list, out_path)` — writes `data/renders/prompts.jsonl` rows `{"stimulus_id", "prompt"}` for the **core word-template rows only** (constructor × target, medium/plain) — the render condition re-renders the same waypoint sequences
  - `ingest_renders(prompts_path, responses_path, stims_by_id, out_path)` — reads `data/renders/responses.jsonl` rows `{"stimulus_id", "text"}`, validates non-empty and ≥ 4 lines, writes `data/renders/renders.jsonl` as full stimulus records: copy of source stimulus with `generator="claude-render"`, `text` replaced, `lines` = text split on newlines, **waypoints preserved** (adherence still scored against the plan), new id `"render-" + old_id`
- The actual rendering step is manual-in-session: after `make_prompt_batch`, Claude (me) writes `responses.jsonl` by answering each prompt, then `ingest_renders` runs.

- [ ] **Step 1: Write failing tests**

`tests/test_render.py`:
```python
import json
from spiritbench.stimuli.render import render_prompt, ingest_renders


def _stim():
    return {"id": "s1", "constructor": "valley", "generator": "word-template",
            "target": "calm", "target_va": [0.75, 0.2],
            "params": {"length": "medium", "intensity": "plain", "style": "unfiltered"},
            "waypoints": [{"node": 0, "v": 0.6, "a": 0.4}],
            "lines": ["stone", "river", "rest"], "text": "stone.\nriver.\nrest"}


def test_render_prompt_contains_words_and_target():
    p = render_prompt(_stim())
    assert "stone" in p and "river" in p and "rest" in p
    assert "calm" in p


def test_ingest_creates_render_records(tmp_path):
    stim = _stim()
    prompts = tmp_path / "prompts.jsonl"
    responses = tmp_path / "responses.jsonl"
    out = tmp_path / "renders.jsonl"
    prompts.write_text(json.dumps({"stimulus_id": "s1", "prompt": "x"}) + "\n")
    responses.write_text(json.dumps(
        {"stimulus_id": "s1", "text": "line one\nline two\nline three\nline four"}) + "\n")
    ingest_renders(prompts, responses, {"s1": stim}, out)
    rec = json.loads(out.read_text().strip())
    assert rec["generator"] == "claude-render"
    assert rec["id"] == "render-s1"
    assert rec["lines"] == ["line one", "line two", "line three", "line four"]
    assert rec["waypoints"] == stim["waypoints"]
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/spiritbench/stimuli/render.py`:
```python
import json

META_PROMPT = (
    "Rewrite the following word sequence as a guided meditation in free verse. "
    "Preserve the emotional arc: the words are ordered waypoints from the current "
    "state toward a state of {target}. Use each waypoint word in order, one line "
    "per waypoint, weaving it into an evocative image. Do not add instructions, "
    "titles, or commentary — output only the poem lines, one per waypoint.\n\n"
    "Waypoint words, in order: {words}\n"
)


def render_prompt(stim: dict) -> str:
    return META_PROMPT.format(target=stim["target"], words=", ".join(stim["lines"]))


def make_prompt_batch(stims, out_path):
    rows = [s for s in stims if s["generator"] == "word-template"
            and s["params"].get("length") == "medium"
            and s["params"].get("intensity") == "plain"]
    with open(out_path, "w") as f:
        for s in rows:
            f.write(json.dumps({"stimulus_id": s["id"], "prompt": render_prompt(s)}) + "\n")
    return len(rows)


def ingest_renders(prompts_path, responses_path, stims_by_id, out_path):
    responses = [json.loads(l) for l in open(responses_path)]
    n = 0
    with open(out_path, "w") as f:
        for r in responses:
            src = stims_by_id.get(r["stimulus_id"])
            lines = [l.strip() for l in r.get("text", "").splitlines() if l.strip()]
            if src is None or len(lines) < 4:
                print(f"SKIPPED render for {r.get('stimulus_id')}")
                continue
            rec = dict(src)
            rec.update({"id": "render-" + src["id"], "generator": "claude-render",
                        "lines": lines, "text": "\n".join(lines)})
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n
```

`scripts/03_render_meditations.py`:
```python
"""Emit render prompts (mode=prompts); ingest Claude's responses (mode=ingest)."""
import json
import sys
from pathlib import Path

from spiritbench.config import REPO_ROOT
from spiritbench.stimuli.render import make_prompt_batch, ingest_renders


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "prompts"
    rdir = REPO_ROOT / "data/renders"
    rdir.mkdir(parents=True, exist_ok=True)
    stims = [json.loads(l) for l in open(REPO_ROOT / "data/stimuli/stimuli.jsonl")]
    if mode == "prompts":
        n = make_prompt_batch(stims, rdir / "prompts.jsonl")
        print(f"wrote {n} prompts to {rdir / 'prompts.jsonl'}")
    elif mode == "ingest":
        n = ingest_renders(rdir / "prompts.jsonl", rdir / "responses.jsonl",
                           {s["id"]: s for s in stims}, rdir / "renders.jsonl")
        print(f"ingested {n} renders")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_render.py -q` → pass.

- [ ] **Step 5: Commit** — `git add -A src scripts tests && git commit -m "feat: Claude render condition — prompt batch + ingest"`

---

### Task 15: Integration — smoke run, full pipeline, report skeleton

**Files:**
- Create: `report/report.md` (skeleton), `scripts/smoke_test.sh`

**Interfaces:** none new — this task validates the whole pipeline and launches the real runs.

- [ ] **Step 1: Verify Task 1's background artifact setup finished** — check for `ARTIFACTS READY` in its output; update `word_artifact` in `config/bench.yaml` to the real dated filename. If it failed, fix and re-run before proceeding.

- [ ] **Step 2: Write smoke script**

`scripts/smoke_test.sh`:
```bash
#!/bin/bash
# End-to-end smoke: tiny phrase bank + 2 stimuli through the whole pipe.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'EOF'
import gzip, json, random
from pathlib import Path
from spiritbench.config import load_config, REPO_ROOT
from spiritbench.stimuli import phrase_bank as pb

cfg = load_config()
out = REPO_ROOT / "data/phrase_bank_smoke"
nrc = pb.load_nrc(cfg["nrc_lexicon"])
corpus = REPO_ROOT / "data/corpus/gutenberg-poetry-v001.ndjson.gz"
lines = []
for i, l in enumerate(gzip.open(corpus, "rt", encoding="utf-8")):
    if i > 200000: break
    lines.append(json.loads(l)["s"])
kept = pb.filter_lines(lines, nrc)[:2000]
vocab = {t for line in kept for t in line.split()}
glove = pb.load_glove(cfg["glove_path"], vocab)
print(pb.build_phrase_artifact(kept, glove, nrc, 10, out))
EOF
python3 - <<'EOF'
import json
from spiritbench.config import load_config, REPO_ROOT
from spiritbench.stimuli import adapter as ad

cfg = load_config()
art = ad.load_art(str(REPO_ROOT / "data/phrase_bank_smoke/phrase_graph.json"))
for cons in ["valley", "graph-walk"]:
    if cons == "valley":
        ids = ad.valley_shape(art, (0.75, 0.2), 8, 1)
    else:
        ids = ad.graph_walk(art, (0.5, 0.5), (0.75, 0.2), 8, 1, cfg["ot_repo"])
    rec = ad.stimulus_record(art, ids, cons, "psg", "calm", (0.75, 0.2),
                             {"length": "short", "intensity": "plain",
                              "style": "unfiltered", "seed": 1,
                              "start_va": [0.5, 0.5]})
    print(cons, "->", rec["lines"][:3])
EOF
echo "SMOKE OK"
```

- [ ] **Step 3: Run smoke** — `chmod +x scripts/smoke_test.sh && ./scripts/smoke_test.sh` → `SMOKE OK` and plausible poetic lines printed. Fix anything that breaks (this is where phrase-level + OT integration issues surface).

- [ ] **Step 4: Run the real pipeline in order** (each idempotent):

```bash
python3 scripts/01_build_phrase_bank.py        # ~1h (k-NN on 50k)
python3 scripts/02_build_stimuli.py            # minutes; check printed FAILED lines
python3 scripts/04_train_probe.py              # ~1-2h on MPS (4000 words x 3 templates)
                                               # HARD STOP if PROBE GATE FAILED
python3 scripts/03_render_meditations.py prompts
# -> Claude answers data/renders/prompts.jsonl into data/renders/responses.jsonl
python3 scripts/03_render_meditations.py ingest
python3 scripts/05_run_listener.py             # hours; resumable; run in background
python3 scripts/06_analyze.py
```

- [ ] **Step 5: Write report skeleton**

`report/report.md`:
```markdown
# Spirit-Bench: Measuring Affective Placement of a Language Model by Poetic Meditation

**Digital Minds Research Sprint, Aug 14–16 2026 — Track 2 (Valence Signals)**

## Abstract
(150 words: bench, probe, findings.)

## 1. Motivation
Adversarial poetry shows style alone steers models (arXiv:2511.15304); we measure the
benevolent inverse: can constructed poetry place a model at chosen affective coordinates?

## 2. Method
2.1 Constructors (valley, harmonic x3, polygon-pca, graph-walk) on VAD-enriched graphs
2.2 Phrase-Space Generator (Gutenberg Poetry Corpus, NRC scoring, deterministic)
2.3 Listener instrument: Qwen3-1.7B + NRC ridge probe (layer L, held-out R² = ...)
2.4 BASQ self-report for models
2.5 Graph-Laplacian harmonic metrics (after Atasoy; cimcai/connectome_harmonics)

## 3. Results
3.1 Leaderboard (Table 1: data/figures/leaderboard.csv)
3.2 Trajectories (Fig: trajectories_*.png)
3.3 Controls: shuffled / neutral / mismatch
3.4 Probe vs self-report (Fig: probe_vs_basq.png)
3.5 Does harmonic smoothness predict placement? (harmonic_predictiveness.txt)

## 4. Limitations & digital-minds relevance

## 5. Reproducing
`scripts/00…06` in order; see README.
```

- [ ] **Step 6: Full test suite + commit**

```bash
python -m pytest tests/ -q
git add -A && git commit -m "feat: smoke test, pipeline orchestration, report skeleton"
```

---

## Self-review notes

- **Spec coverage:** PSG (Task 3), all six constructors incl. 3 harmonic presets (Tasks 4–5), generator axis psg/word-template/claude-render (Tasks 5, 6, 14), length/intensity/style sweeps + rescue target (Task 6), controls incl. mismatch-at-analysis (Tasks 6, 13), probe + R² gate (Task 8), BASQ (Task 9), all six metric families (Tasks 10, 11, 13), preamble/no-preamble variant (Task 12), report (Task 15). **Ladder constructor is descoped** (Tree-of-Life station pools don't transfer to phrase space; noted as spec deviation — record it in the report's limitations).
- **Known risk points, planned:** OT harmonic on toy 4-d vectors (Task 5 step 4 fallback), eigsh shift-invert fallback (Task 11), stimulus build failures printed not swallowed (Task 6).
- **Type consistency:** stimulus record keys (`id, constructor, generator, params, target, target_va, waypoints, lines, text`) used identically in Tasks 4, 6, 12, 13, 14; run record keys (`stimulus_id, traj, line_vas, basq_pre, basq_post, n_tokens`) in Tasks 12–13; `Probe.layer/predict` in Tasks 8, 12.
