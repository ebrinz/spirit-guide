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
python3 - <<'EOF'
import json
from spiritbench.config import load_config, REPO_ROOT
from spiritbench.stimuli import adapter as ad

cfg = load_config()
art = ad.load_art(str(REPO_ROOT / "data/phrase_bank_smoke/phrase_graph.json"))
ids = ad.harmonic(art, str(REPO_ROOT / "data/phrase_bank_smoke/phrase_graph.json"),
                  (0.5, 0.5), (0.75, 0.2), 8, "golden", 1,
                  cfg["ot_repo"], cfg["semantic_axes"])
rec = ad.stimulus_record(art, ids, "harmonic-golden", "psg", "calm", (0.75, 0.2),
                         {"length": "short", "intensity": "plain",
                          "style": "unfiltered", "seed": 1,
                          "start_va": [0.5, 0.5]})
print("harmonic-golden", "->", rec["lines"][:3])
EOF
echo "SMOKE OK"
