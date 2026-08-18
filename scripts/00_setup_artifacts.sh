#!/bin/bash
# Idempotent, self-contained artifact setup. Downloads external DATA into this
# repo's data/ dir and builds the word graph with the vendored build scripts.
# Safe to re-run; each block skips if its output exists.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/glove data/nrc_vad data/corpus data/artifacts

# 1. GloVe 6B 300d (public domain)
if [ ! -f data/glove/glove.6B.300d.txt ]; then
  curl -L --retry 3 -o data/glove/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
  unzip -o data/glove/glove.6B.zip glove.6B.300d.txt -d data/glove/
  rm data/glove/glove.6B.zip
fi

# 2. NRC-VAD lexicon (research license — downloaded, NOT redistributed by this repo)
if [ ! -f data/nrc_vad/NRC-VAD-Lexicon.txt ]; then
  curl -L --retry 3 -o /tmp/nrc-vad.zip https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip
  unzip -j -o /tmp/nrc-vad.zip 'NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt' -d data/nrc_vad/
  rm -f /tmp/nrc-vad.zip
fi

# 3. Gutenberg Poetry Corpus (public domain)
if [ ! -f data/corpus/gutenberg-poetry-v001.ndjson.gz ]; then
  curl -L --retry 3 -o data/corpus/gutenberg-poetry-v001.ndjson.gz \
    http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz
fi

# 4. Word graph — build + enrich locally with the vendored scripts
if [ ! -f data/artifacts/word_graph_enriched.json ]; then
  python3 vendor/traversal/build_word_graph.py \
      --glove data/glove/glove.6B.300d.txt --output-dir data/artifacts/
  RAW=$(ls data/artifacts/word_graph_*.json | grep -v enriched | head -1)
  python3 vendor/traversal/enrich_artifact.py \
      --artifact "$RAW" \
      --nrc-vad data/nrc_vad/NRC-VAD-Lexicon.txt \
      --output "${RAW%.json}_enriched.json"
  # stable name the config points at (build is date-stamped)
  ln -sf "$(basename "${RAW%.json}_enriched.json")" data/artifacts/word_graph_enriched.json
  # the enriched .json references its vectors .npy by relative name — keep alongside
fi

echo "ARTIFACTS READY"
ls data/artifacts/ data/glove/ data/corpus/ data/nrc_vad/
