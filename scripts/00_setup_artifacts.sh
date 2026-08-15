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
