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
