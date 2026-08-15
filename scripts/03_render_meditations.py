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
