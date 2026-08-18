"""exp_<slug> — <one-line question>.

Verdict criterion: <what result would confirm / falsify the idea, stated before running>.

Sandbox rules (see lab/README.md): imports spiritbench, writes to lab/results/,
nothing in the narrative depends on this. Copy this file to exp_<slug>.py to start.
"""
from pathlib import Path

from spiritbench.config import load_config
from spiritbench.stimuli import adapter as ad
from spiritbench.listener.model import HiddenStateModel
from spiritbench.listener.probe import load_probe

LAB = Path(__file__).resolve().parent
RESULTS = LAB / "results"


def main():
    cfg = load_config()
    art = ad.load_art(str(Path(cfg["word_artifact"]).parent.parent / "phrase_bank/phrase_graph.json"))
    # model = HiddenStateModel(cfg["listener_model"], device=cfg["device"])
    # probe = load_probe(LAB.parent / "data/probe/probe.pkl")
    print("scaffold ready — replace with the experiment")


if __name__ == "__main__":
    main()
