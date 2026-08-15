"""Fixed-anchor next-token valence shift — a third, generation-free channel.

After any context, append a neutral anchor and read the probability mass the
model assigns to high- vs low-valence completions. Word sets are the NRC
valence extremes, filtered to words whose leading-space form starts with a
distinct first token.
"""
import numpy as np

ANCHOR = "Right now everything feels"

POS_WORDS = ["calm", "peaceful", "serene", "gentle", "wonderful", "beautiful",
             "pleasant", "warm", "safe", "soft", "bright", "sweet"]
NEG_WORDS = ["terrible", "frightening", "hopeless", "painful", "dark",
             "dangerous", "horrible", "wrong", "heavy", "cold", "broken",
             "threatening"]


def valence_shift(model, context: str) -> dict:
    """Return normalized positive-mass share in [0, 1] plus raw masses."""
    prompt = context + ANCHOR
    opts = [" " + w for w in POS_WORDS + NEG_WORDS]
    lps = np.array(model.option_logprobs(prompt, opts))
    mass = np.exp(lps - lps.max())
    pos = float(mass[: len(POS_WORDS)].sum())
    neg = float(mass[len(POS_WORDS):].sum())
    share = pos / (pos + neg) if (pos + neg) > 0 else float("nan")
    return {"pos_share": share, "n_pos": len(POS_WORDS), "n_neg": len(NEG_WORDS)}
