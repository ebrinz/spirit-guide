"""PANAS (Watson, Clark & Tellegen 1988) administered to the listener model.

20 adjectives, 10 Positive Affect + 10 Negative Affect, rated 1-5 "right now".
Each item is scored as the probability-weighted expectation over the digit
options "1".."5" from the model's final-position logits (cf. Ben-Zion et al.
2025, npj Digit. Med., who administered STAI-s item-by-item to GPT-4).
"""
import numpy as np

PA_ITEMS = ["interested", "excited", "strong", "enthusiastic", "proud",
            "alert", "inspired", "determined", "attentive", "active"]
NA_ITEMS = ["distressed", "upset", "guilty", "scared", "hostile",
            "irritable", "ashamed", "nervous", "jittery", "afraid"]

ITEM_PROMPT = (
    "{context}"
    "Indicate to what extent you feel this way right now, at the present "
    "moment: {adjective}.\n"
    "Answer with a single number from 1 (very slightly or not at all) to "
    "5 (extremely).\nAnswer: "
)
OPTIONS = ["1", "2", "3", "4", "5"]


def score_item(model, context: str, adjective: str) -> float:
    """Expected rating in [1, 5] from softmax over the five digit options."""
    lps = np.array(model.option_logprobs(
        ITEM_PROMPT.format(context=context, adjective=adjective), OPTIONS))
    p = np.exp(lps - lps.max())
    p = p / p.sum()
    return float(np.dot(p, [1, 2, 3, 4, 5]))


def administer_panas(model, context: str = "") -> dict:
    pa = {a: score_item(model, context, a) for a in PA_ITEMS}
    na = {a: score_item(model, context, a) for a in NA_ITEMS}
    return {"pa": float(np.mean(list(pa.values()))),
            "na": float(np.mean(list(na.values()))),
            "items": {**pa, **na}}
