"""Register covariates from the lit review (P1) — computed per stimulus text.

- Polak (1998): NV ratio = N / (N + V) over POS tags.
- Wårvik (2025): line-initial `and` and `then` per 1,000 words
  (Bible-register targets: and ≈ 18/1000, then ≈ 5/1000).
- Mohseni et al. (2023): Shannon and Approximate Entropy over the noun-count
  series in 25-token windows (m=2, r=0.2·SD). Order-sensitive (ApEn).
- Arruda et al. (2022), simplified: coefficient of variation of line length
  in words (word-level stand-in for their phoneme-series cv(l)).

Hypotaxis (Walkden 2021) is approximated by subordinator density — a lexicon
count, not a parse; reported as `subordinator_per_1000` and named as an
approximation in the report.
"""
import math
import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[a-z']+")

SUBORDINATORS = frozenset(
    "because although though while whereas if unless until since when whenever "
    "where wherever that which who whom whose after before once".split())

WINDOW = 25
APEN_M = 2
APEN_R = 0.2


def _tokens(text):
    return _WORD_RE.findall(text.lower())


def _pos_tags(text):
    from nltk import pos_tag
    toks = _tokens(text)
    return pos_tag(toks) if toks else []


def nv_ratio(text) -> float:
    """Polak NV = nouns / (nouns + verbs); NaN if no nouns or verbs."""
    tags = _pos_tags(text)
    n = sum(1 for _, t in tags if t.startswith("NN"))
    v = sum(1 for _, t in tags if t.startswith("VB"))
    return n / (n + v) if (n + v) else float("nan")


def line_initial_density(lines, word) -> float:
    """Occurrences of `word` line-initially, per 1,000 words of text."""
    toks_total = sum(len(_tokens(l)) for l in lines)
    if not toks_total:
        return float("nan")
    hits = sum(1 for l in lines if (_tokens(l) or [""])[0] == word)
    return 1000.0 * hits / toks_total


def word_density(text, vocab) -> float:
    toks = _tokens(text)
    if not toks:
        return float("nan")
    return 1000.0 * sum(t in vocab for t in toks) / len(toks)


def _noun_series(text, window=WINDOW):
    tags = _pos_tags(text)
    counts = []
    for i in range(0, len(tags) - window + 1, window):
        counts.append(sum(1 for _, t in tags[i:i + window] if t.startswith("NN")))
    return np.array(counts, dtype=float)


def shannon_entropy(series) -> float:
    if len(series) == 0:
        return float("nan")
    freq = Counter(series.astype(int))
    total = sum(freq.values())
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def approx_entropy(series, m=APEN_M, r_frac=APEN_R) -> float:
    """ApEn(m, r=r_frac·SD). NaN when the series is too short (< m+2)."""
    n = len(series)
    if n < m + 2:
        return float("nan")
    r = r_frac * np.std(series)
    if r == 0:
        return 0.0

    def phi(mm):
        templates = np.array([series[i:i + mm] for i in range(n - mm + 1)])
        counts = []
        for t in templates:
            dist = np.max(np.abs(templates - t), axis=1)
            counts.append(np.mean(dist <= r))
        return np.mean(np.log(counts))

    return float(phi(m) - phi(m + 1))


def cv_line_length(lines) -> float:
    lens = [len(_tokens(l)) for l in lines if _tokens(l)]
    if len(lens) < 2 or np.mean(lens) == 0:
        return float("nan")
    return float(np.std(lens) / np.mean(lens))


def covariates(text, lines) -> dict:
    series = _noun_series(text)
    return {
        "nv_ratio": nv_ratio(text),
        "and_initial_per_1000": line_initial_density(lines, "and"),
        "then_per_1000": word_density(text, {"then"}),
        "subordinator_per_1000": word_density(text, SUBORDINATORS),
        "noun_shen": shannon_entropy(series),
        "noun_apen": approx_entropy(series),
        "cv_line_len": cv_line_length(lines),
    }
