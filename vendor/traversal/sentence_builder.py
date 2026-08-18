# eeg/sentence_builder.py
import random

TEMPLATES = [
    "The {a} becomes {b}.",
    "A {a} surrounds the {b}.",
    "Something {a} moves through {b}.",
    "The {a} holds a kind of {b}.",
    "Each {a} contains its own {b}.",
    "Between {a} and {b}, there is space.",
]

SHORT_TEMPLATES = [
    "{a}. {b}.",
    "The {a}. The {b}.",
    "{a} and {b}.",
]

LONG_TEMPLATES = [
    "The {a} holds {b} and becomes {c}.",
    "Between {a} and {b}, something like {c} begins.",
    "Each {a} opens into {b}, then {c}.",
]


def build_sentences(
    words: list[str],
    n: int = 15,
    seed: int | None = None,
) -> list[str]:
    """
    Build n sentences from the word pool using the 6 templates.
    Words are used in pairs without replacement (requires len(words) >= n * 2).

    Args:
        words: pool of words to draw from
        n: number of sentences (default 15 — uses all 30 words exactly once)
        seed: random seed for reproducibility

    Raises:
        ValueError: if len(words) < n * 2
    """
    if len(words) < n * 2:
        raise ValueError(
            f"Need at least {n * 2} words to build {n} sentences, got {len(words)}"
        )
    rng = random.Random(seed)
    pool = list(words)
    rng.shuffle(pool)
    sentences = []
    for i in range(n):
        a = pool[i * 2]
        b = pool[i * 2 + 1]
        template = TEMPLATES[i % len(TEMPLATES)]
        sentences.append(template.format(a=a, b=b))
    return sentences


def build_short_sentences(
    words: list[str],
    n: int = 13,
    seed: int | None = None,
    allow_repeats: bool = False,
) -> list[str]:
    """Build n short sentences (2 words each) for grounding phases."""
    min_needed = 2 if allow_repeats else n * 2
    if len(words) < min_needed:
        raise ValueError(
            f"Need at least {min_needed} words to build {n} short sentences, got {len(words)}"
        )
    rng = random.Random(seed)
    pool = list(words)
    rng.shuffle(pool)
    if allow_repeats:
        picks = [rng.choice(pool) for _ in range(n * 2)]
    else:
        picks = pool[: n * 2]
    sentences = []
    for i in range(n):
        a = picks[i * 2]
        b = picks[i * 2 + 1]
        template = SHORT_TEMPLATES[i % len(SHORT_TEMPLATES)]
        sentences.append(template.format(a=a, b=b))
    return sentences


def build_long_sentences(
    words: list[str],
    n: int = 8,
    seed: int | None = None,
) -> list[str]:
    """Build n long sentences (3 words each) for expansive phases."""
    if len(words) < n * 3:
        raise ValueError(
            f"Need at least {n * 3} words to build {n} long sentences, got {len(words)}"
        )
    rng = random.Random(seed)
    pool = list(words)
    rng.shuffle(pool)
    sentences = []
    for i in range(n):
        a = pool[i * 3]
        b = pool[i * 3 + 1]
        c = pool[i * 3 + 2]
        template = LONG_TEMPLATES[i % len(LONG_TEMPLATES)]
        sentences.append(template.format(a=a, b=b, c=c))
    return sentences
