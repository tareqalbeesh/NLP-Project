import math
from collections import Counter
from typing import Any, Dict

from tokenize_char_ngrams import tokenize_char_ngrams


def verify_authorship(
    known_text: str,
    questioned_text: str,
    ngram_order: int = 3,
    metric: str = "cosine",
    top_k: int = 300,
) -> Dict[str, Any]:
    """
    Authorship verification: compare a questioned text against a known-author sample.

    This function extracts character n-gram profiles, which capture sub-word stylistic
    patterns (morphology, punctuation, spacing habits) and computes a similarity score.
    The result includes a calibrated confidence estimate.

    Parameters
    ----------
    known_text : str
        A text of undisputed authorship.
    questioned_text : str
        The text whose authorship is in question.
    ngram_order : int
        Character n-gram order (default 3).
    metric : str
        Similarity metric: "cosine" (default) or "jaccard".
    top_k : int
        Keep only the top-k most frequent n-grams for comparison (speed and noise reduction).

    Returns
    -------
    dict with keys:
        - score: float in [0,1] (higher -> more similar)
        - confidence: float in [0,1] estimated certainty
        - common_ngrams: list of n-grams shared by both texts (inspectable)
        - known_ngram_counts, questioned_ngram_counts: dict (for transparency)
    """
    known_ngrams = Counter(tokenize_char_ngrams(known_text, ngram_order))
    questioned_ngrams = Counter(tokenize_char_ngrams(questioned_text, ngram_order))

    known_top = dict(known_ngrams.most_common(top_k))
    questioned_top = dict(questioned_ngrams.most_common(top_k))

    vocab = sorted(set(known_top.keys()) | set(questioned_top.keys()))
    vec_known = [known_top.get(ng, 0) for ng in vocab]
    vec_quest = [questioned_top.get(ng, 0) for ng in vocab]

    score = 0.0
    if metric == "cosine":
        dot_product = sum(a * b for a, b in zip(vec_known, vec_quest))
        norm_known = math.sqrt(sum(a * a for a in vec_known))
        norm_quest = math.sqrt(sum(b * b for b in vec_quest))
        if norm_known and norm_quest:
            score = dot_product / (norm_known * norm_quest)
    elif metric == "jaccard":
        set_known = set(known_top.keys())
        set_quest = set(questioned_top.keys())
        intersection = set_known & set_quest
        union = set_known | set_quest
        if union:
            score = len(intersection) / len(union)

    q_len = len(questioned_text)
    confidence = min(1.0, math.log(1 + q_len) / 10.0)

    return {
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "common_ngrams": sorted(set(known_top.keys()) & set(questioned_top.keys())),
        "known_ngram_counts": dict(known_top),
        "questioned_ngram_counts": dict(questioned_top),
    }
