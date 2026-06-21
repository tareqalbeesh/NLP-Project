import re
from typing import List


def tokenize_char_ngrams(text: str, n: int = 3, normalize_ws: bool = True) -> List[str]:
    """Tokenize input text into overlapping character n-grams.

    Parameters
    ----------
    text:
        Input string.
    n:
        N-gram order.
    normalize_ws:
        If True, collapse all whitespace to a single space before tokenization.

    Returns
    -------
    list[str]
        Overlapping character n-grams (including spaces).
    """
    if normalize_ws:
        text = re.sub(r'\s+', ' ', text)
    if n <= 0 or len(text) < n:
        return []
    return [text[i:i + n] for i in range(len(text) - n + 1)]
