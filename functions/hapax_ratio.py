from collections import Counter


def hapax_ratio(tokens: list[str]) -> float:
    counts = Counter(tokens)
    hapaxes = sum(1 for c in counts.values() if c == 1)
    return hapaxes / len(counts)
