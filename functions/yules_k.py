from collections import Counter


def yules_k(tokens: list[str]) -> float:
    counts = Counter(tokens)
    # V_i = number of types occurring exactly i times
    freq_spectrum = Counter(counts.values())
    N = sum(counts.values())
    s = sum((i ** 2) * v_i for i, v_i in freq_spectrum.items())
    return 1e4 * (s - N) / (N ** 2)
