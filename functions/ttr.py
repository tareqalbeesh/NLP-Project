def ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens)
