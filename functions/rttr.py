import math


def rttr(tokens: list[str]) -> float:
    return len(set(tokens)) / math.sqrt(len(tokens))
