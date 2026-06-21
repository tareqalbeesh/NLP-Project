from ttr import ttr
from rttr import rttr
from hapax_ratio import hapax_ratio
from yules_k import yules_k


def report(name: str, tokens: list[str]) -> None:
    print(f"--- {name} ---")
    print(f"  tokens (N)      : {len(tokens)}")
    print(f"  types  (V)      : {len(set(tokens))}")
    print(f"  TTR             : {ttr(tokens):.4f}")
    print(f"  RTTR            : {rttr(tokens):.4f}")
    print(f"  hapax ratio     : {hapax_ratio(tokens):.4f}")
    print(f"  Yule's K (lower = richer): {yules_k(tokens):.2f}")
    print()
