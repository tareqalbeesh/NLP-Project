from typing import Any, Dict, List, Tuple


def resolve_references(text: str) -> Dict[str, List[List[Tuple[int, int, str]]]]:
    """
    Perform coreference resolution: cluster mentions (e.g., pronouns, names) referring
    to the same real-world entity.

    The result captures how an author manages reference chains - a rich stylistic feature
    (e.g., pronoun density, switching patterns, repeated naming).

    This function tries to use spaCy's experimental coref model (`en_coreference_web_trf`).
    If unavailable, it falls back to a simple rule-based pronoun-to-nearest-entity mapping
    (for demonstration only).

    Parameters
    ----------
    text : str
        Input English text.

    Returns
    -------
    dict with keys:
        - "clusters": list of clusters, each cluster is a list of mention spans [start, end, text].
        - "method": str indicating which resolver was used.
        - "raw_output": optional raw resolver output for transparency.
    """
    result: Dict[str, Any] = {"clusters": [], "method": "none", "raw_output": None}

    try:
        import spacy
        nlp = spacy.load("en_coreference_web_trf")
        doc = nlp(text)
        if doc.spans:
            clusters = []
            for cluster_id, span_list in doc.spans.items():
                cluster = [(span.start_char, span.end_char, span.text) for span in span_list]
                clusters.append(cluster)
            result["clusters"] = clusters
            result["method"] = "spacy_coref"
            result["raw_output"] = doc.spans
            return result
    except (ImportError, OSError):
        pass

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        print("spaCy not available. Returning empty result.")
        return result

    doc = nlp(text)
    pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their"}
    persons = {"PERSON", "ORG", "GPE"}

    clusters = []
    for token in doc:
        if token.lower_ in pronouns:
            for ent in reversed(list(doc.ents)):
                if ent.start_char < token.idx and ent.label_ in persons:
                    clusters.append([
                        (ent.start_char, ent.end_char, ent.text),
                        (token.idx, token.idx + len(token.text), token.text),
                    ])
                    break

    result["clusters"] = clusters
    result["method"] = "rule_based_fallback"
    return result
