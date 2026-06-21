from typing import List, Tuple


def extract_entity_relations(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples as a simple relation extraction.

    This implementation uses spaCy's dependency parser to identify subject-verb-object
    structures. The extracted triples capture how an author connects entities syntactically,
    which is a stylistic fingerprint (e.g., preferred relational patterns, passive/active voice).

    Requires spaCy: pip install spacy && python -m spacy download en_core_web_sm

    Parameters
    ----------
    text : str
        Input text (English).

    Returns
    -------
    List[Tuple[str, str, str]]
        List of (subject_phrase, relation_phrase, object_phrase) triples.
        If spaCy is not available, returns an empty list and prints a warning.
    """
    try:
        import spacy
    except ImportError:
        print("spaCy not installed. Returning empty list.")
        return []

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    triples = []

    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "ccomp", "advcl"):
            subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
            objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
            for subj in subjects:
                for obj in objects:
                    subj_phrase = " ".join(t.text for t in subj.subtree).strip()
                    obj_phrase = " ".join(t.text for t in obj.subtree).strip()
                    triples.append((subj_phrase, token.lemma_, obj_phrase))
    return triples
