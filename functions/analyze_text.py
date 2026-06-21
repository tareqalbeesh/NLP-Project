import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")


def analyze_text(text):
    doc = nlp(text)

    print(f"Original Text: '{text}'\n")

    # --- PART 1: POS Tagging ---
    print("## Part-of-Speech (POS) Tags:")
    pos_data = [(token.text, token.pos_, token.tag_) for token in doc]
    print(pos_data)
    print("-" * 40)

    # --- PART 2: Named Entity Recognition (NER) ---
    print("## Named Entities:")
    if not doc.ents:
        print("No entities found.")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_} -> {spacy.explain(ent.label_)})")
    print("-" * 40)

    # --- PART 3: Error Patterns (Linguistic Rules) ---
    print("## Detected Error Patterns:")
    matcher = Matcher(nlp.vocab)

    double_negative_pattern = [
        {"LOWER": {"IN": ["don't", "dont", "cannot", "can't", "cant", "never", "not"]}},
        {"OP": "*"},
        {"LOWER": {"IN": ["no", "nothing", "none", "never"]}},
    ]

    homophone_error_pattern = [
        {"LOWER": "they"},
        {"LOWER": "'re"},
        {"POS": "NOUN"},
    ]

    sv_disagreement_pattern = [
        {"LOWER": {"IN": ["he", "she", "it"]}},
        {"LOWER": "do"},
    ]

    matcher.add("DOUBLE_NEGATIVE", [double_negative_pattern])
    matcher.add("HOMOPHONE_ERROR", [homophone_error_pattern])
    matcher.add("SUBJECT_VERB_ERROR", [sv_disagreement_pattern])

    matches = matcher(doc)
    if not matches:
        print("No errors detected.")
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        print(f" - [{string_id}] Found problematic phrase: '{span.text}'")
