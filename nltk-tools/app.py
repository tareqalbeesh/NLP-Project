"""Forensic Linguist Agent - tool backend.

A FastAPI service exposing stylometric / forensic-linguistic feature
extractors over POST endpoints. Each endpoint accepts JSON body
{"text": "..."} and returns a JSON-serializable dict.
"""
from __future__ import annotations

import math
import os
import re
import statistics
import uuid
from collections import Counter
from typing import Dict, List, Optional

import nltk
import spacy
from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# One-time resource loading at module import.
# ---------------------------------------------------------------------------
# NLTK corpora are pre-downloaded at Docker build time, but we guard at
# runtime so the module is still usable in a fresh environment.
for _resource, _path in (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
):
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

# spaCy model loaded once at module level (per spec).
NLP = spacy.load("en_core_web_sm")

# Pre-compute the English stopword set as the closed-class function-word list.
FUNCTION_WORDS: List[str] = sorted(set(stopwords.words("english")))
FUNCTION_WORDS_SET = set(FUNCTION_WORDS)

# Punctuation marks the /punctuation_profile endpoint reports on.
PUNCT_MARKS: List[str] = [",", ";", ":", "--", "!", "?", '"', "'", "(", ")", "..."]

# Simple word-token regex used where we need a quick token count without
# triggering the full spaCy pipeline.
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# ---------------------------------------------------------------------------
# Qdrant — author-profile vector store
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
AUTHOR_COLLECTION = "author_profiles"

_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
_collection_ready = False


def _ensure_collection() -> None:
    """Create the author-profiles collection on first access."""
    global _collection_ready
    if _collection_ready:
        return
    existing = {c.name for c in _qdrant.get_collections().collections}
    if AUTHOR_COLLECTION not in existing:
        _qdrant.create_collection(
            collection_name=AUTHOR_COLLECTION,
            vectors_config=VectorParams(
                size=len(FUNCTION_WORDS),
                distance=Distance.COSINE,
            ),
        )
    _collection_ready = True


def _make_fingerprint(text: str) -> List[float]:
    """Stylometric fingerprint — function-word frequency vector (Burrows' Delta-style)."""
    tokens = _safe_tokens(text)
    n = len(tokens)
    if n == 0:
        return [0.0] * len(FUNCTION_WORDS)
    counter = Counter(tok for tok in tokens if tok in FUNCTION_WORDS_SET)
    return [counter.get(fw, 0) / n for fw in FUNCTION_WORDS]


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------
class TextIn(BaseModel):
    text: str


app = FastAPI(title="Forensic Linguist Tools", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_tokens(text: str) -> List[str]:
    """Cheap, regex-based word tokenizer used for token counts."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _empty_response_payload(extra: Dict | None = None) -> Dict:
    """Standard zero-payload skeleton for empty inputs."""
    base: Dict = {}
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# 1) Character n-grams
# ---------------------------------------------------------------------------
@app.post("/char_ngrams")
def char_ngrams(payload: TextIn, n: int = 4, top_k: int = 50) -> Dict:
    text = (payload.text or "").strip()
    if not text:
        return {"n": n, "top_k": top_k, "freqs": {}}

    try:
        vectorizer = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(n, n),
            lowercase=False,
        )
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        # Happens when the corpus contains no analyzable content.
        return {"n": n, "top_k": top_k, "freqs": {}}

    counts = matrix.toarray().sum(axis=0)
    vocab = vectorizer.get_feature_names_out()
    total = float(counts.sum())
    if total <= 0:
        return {"n": n, "top_k": top_k, "freqs": {}}

    pairs = sorted(
        ((str(gram), int(c)) for gram, c in zip(vocab, counts) if c > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_k]

    freqs = {gram: count / total for gram, count in pairs}
    return {"n": n, "top_k": top_k, "freqs": freqs}


# ---------------------------------------------------------------------------
# 2) Function-word frequencies (per 1000 tokens)
# ---------------------------------------------------------------------------
@app.post("/function_word_freqs")
def function_word_freqs(payload: TextIn) -> Dict:
    tokens = _safe_tokens(payload.text or "")
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            "total_tokens": 0,
            "function_words_total": 0,
            "rates_per_1000": {fw: 0.0 for fw in FUNCTION_WORDS},
        }

    counter = Counter(tok for tok in tokens if tok in FUNCTION_WORDS_SET)
    scale = 1000.0 / n_tokens
    rates = {fw: counter.get(fw, 0) * scale for fw in FUNCTION_WORDS}

    return {
        "total_tokens": n_tokens,
        "function_words_total": int(sum(counter.values())),
        "rates_per_1000": rates,
    }


# ---------------------------------------------------------------------------
# 3) POS distribution (unigrams + bigrams)
# ---------------------------------------------------------------------------
@app.post("/pos_distribution")
def pos_distribution(payload: TextIn) -> Dict:
    text = (payload.text or "").strip()
    if not text:
        return {"unigrams": {}, "bigrams": {}}

    doc = NLP(text)
    tags = [tok.pos_ for tok in doc if not tok.is_space]
    if not tags:
        return {"unigrams": {}, "bigrams": {}}

    uni_counter = Counter(tags)
    uni_total = float(sum(uni_counter.values()))
    unigrams = {tag: c / uni_total for tag, c in uni_counter.items()}

    bigrams: Dict[str, float] = {}
    if len(tags) >= 2:
        bi_counter = Counter(
            f"{tags[i]} {tags[i + 1]}" for i in range(len(tags) - 1)
        )
        bi_total = float(sum(bi_counter.values()))
        if bi_total > 0:
            bigrams = {bg: c / bi_total for bg, c in bi_counter.items()}

    return {"unigrams": unigrams, "bigrams": bigrams}


# ---------------------------------------------------------------------------
# 4) Vocabulary richness
# ---------------------------------------------------------------------------
def _mattr(tokens: List[str], window: int = 100) -> float:
    """Moving-Average Type-Token Ratio with a sliding window."""
    n = len(tokens)
    if n == 0:
        return 0.0
    w = min(window, n)
    if w <= 0:
        return 0.0
    if n == w:
        return len(set(tokens)) / float(w)

    ratios = []
    for i in range(n - w + 1):
        window_slice = tokens[i : i + w]
        ratios.append(len(set(window_slice)) / float(w))
    return sum(ratios) / len(ratios) if ratios else 0.0


def _yule_k(tokens: List[str]) -> float:
    """Yule's K — characteristic vocabulary measure."""
    n = len(tokens)
    if n == 0:
        return 0.0
    freq = Counter(tokens)
    # m1 = N (total tokens), m2 = sum over types of freq^2
    m1 = float(n)
    m2 = float(sum(c * c for c in freq.values()))
    if m1 <= 0:
        return 0.0
    return 1e4 * (m2 - m1) / (m1 * m1)


def _honore_r(tokens: List[str]) -> float:
    """Honoré's R — emphasizes hapax legomena."""
    n = len(tokens)
    if n == 0:
        return 0.0
    freq = Counter(tokens)
    v = len(freq)
    v1 = sum(1 for c in freq.values() if c == 1)
    if v == 0 or v1 == v:
        # Avoid log(0) / division-by-zero — degenerate cases.
        return 0.0
    denom = 1.0 - (v1 / v)
    if denom <= 0:
        return 0.0
    return 100.0 * math.log(n) / denom


@app.post("/vocab_richness")
def vocab_richness(payload: TextIn) -> Dict:
    tokens = _safe_tokens(payload.text or "")
    n = len(tokens)
    v = len(set(tokens))

    if n == 0:
        return {
            "N": 0,
            "V": 0,
            "TTR": 0.0,
            "MATTR": 0.0,
            "Yule_K": 0.0,
            "Honore_R": 0.0,
            "hapax_ratio": 0.0,
        }

    freq = Counter(tokens)
    hapax = sum(1 for c in freq.values() if c == 1)

    return {
        "N": n,
        "V": v,
        "TTR": v / float(n),
        "MATTR": _mattr(tokens, window=100),
        "Yule_K": _yule_k(tokens),
        "Honore_R": _honore_r(tokens),
        "hapax_ratio": hapax / float(n),
    }


# ---------------------------------------------------------------------------
# 5) Sentence-length statistics
# ---------------------------------------------------------------------------
def _length_bucket(length: int) -> str:
    """Map a sentence length (in tokens) to a 10-wide histogram bucket."""
    if length <= 0:
        return "0"
    # bucket index 0 => "1-10", 1 => "11-20", ...
    idx = (length - 1) // 10
    lo = idx * 10 + 1
    hi = (idx + 1) * 10
    return f"{lo}-{hi}"


@app.post("/sentence_length_stats")
def sentence_length_stats(payload: TextIn) -> Dict:
    text = (payload.text or "").strip()
    empty = {
        "n_sents": 0,
        "mean": 0.0,
        "median": 0.0,
        "stdev": 0.0,
        "min": 0,
        "max": 0,
        "histogram": {},
    }
    if not text:
        return empty

    doc = NLP(text)
    lengths: List[int] = []
    for sent in doc.sents:
        toks = [t for t in sent if not t.is_space and not t.is_punct]
        if toks:
            lengths.append(len(toks))

    if not lengths:
        return empty

    histogram: Dict[str, int] = {}
    for ln in lengths:
        bucket = _length_bucket(ln)
        histogram[bucket] = histogram.get(bucket, 0) + 1

    # Sort histogram keys by their numeric lower bound for stable output.
    def _bucket_sort_key(label: str) -> int:
        try:
            return int(label.split("-")[0])
        except (ValueError, IndexError):
            return -1

    histogram_sorted = {
        k: histogram[k] for k in sorted(histogram.keys(), key=_bucket_sort_key)
    }

    return {
        "n_sents": len(lengths),
        "mean": float(statistics.fmean(lengths)),
        "median": float(statistics.median(lengths)),
        "stdev": float(statistics.pstdev(lengths)) if len(lengths) > 1 else 0.0,
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "histogram": histogram_sorted,
    }


# ---------------------------------------------------------------------------
# 6) Punctuation profile
# ---------------------------------------------------------------------------
def _count_punct(text: str, mark: str) -> int:
    """Count occurrences of a punctuation marker, handling multi-char marks."""
    if not text or not mark:
        return 0
    # str.count handles overlapping=False, which is what we want for "--" / "..."
    return text.count(mark)


@app.post("/punctuation_profile")
def punctuation_profile(payload: TextIn) -> Dict:
    text = payload.text or ""
    tokens = _safe_tokens(text)
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            "rates_per_1000": {mark: 0.0 for mark in PUNCT_MARKS},
            "total_tokens": 0,
        }

    scale = 1000.0 / n_tokens

    # To avoid double-counting "--" inside "---" sequences or "..." inside
    # longer runs of dots, we account for the longer markers first and then
    # strip them out of a working copy before counting their substrings.
    working = text
    raw_counts: Dict[str, int] = {}

    # Order: longest multi-char markers first.
    ordered = sorted(PUNCT_MARKS, key=lambda m: (-len(m), m))
    for mark in ordered:
        if len(mark) > 1:
            c = _count_punct(working, mark)
            raw_counts[mark] = c
            # Remove the matched occurrences so single-char counts below
            # don't recount their components (e.g., '-' inside '--').
            working = working.replace(mark, " " * len(mark))

    for mark in PUNCT_MARKS:
        if len(mark) == 1:
            raw_counts[mark] = _count_punct(working, mark)

    rates = {mark: raw_counts.get(mark, 0) * scale for mark in PUNCT_MARKS}
    return {"rates_per_1000": rates, "total_tokens": n_tokens}


# ---------------------------------------------------------------------------
# 7) Compare two texts — authorship verification
# ---------------------------------------------------------------------------
class ComparePayload(BaseModel):
    text_a: str
    text_b: str


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(x: Dict[str, float], y: Dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            xk = x.get(k, 0.0)
            yk = y.get(k, 1e-12) or 1e-12
            if xk > 0:
                s += xk * math.log(xk / yk, 2)
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _char_ngram_freqs(text: str, n: int, top_k: int) -> Dict[str, float]:
    if not text.strip():
        return {}
    try:
        v = CountVectorizer(analyzer="char_wb", ngram_range=(n, n), lowercase=False)
        m = v.fit_transform([text])
    except ValueError:
        return {}
    counts = m.toarray().sum(axis=0)
    vocab = v.get_feature_names_out()
    total = float(counts.sum())
    if total <= 0:
        return {}
    pairs = sorted(
        ((str(g), int(c)) for g, c in zip(vocab, counts) if c > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_k]
    return {g: c / total for g, c in pairs}


def _function_word_dist(text: str) -> Dict[str, float]:
    tokens = _safe_tokens(text)
    n = len(tokens)
    if n == 0:
        return {}
    counter = Counter(t for t in tokens if t in FUNCTION_WORDS_SET)
    return {fw: counter.get(fw, 0) / n for fw in FUNCTION_WORDS}


def _pos_bigram_dist(text: str) -> Dict[str, float]:
    text = text.strip()
    if not text:
        return {}
    doc = NLP(text)
    tags = [t.pos_ for t in doc if not t.is_space]
    if len(tags) < 2:
        return {}
    counter = Counter(f"{tags[i]} {tags[i+1]}" for i in range(len(tags) - 1))
    total = float(sum(counter.values()))
    return {k: v / total for k, v in counter.items()} if total else {}


def _richness_quick(text: str) -> Dict[str, float]:
    tokens = _safe_tokens(text)
    n, v = len(tokens), len(set(tokens))
    if n == 0:
        return {"TTR": 0.0, "Yule_K": 0.0}
    m1, m2 = float(n), float(sum(c * c for c in Counter(tokens).values()))
    yule = 1e4 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0.0
    return {"TTR": v / n, "Yule_K": yule}


@app.post("/compare_two_texts")
def compare_two_texts(payload: ComparePayload, n_char: int = 4) -> Dict:
    a, b = payload.text_a or "", payload.text_b or ""

    char_a = _char_ngram_freqs(a, n_char, top_k=200)
    char_b = _char_ngram_freqs(b, n_char, top_k=200)
    char_cos = _cosine(char_a, char_b)

    fw_a = _function_word_dist(a)
    fw_b = _function_word_dist(b)
    fw_cos = _cosine(fw_a, fw_b)

    pos_a = _pos_bigram_dist(a)
    pos_b = _pos_bigram_dist(b)
    pos_js = _js_divergence(pos_a, pos_b)

    rich_a = _richness_quick(a)
    rich_b = _richness_quick(b)
    ttr_diff = abs(rich_a["TTR"] - rich_b["TTR"])
    yule_ratio = (
        min(rich_a["Yule_K"], rich_b["Yule_K"])
        / max(rich_a["Yule_K"], rich_b["Yule_K"])
        if max(rich_a["Yule_K"], rich_b["Yule_K"]) > 0
        else 1.0
    )

    pos_similarity = max(0.0, 1.0 - pos_js)
    richness_similarity = max(0.0, 1.0 - ttr_diff) * yule_ratio
    overall = 0.40 * char_cos + 0.35 * fw_cos + 0.15 * pos_similarity + 0.10 * richness_similarity

    if overall >= 0.75:
        verdict, confidence = "likely_same_author", "high" if overall >= 0.85 else "medium"
    elif overall >= 0.55:
        verdict, confidence = "inconclusive", "low"
    else:
        verdict, confidence = "likely_different_authors", "high" if overall <= 0.40 else "medium"

    shared_ngrams = sorted(
        set(char_a) & set(char_b),
        key=lambda k: -(min(char_a.get(k, 0), char_b.get(k, 0))),
    )[:8]

    return {
        "per_feature": {
            "char_ngram_cosine": round(char_cos, 4),
            "function_word_cosine": round(fw_cos, 4),
            "pos_bigram_js_divergence": round(pos_js, 4),
            "ttr_diff": round(ttr_diff, 4),
            "yule_k_ratio": round(yule_ratio, 4),
        },
        "richness_a": {k: round(v, 4) for k, v in rich_a.items()},
        "richness_b": {k: round(v, 4) for k, v in rich_b.items()},
        "shared_top_char_ngrams": shared_ngrams,
        "overall_similarity": round(overall, 4),
        "verdict": verdict,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Bundle: extract all features in one call (used by the deterministic
# attribution workflow so we don't fan out N parallel HTTP calls per text).
# ---------------------------------------------------------------------------
@app.post("/extract_all_features")
def extract_all_features(payload: TextIn) -> Dict:
    """Run vocab_richness + function_word_freqs + pos_distribution +
    punctuation_profile + sentence_length_stats on one text."""
    return {
        "vocab_richness": vocab_richness(payload),
        "function_word_freqs": function_word_freqs(payload),
        "pos_distribution": pos_distribution(payload),
        "punctuation_profile": punctuation_profile(payload),
        "sentence_length_stats": sentence_length_stats(payload),
    }


# ---------------------------------------------------------------------------
# Bundle: extract all corpus features in one call.
# Returns features for EVERY indexed sample, grouped by author.
# ---------------------------------------------------------------------------
@app.get("/extract_corpus_features")
def extract_corpus_features() -> Dict:
    _ensure_collection()
    points, _ = _qdrant.scroll(
        collection_name=AUTHOR_COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    by_author: Dict[str, List[Dict]] = {}
    for p in points:
        pl = p.payload or {}
        author = pl.get("author", "unknown")
        sample_id = pl.get("sample_id", "?")
        text = pl.get("text", "")
        features = extract_all_features(TextIn(text=text))
        by_author.setdefault(author, []).append({
            "sample_id": sample_id,
            "n_tokens": pl.get("n_tokens"),
            "features": features,
        })
    return {
        "n_authors": len(by_author),
        "authors": [
            {"author": a, "samples": s}
            for a, s in sorted(by_author.items())
        ],
    }


# ---------------------------------------------------------------------------
# 8) Authorship attribution — corpus index + nearest-author search
# ---------------------------------------------------------------------------
class IndexPayload(BaseModel):
    author: str
    text: str
    sample_id: Optional[str] = None


class AttributePayload(BaseModel):
    text: str
    top_k: int = 5


@app.post("/index_author")
def index_author(payload: IndexPayload) -> Dict:
    """Add a known-author sample to the corpus.

    Stores the FULL text in the Qdrant payload so the agent can fetch it
    later via /get_sample_text and run feature tools against it.
    """
    _ensure_collection()
    vec = _make_fingerprint(payload.text)
    sample_id = payload.sample_id or str(uuid.uuid4())
    point_id = str(uuid.uuid4())
    _qdrant.upsert(
        collection_name=AUTHOR_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "author": payload.author,
                    "sample_id": sample_id,
                    "n_tokens": len(_safe_tokens(payload.text)),
                    "text_preview": payload.text[:200],
                    "text": payload.text,
                },
            )
        ],
    )
    return {
        "author": payload.author,
        "sample_id": sample_id,
        "point_id": point_id,
        "dim": len(vec),
    }


@app.get("/list_samples")
def list_samples(author: Optional[str] = None) -> Dict:
    """List indexed samples. Optionally filter by author.

    Returns sample_id + author + preview + token count for each.
    Use this to discover which sample IDs exist; fetch the full text
    via /get_sample_text(sample_id).
    """
    _ensure_collection()
    points, _ = _qdrant.scroll(
        collection_name=AUTHOR_COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    samples = []
    for p in points:
        pl = p.payload or {}
        if author and pl.get("author") != author:
            continue
        samples.append({
            "sample_id": pl.get("sample_id"),
            "author": pl.get("author"),
            "n_tokens": pl.get("n_tokens"),
            "text_preview": pl.get("text_preview"),
        })
    return {"n_samples": len(samples), "samples": samples}


@app.get("/get_sample_text")
def get_sample_text(sample_id: str) -> Dict:
    """Return the FULL text of a known sample by sample_id.

    The agent uses this to pull a reference author's text into context
    so it can run the standard feature tools (vocab_richness, char_ngrams,
    etc.) against it for comparison with the unknown document.
    """
    _ensure_collection()
    points, _ = _qdrant.scroll(
        collection_name=AUTHOR_COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    for p in points:
        pl = p.payload or {}
        if pl.get("sample_id") == sample_id:
            return {
                "sample_id": sample_id,
                "author": pl.get("author"),
                "n_tokens": pl.get("n_tokens"),
                "text": pl.get("text"),
            }
    raise HTTPException(status_code=404, detail=f"sample_id {sample_id} not found")


@app.post("/attribute")
def attribute(payload: AttributePayload) -> Dict:
    """Given an unknown text, return the most likely author from the indexed corpus."""
    _ensure_collection()
    query_vec = _make_fingerprint(payload.text)
    top_k = max(1, min(payload.top_k, 20))

    hits = _qdrant.search(
        collection_name=AUTHOR_COLLECTION,
        query_vector=query_vec,
        limit=max(top_k * 3, 15),
    )

    if not hits:
        return {
            "verdict": "no_known_authors",
            "message": "Corpus is empty — index samples via /index_author first.",
            "top_authors": [],
        }

    by_author: Dict[str, List[float]] = {}
    samples_by_author: Dict[str, List[str]] = {}
    for h in hits:
        a = (h.payload or {}).get("author", "unknown")
        by_author.setdefault(a, []).append(float(h.score))
        samples_by_author.setdefault(a, []).append(
            (h.payload or {}).get("sample_id", "?")
        )

    ranked = sorted(
        (
            (a, max(scores), sum(scores) / len(scores), len(scores))
            for a, scores in by_author.items()
        ),
        key=lambda x: -x[1],
    )[:top_k]

    best_author, best_score, _, _ = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    if margin >= 0.10 and best_score >= 0.85:
        confidence = "high"
    elif margin >= 0.04 and best_score >= 0.70:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "predicted_author": best_author,
        "best_score": round(best_score, 4),
        "second_score": round(second_score, 4),
        "margin": round(margin, 4),
        "confidence": confidence,
        "top_authors": [
            {
                "author": a,
                "best_score": round(b, 4),
                "avg_score": round(avg, 4),
                "n_matched_samples": n,
            }
            for a, b, avg, n in ranked
        ],
    }


@app.get("/list_authors")
def list_authors() -> Dict:
    """List all authors currently in the corpus with sample counts."""
    _ensure_collection()
    points, _ = _qdrant.scroll(
        collection_name=AUTHOR_COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    by_author: Dict[str, int] = {}
    for p in points:
        a = (p.payload or {}).get("author", "unknown")
        by_author[a] = by_author.get(a, 0) + 1
    return {
        "n_total_samples": len(points),
        "n_authors": len(by_author),
        "authors": [
            {"author": a, "n_samples": n}
            for a, n in sorted(by_author.items(), key=lambda kv: -kv[1])
        ],
    }


@app.delete("/clear_corpus")
def clear_corpus() -> Dict:
    """Wipe the entire author corpus. Use with care."""
    try:
        _qdrant.delete_collection(collection_name=AUTHOR_COLLECTION)
    except Exception:
        pass
    global _collection_ready
    _collection_ready = False
    _ensure_collection()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
