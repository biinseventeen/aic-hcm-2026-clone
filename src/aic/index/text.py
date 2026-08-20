"""P4 + P5 — the sparse text index: BM25 plus fuzzy matching over character n-grams.

This channel takes text from three sources, merged into *one* index but tagged by source so
that per-channel contribution stays diagnosable:

* ``media-info`` — YouTube title, description and keywords. **Available for 873/873 videos of
  batch 1** at no GPU cost. Granularity is per *video*, not per shot.
* ``ocr``        — on-screen text (P4). Shot-level granularity. Needs GPU.
* ``asr``        — transcribed speech (P5). Segment-level granularity. Needs GPU.

Basis for implementing BM25 here
--------------------------------
``rank_bm25`` operates on Python lists and reallocates on every query. Over 176,707 shot-level
documents, an inverted-index plus numpy implementation is orders of magnitude faster and adds
no dependency. The code is short, readable, and holds no mystery.

Basis for the fuzzy layer
-------------------------
OCR output contains character noise and dropped diacritics — "Nguyễn" becomes "Nguyên",
"Nguyen" or "Nguvễn". Exact token matching would fall silent on exactly the entity-rich
queries where this channel should be strongest. The index therefore has two faces: BM25 over
tokens for broad recall, and character n-grams for fuzzy matching on proper nouns.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..log import log

__all__ = [
    "NOISE_PHRASES",
    "STOPWORDS",
    "TextDoc",
    "TextIndex",
    "char_ngrams",
    "docs_from_media_info",
    "normalize_vi",
    "strip_diacritics",
    "tokenize_vi",
]

Source = Literal["media-info", "ocr", "asr", "other"]

#: Tokeniser: keep letters (including accented ones) and digits; drop punctuation.
_TOKEN_RE = re.compile(r"[0-9\w]+", re.UNICODE)

#: Vietnamese function words plus YouTube template noise. The list is deliberately short —
#: BM25 already down-weights frequent terms through IDF, so this only blocks template noise.
#:
#: Caution — do NOT add content words here. The first version contained "ký" (taken from the
#: phrase "đăng ký" in channel-subscribe boilerplate), and it deleted the discriminative token
#: of the query "lễ ký kết hợp tác": the sparse channel fell silent on exactly the query type
#: it is best at. Multi-word noise must be handled by NOISE_PHRASES, never by stop-words.
STOPWORDS: frozenset[str] = frozenset(
    """
    và của có là các một những trong cho được với người khi đã cũng này đó tại về
    từ đến trên dưới ra vào theo như thì mà nhưng hoặc bị nên rất sẽ không chưa
    https http www com vn youtube youtu be bit ly fb facebook
    """.split()  # noqa: SIM905 — a block string keeps this word list reviewable as data
)

#: Template noise phrases, removed BEFORE tokenisation so no content word is lost.
NOISE_PHRASES: tuple[str, ...] = (
    "đăng ký kênh",
    "đăng ký để xem",
    "bấm đăng ký",
    "like và đăng ký",
    "xem thêm tại",
    "theo dõi kênh",
    "mới nhất 2024",
    "mới nhất 2025",
)


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics: "Nguyễn" -> "Nguyen". Used by the fuzzy layer.

    >>> strip_diacritics("Nguyễn Văn Đức")
    'Nguyen Van Duc'
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )


def normalize_vi(text: str) -> str:
    """Normalise to NFC, lowercase, and collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").lower()).strip()


def tokenize_vi(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokenise into syllables.

    Compound words are not segmented: BM25 does not need it, and every Vietnamese word
    segmenter adds both a dependency and a new source of error.

    >>> tokenize_vi("Lễ ký kết hợp tác tại TP.HCM năm 2024")
    ['lễ', 'ký', 'kết', 'hợp', 'tác', 'tp', 'hcm', 'năm', '2024']
    >>> tokenize_vi("Đăng ký kênh để xem tin mới")
    ['để', 'xem', 'tin', 'mới']
    """
    normalised = normalize_vi(text)
    for phrase in NOISE_PHRASES:
        normalised = normalised.replace(phrase, " ")
    tokens = _TOKEN_RE.findall(normalised)
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams over the diacritics-stripped string — the basis of fuzzy matching.

    >>> sorted(char_ngrams("Nguyen", 3))[:3]
    ['guy', 'ngu', 'uye']
    """
    flattened = re.sub(r"\s+", " ", strip_diacritics(text).lower())
    if len(flattened) < n:
        return {flattened} if flattened else set()
    return {flattened[i : i + n] for i in range(len(flattened) - n + 1)}


@dataclass(slots=True)
class TextDoc:
    """One retrievable unit of text.

    ``start``/``end`` is the frame range this text *talks about*. For ``media-info`` that is
    the whole video; for OCR or ASR it is the shot or the speech segment.
    """

    doc_id: int
    video_id: str
    text: str
    source: Source = "other"
    start: int = 0
    end: int = 0

    @property
    def is_video_level(self) -> bool:
        return self.source == "media-info"


@dataclass
class TextIndex:
    """BM25 over an inverted index, plus a character n-gram layer for fuzzy matching."""

    docs: list[TextDoc] = field(default_factory=list)
    k1: float = 1.2
    b: float = 0.75
    ngram: int = 3

    # derived structures
    _postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict, repr=False)
    _doc_len: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _avg_len: float = 0.0
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _ngram_postings: dict[str, set[int]] = field(default_factory=dict, repr=False)
    _ngram_idf: dict[str, float] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.docs)

    # -- building ---------------------------------------------------------

    def build(self, *, with_fuzzy: bool = True, verbose: bool = False) -> TextIndex:
        n_docs = len(self.docs)
        if n_docs == 0:
            raise ValueError("no documents to index")
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(n_docs, dtype=np.float32)
        for doc in self.docs:
            term_freq = Counter(tokenize_vi(doc.text))
            lengths[doc.doc_id] = sum(term_freq.values())
            for term, count in term_freq.items():
                postings[term].append((doc.doc_id, count))
        self._postings = dict(postings)
        self._doc_len = lengths
        self._avg_len = float(lengths.mean()) if n_docs else 0.0
        # Robertson IDF, smoothed so it never goes negative.
        self._idf = {
            term: math.log(1.0 + (n_docs - len(entries) + 0.5) / (len(entries) + 0.5))
            for term, entries in self._postings.items()
        }
        if with_fuzzy:
            self._build_fuzzy(n_docs, verbose=verbose)
        if verbose:
            log.info(
                "  BM25: %s documents, %s tokens, mean length %.1f",
                f"{n_docs:,}",
                f"{len(self._postings):,}",
                self._avg_len,
            )
        return self

    def _build_fuzzy(self, n_docs: int, *, verbose: bool = False) -> None:
        """Build the character n-gram layer.

        No hard document-frequency cut-off is applied. IDF weighting balances the layer
        instead: an n-gram present in every document (channel logo, hashtag, boilerplate)
        naturally receives a weight near zero, while a gram from a rare proper noun receives a
        high one. A hard df cut-off once emptied the results entirely — it removed the
        discriminative grams, and the coverage threshold was then computed over the full gram
        count, so it could never be reached.
        """
        gram_postings: dict[str, set[int]] = defaultdict(set)
        for doc in self.docs:
            for gram in char_ngrams(doc.text, self.ngram):
                gram_postings[gram].add(doc.doc_id)
        self._ngram_postings = dict(gram_postings)
        self._ngram_idf = {
            gram: math.log(1.0 + n_docs / len(doc_ids)) for gram, doc_ids in gram_postings.items()
        }
        if verbose:
            idfs = sorted(self._ngram_idf.values())
            log.info(
                "  n-grams: %s grams, IDF p10=%.2f p90=%.2f",
                f"{len(gram_postings):,}",
                idfs[len(idfs) // 10],
                idfs[9 * len(idfs) // 10],
            )

    # -- querying ---------------------------------------------------------

    def search_bm25(self, query: str, *, top_k: int = 1000) -> list[tuple[int, float]]:
        """BM25. Returns ``[(doc_id, score)]`` in decreasing score."""
        tokens = tokenize_vi(query)
        if not tokens or not self._postings:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokens):
            entries = self._postings.get(term)
            if not entries:
                continue
            idf = self._idf[term]
            query_freq = tokens.count(term)
            for doc_id, count in entries:
                doc_len = self._doc_len[doc_id]
                denominator = count + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self._avg_len, 1e-9)
                )
                scores[doc_id] += (
                    idf * query_freq * (count * (self.k1 + 1)) / max(denominator, 1e-9)
                )
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]

    def search_fuzzy(
        self, entities: Sequence[str], *, top_k: int = 1000, min_coverage: float = 0.34
    ) -> list[tuple[int, float]]:
        """Fuzzy character n-gram match for a list of named entities.

        The score for one entity is its **IDF-weighted n-gram coverage**, a value in [0, 1]:
        the summed IDF of matching grams divided by the summed IDF of all grams in the entity.
        The document score is the *sum across entities* — a document containing two of the
        entities named in the query is markedly stronger evidence than one containing a single
        entity.

        IDF weighting (rather than raw counts) is required: a diacritics-stripped OCR string
        still preserves the *rare* grams of a proper noun, and those carry all the
        discriminative signal.
        """
        if not self._ngram_postings:
            return []
        scores: dict[int, float] = defaultdict(float)
        for entity in entities:
            grams = char_ngrams(entity, self.ngram)
            if not grams:
                continue
            total_idf = sum(self._ngram_idf.get(gram, 0.0) for gram in grams)
            if total_idf <= 0:
                continue
            matched: dict[int, float] = defaultdict(float)
            for gram in grams:
                weight = self._ngram_idf.get(gram)
                if not weight:
                    continue
                for doc_id in self._ngram_postings.get(gram, ()):
                    matched[doc_id] += weight
            for doc_id, accumulated in matched.items():
                coverage = accumulated / total_idf
                if coverage >= min_coverage:
                    scores[doc_id] += coverage
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]

    # -- save / load ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "ngram": self.ngram,
                    "docs": [
                        [doc.doc_id, doc.video_id, doc.text, doc.source, doc.start, doc.end]
                        for doc in self.docs
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path, *, with_fuzzy: bool = True) -> TextIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(
            docs=[
                TextDoc(doc_id, video_id, text, source, start, end)
                for doc_id, video_id, text, source, start, end in payload["docs"]
            ],
            k1=payload.get("k1", 1.2),
            b=payload.get("b", 0.75),
            ngram=payload.get("ngram", 3),
        )
        return index.build(with_fuzzy=with_fuzzy)


def docs_from_media_info(
    media_info: dict[str, dict], *, last_frames: dict[str, int] | None = None
) -> list[TextDoc]:
    """Build video-level documents from ``media-info``.

    Concatenates ``title``, ``author``, ``publish_date``, ``description`` and ``keywords``. The
    broadcast date is kept because some queries anchor on a point in time ("the bulletin of
    01/08/2024") and this is the only field carrying that. ``author`` is kept too: each channel
    in the corpus publishes exactly one genre (see :mod:`aic.index.priors`), so it doubles as a
    genre signal.
    """
    docs: list[TextDoc] = []
    for doc_id, (video_id, info) in enumerate(sorted(media_info.items())):
        parts = [
            str(info.get("title") or ""),
            str(info.get("author") or ""),
            str(info.get("publish_date") or ""),
            str(info.get("description") or "")[:2000],
            " ".join(str(keyword) for keyword in (info.get("keywords") or [])[:60]),
        ]
        docs.append(
            TextDoc(
                doc_id=doc_id,
                video_id=video_id,
                text=" \n ".join(part for part in parts if part),
                source="media-info",
                start=0,
                end=(last_frames or {}).get(video_id, 0),
            )
        )
    return docs
