"""P7 — query understanding: turning a free-form Vietnamese query into a control structure.

This is the single failure point with the **largest blast radius** in the pipeline: a wrong
task classification routes the query into the wrong branch, a wrong translation breaks the
main retrieval channel, and invented keywords inject noise into the sparse channel.

Mitigated architecturally, not by model quality
-----------------------------------------------
The **raw, unprocessed** query is always retained as an independent retrieval channel, running
in parallel with the expanded one. If query understanding mangles a query, the raw channel is
still intact. The cost is one extra vector search, about 5 ms.

Two modes
---------
* :class:`RuleParser` — dependency-free, deterministic, needs no network. Sufficient for task
  classification, entity extraction and negation extraction. It does **not** translate.
* An LLM callable — an external API call, consuming no local GPU. Gives better translation and
  better TRAKE moment splitting.

Negation and spatial relations are **never** fed into the embedding step: the vector for "a
scene without a hat" sits close to the vector for "a scene with a hat". They are extracted as
exclusion constraints and applied only in the verification layer (P9).

The Vietnamese string constants below are linguistic *data*, not prose, and stay in Vietnamese:
marker phrases, spatial words, domain keywords, and the LLM prompt that analyses Vietnamese
queries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from ..index.text import _LABELLED_MOMENT_PATTERN, normalize_vi, tokenize_vi

__all__ = ["ParsedQuery", "RuleParser", "TaskKind", "parse_query"]

TaskKind = Literal["kis", "qa", "trake"]


@dataclass
class ParsedQuery:
    """The control structure for every downstream layer."""

    raw: str
    task: TaskKind = "kis"
    #: English translation or paraphrase for the dense channel (CLIP is strongest in English).
    english: str = ""
    #: keywords for the sparse channel (BM25).
    keywords: list[str] = field(default_factory=list)
    #: named entities for fuzzy OCR matching (people, places, figures).
    entities: list[str] = field(default_factory=list)
    #: for Q&A: the question, separated from the event description.
    question: str = ""
    #: for TRAKE: N ordered moment descriptions.
    moments: list[str] = field(default_factory=list)
    #: exclusion constraints from negation — verification only, NEVER used for embedding.
    exclude: list[str] = field(default_factory=list)
    #: spatial relations — also verification only.
    spatial: list[str] = field(default_factory=list)
    #: content domain hints, used by the L-group prior (see aic.index.priors).
    domain_hints: list[str] = field(default_factory=list)
    #: confidence of the task classification; low means run both branches.
    task_confidence: float = 1.0
    parser: str = "rule"

    @property
    def n_moments(self) -> int:
        return len(self.moments)

    def dense_texts(self) -> list[str]:
        """Strings for the dense channel: the translation if any, and **always** the raw query."""
        texts = []
        if self.english:
            texts.append(self.english)
        texts.append(self.raw)
        return texts

    def summary(self) -> str:
        lines = [f"task={self.task} (conf {self.task_confidence:.2f}) parser={self.parser}"]
        if self.english:
            lines.append(f"  en       : {self.english}")
        if self.question:
            lines.append(f"  question : {self.question}")
        for i, moment in enumerate(self.moments, 1):
            lines.append(f"  moment {i} : {moment}")
        if self.keywords:
            lines.append(f"  keywords : {', '.join(self.keywords[:12])}")
        if self.entities:
            lines.append(f"  entities : {', '.join(self.entities)}")
        if self.exclude:
            lines.append(f"  EXCLUDE  : {', '.join(self.exclude)}  (verification only)")
        if self.spatial:
            lines.append(f"  spatial  : {', '.join(self.spatial)}")
        if self.domain_hints:
            lines.append(f"  domain   : {', '.join(self.domain_hints)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Rule-based parser
#
# Every constant in this section is Vietnamese *data*: surface markers of the language the
# queries are written in. Translating them would break the parser.
# --------------------------------------------------------------------------

#: Markers of a Q&A question.
_QA_MARKERS = (
    "bao nhiêu",
    "mấy",
    "màu gì",
    "màu nào",
    "là ai",
    "tên gì",
    "ở đâu",
    "khi nào",
    "vì sao",
    "tại sao",
    "thế nào",
    "cái gì",
    "gì?",
    "?",
    "hãy cho biết",
    "cho biết",
)
#: Markers of TRAKE — a structured sequence of moments.
_TRAKE_MARKERS = (
    "khoảnh khắc",
    "các bước",
    "giai đoạn",
    "lần lượt",
    "theo thứ tự",
    "chuỗi sự kiện",
    "các pha",
    "trình tự",
    "bước 1",
    "(1)",
    "thứ nhất",
)
#: Negation markers.
_NEGATION_RE = re.compile(
    r"\b(không|chẳng|chưa|đừng|ngoại trừ|trừ|không có|không phải|no |not )\s*([^,.;]{1,40})",
    re.IGNORECASE,
)
#: Spatial relation markers.
_SPATIAL_WORDS = (
    "bên trái",
    "bên phải",
    "phía trước",
    "phía sau",
    "ở giữa",
    "trên",
    "dưới",
    "cạnh",
    "kế bên",
    "góc",
    "nền",
    "hậu cảnh",
    "tiền cảnh",
)
#: Domain hints. The keys must match the label set of ``aic.index.priors.GROUP_DOMAIN``,
#: otherwise a hint inferred from the query matches no video and the prior silently does
#: nothing. ``tests/test_priors.py`` locks that consistency. Genres: docs/DATA_AUDIT.md §2.1.
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sport_cycling": (
        "đua xe đạp",
        "cua-rơ",
        "cúp truyền hình",
        "chặng đua",
        "vòng đua",
        "về đích",
        "nước rút",
        "xe đạp",
    ),
    "sport_liondance": ("lân sư rồng", "múa lân", "múa rồng", "mai hoa thung", "đoàn lân"),
    "cooking": (
        "món",
        "nấu",
        "chiên",
        "xào",
        "nêm",
        "công thức",
        "nguyên liệu",
        "đầu bếp",
        "chảo",
        "nồi",
        "ướp",
        "bánh",
    ),
    "news": (
        "bản tin",
        "phóng viên",
        "họp báo",
        "phát biểu",
        "hội nghị",
        "lễ",
        "khai mạc",
        "trao giải",
        "60 giây",
        "thời sự",
    ),
    # L25 — 36.2 hours of exam revision, 27.7 % of corpus duration.
    "education": (
        "ôn thi",
        "luyện thi",
        "bí quyết ôn",
        "chuyên đề",
        "bài giảng",
        "thí sinh",
        "môn toán",
        "môn văn",
        "thpt",
        "giáo viên",
        "bảng đen",
    ),
    # L30 — 95 of 96 videos belong to one talk-show series.
    "talkshow": (
        "talkshow",
        "toạ đàm",
        "tọa đàm",
        "khách mời",
        "diễn giả",
        "năng lượng tích cực",
        "trò chuyện",
    ),
    "documentary": ("phóng sự", "tài liệu", "ký sự", "mê kông", "làng nghề"),
    "travel": ("du lịch", "miệt vườn", "điểm đến", "khám phá", "đặc sản"),
}

#: Splitting TRAKE moments: "(1) ... (2) ...", "1. ... 2. ...", or "bước 1: ...".
_MOMENT_SPLIT = re.compile(r"(?:\(\s*\d+\s*\)|\b\d+\s*[.):]|\bbước\s+\d+\s*[:.]?)", re.IGNORECASE)

#: Explicit moment labels, the form the organisers actually publish: each moment on its own line,
#: opened by ``E1:``, ``E2:``, ... Where these are present they are unambiguous — the number of
#: labels *is* N — so they take precedence over the heuristic split above, which would otherwise
#: miss them entirely (routing a TRAKE query into the KIS branch) or mistake the lead-in sentence
#: for a moment. The label **values** are not trusted: the published set contains a query numbered
#: E1, E2, E2, E4, and that query still has four moments.
#: Punctuation after the number is optional: the mock set writes "E1:", the first real set writes
#: "E1 " with nothing but a space. The pattern lives in aic.index.text beside the other
#: query-surface data.
_LABELLED_MOMENT = re.compile(_LABELLED_MOMENT_PATTERN, re.MULTILINE)


def _vietnamese_uppercase() -> str:
    """The Vietnamese uppercase letters, as the body of a regex character class.

    Computed rather than written as the range ``À-Ỹ``. That range spans the Latin-1 Supplement
    and the Latin Extended blocks, where uppercase and lowercase code points interleave, so it
    matches every lowercase accented letter too — and the all-lowercase query
    "áo đỏ đang nấu ăn" then yields "áo đỏ đang" as a proper noun, which the fuzzy entity
    channel goes on to treat as a name.

    >>> upper = _vietnamese_uppercase()
    >>> "Đ" in upper and "Ỹ" in upper
    True
    >>> any(character.islower() for character in upper)
    False
    """
    # Latin-1 Supplement + Latin Extended-A/B, then Latin Extended Additional: between them they
    # hold every precomposed Vietnamese letter.
    blocks = ((0x00C0, 0x0250), (0x1E00, 0x1F00))
    letters = "".join(
        chr(code) for start, stop in blocks for code in range(start, stop) if chr(code).isupper()
    )
    return "A-Z" + letters


#: Named entities: runs of capitalised words, figures, dates.
_UPPERCASE = _vietnamese_uppercase()
_PROPER_NOUN_RE = re.compile(rf"\b(?:[{_UPPERCASE}]\w*(?:\s+[{_UPPERCASE}]\w*){{0,4}})\b")
#: Figures, dates, and Vietnamese licence plates ("51F-123.45"): the optional letter group
#: keeps the plate in one piece, which is what OCR reads off the vehicle.
_NUMERIC_RE = re.compile(r"\b\d{1,4}[A-Z]?(?:[/.\-]\d{1,4}){0,2}\b")

#: Vietnamese imperatives that get capitalised at the start of a query but are not entities.
#: Also generic nouns that open a capitalised name: "Thành phố Hồ Chí Minh" breaks into
#: "Thành" + "Hồ Chí Minh" because "phố" is lowercase, and the fragment "Thành" would be handed
#: to the fuzzy channel as a name. The filter applies to **single-word** candidates only, so
#: "Hồ Chí Minh" survives while a lone "Hồ" does not.
_NON_ENTITY_WORDS = frozenset({"tìm", "trong", "hãy", "cho"})
_GENERIC_HEAD_NOUNS = frozenset(
    {
        "thành",
        "thủ",
        "tỉnh",
        "huyện",
        "quận",
        "phường",
        "xã",
        "sông",
        "núi",
        "hồ",
        "biển",
        "đường",
        "phố",
        "chợ",
        "trường",
        "bệnh",
        "công",
        "nhà",
    }
)


# Temporal markers commonly used by real KIS queries.  KIS remains a KIS task:
# these are only ordered visual sub-moments used by retrieval/localisation.
_KIS_TEMPORAL_SPLIT = re.compile(
    r"""
    \s*(?:
        (?<=\.)\s+
        |
        \b(?:sau\s+đó|tiếp\s+theo|rồi|sau\s+khi)\b
        |
        \b(?:đoạn\s+clip|cảnh\s+quay)\s+kết\s+thúc\s+(?:với|bằng)?\b
        |
        \bkết\s+thúc\s+(?:với|bằng)?\b
        |
        # English temporal KIS markers.  The devset/manual translations are
        # English, so without these the temporal retriever is silently disabled.
        \b(?:then|after\s+that|afterwards?|subsequently|next|finally)\b
        |
        \b(?:the\s+)?(?:scene|clip)\s+(?:then\s+)?(?:cuts?|transitions?)\s+to\b
        |
        \b(?:the\s+)?(?:scene|clip)\s+ends?\s+(?:with|on)?\b
        |
        \bends?\s+(?:with|on)\b
    )\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

_KIS_START_PREFIX = re.compile(
    r"""
    ^\s*
    (?:
        đoạn\s+clip\s+bắt\s+đầu\s+(?:với|bằng)? |
        cảnh\s+quay\s+bắt\s+đầu\s+(?:với|bằng)? |
        bắt\s+đầu\s+(?:với|bằng)? |
        (?:the\s+)?(?:scene|clip)\s+(?:begins?|starts?)\s+(?:with|on)? |
        (?:begins?|starts?)\s+(?:with|on)?
    )
    \s*
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _split_kis_temporal_moments(raw: str) -> list[str]:
    """Split a temporal KIS description into ordered visual moments.

    The task stays ``kis``.  These moments are retrieval hints for finding a
    short ordered segment inside one video, not TRAKE output moments.
    """
    cleaned = _KIS_START_PREFIX.sub("", raw.strip())

    parts = [
        part.strip(" ,.;:–—-")
        for part in _KIS_TEMPORAL_SPLIT.split(cleaned)
    ]
    parts = [part for part in parts if len(part.split()) >= 3]

    # Only call it temporal when the query genuinely decomposes into >=2
    # meaningful visual clauses.
    if len(parts) < 2:
        return []

    # Keep a conservative cap so a verbose query cannot explode retrieval cost.
    return parts[:6]


@dataclass
class RuleParser:
    """Deterministic parsing, with no dependency on a network or a model.

    It does **not translate**. Without a translation, the dense channel uses only the raw
    query — lower quality, but no additional failure point.
    """

    parser_name: str = "rule"

    def parse(self, raw: str, *, task_hint: TaskKind | None = None) -> ParsedQuery:
        query = ParsedQuery(raw=raw.strip(), parser=self.parser_name)
        lowered = normalize_vi(raw)

        # -- task classification -----------------------------------------
        if task_hint is not None:
            query.task, query.task_confidence = task_hint, 1.0
        else:
            n_trake = sum(1 for marker in _TRAKE_MARKERS if marker in lowered)
            n_qa = sum(1 for marker in _QA_MARKERS if marker in lowered)
            moments = self._split_moments(raw)
            # Explicit E-labels are evidence on their own: a query listing "E1: ... E2: ..." is a
            # moment sequence whether or not it also uses one of the Vietnamese marker phrases.
            labelled = len(_LABELLED_MOMENT.findall(raw)) >= 2
            if (n_trake > 0 or labelled) and len(moments) >= 2:
                query.task = "trake"
                query.task_confidence = min(1.0, (0.75 if labelled else 0.6) + 0.15 * n_trake)
            elif n_qa > 0:
                query.task = "qa"
                query.task_confidence = min(1.0, 0.55 + 0.15 * n_qa)
            else:
                query.task, query.task_confidence = "kis", 0.6

        # -- moment / question extraction --------------------------------
        if query.task == "trake":
            query.moments = self._split_moments(raw)
        elif query.task == "kis":
            # Real KIS queries are often short temporal segment descriptions
            # ("bắt đầu ... sau đó ... kết thúc ..."). Preserve the task as KIS
            # but expose ordered sub-moments for temporal retrieval/localisation.
            query.moments = _split_kis_temporal_moments(raw)

        if query.task == "qa":
            query.question = self._extract_question(raw)

        # -- sparse channel and entity matching --------------------------
        query.keywords = self._keywords(raw)
        query.entities = self._entities(raw)

        # -- verification-only constraints -------------------------------
        # Important: negation must NOT reach the embedding. Retrieve on positive constraints,
        # then eliminate on negative ones in P9.
        query.exclude = [
            match.group(2).strip() for match in _NEGATION_RE.finditer(raw) if match.group(2).strip()
        ]
        query.spatial = [word for word in _SPATIAL_WORDS if word in lowered]

        # -- domain hints ------------------------------------------------
        query.domain_hints = [
            domain
            for domain, keywords in _DOMAIN_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        return query

    # -- details ----------------------------------------------------------

    @staticmethod
    def _split_moments(raw: str) -> list[str]:
        # Labelled form first: the label count is N, and whatever precedes the first label is the
        # lead-in sentence, not a moment.
        labels = list(_LABELLED_MOMENT.finditer(raw))
        if len(labels) >= 2:
            starts = [match.end() for match in labels]
            stops = [match.start() for match in labels[1:]] + [len(raw)]
            return [
                raw[start:stop].strip(" ,.;:\n–-")
                for start, stop in zip(starts, stops, strict=True)
            ]
        parts = [part.strip(" ,.;:–-") for part in _MOMENT_SPLIT.split(raw)]
        parts = [part for part in parts if len(part) >= 3]
        # The first part is usually the lead-in ("Find the 4 moments when..."), not a moment.
        if len(parts) >= 3 and re.search(
            r"khoảnh khắc|các bước|giai đoạn|pha", parts[0], re.IGNORECASE
        ):
            parts = parts[1:]
        return parts if len(parts) >= 2 else []

    @staticmethod
    def _extract_question(raw: str) -> str:
        # The sentence containing a question mark, or the last sentence if there is none.
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", raw) if s.strip()]
        for sentence in sentences:
            if "?" in sentence:
                return sentence
        lowered = normalize_vi(raw)
        for marker in _QA_MARKERS:
            if marker in lowered and marker != "?":
                for sentence in sentences:
                    if marker in normalize_vi(sentence):
                        return sentence
        return sentences[-1] if sentences else raw

    @staticmethod
    def _keywords(raw: str, *, max_keywords: int = 24) -> list[str]:
        counts: dict[str, int] = {}
        for token in tokenize_vi(raw):
            if len(token) >= 2 or token.isdigit():
                counts[token] = counts.get(token, 0) + 1
        return list(counts)[:max_keywords]

    @staticmethod
    def _entities(raw: str, *, max_entities: int = 12) -> list[str]:
        candidates: list[str] = []
        for match in _PROPER_NOUN_RE.finditer(raw):
            text = match.group(0).strip()
            # Skip the first word of the query: it is capitalised because it starts the
            # sentence, not because it is a proper noun.
            if match.start() == 0 and " " not in text:
                continue
            lowered = text.lower()
            if len(text) < 3 or lowered in _NON_ENTITY_WORDS:
                continue
            if " " not in text and lowered in _GENERIC_HEAD_NOUNS:
                continue
            candidates.append(text)
        candidates.extend(match.group(0) for match in _NUMERIC_RE.finditer(raw))
        # Deduplicate, preserving order.
        seen: set[str] = set()
        unique = []
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique[:max_entities]


def parse_query(raw: str, *, task_hint: TaskKind | None = None, llm=None) -> ParsedQuery:
    """Parse one query. When ``llm`` is given, use it to enrich the result.

    ``llm`` must be a callable taking a prompt and returning a JSON string. Without it, the
    rule parser is used alone. The rule result is *always* kept as the base: the LLM may only
    **add** to it, never overwrite a field the rules already know for certain (such as ``raw``).
    """
    query = RuleParser().parse(raw, task_hint=task_hint)
    if llm is None:
        return query
    try:
        return _enrich_with_llm(query, llm)
    except Exception as exc:
        # Query understanding must never bring down the pipeline. Fall back to the rules.
        query.parser = f"rule (LLM failed: {type(exc).__name__})"
        return query


#: The prompt stays in Vietnamese: it instructs a model to analyse Vietnamese queries, and the
#: task markers it must recognise are Vietnamese surface forms.
_PROMPT = """Bạn phân tích truy vấn tìm kiếm video tiếng Việt cho cuộc thi AIC 2026.
Trả về DUY NHẤT một JSON object, không giải thích, với các khoá:
  task: "kis" | "qa" | "trake"
  english: bản dịch/diễn giải tiếng Anh, mô tả THỊ GIÁC những gì thấy trong khung hình
  keywords: mảng từ khoá tiếng Việt để tìm kiếm văn bản
  entities: mảng tên riêng / địa danh / số liệu xuất hiện trong truy vấn
  question: câu hỏi (chỉ khi task = "qa"), rỗng nếu không
  moments: mảng mô tả từng khoảnh khắc theo thứ tự (chỉ khi task = "trake"), rỗng nếu không
  exclude: mảng điều kiện PHỦ ĐỊNH (những gì KHÔNG được có trong khung hình)
Quy tắc quan trọng: trường "english" chỉ chứa ràng buộc DƯƠNG. Không đưa phủ định
vào "english" — mô hình nhúng vector không biểu diễn được phủ định.

Truy vấn: {raw}
JSON:"""

#: Fields the LLM may extend. Each is a list that gets merged, never replaced.
_LLM_LIST_FIELDS = ("keywords", "entities", "moments", "exclude")


def _enrich_with_llm(query: ParsedQuery, llm) -> ParsedQuery:
    response = llm(_PROMPT.format(raw=query.raw))
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError(f"the LLM returned no JSON: {response[:200]!r}")
    payload = json.loads(match.group(0))
    if payload.get("task") in ("kis", "qa", "trake"):
        query.task = payload["task"]
        query.task_confidence = 0.9
    query.english = str(payload.get("english") or "").strip()
    for field_name in _LLM_LIST_FIELDS:
        values = payload.get(field_name) or []
        if not isinstance(values, list):
            continue
        merged = list(getattr(query, field_name))
        for value in values:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
        setattr(query, field_name, merged)
    query.question = str(payload.get("question") or query.question).strip()
    query.parser = "rule+llm"
    return query
