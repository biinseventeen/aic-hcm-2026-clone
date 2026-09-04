"""The request and response contract, written with standard-library dataclasses.

``pydantic`` is deliberately not used: these structures are the contract of the service layer, and
the service layer must not depend on a web library. FastAPI reads plain dataclasses through their
type annotations, and anybody using a different framework can reuse them unchanged.

Input validation lives in :meth:`SolveRequest.validated` rather than in a decorator: the same rules
must apply to HTTP callers and to in-process callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..data.layout import VIDEO_ID_RE
from ..submit.writer import MAX_ANSWER_CHARS

__all__ = [
    "MAX_BATCH",
    "MAX_QUERY_CHARS",
    "AnswerHypothesisIn",
    "BatchSolveRequest",
    "ErrorResponse",
    "HealthResponse",
    "PackageRequest",
    "PinIn",
    "ReviewRequest",
    "SolveRequest",
    "SolveResponse",
    "SubmitRequest",
    "normalise_query_id",
    "normalise_task",
]

#: A judge's query is a description, not a document. This ceiling exists so one unusual request
#: cannot drag the engine through several pages of text.
MAX_QUERY_CHARS = 2000
#: Each query costs hundreds of milliseconds to seconds; an oversized batch stalls a worker.
MAX_BATCH = 50
#: Characters rejected in a query id, because the id becomes part of a submission filename.
_UNSAFE_ID_CHARS = frozenset(r'/\:*?"<>|')
#: Frontend pages use ``qna``; submission filenames and the engine use ``qa``.
_TASK_ALIAS = {"qna": "qa"}
_TASKS = frozenset({"kis", "qa", "trake"})


def normalise_query_id(query_id: str | None) -> str:
    value = (query_id or "").strip() or "1"
    if any(char in value for char in _UNSAFE_ID_CHARS) or value in (".", ".."):
        raise ValueError(f"`query_id` contains characters unusable in a filename: {value!r}")
    return value


def normalise_task(task: str | None) -> str | None:
    if task is None:
        return None
    mapped = _TASK_ALIAS.get(task, task)
    if mapped not in _TASKS:
        raise ValueError(f"invalid `task`: {task!r}")
    return mapped


def _normalise_text(text: str | None) -> str:
    value = (text or "").strip()
    if not value:
        raise ValueError("missing `text`: a query body is required")
    if len(value) > MAX_QUERY_CHARS:
        raise ValueError(f"`text` is {len(value)} characters, over the limit of {MAX_QUERY_CHARS}")
    return value


@dataclass
class AnswerHypothesisIn:
    """One Q&A answer string a human (or a future VQA model) supplies."""

    text: str
    prob: float = 1.0

    def validated(self) -> AnswerHypothesisIn:
        # Whitespace is significant under the strict Q&A reading — do not strip.
        if self.text is None or self.text == "":
            raise ValueError("an answer hypothesis needs non-empty `text`")
        if len(self.text) > MAX_ANSWER_CHARS:
            raise ValueError(
                f"answer text is {len(self.text)} characters, over the limit of {MAX_ANSWER_CHARS}"
            )
        try:
            prob = float(self.prob)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"answer `prob` must be a number, got {self.prob!r}") from exc
        if prob < 0:
            raise ValueError(f"answer `prob` must be >= 0, got {prob}")
        return AnswerHypothesisIn(text=self.text, prob=prob)


@dataclass
class PinIn:
    """A ``(video_id, frame)`` row a human has verified by looking at the frame."""

    video_id: str
    frame: int

    def validated(self) -> PinIn:
        video_id = (self.video_id or "").strip()
        if not VIDEO_ID_RE.match(video_id):
            raise ValueError(
                f"pin `video_id` does not match L<group>_V<number>: {self.video_id!r}"
            )
        try:
            frame = int(self.frame)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pin `frame` must be an integer, got {self.frame!r}") from exc
        if frame < 0:
            raise ValueError(f"pin `frame` must be >= 0, got {frame}")
        return PinIn(video_id=video_id, frame=frame)


def _coerce_answers(raw: Any) -> list[AnswerHypothesisIn]:
    if not raw:
        return []
    items: list[AnswerHypothesisIn] = []
    for entry in raw:
        if isinstance(entry, AnswerHypothesisIn):
            items.append(entry)
        elif isinstance(entry, dict):
            items.append(
                AnswerHypothesisIn(text=entry.get("text", ""), prob=entry.get("prob", 1.0))
            )
        elif isinstance(entry, (list, tuple)) and entry:
            items.append(
                AnswerHypothesisIn(
                    text=str(entry[0]),
                    prob=float(entry[1]) if len(entry) > 1 else 1.0,
                )
            )
        else:
            raise ValueError(f"invalid answer hypothesis: {entry!r}")
    return [item.validated() for item in items]


def _coerce_pins(raw: Any) -> list[PinIn]:
    if not raw:
        return []
    items: list[PinIn] = []
    for entry in raw:
        if isinstance(entry, PinIn):
            items.append(entry)
        elif isinstance(entry, dict):
            items.append(PinIn(video_id=entry.get("video_id", ""), frame=entry.get("frame", -1)))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            items.append(PinIn(video_id=str(entry[0]), frame=entry[1]))
        else:
            raise ValueError(f"invalid pin: {entry!r}")
    validated = [item.validated() for item in items]
    seen: set[tuple[str, int]] = set()
    for pin in validated:
        key = (pin.video_id, pin.frame)
        if key in seen:
            raise ValueError(f"duplicate pin: {pin.video_id} frame {pin.frame}")
        seen.add(key)
    return validated


@dataclass
class SolveRequest:
    """One query to solve."""

    text: str
    query_id: str = "1"
    #: ``qna`` is accepted as an alias of ``qa`` (frontend page name vs submission task).
    task: Literal["kis", "qa", "qna", "trake"] | None = None
    #: caps how many answers the response carries; None returns all 100. Does not affect scoring.
    top: int | None = None
    #: Q&A: False when the organisers accept only one answer per (video_id, frame_id).
    hedge_answers: bool = True
    answers: list[AnswerHypothesisIn] = field(default_factory=list)
    pins: list[PinIn] = field(default_factory=list)

    def validated(self) -> SolveRequest:
        """Raise :class:`ValueError` with a readable message. Never silently repairs the input."""
        if self.top is not None and not 1 <= self.top <= 100:
            raise ValueError(f"`top` must lie in [1, 100], got {self.top}")
        return SolveRequest(
            text=_normalise_text(self.text),
            query_id=normalise_query_id(self.query_id),
            task=normalise_task(self.task),
            top=self.top,
            hedge_answers=self.hedge_answers,
            answers=_coerce_answers(self.answers),
            pins=_coerce_pins(self.pins),
        )

    def engine_answers(self) -> list[tuple[str, float]] | None:
        if not self.answers:
            return None
        return [(item.text, item.prob) for item in self.answers]

    def engine_pins(self) -> list[tuple[str, int]] | None:
        if not self.pins:
            return None
        return [(item.video_id, item.frame) for item in self.pins]


@dataclass
class BatchSolveRequest:
    queries: list[SolveRequest] = field(default_factory=list)

    def validated(self) -> BatchSolveRequest:
        if not self.queries:
            raise ValueError("`queries` is empty")
        if len(self.queries) > MAX_BATCH:
            raise ValueError(f"{len(self.queries)} queries, over the limit of {MAX_BATCH}")
        validated = [query.validated() for query in self.queries]
        ids = [query.query_id for query in validated]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate `query_id` in the batch — each query needs its own id")
        return BatchSolveRequest(validated)


@dataclass
class ReviewRequest:
    """Retrieve a human-review shortlist. Does not run the solver or write a submission."""

    text: str
    query_id: str = "1"
    task: Literal["kis", "qa", "qna", "trake"] | None = None

    def validated(self) -> ReviewRequest:
        return ReviewRequest(
            text=_normalise_text(self.text),
            query_id=normalise_query_id(self.query_id),
            task=normalise_task(self.task),
        )


@dataclass
class SubmitRequest(SolveRequest):
    """Solve one query and write its CSV. ``strict`` controls whether errors still write."""

    #: True (default): error-level issues become 422 and the file is not written.
    #: False: write anyway and return the issues (warnings always write).
    strict: bool = True

    def validated(self) -> SubmitRequest:
        base = SolveRequest.validated(self)
        return SubmitRequest(
            text=base.text,
            query_id=base.query_id,
            task=base.task,
            top=base.top,
            hedge_answers=base.hedge_answers,
            answers=base.answers,
            pins=base.pins,
            strict=bool(self.strict),
        )


@dataclass
class PackageRequest:
    """Zip already-written CSVs into a ``submission/`` archive."""

    query_ids: list[str] = field(default_factory=list)
    #: optional subdirectory under the submission dir (one directory per query set).
    set_name: str | None = None
    zip_name: str = "submission.zip"

    def validated(self) -> PackageRequest:
        if not self.query_ids:
            raise ValueError("`query_ids` is empty")
        query_ids = [normalise_query_id(query_id) for query_id in self.query_ids]
        set_name = (self.set_name or "").strip() or None
        if set_name is not None and (
            any(char in set_name for char in _UNSAFE_ID_CHARS) or set_name in (".", "..")
        ):
            raise ValueError(f"`set_name` contains characters unusable in a path: {set_name!r}")
        zip_name = (self.zip_name or "").strip() or "submission.zip"
        if any(char in zip_name for char in _UNSAFE_ID_CHARS) or zip_name in (".", ".."):
            raise ValueError(f"`zip_name` contains characters unusable in a filename: {zip_name!r}")
        if not zip_name.endswith(".zip"):
            zip_name = f"{zip_name}.zip"
        return PackageRequest(query_ids=query_ids, set_name=set_name, zip_name=zip_name)


@dataclass
class SolveResponse:
    """Wraps the result of :meth:`aic.service.Engine.solve`.

    ``degraded`` is deliberately lifted to the top level of the response: the client must see
    immediately that a result is **not** usable for a submission, rather than having to dig
    through ``notes``.
    """

    result: dict[str, Any]
    degraded: bool

    @classmethod
    def of(cls, result, *, top: int | None = None) -> SolveResponse:
        return cls(result=result.to_dict(top=top), degraded=result.degraded)

    def to_dict(self) -> dict:
        return {"degraded": self.degraded, **self.result}


@dataclass
class HealthResponse:
    status: Literal["ready", "degraded", "loading", "error"]
    engine: dict[str, Any] | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class ErrorResponse:
    error: str
    detail: str | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}
