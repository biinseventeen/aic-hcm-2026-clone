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

__all__ = [
    "MAX_BATCH",
    "MAX_QUERY_CHARS",
    "BatchSolveRequest",
    "ErrorResponse",
    "HealthResponse",
    "SolveRequest",
    "SolveResponse",
]

#: A judge's query is a description, not a document. This ceiling exists so one unusual request
#: cannot drag the engine through several pages of text.
MAX_QUERY_CHARS = 2000
#: Each query costs hundreds of milliseconds to seconds; an oversized batch stalls a worker.
MAX_BATCH = 50
#: Characters rejected in a query id, because the id becomes part of a submission filename.
_UNSAFE_ID_CHARS = frozenset(r'/\:*?"<>|')


@dataclass
class SolveRequest:
    """One query to solve."""

    text: str
    query_id: str = "1"
    task: Literal["kis", "qa", "trake"] | None = None
    #: caps how many answers the response carries; None returns all 100. Does not affect scoring.
    top: int | None = None
    #: Q&A: False when the organisers accept only one answer per (video_id, frame_id).
    hedge_answers: bool = True

    def validated(self) -> SolveRequest:
        """Raise :class:`ValueError` with a readable message. Never silently repairs the input."""
        text = (self.text or "").strip()
        if not text:
            raise ValueError("missing `text`: a query body is required")
        if len(text) > MAX_QUERY_CHARS:
            raise ValueError(
                f"`text` is {len(text)} characters, over the limit of {MAX_QUERY_CHARS}"
            )
        if self.task is not None and self.task not in ("kis", "qa", "trake"):
            raise ValueError(f"invalid `task`: {self.task!r}")
        if self.top is not None and not 1 <= self.top <= 100:
            raise ValueError(f"`top` must lie in [1, 100], got {self.top}")
        query_id = (self.query_id or "").strip() or "1"
        # The query id becomes a submission filename; block path characters at the boundary.
        if any(char in query_id for char in _UNSAFE_ID_CHARS) or query_id in (".", ".."):
            raise ValueError(f"`query_id` contains characters unusable in a filename: {query_id!r}")
        return SolveRequest(
            text=text,
            query_id=query_id,
            task=self.task,
            top=self.top,
            hedge_answers=self.hedge_answers,
        )


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
