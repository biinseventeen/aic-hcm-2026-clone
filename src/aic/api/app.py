"""The FastAPI application. A transport layer only — all logic lives in :mod:`aic.service`.

    uv sync --extra backend
    uvicorn aic.api.app:app --port 8000
    # or: aic serve --port 8000

Three decisions worth stating
-----------------------------
**The engine is a lazily loaded singleton, behind a lock.** Loading takes seconds to tens of
seconds (a 173 MiB dense index plus the encoder). Letting each request load its own copy would
mean the first concurrent requests load several copies and exhaust RAM. :func:`get_engine` wraps
loading in a ``threading.Lock`` so exactly one copy is ever built.

**Computation is bounded by a semaphore.** ``Engine.solve`` is CPU-bound: a 177,321 x 512 matrix
multiplication plus greedy over 500 candidates. Running unlimited requests in parallel does not
make it faster, it makes every request uniformly slower and risks exhausting RAM. The default
limit comes from ``AIC_MAX_CONCURRENCY`` (default 2). The synchronous endpoints run in FastAPI's
threadpool, so this semaphore is the real control.

**``/health`` distinguishes "running" from "usable".** It returns 200 with
``status: "degraded"`` when the engine loaded but the encoder is a stub. A backend returning noise
while reporting ``ready`` is a worse failure than one reporting an error.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from ..log import log
from ..service import Engine
from .schemas import (
    BatchSolveRequest,
    ErrorResponse,
    HealthResponse,
    SolveRequest,
    SolveResponse,
)

__all__ = ["create_app", "get_engine", "reset_engine"]

_engine: Engine | None = None
_engine_error: str | None = None
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(int(os.environ.get("AIC_MAX_CONCURRENCY", "2")))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def get_engine(**kwargs) -> Engine:
    """The shared engine, loaded on first call. Safe to call from several threads."""
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is None:
            allow_stub = _env_flag("AIC_ALLOW_STUB")
            log.info("loading engine (allow_stub=%s)...", allow_stub)
            try:
                _engine = Engine.load(allow_stub=allow_stub, **kwargs)
                _engine_error = None
            except Exception as exc:
                _engine_error = f"{type(exc).__name__}: {exc}"
                raise
    return _engine  # type: ignore[return-value]


def reset_engine() -> None:
    """Drop the held engine. Used by tests and to reload after rebuilding the index."""
    global _engine, _engine_error
    with _lock:
        if _engine is not None:
            _engine.close()
        _engine = None
        _engine_error = None


def create_app(*, eager: bool | None = None) -> Any:
    """Create the application. ``eager=True`` loads the engine during startup.

    Eager loading is the right default for a real deployment: it turns "index missing" into a
    startup error visible immediately, instead of a 500 on the first user request. The default
    comes from ``AIC_EAGER_LOAD``.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    if eager is None:
        eager = _env_flag("AIC_EAGER_LOAD")

    @asynccontextmanager
    async def lifespan(_app):
        if eager:
            try:
                # Load in the threadpool: Engine.load is synchronous and takes tens of seconds,
                # so running it on the event loop would block the whole process.
                import anyio

                await anyio.to_thread.run_sync(get_engine)
            except Exception as exc:
                # Do not block startup: /health must stay answerable so an operator can read the
                # reason, rather than finding a process that died without a trace.
                log.error("loading the engine at startup failed: %s", exc)
        yield
        reset_engine()

    api = FastAPI(
        title="AIC 2026 — video retrieval",
        version="0.1.0",
        description=(
            "Multi-task video retrieval (Textual KIS / Q&A / TRAKE) for the AI Challenge HCM 2026 "
            "preliminary round. Each query returns up to 100 ordered answers; the order is "
            "decisive, because the score is the mean of R@k at k in {1, 5, 20, 50, 100}."
        ),
        lifespan=lifespan,
    )

    def engine_or_503() -> Engine:
        try:
            return get_engine()
        except Exception as exc:
            raise HTTPException(
                503,
                ErrorResponse(
                    "engine not usable",
                    f"{type(exc).__name__}: {exc}. Check `aic validate` and `aic build-index`; "
                    "see docs/CONSTRAINTS.md.",
                ).to_dict(),
            ) from exc

    def validated_or_422(request: SolveRequest) -> SolveRequest:
        try:
            return request.validated()
        except ValueError as exc:
            raise HTTPException(422, ErrorResponse("invalid input", str(exc)).to_dict()) from exc

    @api.get("/health", tags=["operations"])
    def health() -> dict:
        if _engine is None:
            if _engine_error:
                return HealthResponse("error", detail=_engine_error).to_dict()
            return HealthResponse("loading", detail="the engine has not been loaded").to_dict()
        status = _engine.status()
        return HealthResponse(
            "ready" if status.ready else "degraded", engine=status.to_dict()
        ).to_dict()

    @api.post("/solve", tags=["queries"])
    def solve(request: SolveRequest) -> dict:
        """Solve one query and return the ordered answer list."""
        validated = validated_or_422(request)
        engine = engine_or_503()
        with _slots:
            try:
                result = engine.solve(
                    validated.text,
                    query_id=validated.query_id,
                    task_hint=validated.task,
                    hedge_answers=validated.hedge_answers,
                )
            except ValueError as exc:
                # Syntactically valid but unsolvable (for example a TRAKE query with no moments) —
                # an input error, not a system error.
                raise HTTPException(
                    422, ErrorResponse("query could not be solved", str(exc)).to_dict()
                ) from exc
        return SolveResponse.of(result, top=validated.top).to_dict()

    @api.post("/solve/batch", tags=["queries"])
    def solve_batch(request: BatchSolveRequest) -> dict:
        """Solve a batch of queries. One failing query does **not** fail the whole batch."""
        try:
            batch = request.validated()
        except ValueError as exc:
            raise HTTPException(422, ErrorResponse("invalid input", str(exc)).to_dict()) from exc
        engine = engine_or_503()
        results, errors = [], []
        for query in batch.queries:
            with _slots:
                try:
                    result = engine.solve(
                        query.text,
                        query_id=query.query_id,
                        task_hint=query.task,
                        hedge_answers=query.hedge_answers,
                    )
                    results.append(SolveResponse.of(result, top=query.top).to_dict())
                except Exception as exc:
                    errors.append(
                        {
                            "query_id": query.query_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        return {
            "results": results,
            "errors": errors,
            "n_ok": len(results),
            "n_failed": len(errors),
        }

    @api.post("/parse", tags=["queries"])
    def parse(request: SolveRequest) -> dict:
        """Parse a query only — fast, no retrieval. Used to inspect query understanding."""
        validated = validated_or_422(request)
        query = engine_or_503().parse(validated.text, task_hint=validated.task)
        return {
            "raw": query.raw,
            "task": query.task,
            "task_confidence": round(float(query.task_confidence), 3),
            "parser": query.parser,
            "english": query.english,
            "keywords": list(query.keywords),
            "entities": list(query.entities),
            "question": query.question,
            "moments": list(query.moments),
            "exclude": list(query.exclude),
            "spatial": list(query.spatial),
            "domain_hints": list(query.domain_hints),
        }

    @api.get("/videos/{video_id}", tags=["data"])
    def video(video_id: str) -> dict:
        """Frame metadata for one video — enough for a client to convert frame_id to a time."""
        engine = engine_or_503()
        table = engine.tables.get(video_id)
        if table is None:
            raise HTTPException(404, ErrorResponse(f"no video {video_id!r}").to_dict())
        return {
            "video_id": video_id,
            "fps": table.fps,
            "n_keyframes": len(table),
            "duration_frames": table.duration_frames,
            "duration_seconds": round(table.duration_frames / (table.fps or 25.0), 2),
            "n_shots": len(engine.shots.of(video_id)),
        }

    @api.exception_handler(Exception)
    def unhandled(_request, exc: Exception) -> Any:
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse("internal error", f"{type(exc).__name__}: {exc}").to_dict(),
        )

    return api


def __getattr__(name: str):
    """``uvicorn aic.api.app:app`` — the application is created when actually touched."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
