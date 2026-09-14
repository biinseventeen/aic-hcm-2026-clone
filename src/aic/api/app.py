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

import json
import os
import threading
from pathlib import Path
from typing import Any

from ..log import log
from ..service import Engine
from .preview import (
    PreviewLookupError,
    enrich_for_ui,
    enrich_review,
    keyframe_local_path,
    local_data_path,
    nearest_keyframe,
    public_base_url,
    require_video_id,
)
from .schemas import (
    BatchSolveRequest,
    ErrorResponse,
    HealthResponse,
    PackageRequest,
    ReviewRequest,
    SolveRequest,
    SolveResponse,
    SubmitRequest,
    normalise_query_id,
    normalise_task,
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

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

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

    # Vite may hop off :5173 when that port is taken (this machine used :5174).
    # A fixed origin list would make the browser treat /health as a network failure
    # (BACKEND_DISCONNECTED) even though the engine is up.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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

    def preview_http(exc: PreviewLookupError) -> HTTPException:
        return HTTPException(exc.status, ErrorResponse(exc.message, exc.detail).to_dict())

    def solve_engine(engine: Engine, request: SolveRequest):
        return engine.solve(
            request.text,
            query_id=request.query_id,
            task_hint=request.task,  # type: ignore[arg-type]
            hedge_answers=request.hedge_answers,
            answers=request.engine_answers(),
            pins=request.engine_pins(),
        )

    def solve_payload(engine: Engine, result: Any, *, top: int | None) -> dict[str, Any]:
        payload = SolveResponse.of(result, top=top).to_dict()
        return enrich_for_ui(payload, engine, base_url=public_base_url())

    def issues_json(issues: list) -> list[dict[str, Any]]:
        return [
            {
                "severity": issue.severity,
                "query_id": issue.query_id,
                "message": issue.message,
                "row": issue.row,
            }
            for issue in issues
        ]

    def submission_matches(directory: Path, query_id: str, task: str | None) -> list[Path]:
        from ..submit.writer import SubmissionNaming

        if task:
            path = directory / SubmissionNaming().file_for(query_id, task)  # type: ignore[arg-type]
            return [path] if path.is_file() else []
        pattern = f"query-{query_id}-*.csv"
        return sorted({*directory.glob(pattern), *directory.glob(f"*/{pattern}")})

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
                result = solve_engine(engine, validated)
            except ValueError as exc:
                # Syntactically valid but unsolvable (for example a TRAKE query with no moments) —
                # an input error, not a system error.
                raise HTTPException(
                    422, ErrorResponse("query could not be solved", str(exc)).to_dict()
                ) from exc
        return solve_payload(engine, result, top=validated.top)

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
                    result = solve_engine(engine, query)
                    results.append(solve_payload(engine, result, top=query.top))
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
        query = engine_or_503().parse(validated.text, task_hint=validated.task)  # type: ignore[arg-type]
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

    @api.post("/review", tags=["queries"])
    def review(request: ReviewRequest) -> dict:
        """Human-review shortlist. Does not run P10/P13 or write a submission."""
        try:
            validated = request.validated()
        except ValueError as exc:
            raise HTTPException(422, ErrorResponse("invalid input", str(exc)).to_dict()) from exc
        engine = engine_or_503()
        with _slots:
            result = engine.review(validated.text, task_hint=validated.task)  # type: ignore[arg-type]
        payload = result.to_dict()
        payload["query_id"] = validated.query_id
        return enrich_review(payload, engine, base_url=public_base_url())

    @api.post("/submit", tags=["submission"])
    def submit(request: SubmitRequest) -> dict:
        """Solve one query and write ``query-<id>-<task>.csv``. Errors do not write."""
        try:
            validated = request.validated()
        except ValueError as exc:
            raise HTTPException(422, ErrorResponse("invalid input", str(exc)).to_dict()) from exc
        engine = engine_or_503()
        with _slots:
            try:
                result = solve_engine(engine, validated)
            except ValueError as exc:
                raise HTTPException(
                    422, ErrorResponse("query could not be solved", str(exc)).to_dict()
                ) from exc
        try:
            path, issues = engine.write(result, strict=validated.strict)
        except ValueError as exc:
            raise HTTPException(
                422, ErrorResponse("submission not written", str(exc)).to_dict()
            ) from exc
        payload = solve_payload(engine, result, top=validated.top)
        payload["path"] = str(path)
        payload["filename"] = path.name
        payload["issues"] = issues_json(issues)
        return payload

    @api.get("/submit/{query_id}", tags=["submission"])
    def download_submission(query_id: str, task: str | None = Query(default=None)):
        """Download a CSV already written by ``POST /submit`` or ``aic run``."""
        try:
            query_id = normalise_query_id(query_id)
            task = normalise_task(task)
        except ValueError as exc:
            raise HTTPException(422, ErrorResponse("invalid input", str(exc)).to_dict()) from exc
        engine = engine_or_503()
        directory = Path(engine.cfg.paths.submission_dir)
        matches = submission_matches(directory, query_id, task)
        if not matches:
            raise HTTPException(
                404, ErrorResponse(f"no submission file for query {query_id!r}").to_dict()
            )
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise HTTPException(
                409,
                ErrorResponse(
                    "ambiguous query_id",
                    f"several files match {query_id!r}: {names}. Pass ?task=kis|qa|trake.",
                ).to_dict(),
            )
        path = matches[0]
        return FileResponse(path, media_type="text/csv", filename=path.name)

    @api.post("/submit/package", tags=["submission"])
    def package(request: PackageRequest) -> dict:
        """Zip written CSVs into an archive that contains a ``submission/`` directory."""
        try:
            validated = request.validated()
        except ValueError as exc:
            raise HTTPException(422, ErrorResponse("invalid input", str(exc)).to_dict()) from exc
        engine = engine_or_503()
        from ..submit.writer import package_submission

        directory = Path(engine.cfg.paths.submission_dir)
        if validated.set_name:
            directory = directory / validated.set_name
        files: list[Path] = []
        missing: list[str] = []
        for query_id in validated.query_ids:
            matches = submission_matches(directory, query_id, None)
            if not matches:
                missing.append(query_id)
            else:
                files.extend(matches)
        if missing:
            raise HTTPException(
                404,
                ErrorResponse(
                    "submission files missing",
                    f"no CSV for query_id(s): {', '.join(missing)}",
                ).to_dict(),
            )
        archive = package_submission(files, directory / validated.zip_name)
        return {
            "path": str(archive),
            "filename": archive.name,
            "n_files": len(files),
        }

    @api.get("/videos/{video_id}", tags=["data"])
    def video(video_id: str) -> dict:
        """Frame metadata for one video — enough for a client to convert frame_id to a time."""
        try:
            require_video_id(video_id)
        except PreviewLookupError as exc:
            raise preview_http(exc) from exc
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

    @api.get("/frames/{video_id}/{frame_id}", tags=["data"])
    def frame_preview(video_id: str, frame_id: int):
        """JPEG nearest to a requested source-video frame for human inspection."""
        try:
            require_video_id(video_id)
        except PreviewLookupError as exc:
            raise preview_http(exc) from exc
        engine = engine_or_503()
        try:
            nearest = nearest_keyframe(engine, video_id, frame_id)
            path = keyframe_local_path(engine, video_id, int(nearest["n"]))
        except PreviewLookupError as exc:
            raise preview_http(exc) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                404,
                ErrorResponse("keyframe preview unavailable", str(exc)).to_dict(),
            ) from exc

        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={
                "X-AIC-Requested-Frame": str(frame_id),
                "X-AIC-Preview-Frame": str(nearest["preview_frame"]),
                "X-AIC-Keyframe-N": str(nearest["n"]),
            },
        )

    @api.get("/detections", tags=["data"])
    def detections(
        video_id: str,
        frame_id: int,
        min_score: float = Query(0.4, ge=0.0, le=1.0),
        limit: int = Query(8, ge=1, le=50),
    ) -> list[dict[str, Any]]:
        """Real organiser-supplied Open Images detections near one frame."""
        try:
            require_video_id(video_id)
        except PreviewLookupError as exc:
            raise preview_http(exc) from exc
        engine = engine_or_503()
        try:
            nearest = nearest_keyframe(engine, video_id, frame_id)
        except PreviewLookupError as exc:
            raise preview_http(exc) from exc
        relative = f"objects/{video_id}/{nearest['n']:03d}.json"
        path = local_data_path(engine, "objects", relative)
        if path is None:
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        entities = list(payload.get("detection_class_entities") or [])
        scores = list(payload.get("detection_scores") or [])
        boxes = list(payload.get("detection_boxes") or [])
        image_url = f"{public_base_url()}/frames/{video_id}/{frame_id}"

        rows: list[dict[str, Any]] = []
        for index, (entity, score_raw) in enumerate(zip(entities, scores, strict=False)):
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            if score < min_score:
                continue

            bbox = None
            if index < len(boxes):
                try:
                    values = [float(value) for value in boxes[index]]
                    if len(values) == 4:
                        bbox = values
                except (TypeError, ValueError):
                    bbox = None

            label = str(entity).strip() or "UNKNOWN"
            rows.append(
                {
                    "id": f"{video_id}-{nearest['n']}-{index}",
                    "videoSource": video_id,
                    "frameId": str(nearest["preview_frame"]),
                    "requestedFrameId": str(frame_id),
                    "timestamp": f'{nearest["pts_time"]:.2f}s',
                    "confidence": f"{score * 100:.1f}%",
                    "isAlert": score >= 0.8,
                    "imageSrc": image_url,
                    "tags": [
                        {
                            "label": label.upper().replace(" ", "_"),
                            "variant": "solid",
                        }
                    ],
                    "isSelected": len(rows) == 0,
                    "bbox": bbox,
                }
            )
            if len(rows) >= limit:
                break

        return rows

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
