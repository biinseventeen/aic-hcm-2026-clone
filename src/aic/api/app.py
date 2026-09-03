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
import zipfile
from pathlib import Path
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

    # Vite dev server runs on a different origin (:5173) from FastAPI (:8000).
    # Without CORS middleware the browser blocks even successful /health responses,
    # and POST /solve fails at the OPTIONS preflight with 405 Method Not Allowed.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
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

    def _nearest_keyframe(engine: Engine, video_id: str, frame_id: int) -> dict[str, Any]:
        """Map an arbitrary submission frame to the nearest supplied keyframe."""
        table = engine.tables.get(video_id)
        if table is None:
            raise HTTPException(
                404,
                ErrorResponse(f"no video {video_id!r}").to_dict(),
            )

        frames = [int(value) for value in table.frame_idx]
        if not frames:
            raise HTTPException(
                404,
                ErrorResponse(f"video {video_id!r} has no keyframes").to_dict(),
            )

        index = min(
            range(len(frames)),
            key=lambda i: abs(frames[i] - int(frame_id)),
        )
        n = int(table.n[index])
        preview_frame = frames[index]
        fps = float(table.fps or 25.0)

        pts_values = getattr(table, "pts_time", None)
        if pts_values is not None and len(pts_values) > index:
            pts_time = float(pts_values[index])
        else:
            pts_time = preview_frame / fps

        return {
            "n": n,
            "preview_frame": preview_frame,
            "requested_frame": int(frame_id),
            "fps": fps,
            "pts_time": pts_time,
        }

    def _local_data_path(engine: Engine, family: str, relative: str) -> Path | None:
        """Find an extracted organiser artefact without fabricating data."""
        source = getattr(engine.root, family, None)
        if source is not None:
            local_path = getattr(source, "local_path", None)
            if callable(local_path):
                try:
                    path = local_path(relative)
                    if path is not None and Path(path).is_file():
                        return Path(path)
                except Exception:
                    pass

        data_root = Path(engine.cfg.paths.data_root)
        for path in (
            data_root / "extracted" / relative,
            data_root / relative,
        ):
            if path.is_file():
                return path
        return None

    def _public_base_url() -> str:
        """URL the browser should use for preview/detection images."""
        return os.environ.get(
            "AIC_API_PUBLIC_URL",
            "http://127.0.0.1:8000",
        ).rstrip("/")

    def _keyframe_local_path(
        engine: Engine,
        video_id: str,
        n: int,
    ) -> Path:
        """Return a real JPEG path for one organiser keyframe.

        The corpus can be in several equivalent layouts:
        - extracted/keyframes/<video>/<nnn>.jpg
        - keyframes/<video>/<nnn>.jpg
        - Keyframes_*.zip

        ``DataRoot`` abstracts these for the retrieval code, but ``FileResponse``
        needs an actual filesystem path.  If the keyframe exists only in a zip,
        extract exactly that one JPEG into a small UI cache.
        """
        filename = f"{n:03d}.jpg"
        key = f"keyframes/{video_id}/{filename}"
        flat_key = f"{video_id}/{filename}"
        relative_flat = Path(video_id) / filename

        # 1) Ask the repository source abstraction first. Different source
        # implementations may expect either the full key or a family-relative key.
        source = engine.root.keyframes
        local_path = getattr(source, "local_path", None)
        if callable(local_path):
            for source_key in (key, flat_key):
                try:
                    path = local_path(source_key)
                    if path is not None and Path(path).is_file():
                        return Path(path)
                except Exception:
                    pass

        # 2) Check the known extracted layouts. Include both the resolved config
        # root and the repository-local junction because either may be used.
        roots: list[Path] = []
        for candidate_root in (
            Path(engine.cfg.paths.data_root),
            Path("data/batch1"),
        ):
            try:
                candidate_root = candidate_root.resolve()
            except OSError:
                pass
            if candidate_root not in roots:
                roots.append(candidate_root)

        for data_root in roots:
            candidates = (
                data_root / "extracted" / "keyframes" / relative_flat,
                data_root / "extracted" / key,
                data_root / "keyframes" / relative_flat,
                data_root / key,
            )
            for path in candidates:
                if path.is_file():
                    return path

        # 3) Archive fallback: extract only the requested keyframe, never the
        # whole 28+ GiB family.
        cache_path = (
            Path("data/processed/keyframe_cache")
            / video_id
            / filename
        )
        if cache_path.is_file():
            return cache_path

        archive_entries = (key, flat_key)
        for data_root in roots:
            for archive_path in sorted(data_root.glob("Keyframes_*.zip")):
                try:
                    with zipfile.ZipFile(archive_path) as archive:
                        names = set(archive.namelist())
                        entry = next(
                            (
                                candidate
                                for candidate in archive_entries
                                if candidate in names
                            ),
                            None,
                        )
                        if entry is None:
                            continue

                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_bytes(archive.read(entry))
                        return cache_path
                except (OSError, zipfile.BadZipFile):
                    continue

        raise FileNotFoundError(
            f"cannot locate {key}; checked DataRoot.local_path, extracted "
            "keyframe layouts, and Keyframes_*.zip"
        )

    def _preview_metadata(
        engine: Engine,
        video_id: str,
        frame_id: int,
        *,
        base_url: str,
    ) -> dict[str, Any]:
        nearest = _nearest_keyframe(engine, video_id, frame_id)
        return {
            "timestamp_seconds": round(int(frame_id) / nearest["fps"], 3),
            "timestamp": f'{int(frame_id) / nearest["fps"]:.2f}s',
            "image_url": f"{base_url}/frames/{video_id}/{int(frame_id)}",
            "preview_frame_id": nearest["preview_frame"],
            "preview_keyframe_n": nearest["n"],
        }

    def _enrich_answer_previews(
        payload: dict[str, Any],
        engine: Engine,
        *,
        base_url: str,
    ) -> dict[str, Any]:
        """Add UI-only preview metadata without changing ranking/submission fields."""
        for answer in payload.get("answers", []):
            video_id = answer.get("video_id")
            frame_id = answer.get("frame_id")
            if not video_id or frame_id is None:
                continue
            try:
                answer.update(
                    _preview_metadata(
                        engine,
                        str(video_id),
                        int(frame_id),
                        base_url=base_url,
                    )
                )
            except (TypeError, ValueError, HTTPException):
                continue
        return payload

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

        payload = SolveResponse.of(result, top=validated.top).to_dict()
        return _enrich_answer_previews(
            payload,
            engine,
            base_url=_public_base_url(),
        )

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

    @api.get("/frames/{video_id}/{frame_id}", tags=["data"])
    def frame_preview(video_id: str, frame_id: int):
        """JPEG nearest to a requested source-video frame for human inspection."""
        engine = engine_or_503()
        nearest = _nearest_keyframe(engine, video_id, frame_id)
        relative = f"keyframes/{video_id}/{nearest['n']:03d}.jpg"
        try:
            path = _keyframe_local_path(
                engine,
                video_id,
                int(nearest["n"]),
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                404,
                ErrorResponse(
                    "keyframe preview unavailable",
                    str(exc),
                ).to_dict(),
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
        engine = engine_or_503()
        nearest = _nearest_keyframe(engine, video_id, frame_id)
        relative = f"objects/{video_id}/{nearest['n']:03d}.json"
        path = _local_data_path(engine, "objects", relative)
        if path is None:
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        entities = list(payload.get("detection_class_entities") or [])
        scores = list(payload.get("detection_scores") or [])
        boxes = list(payload.get("detection_boxes") or [])

        image_url = (
            f"{_public_base_url()}/frames/{video_id}/{frame_id}"
        )

        rows: list[dict[str, Any]] = []
        for index, (entity, score_raw) in enumerate(zip(entities, scores)):
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
