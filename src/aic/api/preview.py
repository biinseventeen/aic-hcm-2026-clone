"""Keyframe lookup and UI-only enrichment for the HTTP layer.

These helpers must not change ranking or submission fields. They map a requested source-video
frame onto the nearest organiser keyframe, resolve a real JPEG path, and attach preview URLs.

``video_id`` is checked against :data:`aic.data.layout.VIDEO_ID_RE` **before** any path is
joined, so a client cannot walk out of the corpus directory.
"""

from __future__ import annotations

import os
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..config import find_project_root
from ..data.layout import VIDEO_ID_RE

__all__ = [
    "PreviewLookupError",
    "enrich_for_ui",
    "enrich_review",
    "keyframe_local_path",
    "local_data_path",
    "nearest_keyframe",
    "preview_metadata",
    "public_base_url",
    "require_video_id",
]

_EVIDENCE_CAP = 8


class PreviewLookupError(Exception):
    """A preview cannot be resolved. ``status`` is the HTTP status a transport should return."""

    def __init__(self, status: int, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail


def require_video_id(video_id: str, *, known: set[str] | None = None) -> str:
    """Return ``video_id`` if it is a well-formed corpus id, else raise :class:`PreviewLookupError`.

    The 404 is deliberate: a traversal attempt must not look different from a missing video.
    """
    if not isinstance(video_id, str) or not VIDEO_ID_RE.match(video_id):
        raise PreviewLookupError(404, f"no video {video_id!r}")
    if known is not None and video_id not in known:
        raise PreviewLookupError(404, f"no video {video_id!r}")
    return video_id


def public_base_url() -> str:
    return os.environ.get("AIC_API_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")


def nearest_keyframe(engine: Any, video_id: str, frame_id: int) -> dict[str, Any]:
    """Map an arbitrary submission frame to the nearest supplied keyframe."""
    require_video_id(video_id, known=set(engine.tables))
    table = engine.tables.get(video_id)
    if table is None or not table.frame_idx:
        raise PreviewLookupError(404, f"video {video_id!r} has no keyframes")

    n = int(table.nearest_keyframe(int(frame_id)))
    index = int(table.feature_row_of_n(n))
    preview_frame = int(table.frame_idx[index])
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


def local_data_path(engine: Any, family: str, relative: str) -> Path | None:
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


def keyframe_local_path(engine: Any, video_id: str, n: int) -> Path:
    """Return a real JPEG path for one organiser keyframe.

    The corpus can be in several equivalent layouts:
    - extracted/keyframes/<video>/<nnn>.jpg
    - keyframes/<video>/<nnn>.jpg
    - Keyframes_*.zip

    ``DataRoot`` abstracts these for the retrieval code, but a file response needs an actual
    filesystem path. If the keyframe exists only in a zip, extract exactly that one JPEG into
    a small UI cache under the project root.
    """
    require_video_id(video_id, known=set(engine.tables))
    n = int(n)
    if n < 1:
        raise PreviewLookupError(404, f"no keyframe n={n} for {video_id!r}")

    filename = f"{n:03d}.jpg"
    key = f"keyframes/{video_id}/{filename}"
    flat_key = f"{video_id}/{filename}"
    relative_flat = Path(video_id) / filename

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

    roots: list[Path] = []
    for candidate_root in (
        Path(engine.cfg.paths.data_root),
        find_project_root() / "data" / "batch1",
    ):
        with suppress(OSError):
            candidate_root = candidate_root.resolve()
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

    cache_path = find_project_root() / "data" / "processed" / "keyframe_cache" / video_id / filename
    if cache_path.is_file():
        return cache_path

    archive_entries = (key, flat_key)
    for data_root in roots:
        for archive_path in sorted(data_root.glob("Keyframes_*.zip")):
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    names = set(archive.namelist())
                    entry = next(
                        (candidate for candidate in archive_entries if candidate in names),
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


def preview_metadata(
    engine: Any,
    video_id: str,
    frame_id: int,
    *,
    base_url: str,
) -> dict[str, Any]:
    nearest = nearest_keyframe(engine, video_id, frame_id)
    fps = nearest["fps"] or 25.0
    seconds = int(frame_id) / fps
    return {
        "timestamp_seconds": round(seconds, 3),
        "timestamp": f"{seconds:.2f}s",
        "image_url": f"{base_url}/frames/{video_id}/{int(frame_id)}",
        "preview_frame_id": nearest["preview_frame"],
        "preview_keyframe_n": nearest["n"],
    }


def _attach_preview(engine: Any, video_id: Any, frame_id: Any, *, base_url: str) -> dict[str, Any]:
    return preview_metadata(engine, str(video_id), int(frame_id), base_url=base_url)


def _enrich_one_answer(answer: dict[str, Any], engine: Any, *, base_url: str) -> None:
    video_id = answer.get("video_id")
    frame_ids = list(answer.get("frame_ids") or [])
    frame_id = answer.get("frame_id")
    if frame_id is None and frame_ids:
        frame_id = frame_ids[0]
        answer["frame_id"] = frame_id
    if not video_id or frame_id is None:
        return
    with suppress(TypeError, ValueError, PreviewLookupError):
        answer.update(_attach_preview(engine, video_id, frame_id, base_url=base_url))

    if not frame_ids:
        return
    # Moments live on the payload, not the row; the caller copies them in when needed.
    supplied_moments = answer.get("_moments") or []
    milestones: list[dict[str, Any]] = []
    for index, moment_frame in enumerate(frame_ids):
        step_name = ""
        if index < len(supplied_moments):
            step_name = str(supplied_moments[index])
        row: dict[str, Any] = {
            "stepId": f"{index + 1:02d}",
            "stepName": step_name,
            "frameId": moment_frame,
            "timestamp": "",
            "confidence": "",
        }
        with suppress(TypeError, ValueError, PreviewLookupError):
            meta = _attach_preview(engine, video_id, moment_frame, base_url=base_url)
            row["timestamp"] = meta["timestamp"]
            row["image_url"] = meta["image_url"]
        milestones.append(row)
    answer["milestones"] = milestones
    answer.pop("_moments", None)


def enrich_for_ui(payload: dict[str, Any], engine: Any, *, base_url: str) -> dict[str, Any]:
    """Add UI-only preview metadata without changing ranking/submission fields."""
    moments = list((payload.get("query") or {}).get("moments") or [])
    answers = payload.get("answers") or []
    for answer in answers:
        if moments and answer.get("frame_ids"):
            answer["_moments"] = moments
        _enrich_one_answer(answer, engine, base_url=base_url)

    if payload.get("task") == "qa" and answers:
        first = answers[0]
        evidence = []
        for row in answers[:_EVIDENCE_CAP]:
            frame_id = row.get("frame_id")
            evidence.append(
                {
                    "id": f"{row.get('video_id')}/{frame_id}",
                    "frame_id": frame_id,
                    "time": row.get("timestamp") or "",
                    "confidence": (
                        f"{row['gain']:.4f}" if isinstance(row.get("gain"), (int, float)) else ""
                    ),
                    "image_url": row.get("image_url"),
                }
            )
        payload["qna_answer"] = {
            "answer_text": first.get("answer") or "",
            "confidence": evidence[0]["confidence"] if evidence else "",
            "source_segment": first.get("video_id") or "",
            "interval": first.get("timestamp") or "",
            "evidence_frames": evidence,
        }
    return payload


def enrich_review(payload: dict[str, Any], engine: Any, *, base_url: str) -> dict[str, Any]:
    """Attach ``image_url`` to every frame in a review shortlist."""
    review = payload.get("review") or {}

    def stamp(video_id: Any, frame_obj: dict[str, Any]) -> None:
        frame = frame_obj.get("frame")
        if video_id is None or frame is None:
            return
        try:
            meta = _attach_preview(engine, video_id, frame, base_url=base_url)
        except (TypeError, ValueError, PreviewLookupError):
            return
        frame_obj["image_url"] = meta["image_url"]
        frame_obj["timestamp"] = meta["timestamp"]

    for item in review.get("scan_order") or []:
        frame_obj = item.get("frame") or {}
        stamp(item.get("video_id"), frame_obj)
        if frame_obj.get("image_url"):
            item["image_url"] = frame_obj["image_url"]

    for video in review.get("videos") or []:
        for frame_obj in video.get("frames") or []:
            stamp(video.get("video_id"), frame_obj)
    return payload
