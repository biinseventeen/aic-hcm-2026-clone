"""P2 — temporal segmentation: dividing videos into index units.

Design decision: **derive shot boundaries from the organiser's own keyframes** rather than
running TransNetV2.

Basis: the rules state that keyframes are "extracted from the video", and DESIGN.md R1 records
that they are sampled at shot boundaries. The batch 1 measurements match that hypothesis — the
gap between consecutive keyframes has a median of 55 frames (2.2 s at 25 fps) and a strongly
skewed distribution (p99 = 183 f, max = 211 f). The skew rules out uniform sampling, which
would make every gap identical; it is consistent with the shot-length distribution of edited
content.

What this buys: P2 costs **zero** GPU hours, and every shot maps one-to-one onto an existing
CLIP vector, so the dense retrieval layer and the segmentation layer share one index space.

What it costs, and the compensation
-----------------------------------
1. No control over the detection threshold. Cross-dissolve transitions may already have
   produced spurious boundaries in the source data, and that cannot be undone here.
2. Long shots contain several topics — a presenter at a desk in a news bulletin, or one
   continuous cooking step in L26. Compensated by :func:`split_long_shots`, which force-splits
   any shot longer than ``max_seconds``.
3. Events spanning a shot boundary. Compensated by :func:`sliding_windows`, a second index
   over sliding windows that runs **in parallel** as its own retrieval channel rather than as
   a series stage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..core.frameidx import seconds_to_frames
from ..data.keyframes import KeyframeTable

__all__ = [
    "Shot",
    "ShotTable",
    "build_shot_table",
    "shots_from_keyframes",
    "sliding_windows",
    "split_long_shots",
]


@dataclass(slots=True, frozen=True)
class Shot:
    """One index unit: a frame range closed at both ends, plus the keyframes inside it."""

    video_id: str
    shot_id: int
    start: int
    end: int
    #: the ``n`` of each keyframe inside this shot (1-based; may be empty after a split).
    keyframes: tuple[int, ...] = ()
    #: "shot" (derived from keyframes), "split" (cut from a long shot), "window" (sliding).
    kind: str = "shot"

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def duration(self, fps: float) -> float:
        return self.length / fps

    @property
    def anchor_keyframe(self) -> int | None:
        """The representative keyframe — the middle one when the shot has several."""
        if not self.keyframes:
            return None
        return self.keyframes[len(self.keyframes) // 2]

    def contains(self, frame: int) -> bool:
        return self.start <= frame <= self.end


def shots_from_keyframes(table: KeyframeTable, *, last_frame: int | None = None) -> list[Shot]:
    """Shot i spans [frame_idx[i], frame_idx[i+1] - 1].

    The final shot extends to ``last_frame`` when known; otherwise it is given the same length
    as its predecessor, which is the safe estimate — running long is better than truncating
    away the answer.

    Keyframes sharing a ``frame_idx`` (192 of 873 videos show this) would produce zero-length
    shots; they are merged into the following shot rather than dropped, because their CLIP
    vectors are still valid retrieval signal.
    """
    frame_indices = table.frame_idx
    if not frame_indices:
        return []

    # Merge keyframes that share a frame_idx into a single boundary.
    boundaries: list[int] = []
    keyframe_groups: list[list[int]] = []
    for frame, n in zip(frame_indices, table.n, strict=True):
        if boundaries and frame == boundaries[-1]:
            keyframe_groups[-1].append(n)
        else:
            boundaries.append(frame)
            keyframe_groups.append([n])

    shots: list[Shot] = []
    for i, (start, keyframes) in enumerate(zip(boundaries, keyframe_groups, strict=True)):
        if i + 1 < len(boundaries):
            end = boundaries[i + 1] - 1
        elif last_frame is not None and last_frame >= start:
            end = last_frame
        elif len(boundaries) >= 2:
            end = start + (boundaries[-1] - boundaries[-2]) - 1
        else:
            end = start
        shots.append(
            Shot(
                video_id=table.video_id,
                shot_id=i,
                start=start,
                end=max(end, start),
                keyframes=tuple(keyframes),
            )
        )
    return shots


def split_long_shots(
    shots: list[Shot], fps: float, *, max_seconds: float = 20.0, part_seconds: float = 10.0
) -> list[Shot]:
    """Force-split shots longer than ``max_seconds`` into ``part_seconds`` pieces.

    A 60-second interview shot summarised as a single description loses all the detail inside
    it; splitting preserves granularity for the localisation layer.

    ``shot_id`` is reassigned contiguously after splitting.
    """
    max_frames = seconds_to_frames(max_seconds, fps)
    part_frames = seconds_to_frames(part_seconds, fps)
    pieces: list[Shot] = []
    for shot in shots:
        if shot.length <= max_frames:
            pieces.append(shot)
            continue
        n_parts = math.ceil(shot.length / part_frames)
        for part in range(n_parts):
            start = shot.start + part * part_frames
            end = min(shot.end, start + part_frames - 1)
            if start > end:
                break
            pieces.append(
                Shot(
                    video_id=shot.video_id,
                    shot_id=0,
                    start=start,
                    end=end,
                    # The original keyframes belong only to the piece containing them.
                    keyframes=tuple(shot.keyframes) if part == 0 else (),
                    kind="split",
                )
            )
    return [
        Shot(piece.video_id, i, piece.start, piece.end, piece.keyframes, piece.kind)
        for i, piece in enumerate(pieces)
    ]


def sliding_windows(
    table: KeyframeTable,
    *,
    window_seconds: float = 30.0,
    step_seconds: float = 15.0,
    last_frame: int | None = None,
) -> list[Shot]:
    """A second index: 30-second windows stepped every 15 seconds.

    This exists specifically for queries describing an event that *spans several cuts* — the
    case shot-based segmentation can never catch, because no single unit contains the whole
    event. It is a parallel channel, not a series stage.
    """
    fps = table.fps or 25.0
    window_frames = seconds_to_frames(window_seconds, fps)
    step_frames = seconds_to_frames(step_seconds, fps)
    end_frame = last_frame if last_frame is not None else table.duration_frames
    windows: list[Shot] = []
    start = 0
    index = 0
    while start <= end_frame:
        end = min(end_frame, start + window_frames - 1)
        windows.append(
            Shot(
                video_id=table.video_id,
                shot_id=index,
                start=start,
                end=end,
                keyframes=tuple(table.keyframes_in_range(start, end)),
                kind="window",
            )
        )
        if end >= end_frame:
            break
        start += step_frames
        index += 1
    return windows


@dataclass
class ShotTable:
    """The shot table for the whole corpus, plus reverse frame -> shot lookup."""

    shots: dict[str, list[Shot]] = field(default_factory=dict)
    kind: str = "shot"

    def __len__(self) -> int:
        return sum(len(shots) for shots in self.shots.values())

    def of(self, video_id: str) -> list[Shot]:
        return self.shots.get(video_id, [])

    def find(self, video_id: str, frame: int) -> Shot | None:
        """The shot containing ``frame``. Binary search over ``start``."""
        shots = self.shots.get(video_id)
        if not shots:
            return None
        lo, hi = 0, len(shots) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if shots[mid].end < frame:
                lo = mid + 1
            elif shots[mid].start > frame:
                hi = mid - 1
            else:
                return shots[mid]
        return None

    def stats(self, tables: dict[str, KeyframeTable]) -> dict[str, float]:
        durations: list[float] = []
        for video_id, shots in self.shots.items():
            fps = tables[video_id].fps or 25.0
            durations.extend(shot.length / fps for shot in shots)
        if not durations:
            return {}
        durations.sort()
        n = len(durations)
        return {
            "n_shots": float(n),
            "mean_s": sum(durations) / n,
            "median_s": durations[n // 2],
            "p90_s": durations[int(0.90 * n)],
            "p99_s": durations[int(0.99 * n)],
            "max_s": durations[-1],
            "under_2s_pct": 100.0 * sum(1 for d in durations if d < 2.0) / n,
            "over_20s_pct": 100.0 * sum(1 for d in durations if d > 20.0) / n,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "shots": {
                video_id: [[s.shot_id, s.start, s.end, list(s.keyframes), s.kind] for s in shots]
                for video_id, shots in self.shots.items()
            },
        }
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> ShotTable:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            kind=payload.get("kind", "shot"),
            shots={
                video_id: [
                    Shot(video_id, shot_id, start, end, tuple(keyframes), kind)
                    for shot_id, start, end, keyframes, kind in shots
                ]
                for video_id, shots in payload["shots"].items()
            },
        )


def build_shot_table(
    tables: dict[str, KeyframeTable],
    *,
    fps_default: float = 25.0,
    max_seconds: float = 20.0,
    part_seconds: float = 10.0,
    last_frames: dict[str, int] | None = None,
) -> ShotTable:
    """Build the corpus-wide shot table from the keyframe tables."""
    per_video: dict[str, list[Shot]] = {}
    for video_id, table in tables.items():
        last_frame = (last_frames or {}).get(video_id)
        shots = shots_from_keyframes(table, last_frame=last_frame)
        per_video[video_id] = split_long_shots(
            shots,
            table.fps or fps_default,
            max_seconds=max_seconds,
            part_seconds=part_seconds,
        )
    return ShotTable(shots=per_video, kind="shot+split")
