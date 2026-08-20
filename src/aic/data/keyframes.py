"""Reading ``map-keyframes/<video>.csv`` — the keyframe to frame-index mapping table.

This is the *source of truth* for the organiser's frame index convention, and the input to
the P1 validation step. Four columns:

    n         keyframe ordinal, 1-based. Matches the image filename in Keyframes_*.zip and
              row (n-1) in clip-features-32/<video>.npy.
    pts_time  presentation timestamp in seconds, rounded to 4 decimal places.
    fps       the nominal frame rate of the video.
    frame_idx the frame index used for scoring = floor(pts_time * fps).

Warning about injectivity
-------------------------
``frame_idx`` is **not** injective: 192 of the 873 videos in batch 1 have two keyframes
sharing a ``frame_idx`` (typically the first keyframe at pts 0.0 and the next one at a pts
smaller than 1/fps). Any reverse ``frame_idx -> n`` mapping must handle collisions, and
``n`` is the key used to look up CLIP features.
"""

from __future__ import annotations

import csv
import io
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from ..core.frameidx import DEFAULT_CONVENTION, time_to_frame
from .layout import DataRoot

__all__ = [
    "KeyframeTable",
    "load_all_keyframe_tables",
    "load_keyframe_table",
    "parse_keyframe_csv",
    "recompute_frame_idx",
]


@dataclass(slots=True)
class KeyframeTable:
    """The keyframe table of one video, sorted by increasing ``n``."""

    video_id: str
    #: 1-based, one entry per keyframe.
    n: list[int] = field(default_factory=list)
    pts_time: list[float] = field(default_factory=list)
    frame_idx: list[int] = field(default_factory=list)
    fps: float = 0.0

    def __len__(self) -> int:
        return len(self.n)

    @property
    def duration_frames(self) -> int:
        """The last *known* frame.

        Not the true length of the video — the last keyframe usually sits before the last
        frame. Used as a safe upper bound when clamping indices.
        """
        return self.frame_idx[-1] if self.frame_idx else 0

    def rows(self) -> Iterator[tuple[int, float, float, int]]:
        """Yield ``(n, pts_time, fps, frame_idx)`` — the shape that P1 consumes."""
        for i in range(len(self.n)):
            yield (self.n[i], self.pts_time[i], self.fps, self.frame_idx[i])

    # -- lookups ---------------------------------------------------------

    def feature_row_of_n(self, n: int) -> int:
        """``n`` (1-based) -> row index in ``clip-features-32/<video>.npy`` (0-based)."""
        if not 1 <= n <= len(self.n):
            raise IndexError(f"{self.video_id}: n={n} outside [1, {len(self.n)}]")
        return n - 1

    def nearest_keyframe(self, frame: int) -> int:
        """The keyframe ``n`` whose ``frame_idx`` is closest to ``frame``."""
        if not self.frame_idx:
            raise ValueError(f"{self.video_id}: empty keyframe table")
        position = bisect_left(self.frame_idx, frame)
        if position == 0:
            return self.n[0]
        if position >= len(self.frame_idx):
            return self.n[-1]
        before, after = self.frame_idx[position - 1], self.frame_idx[position]
        return self.n[position - 1] if (frame - before) <= (after - frame) else self.n[position]

    def keyframes_in_range(self, start: int, end: int) -> list[int]:
        """Every ``n`` whose ``frame_idx`` lies in [start, end], both ends inclusive."""
        lo = bisect_left(self.frame_idx, start)
        hi = bisect_right(self.frame_idx, end)
        return self.n[lo:hi]

    def frame_of_n(self, n: int) -> int:
        return self.frame_idx[self.feature_row_of_n(n)]

    def time_of_n(self, n: int) -> float:
        return self.pts_time[self.feature_row_of_n(n)]

    # -- diagnostics -----------------------------------------------------

    def gaps(self) -> list[int]:
        """Distances in frames between consecutive keyframes.

        The median over batch 1 is 55 frames. With TRAKE answer windows under 10 frames, the
        probability that an available keyframe lands inside the window is only about
        10/69 ~ 14 %. That is the number which makes full-fps decoding *mandatory* for TRAKE.
        """
        return [b - a for a, b in zip(self.frame_idx, self.frame_idx[1:], strict=False)]

    def duplicate_frame_indices(self) -> list[int]:
        seen: set[int] = set()
        duplicates: list[int] = []
        for frame in self.frame_idx:
            if frame in seen:
                duplicates.append(frame)
            seen.add(frame)
        return duplicates


def parse_keyframe_csv(video_id: str, text: str) -> KeyframeTable:
    """Parse the contents of one ``map-keyframes/<video>.csv``.

    Rows without a ``frame_idx`` are skipped: some organiser files end with a blank row.
    """
    table = KeyframeTable(video_id=video_id)
    reader = csv.DictReader(io.StringIO(text))
    required = {"n", "pts_time", "fps", "frame_idx"}
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise ValueError(
            f"{video_id}: map-keyframes must have columns {sorted(required)}, "
            f"got {reader.fieldnames}"
        )
    fps_seen: set[float] = set()
    for row in reader:
        if not row.get("frame_idx"):
            continue
        table.n.append(int(row["n"]))
        table.pts_time.append(float(row["pts_time"]))
        table.frame_idx.append(int(row["frame_idx"]))
        fps_seen.add(float(row["fps"]))
    if len(fps_seen) > 1:
        raise ValueError(
            f"{video_id}: multiple fps values within one video {sorted(fps_seen)} — "
            "possible VFR, must be resolved before use (P1)"
        )
    table.fps = fps_seen.pop() if fps_seen else 0.0
    order = sorted(range(len(table.n)), key=lambda i: table.n[i])
    if order != list(range(len(table.n))):
        table.n = [table.n[i] for i in order]
        table.pts_time = [table.pts_time[i] for i in order]
        table.frame_idx = [table.frame_idx[i] for i in order]
    return table


def load_keyframe_table(root: DataRoot, video_id: str) -> KeyframeTable:
    return parse_keyframe_csv(video_id, root.map_keyframes.read_text(f"{video_id}.csv"))


def load_all_keyframe_tables(
    root: DataRoot, *, video_ids: Sequence[str] | None = None
) -> dict[str, KeyframeTable]:
    """Load keyframe tables for every video. Cheap: all of batch 1 is ~1.6 MB compressed."""
    ids = list(video_ids) if video_ids is not None else root.video_ids()
    return {video_id: load_keyframe_table(root, video_id) for video_id in ids}


def recompute_frame_idx(table: KeyframeTable) -> list[int]:
    """Recompute ``frame_idx`` from ``(pts_time, fps)`` using the verified convention."""
    return [time_to_frame(pts, table.fps, DEFAULT_CONVENTION) for pts in table.pts_time]
