"""P1 — frame index convention. **Blocking item.**

An error here causes *silent* score loss: the system runs normally, produces valid indices,
and scores zero. Nothing in the development loop points at the problem unless there is an
explicit cross-check step.

The convention, established empirically
---------------------------------------
Checked against **all 177,321 keyframes of 873 videos** in batch 1
(``map-keyframes-aic25-b1``), comparing the organiser-supplied ``frame_idx`` with three
candidate formulas applied to the ``(pts_time, fps)`` they also supply:

    floor(pts_time * fps)   ->  177321 / 177321   =  100.00 %   <-- CORRECT
    round(pts_time * fps)   ->  154399 / 177321   =   87.07 %
    ceil (pts_time * fps)   ->  138655 / 177321   =   78.19 %

The convention is **floor**, not ``round``. Using ``round`` is off by one frame on 12.93 %
of keyframes. With TRAKE answer windows typically under 10 frames, a systematic one-frame
offset is real score loss, and it never surfaces as a runtime error.

See ``docs/DATA_AUDIT.md`` for the full comparison and the anomalies found.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "COMMON_FPS",
    "DEFAULT_CONVENTION",
    "Convention",
    "FrameIndexReport",
    "check_convention",
    "frame_to_time",
    "grid_alignment",
    "seconds_to_frames",
    "time_to_frame",
]

Convention = Literal["floor", "round", "ceil"]

#: Verified on 100 % of batch 1 keyframes. Any change to this value must come with a
#: fresh run of :func:`check_convention` over the whole corpus.
DEFAULT_CONVENTION: Convention = "floor"

#: fps values that occur in batch 1 (see docs/DATA_AUDIT.md).
COMMON_FPS = (25.0, 29.97, 30.0)


def time_to_frame(pts_time: float, fps: float, convention: Convention = DEFAULT_CONVENTION) -> int:
    """Presentation timestamp (seconds) -> frame index, per the organiser's convention.

    >>> time_to_frame(11.7333, 30.0)          # round() would give 352 -- WRONG
    351
    >>> time_to_frame(3.0, 30.0), time_to_frame(71.0333, 30.0)
    (90, 2130)
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if pts_time < 0:
        raise ValueError(f"pts_time must be >= 0, got {pts_time}")
    scaled = pts_time * fps
    if convention == "floor":
        return math.floor(scaled)
    if convention == "round":
        return round(scaled)
    if convention == "ceil":
        return math.ceil(scaled)
    raise ValueError(f"invalid convention: {convention!r}")


def frame_to_time(frame_idx: int, fps: float, *, at_center: bool = False) -> float:
    """Frame index -> time (seconds). The *approximate* inverse of :func:`time_to_frame`.

    Since ``floor`` is not injective, the inverse returns the *start* mark of the frame.

    The start mark does **not** round-trip reliably: ``f / fps * fps`` can evaluate to
    ``f - epsilon`` in floating point, and ``floor`` then returns ``f - 1``. This is not a
    rare corner case — even at ``fps = 25``, ``29 / 25 * 25 = 28.999999999999996``. Counted
    over the first 200,000 frames: 25 fps is off on 2,730 frames, 29.97 fps on 1,384. This
    is exactly the class of silent error this module exists to prevent.

    Use ``at_center=True`` whenever the time will be fed back to an ffmpeg/ffprobe seek or
    to :func:`time_to_frame`: the mid-frame mark round-trips exactly at every fps.

    >>> time_to_frame(frame_to_time(29, 25.0), 25.0)                  # start mark
    28
    >>> time_to_frame(frame_to_time(29, 25.0, at_center=True), 25.0)
    29
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be >= 0, got {frame_idx}")
    return (frame_idx + 0.5) / fps if at_center else frame_idx / fps


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Duration (seconds) -> duration (frames), rounded up so coverage is never short."""
    return max(1, math.ceil(seconds * fps))


@dataclass(slots=True)
class FrameIndexReport:
    """Result of cross-checking the frame index convention over a set of keyframes."""

    n_checked: int = 0
    #: number of matching keyframes, per candidate formula.
    matches: dict[str, int] = field(default_factory=dict)
    #: (video_id, n, pts_time, fps, expected, got) for each mismatch.
    mismatches: list[tuple[str, int, float, float, int, int]] = field(default_factory=list)
    #: videos carrying more than one fps value -> possible VFR.
    multi_fps: list[tuple[str, list[float]]] = field(default_factory=list)
    #: videos whose fps is outside the common set.
    unusual_fps: list[tuple[str, list[float]]] = field(default_factory=list)
    #: videos where two keyframes share a frame index -> the mapping is not injective.
    duplicate_frame_idx: list[tuple[str, int]] = field(default_factory=list)
    #: videos whose pts_time is not monotonically increasing.
    non_monotonic_pts: list[str] = field(default_factory=list)

    @property
    def best_convention(self) -> str | None:
        if not self.matches:
            return None
        return max(self.matches, key=lambda name: self.matches[name])

    @property
    def passed(self) -> bool:
        """Acceptance condition: exactly **zero** mismatches over every sample."""
        return self.n_checked > 0 and not self.mismatches

    def summary(self) -> str:
        lines = [f"P1 — frame index convention checked over {self.n_checked:,} keyframes", ""]
        for name, hits in sorted(self.matches.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * hits / self.n_checked if self.n_checked else 0.0
            mark = "  <-- use this one" if name == self.best_convention else ""
            lines.append(f"  {name:<6} {hits:>8,} / {self.n_checked:,}  = {pct:6.2f} %{mark}")
        lines.append("")
        verdict = "PASS" if self.passed else "FAIL"
        lines.append(f"  VERDICT: {verdict} ({len(self.mismatches)} mismatches)")
        if self.mismatches:
            lines.append("  First mismatches (video, n, pts, fps, expected, got):")
            lines.extend(f"    {row}" for row in self.mismatches[:5])
        if self.multi_fps:
            lines.append(
                f"  Videos with multiple fps (possible VFR): {len(self.multi_fps)} "
                f"-> {self.multi_fps[:3]}"
            )
        if self.unusual_fps:
            lines.append(f"  Videos with unusual fps: {self.unusual_fps[:5]}")
        if self.duplicate_frame_idx:
            lines.append(
                f"  Videos with duplicate frame_idx: {len(self.duplicate_frame_idx)} "
                "-> the keyframe->frame mapping is NOT injective, look up by n"
            )
        if self.non_monotonic_pts:
            lines.append(f"  Videos with non-monotonic pts: {self.non_monotonic_pts[:5]}")
        return "\n".join(lines)


def check_convention(
    rows_by_video: dict[str, Sequence[tuple[int, float, float, int]]],
) -> FrameIndexReport:
    """Compare three candidate formulas against the organiser's ``frame_idx``.

    ``rows_by_video[video_id]`` is a sequence of ``(n, pts_time, fps, frame_idx)`` — exactly
    the four columns of ``map-keyframes/<video>.csv``.

    The acceptance condition is ``report.passed``: zero absolute mismatches. Any non-zero
    mismatch must be traced to its root cause *before* any other component is built; this
    is a blocking item.
    """
    report = FrameIndexReport(matches={"floor": 0, "round": 0, "ceil": 0})
    for video_id, rows in rows_by_video.items():
        fps_seen: set[float] = set()
        seen_indices: set[int] = set()
        duplicates = 0
        previous_pts = -1.0
        non_monotonic = False
        for n, pts, fps, got in rows:
            report.n_checked += 1
            fps_seen.add(fps)
            for convention in ("floor", "round", "ceil"):
                if time_to_frame(pts, fps, convention) == got:  # type: ignore[arg-type]
                    report.matches[convention] += 1
            expected = time_to_frame(pts, fps, DEFAULT_CONVENTION)
            if expected != got:
                report.mismatches.append((video_id, n, pts, fps, expected, got))
            if got in seen_indices:
                duplicates += 1
            seen_indices.add(got)
            if pts < previous_pts:
                non_monotonic = True
            previous_pts = pts
        if len(fps_seen) > 1:
            report.multi_fps.append((video_id, sorted(fps_seen)))
        if fps_seen - set(COMMON_FPS):
            report.unusual_fps.append((video_id, sorted(fps_seen)))
        if duplicates:
            report.duplicate_frame_idx.append((video_id, duplicates))
        if non_monotonic:
            report.non_monotonic_pts.append(video_id)
    return report


def grid_alignment(
    pts_times: Iterable[float], fps: float, *, decimals: int = 4
) -> dict[str, float]:
    """Measure how well ``pts_time`` sits on the CFR grid of ``fps``.

    This is a VFR test that needs *only* the organiser-supplied data — no ffprobe.

    If a video is true CFR at ``fps``, every frame's presentation time is ``k/fps``, so
    ``pts_time * fps`` must be an integer up to the rounding introduced by storing
    ``decimals`` decimal places: ``tol = 0.5 * 10^-decimals * fps``.

    Measured over all 177,321 keyframes of batch 1: ``frac(pts*fps)`` **never** lands near
    0.5 — it always sits close to 0 (83.6 %) or close to 1 (12.9 %). That is the signature
    of a rounded CFR grid, and it *proves* the ``floor`` convention: the true value of
    ``pts*fps`` always lies in ``[frame_idx, frame_idx+1)``. The fps = 29.97 group has a
    mean ``frac`` of 0.835 because ``1/29.97`` is a repeating decimal, which is why
    ``round()`` is wrong on nearly that entire group.

    An earlier version of this measurement fitted a straight line to ``pts_time`` against
    *keyframe index* and treated the residual as VFR drift. That was conceptually wrong:
    keyframes are sampled at shot boundaries, not at even time intervals, so the residual
    measures editing irregularity rather than VFR. It raised a false alarm on 873/873
    videos. Genuine VFR validation needs per-frame ``pts`` from ``ffprobe -show_frames``;
    see ``docs/CONSTRAINTS.md``, open risk R2.

    Returns ``off_grid_rate`` (fraction of keyframes off the grid) and ``max_dev_frames``.
    """
    values = list(pts_times)
    if not values:
        return {"off_grid_rate": 0.0, "max_dev_frames": 0.0, "n": 0.0, "is_vfr": 0.0}
    tol = 0.5 * (10.0**-decimals) * fps
    # Distance to the nearest frame, in frame units.
    deviations = [abs(pts * fps - round(pts * fps)) for pts in values]
    off_grid = sum(1 for d in deviations if d > tol) / len(deviations)
    return {
        "off_grid_rate": off_grid,
        "max_dev_frames": max(deviations),
        "n": float(len(values)),
        # Scattered off-grid values are normal (rounding); only a *majority* is suspicious.
        "is_vfr": float(off_grid > 0.5 and max(deviations) > 0.25),
    }
