"""P10 — locating frames inside a shot: generating the frame set to submit.

This is an *interval covering* problem, not a point estimation problem. The answer is an
interval [s, e] whose position inside the candidate shot is unknown. While slots remain
free, submitting a single frame at the best point estimate is a dominated strategy.

For a locus of length W and an answer interval of length L, a frame f covers exactly L
start positions, so the minimum number of frames needed for guaranteed coverage is
ceil(W / L) and the optimal sampling step is exactly L. Covering cost grows linearly in
1/L: a 10-second shot at 25 fps (W = 250) with L = 10 frames needs 25 slots.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .coverage import Locus

__all__ = [
    "coverage_of",
    "covering_frames",
    "expand_locus",
    "min_frames_to_cover",
    "order_anchor_first",
    "slots_needed_report",
    "spread_frames",
]


def min_frames_to_cover(locus_len: int, answer_len: int) -> int:
    """Minimum number of frames for guaranteed coverage of one locus.

    >>> min_frames_to_cover(250, 10), min_frames_to_cover(250, 50), min_frames_to_cover(5, 50)
    (25, 5, 1)
    """
    if locus_len < 1 or answer_len < 1:
        raise ValueError(f"lengths must be >= 1, got ({locus_len}, {answer_len})")
    return max(1, math.ceil(locus_len / answer_len))


def covering_frames(
    locus: Locus,
    answer_len: int,
    *,
    max_frame: int | None = None,
    limit: int | None = None,
) -> list[int]:
    """The *minimal* frame set that certainly covers every answer position in the locus.

    Frame i is placed at ``start + (i+1)*L - 1`` — the right edge of window i — so that
    each frame covers exactly L consecutive start positions with no overlap.

    ``max_frame`` is the last frame that exists in the video; generated frames are clamped
    to it, because submitting a frame index that does not exist is a guaranteed miss.

    >>> covering_frames(Locus(0, 99), 25)
    [24, 49, 74, 99]
    >>> covering_frames(Locus(100, 149), 25, max_frame=120)
    [120]
    """
    count = min_frames_to_cover(locus.length, answer_len)
    if limit is not None:
        count = min(count, max(1, limit))
    out: list[int] = []
    for i in range(count):
        frame = locus.start + (i + 1) * answer_len - 1
        frame = min(frame, locus.end)
        if max_frame is not None:
            frame = min(frame, max_frame)
        frame = max(frame, 0)
        if not out or frame != out[-1]:
            out.append(frame)
    return out


def spread_frames(
    locus: Locus,
    step: int,
    *,
    max_frame: int | None = None,
    limit: int | None = None,
) -> list[int]:
    """Spread frames evenly across the locus with a fixed ``step``.

    Used when density is controlled directly rather than derived from ``answer_len``.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    out: list[int] = []
    frame = locus.start + step // 2
    while frame <= locus.end:
        clamped = frame if max_frame is None else min(frame, max_frame)
        if not out or clamped != out[-1]:
            out.append(clamped)
        if limit is not None and len(out) >= limit:
            break
        frame += step
    if not out:
        out.append(min(locus.start, max_frame) if max_frame is not None else locus.start)
    return out


def order_anchor_first(anchor: int, frames: Iterable[int]) -> list[int]:
    """Put ``anchor`` first, then the rest by increasing distance from the anchor.

    Slot 1 always receives the frame with the highest verification score (a consequence of
    H3: position 1 is chosen by argmax, entirely independently of the diversification
    strategy that follows). Subsequent frames fan out in both directions so that the
    neighbourhood of the anchor is covered first.

    >>> order_anchor_first(50, [0, 25, 50, 75, 100])
    [50, 25, 75, 0, 100]
    """
    rest = sorted({f for f in frames if f != anchor}, key=lambda f: (abs(f - anchor), f))
    return [anchor, *rest]


def expand_locus(locus: Locus, pad: int, *, max_frame: int | None = None) -> Locus:
    """Widen a locus by ``pad`` frames on both sides.

    Needed when the target event sits at a *shot boundary*: the shot detector cut through
    the middle of the event, so sampling has to spill into the adjacent shot.
    """
    start = max(0, locus.start - pad)
    end = locus.end + pad
    if max_frame is not None:
        end = min(end, max_frame)
    return Locus(start=start, end=max(end, start), weight=locus.weight, label=locus.label)


def coverage_of(frames: Sequence[int], locus: Locus, answer_len: int) -> float:
    """kappa of a frame set over a single locus — convenient for tests and diagnostics."""
    from .coverage import VideoBelief

    return VideoBelief("_", 1.0, [locus]).kappa(frames, answer_len)


def slots_needed_report(locus_len_frames: int, answer_len_frames: int, fps: float = 25.0) -> str:
    """Table of slot cost for coverage — used when weighing allocation between candidates."""
    count = min_frames_to_cover(locus_len_frames, answer_len_frames)
    return (
        f"locus {locus_len_frames}f ({locus_len_frames / fps:.1f}s), "
        f"L={answer_len_frames}f ({answer_len_frames / fps:.2f}s) "
        f"-> needs {count} slots for guaranteed coverage"
    )
