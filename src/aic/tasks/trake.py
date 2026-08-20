"""P12 + P13 for TRAKE: moment sequence alignment and tuple hedging.

Three facts shape this branch
-----------------------------
1. **The wrong video is a hard zero.** The ``Pr(v = GT_v)`` factor dominates everything.
   Raising alignment accuracy from 0.6 to 0.8 while the probability of the right video is 0.4
   yields less than raising that probability from 0.4 to 0.6. The engineering budget for TRAKE
   must therefore favour the **retrieval layer**, not the alignment layer.

2. **The answer window is under 10 frames.** The organiser's keyframes are 69 frames apart on
   average, so the probability that an existing keyframe lands inside the window is only ~14 %.
   That is measured on the real data, not guessed. **TRAKE requires decoding video at full
   temporal resolution.** The supplied CLIP features are not sufficient for this branch.

3. **Zero-shot is mandatory.** The number of moments and their content are defined by the
   judges at query time; no labelled set exists for these action classes.

Degraded mode
-------------
When video cannot be decoded (no ffmpeg, no archive), this module still runs in
**keyframe-only** mode and says so in ``notes``. The score ceiling in that mode is ~14 % per
moment; it exists so the pipeline runs end to end, not to produce a submission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.align import AlignedPath, alternative_paths, detect_cycles
from ..core.allocator import AllocationTrace, TrakeCandidate, allocate_trake
from ..core.frameidx import seconds_to_frames
from ..core.objective import MAX_ANSWERS
from ..query.retrieve import RetrievalResult
from ..submit.writer import Answer, QuerySubmission

__all__ = [
    "MomentWindow",
    "TrakeSolution",
    "align_in_window",
    "solve_trake",
    "windows_from_keyframes",
]


@dataclass(slots=True)
class MomentWindow:
    """One encoded candidate window, with the column -> real frame index mapping.

    ``frames[t]`` is the *organiser's* frame index for column ``t`` of ``embeddings``. This
    mapping is where every frame-convention error shows up, so it is kept explicit rather than
    re-derived downstream by multiplying by fps.
    """

    video_id: str
    frames: np.ndarray  # (T,) int
    embeddings: np.ndarray  # (T, D), L2-normalised
    source: str = "fullfps"

    def __post_init__(self) -> None:
        if len(self.frames) != self.embeddings.shape[0]:
            raise ValueError(
                f"{self.video_id}: column count mismatch frames={len(self.frames)} "
                f"embeddings={self.embeddings.shape[0]} — the column->frame map is broken"
            )

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def align_in_window(
    window: MomentWindow,
    moment_embeddings: np.ndarray,
    *,
    min_gap_frames: int = 5,
    n_paths: int = 6,
    use_cycles: bool = True,
) -> list[tuple[AlignedPath, tuple[int, ...]]]:
    """Align N moments inside one window. Returns ``[(path, frame_ids)]``.

    ``moment_embeddings`` has shape (N, D), in the same space as ``window.embeddings``.

    When ``use_cycles`` is set and the window has repeating structure (an athlete performing the
    action several times, a lion dance troupe going through several rounds), the window is
    segmented by cycle and aligned **independently within each cycle**, then cycles are ordered
    by total score. Later cycles become hedge candidates for the lower slot bands. Without this
    step, dynamic programming over the whole window can pair moment 1 of the first cycle with
    moment 2 of the last — a high-scoring but meaningless tuple.
    """
    moments = np.asarray(moment_embeddings, dtype=np.float32)
    if moments.ndim != 2:
        raise ValueError(f"moment_embeddings must be (N, D), got {moments.shape}")
    if moments.shape[1] != window.embeddings.shape[1]:
        raise ValueError(
            f"dimension mismatch: moments {moments.shape[1]} vs window "
            f"{window.embeddings.shape[1]} — not the same embedding space"
        )
    n_moments = moments.shape[0]
    # The full similarity matrix S, shape (N, T).
    similarity = moments @ np.asarray(window.embeddings, dtype=np.float32).T

    # Minimum step between moments, measured in *columns* rather than frames.
    stride = max(1, int(np.median(np.diff(window.frames)))) if window.n_frames >= 2 else 1
    delta = max(1, min_gap_frames // stride)

    segments: list[tuple[int, int]] = [(0, window.n_frames - 1)]
    if use_cycles and window.n_frames >= 4 * n_moments * delta:
        detected = detect_cycles(window.embeddings, min_period=max(8, n_moments * delta))
        long_enough = [
            (start, end)
            for start, end in detected
            if (end - start + 1) >= (n_moments - 1) * delta + 1
        ]
        segments = long_enough or segments

    results: list[tuple[AlignedPath, tuple[int, ...]]] = []
    for start, end in segments:
        segment = similarity[:, start : end + 1]
        if segment.shape[1] < (n_moments - 1) * delta + 1:
            continue
        per_segment = max(1, n_paths // max(1, len(segments)))
        try:
            paths = alternative_paths(
                segment, delta=delta, n_paths=per_segment, min_shift=max(1, delta)
            )
        except ValueError:
            continue
        for path in paths:
            frames = tuple(int(window.frames[start + column]) for column in path.indices)
            results.append((path, frames))
    results.sort(key=lambda pair: -pair[0].total)
    return results[:n_paths]


@dataclass
class TrakeSolution:
    submission: QuerySubmission
    trace: AllocationTrace
    n_moments: int = 0
    degraded: bool = False

    def report(self) -> str:
        trace = self.trace
        coverage = ", ".join(f"{k}={v:.3f}" for k, v in sorted(trace.coverage_at_k.items()))
        lines = [
            f"TRAKE {self.submission.query_id}: {len(self.submission.answers)} tuples x "
            f"{self.n_moments} moments, {trace.n_distinct_videos} videos",
            f"  E[max R]@k : {coverage}",
            f"  E[Final] = {trace.expected_final:.4f}",
        ]
        if self.degraded:
            lines.append(
                "  [!] DEGRADED MODE: existing keyframes only (69 frames apart on average). "
                "Score ceiling ~14 % per moment. NOT for submission."
            )
        lines.append("  top 5 tuples:")
        for allocation in trace.allocations[:5]:
            lines.append(
                f"    {allocation.rank:>3} {allocation.video_id:<11} "
                f"{allocation.frames}  gain={allocation.gain:.4f}"
            )
        lines.extend(f"  [i] {note}" for note in trace.notes)
        return "\n".join(lines)


def solve_trake(
    query_id: str,
    result: RetrievalResult,
    windows: dict[str, MomentWindow],
    moment_embeddings: np.ndarray,
    *,
    fps_by_video: dict[str, float],
    answer_len: int = 10,
    min_gap_seconds: float = 0.2,
    n_paths_per_video: int = 6,
    budget: int = MAX_ANSWERS,
    seed: int = 0,
    degraded: bool = False,
) -> TrakeSolution:
    """Solve one TRAKE query.

    ``windows[video_id]`` is the encoded window for each candidate video. Building them is the
    caller's responsibility — either at full fps (correct) or from existing keyframes (degraded,
    for testing only, see :func:`windows_from_keyframes`).
    """
    n_moments = int(np.asarray(moment_embeddings).shape[0])
    candidates: list[TrakeCandidate] = []
    moment_spans: dict[str, list[tuple[int, int]]] = {}

    for video_id, window in windows.items():
        fps = fps_by_video.get(video_id, 25.0)
        gap = seconds_to_frames(min_gap_seconds, fps)
        try:
            aligned = align_in_window(
                window, moment_embeddings, min_gap_frames=gap, n_paths=n_paths_per_video
            )
        except ValueError:
            continue
        for path, frames in aligned:
            candidates.append(
                TrakeCandidate(
                    video_id=video_id,
                    frames=frames,
                    score=float(path.total) * result.video_scores.get(video_id, 0.0),
                    moment_conf=path.confidence,
                    source=window.source,
                )
            )
        # Uncertainty locus per moment: the spread across paths, widened by answer_len so the
        # Monte Carlo model does not become overconfident.
        if aligned:
            moment_spans[video_id] = [
                (
                    min(frames[j] for _, frames in aligned) - answer_len,
                    max(frames[j] for _, frames in aligned) + answer_len,
                )
                for j in range(n_moments)
            ]

    if not candidates:
        return TrakeSolution(
            submission=QuerySubmission(query_id, "trake", [], n_moments=n_moments),
            trace=AllocationTrace(
                notes=[
                    "no tuples could be built — check the candidate windows and the "
                    "embedding dimensionality"
                ]
            ),
            n_moments=n_moments,
            degraded=degraded,
        )

    trace = allocate_trake(
        candidates,
        video_pi=result.video_scores,
        moment_spans=moment_spans,
        answer_len=answer_len,
        budget=min(budget, len(candidates)),
        seed=seed,
    )
    answers = [
        Answer(video_id=allocation.video_id, frames=allocation.frames)
        for allocation in trace.allocations
    ]
    return TrakeSolution(
        submission=QuerySubmission(query_id, "trake", answers, n_moments=n_moments),
        trace=trace,
        n_moments=n_moments,
        degraded=degraded,
    )


def windows_from_keyframes(
    result: RetrievalResult,
    dense,
    *,
    pad_seconds: float = 15.0,
    fps_by_video: dict[str, float] | None = None,
    max_videos: int = 20,
) -> dict[str, MomentWindow]:
    """Build windows from **existing keyframes** — degraded mode, for testing.

    The score ceiling of this mode is ~14 % per moment (keyframes 69 frames apart on average
    against a 10-frame answer window). It exists so the pipeline can run end to end without
    ffmpeg, **not** to produce a submission. The correct mode decodes the video at full fps.
    """
    fps_by_video = fps_by_video or {}
    windows: dict[str, MomentWindow] = {}
    by_video: dict[str, list] = {}
    for candidate in result.candidates:
        by_video.setdefault(candidate.video_id, []).append(candidate)
    ranked = sorted(by_video, key=lambda v: -result.video_scores.get(v, 0.0))[:max_videos]

    for video_id in ranked:
        fps = fps_by_video.get(video_id, 25.0)
        pad = seconds_to_frames(pad_seconds, fps)
        low = min(c.start for c in by_video[video_id]) - pad
        high = max(c.end for c in by_video[video_id]) + pad
        rows = dense.video_rows(video_id)
        if len(rows) == 0:
            continue
        frames = dense.rows[rows, 2]
        inside = (frames >= low) & (frames <= high)
        if inside.sum() < 2:
            inside = np.ones(len(frames), dtype=bool)
        selected = rows[inside]
        embeddings = np.asarray(dense.vectors[selected], dtype=np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        windows[video_id] = MomentWindow(
            video_id=video_id,
            frames=frames[inside],
            embeddings=embeddings,
            source="keyframe-degraded",
        )
    return windows
