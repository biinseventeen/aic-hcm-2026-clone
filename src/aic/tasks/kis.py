"""P10 + P13 for Textual KIS: from candidates to 100 submission rows.

The transformation chain:

    Candidate (a scored shot)
      -> VideoBelief (pi_v, loci)                [hierarchical coverage model]
      -> FrameCandidate (frames spread to cover) [P10]
      -> Allocation x100 (submodular greedy)     [P13]
      -> Answer x100                             [submission writer]

The important point: the last two steps need **no GPU and no model**. They only consume the
``pi_v`` supplied by the retrieval and verification layers. That is why this is the highest
score-per-effort part of the whole system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.allocator import AllocationTrace, FrameCandidate, allocate_kis
from ..core.coverage import CoverageModel, Locus, VideoBelief
from ..core.frameidx import seconds_to_frames
from ..core.objective import MAX_ANSWERS
from ..core.spread import covering_frames, expand_locus, order_anchor_first
from ..query.retrieve import Candidate, RetrievalResult
from ..submit.writer import Answer, QuerySubmission

__all__ = ["KisSolution", "build_beliefs", "build_frame_pool", "solve_kis"]


def build_beliefs(
    result: RetrievalResult,
    *,
    fps_by_video: dict[str, float],
    pad_seconds: float = 0.5,
    max_frames: dict[str, int] | None = None,
) -> list[VideoBelief]:
    """Turn candidates into :class:`VideoBelief` objects — ``pi_v`` plus a locus distribution.

    A video's loci are its candidate shots, each weighted by its fused score. Loci are
    **widened on both sides** by ``pad_seconds`` because the target event may straddle a shot
    boundary: the detector cut through the middle of the event, so sampling has to spill into
    the adjacent shot.

    Near-duplicate clusters: members sharing a ``cluster`` split their weight rather than
    accumulating it, otherwise a shot repeated ten times would absorb all of the video's
    probability mass purely by repeating.
    """
    by_video: dict[str, list[Candidate]] = {}
    for candidate in result.candidates:
        by_video.setdefault(candidate.video_id, []).append(candidate)

    beliefs: list[VideoBelief] = []
    for video_id, candidates in by_video.items():
        fps = fps_by_video.get(video_id, 25.0)
        pad = seconds_to_frames(pad_seconds, fps)
        last_frame = (max_frames or {}).get(video_id)

        cluster_sizes: dict[int, int] = {}
        for candidate in candidates:
            if candidate.cluster >= 0:
                cluster_sizes[candidate.cluster] = cluster_sizes.get(candidate.cluster, 0) + 1

        loci: list[Locus] = []
        for candidate in candidates:
            share = cluster_sizes.get(candidate.cluster, 1) if candidate.cluster >= 0 else 1
            locus = Locus(
                candidate.start,
                candidate.end,
                weight=max(candidate.fused_score, 1e-9) / share,
                label=f"shot#{candidate.shot_id}",
            )
            loci.append(expand_locus(locus, pad, max_frame=last_frame))
        beliefs.append(
            VideoBelief(video_id=video_id, pi=result.video_scores.get(video_id, 0.0), loci=loci)
        )
    return beliefs


def build_frame_pool(
    beliefs: list[VideoBelief],
    *,
    answer_len: int,
    anchors: dict[str, int] | None = None,
    max_frames: dict[str, int] | None = None,
    budget: int = MAX_ANSWERS,
) -> list[FrameCandidate]:
    """Generate candidate frames covering every locus, dense enough never to under-fill.

    Each locus receives the *minimal* frame set that certainly covers it (step =
    ``answer_len``). If the total number of frames generated is below ``budget``, density is
    increased by halving the step — a consequence of H4: the last 50 slots are worth only 0.2
    points but cost almost nothing to produce, so leaving them empty forfeits free expectation.
    """
    if answer_len < 1:
        raise ValueError(f"answer_len must be >= 1, got {answer_len}")
    anchors = anchors or {}

    def generate(step: int) -> list[FrameCandidate]:
        pool: dict[tuple[str, int], FrameCandidate] = {}
        for belief in beliefs:
            last_frame = (max_frames or {}).get(belief.video_id)
            for locus in belief.loci:
                frames = covering_frames(locus, step, max_frame=last_frame)
                anchor = anchors.get(belief.video_id)
                if anchor is not None and locus.start <= anchor <= locus.end:
                    frames = order_anchor_first(anchor, frames)
                for rank, frame in enumerate(frames):
                    key = (belief.video_id, frame)
                    # The raw score is only a tie-breaker: the video's pi_v, discounted by
                    # position within the locus, times the locus weight.
                    score = belief.pi * locus.weight / (1.0 + rank)
                    previous = pool.get(key)
                    if previous is None or score > previous.score:
                        pool[key] = FrameCandidate(
                            video_id=belief.video_id,
                            frame_id=frame,
                            score=score,
                            source=f"{locus.label}@step{step}",
                        )
        return list(pool.values())

    step = answer_len
    pool = generate(step)
    # Increase density until there are enough slots, but never finer than one frame.
    while len(pool) < budget and step > 1:
        step = max(1, step // 2)
        pool = generate(step)
    return pool


@dataclass
class KisSolution:
    submission: QuerySubmission
    trace: AllocationTrace
    beliefs: list[VideoBelief] = field(default_factory=list)

    def report(self) -> str:
        trace = self.trace
        coverage = ", ".join(f"{k}={v:.3f}" for k, v in sorted(trace.coverage_at_k.items()))
        lines = [
            f"KIS {self.submission.query_id}: {len(self.submission.answers)} slots, "
            f"{trace.n_distinct_videos} distinct videos",
            f"  P@k  : {coverage}",
            f"  E[Final] = {trace.expected_final:.4f}",
        ]
        lines.extend(f"  [i] {note}" for note in trace.notes)
        lines.append("  top 5 slots:")
        for allocation in trace.allocations[:5]:
            lines.append(
                f"    {allocation.rank:>3} {allocation.video_id:<11} "
                f"f={allocation.frame_id:<7} gain={allocation.gain:.4f} "
                f"cum={allocation.cumulative:.4f} [{allocation.source}]"
            )
        return "\n".join(lines)


def solve_kis(
    query_id: str,
    result: RetrievalResult,
    *,
    fps_by_video: dict[str, float],
    answer_len: int = 25,
    pad_seconds: float = 0.5,
    max_frames: dict[str, int] | None = None,
    budget: int = MAX_ANSWERS,
) -> KisSolution:
    """Solve one Textual KIS query end to end from a retrieval result."""
    beliefs = build_beliefs(
        result, fps_by_video=fps_by_video, pad_seconds=pad_seconds, max_frames=max_frames
    )
    if not beliefs:
        return KisSolution(
            submission=QuerySubmission(query_id, "kis", []),
            trace=AllocationTrace(notes=["no candidates — the retrieval layer failed"]),
        )
    anchors: dict[str, int] = {}
    for candidate in sorted(result.candidates, key=lambda c: -c.fused_score):
        anchors.setdefault(candidate.video_id, candidate.anchor)

    model = CoverageModel.from_beliefs(beliefs, answer_len=answer_len)
    pool = build_frame_pool(
        beliefs,
        answer_len=answer_len,
        anchors=anchors,
        max_frames=max_frames,
        budget=budget,
    )
    trace = allocate_kis(model, pool, budget=min(budget, len(pool)))
    answers = [
        Answer(video_id=allocation.video_id, frame=allocation.frame_id)
        for allocation in trace.allocations
    ]
    return KisSolution(
        submission=QuerySubmission(query_id, "kis", answers),
        trace=trace,
        beliefs=beliefs,
    )
