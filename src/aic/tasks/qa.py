"""P11 + P13 for Q&A: hedging along the answer axis.

The Q&A R-Score is a **conjunction of three conditions**: right video, frame inside the span,
and an answer that matches semantically. The answer axis is *independent* of the other two and
is **far cheaper to hedge**: adding another answer for the same (video, frame) pair costs one
slot and **adds** ``pi_v * p_a * kappa`` to the coverage probability, whereas adding a new frame
adds only ``pi_v * (kappa_new - kappa_old)``, which is already diminishing.

That is consequence H5: when confidence in (video, frame) is high but the answer is uncertain —
counts, colours, proper nouns — submitting the same (video, frame) pair with several different
answers in consecutive slots moves nearly all of the answer axis's probability mass into P_5.

AN UNCONFIRMED ASSUMPTION
-------------------------
This strategy assumes the scoring system **accepts several rows sharing a ``(video_id,
frame_id)`` with different ``answer`` values**. The rules do not forbid it, but they do not
confirm it either. If the scoring system deduplicates on ``(video_id, frame_id)``, three slots
spent on three answers collapse into one and the strategy is void.

The cost of being wrong is three slots in the 2–5 band — not severe, but the expected benefit
also disappears. Pass ``hedge_answers=False`` for the safe mode: each (video, frame) pair keeps
only its highest-probability answer. See ``docs/SUBMISSION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.allocator import AllocationTrace, AnswerCandidate, allocate_qa
from ..core.coverage import CoverageModel, VideoBelief
from ..core.objective import MAX_ANSWERS
from ..core.spread import covering_frames, order_anchor_first
from ..query.retrieve import RetrievalResult
from ..submit.writer import Answer, QuerySubmission
from .kis import build_beliefs

__all__ = [
    "AnswerHypothesis",
    "QaSolution",
    "hedge_color",
    "hedge_numeric",
    "normalize_hypotheses",
    "solve_qa",
]


@dataclass(slots=True)
class AnswerHypothesis:
    """One candidate answer with its calibrated probability."""

    text: str
    prob: float
    source: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must lie in [0, 1], got {self.prob}")


def normalize_hypotheses(
    hypotheses: list[AnswerHypothesis], *, top_k: int = 4
) -> list[AnswerHypothesis]:
    """Keep the ``top_k`` hypotheses and normalise probabilities to sum to at most 1.

    Answers are **mutually exclusive** (only one is correct), so their probabilities must not
    sum above 1. If they do, that is a sign the generating model is not calibrated.
    """
    kept = sorted(hypotheses, key=lambda h: -h.prob)[:top_k]
    total = sum(h.prob for h in kept)
    if total > 1.0:
        kept = [AnswerHypothesis(h.text, h.prob / total, h.source) for h in kept]
    return [h for h in kept if h.prob > 0.0]


def hedge_numeric(
    estimate: int, *, spread: int = 1, center_prob: float = 0.5
) -> list[AnswerHypothesis]:
    """Hedge around a count: n-1, n, n+1.

    Counting questions are where a small vision-language model is least reliable, and also where
    hedging is cheapest — the answer space is ordered and narrow.

    >>> [(h.text, round(h.prob, 3)) for h in hedge_numeric(5)]
    [('5', 0.5), ('4', 0.25), ('6', 0.25)]
    """
    if estimate < 0:
        raise ValueError(f"estimate must be >= 0, got {estimate}")
    hypotheses = [AnswerHypothesis(str(estimate), center_prob, "count")]
    offsets = [
        delta
        for distance in range(1, spread + 1)
        for delta in (-distance, distance)
        if estimate + delta >= 0
    ]
    if offsets:
        each = (1.0 - center_prob) / len(offsets)
        hypotheses.extend(
            AnswerHypothesis(str(estimate + delta), each, "count-hedge") for delta in offsets
        )
    return normalize_hypotheses(hypotheses, top_k=1 + len(offsets))


#: Perceptually adjacent colours, in Vietnamese because that is the answer language. Hedging
#: along this axis is what a colour question calls for.
_COLOR_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "đỏ": ("cam", "hồng", "nâu"),
    "cam": ("đỏ", "vàng"),
    "vàng": ("cam", "nâu"),
    "xanh": ("xanh lá", "xanh dương", "lam"),
    "xanh lá": ("xanh", "xanh dương"),
    "xanh dương": ("xanh", "lam", "tím"),
    "tím": ("xanh dương", "hồng"),
    "hồng": ("đỏ", "tím"),
    "nâu": ("đỏ", "vàng", "đen"),
    "đen": ("nâu", "xám"),
    "trắng": ("xám", "be"),
    "xám": ("trắng", "đen"),
}


def hedge_color(color: str, *, center_prob: float = 0.55) -> list[AnswerHypothesis]:
    """Hedge over perceptually adjacent colours.

    >>> [h.text for h in hedge_color("đỏ")]
    ['đỏ', 'cam', 'hồng', 'nâu']
    """
    normalised = color.strip().lower()
    neighbours = _COLOR_NEIGHBOURS.get(normalised, ())
    hypotheses = [AnswerHypothesis(normalised, center_prob, "color")]
    if neighbours:
        each = (1.0 - center_prob) / len(neighbours)
        hypotheses.extend(
            AnswerHypothesis(neighbour, each, "color-hedge") for neighbour in neighbours
        )
    return normalize_hypotheses(hypotheses, top_k=1 + len(neighbours))


@dataclass
class QaSolution:
    submission: QuerySubmission
    trace: AllocationTrace
    beliefs: list[VideoBelief] = field(default_factory=list)

    def report(self) -> str:
        trace = self.trace
        coverage = ", ".join(f"{k}={v:.3f}" for k, v in sorted(trace.coverage_at_k.items()))
        lines = [
            f"Q&A {self.submission.query_id}: {len(self.submission.answers)} slots, "
            f"{trace.n_distinct_videos} videos",
            f"  P@k  : {coverage}",
            f"  E[Final] = {trace.expected_final:.4f}",
            "  top 6 slots:",
        ]
        for allocation in trace.allocations[:6]:
            lines.append(
                f"    {allocation.rank:>3} {allocation.video_id:<11} "
                f"f={allocation.frame_id:<7} ans={allocation.answer!r:<14} "
                f"gain={allocation.gain:.4f}"
            )
        lines.extend(f"  [i] {note}" for note in trace.notes)
        return "\n".join(lines)


def solve_qa(
    query_id: str,
    result: RetrievalResult,
    *,
    fps_by_video: dict[str, float],
    hypotheses: dict[str, list[AnswerHypothesis]] | list[AnswerHypothesis],
    answer_len: int = 25,
    pad_seconds: float = 0.5,
    max_frames: dict[str, int] | None = None,
    budget: int = MAX_ANSWERS,
    hedge_answers: bool = True,
    max_frames_per_locus: int = 4,
) -> QaSolution:
    """Solve one Q&A query.

    ``hypotheses`` is either a single list shared by every video, or a dict keyed by
    ``video_id`` when the generating model is called per candidate.

    ``max_frames_per_locus`` caps frames per shot: the Q&A candidate space is a *three-way
    product* (video x frame x answer), so it grows fast. With 4 frames and 4 answers, one shot
    already occupies 16 combinations; greedy still picks correctly, but its runtime grows
    quadratically in the candidate set size.
    """
    beliefs = build_beliefs(
        result, fps_by_video=fps_by_video, pad_seconds=pad_seconds, max_frames=max_frames
    )
    if not beliefs:
        return QaSolution(
            submission=QuerySubmission(query_id, "qa", []),
            trace=AllocationTrace(notes=["no candidates — the retrieval layer failed"]),
        )

    anchors: dict[str, int] = {}
    for candidate in sorted(result.candidates, key=lambda c: -c.fused_score):
        anchors.setdefault(candidate.video_id, candidate.anchor)

    def hypotheses_for(video_id: str) -> list[AnswerHypothesis]:
        raw = hypotheses[video_id] if isinstance(hypotheses, dict) else hypotheses
        normalised = normalize_hypotheses(list(raw))
        return normalised[:1] if not hedge_answers else normalised

    pool: dict[tuple[str, int, str], AnswerCandidate] = {}
    for belief in beliefs:
        answers_for_video = hypotheses_for(belief.video_id)
        if not answers_for_video:
            continue
        last_frame = (max_frames or {}).get(belief.video_id)
        for locus in belief.loci:
            frames = covering_frames(
                locus, answer_len, max_frame=last_frame, limit=max_frames_per_locus
            )
            anchor = anchors.get(belief.video_id)
            if anchor is not None and locus.start <= anchor <= locus.end:
                frames = order_anchor_first(anchor, frames)
            for rank, frame in enumerate(frames):
                for hypothesis in answers_for_video:
                    key = (belief.video_id, frame, hypothesis.text)
                    score = belief.pi * locus.weight * hypothesis.prob / (1.0 + rank)
                    previous = pool.get(key)
                    if previous is None or score > previous.score:
                        pool[key] = AnswerCandidate(
                            video_id=belief.video_id,
                            frame_id=frame,
                            answer=hypothesis.text,
                            answer_prob=hypothesis.prob,
                            score=score,
                            source=hypothesis.source,
                        )

    model = CoverageModel.from_beliefs(beliefs, answer_len=answer_len)
    candidates = list(pool.values())
    trace = allocate_qa(model, candidates, budget=min(budget, len(candidates)))
    if not hedge_answers:
        trace.notes.append(
            "hedge_answers=False — safe mode for a scoring system that deduplicates on "
            "(video_id, frame_id). See docs/SUBMISSION.md."
        )
    answers = [
        Answer(
            video_id=allocation.video_id,
            frame=allocation.frame_id,
            answer=allocation.answer,
        )
        for allocation in trace.allocations
    ]
    return QaSolution(
        submission=QuerySubmission(query_id, "qa", answers),
        trace=trace,
        beliefs=beliefs,
    )
