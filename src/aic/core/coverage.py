"""Hierarchical probability model for *coverage* (DESIGN.md section 1.4).

The task is coverage, not ranking: the correct answer is an *interval* [s, e] hidden
inside a video, and we win if at least one submitted frame falls in that interval.

    Pr(hit | S) = sum_v  pi_v * kappa_v(S_v)

where ``pi_v`` is the probability that video v is the answer and ``kappa_v`` is the
probability of covering the interval *within* video v, given that v is correct. This
factorisation separates the two sources of uncertainty — picking the wrong video, and
picking the wrong frame inside the right video — so that P13 can allocate slots between
the two kinds of hedge quantitatively.

Model for kappa
---------------
Inside a video, belief is represented by a set of *loci* (usually candidate shots), each
with a weight ``q``. Within a locus the answer interval is assumed to have length ``L``
and a uniformly distributed start position. A submitted frame f covers the interval
[s, s+L-1] if and only if s lies in [f-L+1, f]. The set of covered start positions is
therefore a union of intervals of width L, and

    kappa(locus) = measure(union[f-L+1, f] intersect locus) / measure(locus)

An immediate consequence: two frames at least L apart contribute coverage *additively*,
while two frames less than L apart overlap — which is exactly the submodularity that the
greedy allocator in P13 relies on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

__all__ = ["CoverageModel", "Locus", "VideoBelief", "interval_union_measure"]


def interval_union_measure(intervals: Sequence[tuple[int, int]]) -> int:
    """Total length (integer count, both ends closed) of a union of intervals.

    >>> interval_union_measure([(0, 4), (3, 6), (10, 10)])
    8
    """
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end + 1:
            total += current_end - current_start + 1
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start + 1
    return total


@dataclass(slots=True, frozen=True)
class Locus:
    """A candidate time region inside one video, measured in frame indices.

    ``start`` and ``end`` are both inclusive. ``weight`` is the unnormalised belief that
    the answer interval lies inside this locus.
    """

    start: int
    end: int
    weight: float = 1.0
    #: diagnostic label, e.g. shot#123 or asr-window.
    label: str = ""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"empty locus: [{self.start}, {self.end}]")
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def clip(self, start: int, end: int) -> tuple[int, int] | None:
        lo, hi = max(start, self.start), min(end, self.end)
        return (lo, hi) if lo <= hi else None


@dataclass(slots=True)
class VideoBelief:
    """Belief about one video: how likely it is correct, and where inside it."""

    video_id: str
    pi: float
    loci: list[Locus] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.pi <= 1.0:
            raise ValueError(f"pi must lie in [0, 1], got {self.pi} for {self.video_id}")

    @property
    def total_weight(self) -> float:
        return sum(locus.weight for locus in self.loci)

    def kappa(self, frames: Iterable[int], answer_len: int) -> float:
        """Probability of covering the answer inside this video, given ``frames``.

        ``answer_len`` is L, the assumed length of the answer interval in frames.
        """
        if answer_len < 1:
            raise ValueError(f"answer_len must be >= 1, got {answer_len}")
        sorted_frames = sorted(set(frames))
        if not sorted_frames or not self.loci:
            return 0.0
        total = self.total_weight
        if total <= 0:
            return 0.0
        accumulated = 0.0
        for locus in self.loci:
            if locus.weight == 0.0:
                continue
            # A valid start position s lies in the locus; frame f covers s if
            # s is in [f - L + 1, f].
            covered = [
                clipped
                for clipped in (locus.clip(f - answer_len + 1, f) for f in sorted_frames)
                if clipped is not None
            ]
            if covered:
                accumulated += locus.weight * interval_union_measure(covered) / locus.length
        return min(1.0, accumulated / total)


@dataclass(slots=True)
class CoverageModel:
    """Pr(hit | S) over all videos, plus marginal gain in closed form.

    This is the objective that :mod:`aic.core.allocator` maximises. It is **monotone**
    and **submodular** in the set of submitted frames, so greedy carries a (1 - 1/e)
    guarantee.
    """

    beliefs: dict[str, VideoBelief]
    #: L — the assumed length of the answer interval, in frames.
    answer_len: int = 50

    @classmethod
    def from_beliefs(cls, beliefs: Iterable[VideoBelief], **kwargs) -> CoverageModel:
        return cls(beliefs={b.video_id: b for b in beliefs}, **kwargs)

    def normalized(self) -> CoverageModel:
        """A copy with sum(pi) = 1 if the total is positive. P13 assumes pi is a probability."""
        total = sum(b.pi for b in self.beliefs.values())
        if total <= 0:
            return self
        return replace(
            self,
            beliefs={
                video_id: replace(belief, pi=belief.pi / total)
                for video_id, belief in self.beliefs.items()
            },
        )

    def probability(self, selection: dict[str, list[int]]) -> float:
        """Pr(hit | S) where ``selection`` is {video_id: [frame_id, ...]}."""
        total = 0.0
        for video_id, frames in selection.items():
            belief = self.beliefs.get(video_id)
            if belief is None or not frames:
                continue
            total += belief.pi * belief.kappa(frames, self.answer_len)
        return min(1.0, total)

    def marginal_gain(self, selection: dict[str, list[int]], video_id: str, frame_id: int) -> float:
        """Marginal gain of adding (video_id, frame_id) — closed form, O(#loci).

        Only ``video_id`` is affected, so there is no need to recompute the whole sum.
        """
        belief = self.beliefs.get(video_id)
        if belief is None or belief.pi == 0.0:
            return 0.0
        current = selection.get(video_id, [])
        before = belief.kappa(current, self.answer_len)
        after = belief.kappa([*current, frame_id], self.answer_len)
        return belief.pi * (after - before)

    def upper_bound_gain(self, selection: dict[str, list[int]], video_id: str) -> float:
        """Ceiling on the remaining marginal gain of one video: pi_v * (1 - kappa_v).

        Used by lazy greedy: if this ceiling is below the best gain found so far, every
        frame of that video can be skipped without being evaluated.
        """
        belief = self.beliefs.get(video_id)
        if belief is None:
            return 0.0
        return belief.pi * (1.0 - belief.kappa(selection.get(video_id, []), self.answer_len))
