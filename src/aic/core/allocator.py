"""P13 — allocating the 100 answer slots.

The module with the highest score-per-compute ratio in the system: no GPU, no model, runs
in milliseconds, and its effect *multiplies* the quality of everything upstream.

The algorithm is greedy maximisation of the submodular coverage function defined in
:mod:`aic.core.coverage`.

A note on band weights
----------------------
The pseudocode in DESIGN.md section 7 writes ``argmax w(j) * Delta(c)``. Since ``w(j)`` is
constant *across candidates* at slot j, it cancels out of the argmax:

    argmax_c w(j) * Delta(c)  ==  argmax_c Delta(c)

So plain greedy on marginal gain is correct and ``w(j)`` need not appear in the loop. This
does *not* make ``w(j)`` irrelevant: because the coverage function is submodular, the
sequence of marginal gains produced by greedy is non-increasing, and ``w(j)`` is also
non-increasing — by the rearrangement inequality, greedy automatically pairs the largest
gains with the largest weights. The band weights justify the algorithm rather than
participating in it.

Lazy greedy (CELF)
------------------
All three variants use lazy greedy: keep a heap of ``(-gain, candidate)`` pairs whose gains
are *stale*. Pop the top, recompute its gain; if it is still no smaller than the estimated
gain of the next element, accept it, otherwise push it back. Because the coverage function
is **submodular**, gains only shrink as the selected set grows, so a stale gain is always an
upper bound — the procedure yields **exactly the same result** as exhaustive greedy, orders
of magnitude faster. The first implementation rescanned every candidate and re-sorted every
video at *each* slot: 4.86 s for one KIS query, and Q&A did not finish at all because each
candidate evaluation recomputed the whole probability sum.

Three emergent behaviours (no hand-written rule produces them)
--------------------------------------------------------------
1. Slot 1 receives the candidate with the largest ``pi_v * kappa``.
2. As ``kappa_v`` of the leading video approaches 1, the marginal gain of adding another
   frame to that video collapses and the algorithm moves to the runner-up on its own.
3. The last slots land on videos with low ``pi_v`` that are not covered at all, because at
   ``kappa_v = 0`` the marginal gain is still the full ``pi_v``.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .coverage import CoverageModel
from .objective import MAX_ANSWERS, THRESHOLDS, band_weight, expected_final_score

__all__ = [
    "Allocation",
    "AllocationTrace",
    "AnswerCandidate",
    "FrameCandidate",
    "TrakeCandidate",
    "allocate_kis",
    "allocate_qa",
    "allocate_trake",
]

#: Gains equal to within this tolerance are treated as tied. Ties are the common case, not
#: an edge case: any two frames at least L apart cover exactly L start positions each. The
#: tolerance is only used for *reporting*; the selection itself resolves ties through the heap
#: ordering, which is exact — see :func:`_pick_next`.
GAIN_TIE_EPS = 1e-15


def _pick_next(
    heap: list[tuple[float, float, int, Any]],
    *,
    gain_of: Callable[[Any], float],
    score_of: Callable[[Any], float],
    trace: AllocationTrace,
    available: Callable[[Any], bool] | None = None,
) -> tuple[Any, float] | None:
    """One lazy-greedy (CELF) pick. Returns ``(key, gain)``, or ``None`` when nothing is left.

    The heap is ordered by ``(-gain, -score, insertion order)``, so among equal gains the
    higher-scoring candidate surfaces first — the property H3 needs, so that slot 1 lands on the
    anchor frame of a locus rather than an arbitrary frame inside it.

    An entry is accepted only once its gain has been recomputed against the **current**
    selection, tracked here as *clean*. Comparing a freshly computed gain against the *stale*
    gain of the next entry instead is what used to make this loop spin forever: a candidate whose
    true gain was the largest by ~1e-18 but whose score lost the tie-break was pushed back
    unchanged, returned to the top of the heap, and popped again — measured at 399,645 pops of a
    single key on query ``p1-6``, with no way out. Accepting a clean top entry is still the true
    greedy maximum, because a stale gain is an upper bound on the current one (coverage is
    submodular, so gains only shrink as the selection grows). Every iteration either returns or
    turns one stale entry clean, so a slot costs at most ``2 * len(heap)`` pops.
    """
    clean: set[Any] = set()
    while heap:
        neg_gain, _neg_score, order, key = heapq.heappop(heap)
        if available is not None and not available(key):
            continue
        if key in clean:
            return key, -neg_gain
        gain = gain_of(key)
        trace.n_reevaluations += 1
        if not heap:
            return key, gain
        clean.add(key)
        heapq.heappush(heap, (-gain, -score_of(key), order, key))
    return None


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class FrameCandidate:
    """A Textual KIS candidate: one (video, frame) pair."""

    video_id: str
    frame_id: int
    #: raw confidence score, used only for tie-breaking and diagnostics.
    score: float = 0.0
    source: str = ""

    def key(self) -> tuple[str, int]:
        return (self.video_id, self.frame_id)


@dataclass(slots=True, frozen=True)
class AnswerCandidate:
    """A Q&A candidate: one (video, frame, answer) triple.

    ``answer_prob`` is the *calibrated* probability that ``answer`` matches the judges'
    answer semantically. Different answers to the same query are treated as **mutually
    exclusive** — only one can be right — so their coverage probabilities add linearly
    rather than multiplying.
    """

    video_id: str
    frame_id: int
    answer: str
    answer_prob: float = 1.0
    score: float = 0.0
    source: str = ""

    def key(self) -> tuple[str, int, str]:
        return (self.video_id, self.frame_id, self.answer)


@dataclass(slots=True, frozen=True)
class TrakeCandidate:
    """A TRAKE candidate: one video plus exactly N ordered frame indices."""

    video_id: str
    frames: tuple[int, ...]
    score: float = 0.0
    #: per-moment confidence, used to choose which moment to vary.
    moment_conf: tuple[float, ...] = ()
    source: str = ""

    def key(self) -> tuple[str, tuple[int, ...]]:
        return (self.video_id, self.frames)


@dataclass(slots=True)
class Allocation:
    """One row of the submission list, with a diagnostic trail."""

    rank: int
    video_id: str
    frame_id: int | None = None
    frames: tuple[int, ...] = ()
    answer: str | None = None
    #: marginal gain at the moment this candidate was chosen.
    gain: float = 0.0
    #: cumulative Pr(hit) after this slot was filled.
    cumulative: float = 0.0
    source: str = ""

    @property
    def band_weight(self) -> float:
        return band_weight(self.rank)


@dataclass(slots=True)
class AllocationTrace:
    """Diagnostic trail of one allocation run.

    Without this trail, a hyperparameter change cannot be attributed to a cause, so tuning
    loses its measurement basis.
    """

    allocations: list[Allocation] = field(default_factory=list)
    #: Pr(hit) at each cut-off k in THRESHOLDS.
    coverage_at_k: dict[int, float] = field(default_factory=dict)
    expected_final: float = 0.0
    n_distinct_videos: int = 0
    exhausted_at: int | None = None
    #: number of gain recomputations — the efficiency metric of lazy greedy.
    n_reevaluations: int = 0
    notes: list[str] = field(default_factory=list)

    def finalize(self) -> AllocationTrace:
        for allocation in self.allocations:
            if allocation.rank in THRESHOLDS:
                self.coverage_at_k[allocation.rank] = allocation.cumulative
        last = self.allocations[-1].cumulative if self.allocations else 0.0
        running = 0.0
        for k in THRESHOLDS:
            if k in self.coverage_at_k:
                running = self.coverage_at_k[k]
            else:
                # Candidates ran out before cut-off k: hold the last value reached.
                self.coverage_at_k[k] = last if not self.allocations else running or last
        self.expected_final = expected_final_score(self.coverage_at_k)
        self.n_distinct_videos = len({a.video_id for a in self.allocations})
        return self


# --------------------------------------------------------------------------
# KIS
# --------------------------------------------------------------------------


def allocate_kis(
    model: CoverageModel,
    candidates: Sequence[FrameCandidate],
    *,
    budget: int = MAX_ANSWERS,
    normalize: bool = True,
    seed_candidates: Sequence[FrameCandidate] = (),
) -> AllocationTrace:
    """Greedy coverage for Textual KIS, using lazy greedy.

    ``seed_candidates`` are optional forced early picks. They are added to the
    coverage state before normal greedy allocation continues, so subsequent
    marginal gains are computed against the already-seeded selection.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if budget > MAX_ANSWERS:
        raise ValueError(f"the rules cap answers at {MAX_ANSWERS}, requested {budget}")

    coverage = model.normalized() if normalize else model

    # Deduplicate: submitting the same (video, frame) twice burns a slot.
    pool: dict[tuple[str, int], FrameCandidate] = {}
    for candidate in candidates:
        previous = pool.get(candidate.key())
        if previous is None or candidate.score > previous.score:
            pool[candidate.key()] = candidate

    trace = AllocationTrace()
    if not pool:
        trace.notes.append("no candidates")
        return trace.finalize()

    selection: dict[str, list[int]] = {}
    cumulative = 0.0

    # Optional forced coverage floor. Seeds are accounted for by the coverage
    # model before greedy allocation begins, so later marginal gains remain valid.
    seeded = 0

    for seed in seed_candidates:
        if seeded >= budget:
            break

        key = seed.key()
        candidate = pool.pop(key, None)

        # Ignore duplicate / unavailable seeds gracefully.
        if candidate is None:
            continue

        gain = coverage.marginal_gain(
            selection,
            candidate.video_id,
            candidate.frame_id,
        )

        selection.setdefault(candidate.video_id, []).append(candidate.frame_id)

        cumulative += gain
        seeded += 1

        trace.allocations.append(
            Allocation(
                rank=seeded,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                gain=gain,
                cumulative=min(1.0, cumulative),
                source=f"{candidate.source}|video-floor",
            )
        )

    # The heap holds (-stale_gain, -score, stable_order, key). Initial gains are
    # computed against the current selection (including any seeded candidates).
    heap: list[tuple[float, float, int, tuple[str, int]]] = []
    order = {key: i for i, key in enumerate(pool)}

    for key, candidate in pool.items():
        gain = coverage.marginal_gain(
            selection,
            candidate.video_id,
            candidate.frame_id,
        )
        heap.append((-gain, -candidate.score, order[key], key))

    heapq.heapify(heap)

    for rank in range(seeded + 1, budget + 1):
        picked = _pick_next(
            heap,
            gain_of=lambda key: coverage.marginal_gain(
                selection,
                pool[key].video_id,
                pool[key].frame_id,
            ),
            score_of=lambda key: pool[key].score,
            trace=trace,
        )

        chosen, chosen_gain = picked if picked is not None else (None, 0.0)

        if chosen is None:
            trace.exhausted_at = rank
            trace.notes.append(
                f"ran out of distinct candidates at slot {rank}: only {len(pool)} "
                "(video, frame) pairs available. Increase frame spread density to fill "
                "all 100 slots (H4)."
            )
            break

        candidate = pool.pop(chosen)
        selection.setdefault(candidate.video_id, []).append(candidate.frame_id)

        cumulative += chosen_gain

        trace.allocations.append(
            Allocation(
                rank=rank,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                gain=chosen_gain,
                cumulative=min(1.0, cumulative),
                source=candidate.source,
            )
        )

    return trace.finalize()


# --------------------------------------------------------------------------
# Q&A
# --------------------------------------------------------------------------


def allocate_qa(
    model: CoverageModel,
    candidates: Sequence[AnswerCandidate],
    *,
    budget: int = MAX_ANSWERS,
    normalize: bool = True,
) -> AllocationTrace:
    """Greedy coverage for Q&A over the (video, frame, answer) product space.

    The marginal gain has a **closed form**; there is no need to recompute the whole
    probability sum. Because

        Pr(hit | S) = sum_v pi_v * sum_a p_a * kappa_v(F_{v,a})

    and answers are mutually exclusive, adding ``(v, f, a)`` touches exactly one term:

        Delta = pi_v * p_a * [ kappa_v(F_{v,a} + f) - kappa_v(F_{v,a}) ]

    Hedging along the answer axis (a consequence of H5) emerges on its own: once the best
    (video, frame) pair already has high ``kappa``, the largest remaining marginal gain is
    to attach a *different answer* to that same frame, because it opens a new term whose
    ``kappa`` starts again from zero.
    """
    if budget > MAX_ANSWERS:
        raise ValueError(f"the rules cap answers at {MAX_ANSWERS}, requested {budget}")
    coverage = model.normalized() if normalize else model

    pool: dict[tuple[str, int, str], AnswerCandidate] = {}
    for candidate in candidates:
        previous = pool.get(candidate.key())
        if previous is None or candidate.score > previous.score:
            pool[candidate.key()] = candidate

    trace = AllocationTrace()
    if not pool:
        trace.notes.append("no candidates")
        return trace.finalize()

    # frames_by_term[(video, answer)] -> frames already chosen for that term.
    frames_by_term: dict[tuple[str, str], list[int]] = {}

    def gain_of(candidate: AnswerCandidate) -> float:
        belief = coverage.beliefs.get(candidate.video_id)
        if belief is None or belief.pi == 0.0 or candidate.answer_prob == 0.0:
            return 0.0
        current = frames_by_term.get((candidate.video_id, candidate.answer), ())
        before = belief.kappa(current, coverage.answer_len) if current else 0.0
        after = belief.kappa([*current, candidate.frame_id], coverage.answer_len)
        return belief.pi * candidate.answer_prob * (after - before)

    heap: list[tuple[float, float, int, tuple[str, int, str]]] = []
    order = {key: i for i, key in enumerate(pool)}
    for key, candidate in pool.items():
        heap.append((-gain_of(candidate), -candidate.score, order[key], key))
    heapq.heapify(heap)

    cumulative = 0.0
    for rank in range(1, budget + 1):
        picked = _pick_next(
            heap,
            gain_of=lambda key: gain_of(pool[key]),
            score_of=lambda key: pool[key].score,
            trace=trace,
        )
        chosen, chosen_gain = picked if picked is not None else (None, 0.0)
        if chosen is None:
            trace.exhausted_at = rank
            trace.notes.append(f"ran out of distinct (video, frame, answer) triples at slot {rank}")
            break
        candidate = pool.pop(chosen)
        frames_by_term.setdefault((candidate.video_id, candidate.answer), []).append(
            candidate.frame_id
        )
        cumulative += chosen_gain
        trace.allocations.append(
            Allocation(
                rank=rank,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                answer=candidate.answer,
                gain=chosen_gain,
                cumulative=min(1.0, cumulative),
                source=candidate.source,
            )
        )
    return trace.finalize()


# --------------------------------------------------------------------------
# TRAKE
# --------------------------------------------------------------------------


def _moment_spans_from_candidates(
    candidates: Sequence[TrakeCandidate],
    video_ids: Sequence[str],
    n_moments: int,
    answer_len: int,
) -> dict[str, list[tuple[int, int]]]:
    """Infer each moment's uncertainty locus from the spread of candidate tuples."""
    spans: dict[str, list[tuple[int, int]]] = {}
    for video_id in video_ids:
        frames_for_video = [c.frames for c in candidates if c.video_id == video_id]
        spans[video_id] = [
            (
                min(frames[j] for frames in frames_for_video) - answer_len,
                max(frames[j] for frames in frames_for_video) + answer_len,
            )
            for j in range(n_moments)
        ]
    return spans


def allocate_trake(
    candidates: Sequence[TrakeCandidate],
    *,
    video_pi: dict[str, float],
    moment_spans: dict[str, list[tuple[int, int]]] | None = None,
    answer_len: int = 10,
    budget: int = MAX_ANSWERS,
    n_samples: int = 2000,
    seed: int = 0,
) -> AllocationTrace:
    """Greedy on ``E[max_i R_i]`` for TRAKE, estimated by vectorised Monte Carlo.

    The TRAKE R-Score is *continuous*: ``(1/N) * sum_j I(f_j in [s_j, e_j])`` on the right
    video, a hard 0 on the wrong one. Because the ``max`` of an average does not decompose
    per moment, there is no closed form for ``E[max_i R_i]`` over a *set* of tuples. So we
    sample the true positions of the N answer windows, build the matrix ``R``
    (n_candidates x n_samples) **once**, and greedy reduces to a numpy ``maximum``: the
    marginal gain of candidate i is ``mean(maximum(best, R[i])) - mean(best)``.

    ``moment_spans[v][j]`` is the uncertainty locus of moment j in video v. When omitted, it
    is inferred from the spread of candidate tuples, widened by ``answer_len``.

    Emergent behaviour: greedy picks tuples that differ at the moment with the highest
    entropy, because that is where ``max`` still has room to grow — exactly the H6 strategy.
    It also reserves an early slot for the runner-up *video*, because the wrong video is a
    total loss.
    """
    if budget > MAX_ANSWERS:
        raise ValueError(f"the rules cap answers at {MAX_ANSWERS}, requested {budget}")
    trace = AllocationTrace()
    if not candidates:
        trace.notes.append("no TRAKE candidates")
        return trace.finalize()

    n_moments = len(candidates[0].frames)
    for candidate in candidates:
        if len(candidate.frames) != n_moments:
            raise ValueError(
                f"every TRAKE tuple must have the same N moments; {candidate.video_id} has "
                f"{len(candidate.frames)} instead of {n_moments}"
            )

    video_ids = sorted({c.video_id for c in candidates})
    total_pi = sum(max(0.0, video_pi.get(v, 0.0)) for v in video_ids)
    if total_pi <= 0:
        trace.notes.append("every pi_v is 0; the retrieval layer failed")
        return trace.finalize()
    pi = {v: max(0.0, video_pi.get(v, 0.0)) / total_pi for v in video_ids}

    spans = dict(moment_spans or {})
    missing = [v for v in video_ids if v not in spans]
    if missing:
        spans.update(_moment_spans_from_candidates(candidates, missing, n_moments, answer_len))

    rng = np.random.default_rng(seed)
    # One sample = the correct video, plus the start positions of its N answer windows.
    index_of_video = {v: i for i, v in enumerate(video_ids)}
    probabilities = np.array([pi[v] for v in video_ids], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    sampled_video = rng.choice(len(video_ids), size=n_samples, p=probabilities)
    starts = np.zeros((n_samples, n_moments), dtype=np.int64)
    for video_id, index in index_of_video.items():
        mask = sampled_video == index
        count = int(mask.sum())
        if count == 0:
            continue
        for j, (lo, hi) in enumerate(spans[video_id]):
            starts[mask, j] = rng.integers(lo, max(lo, hi) + 1, size=count)

    # The R matrix (n_candidates, n_samples), computed once.
    pool = list(candidates)
    r_matrix = np.zeros((len(pool), n_samples), dtype=np.float32)
    for i, candidate in enumerate(pool):
        same_video = sampled_video == index_of_video[candidate.video_id]
        if not same_video.any():
            continue
        frames = np.asarray(candidate.frames, dtype=np.int64)[None, :]  # (1, N)
        sampled_starts = starts[same_video]  # (k, N)
        hit = (sampled_starts <= frames) & (frames <= sampled_starts + answer_len - 1)
        r_matrix[i, same_video] = hit.sum(axis=1) / n_moments

    best = np.zeros(n_samples, dtype=np.float32)
    cumulative = 0.0
    remaining = set(range(len(pool)))
    # The fourth element is the heap key; here it is the candidate index, which doubles as the
    # insertion order, so the tie-break is unchanged.
    heap = [(-float(r_matrix[i].mean()), -pool[i].score, i, i) for i in remaining]
    heapq.heapify(heap)

    for rank in range(1, budget + 1):
        picked = _pick_next(
            heap,
            # best and cumulative are rebound each slot, so they are bound as defaults rather
            # than captured by reference.
            gain_of=lambda index, best=best, cumulative=cumulative: (
                float(np.maximum(best, r_matrix[index]).mean()) - cumulative
            ),
            score_of=lambda index: pool[index].score,
            trace=trace,
            available=remaining.__contains__,
        )
        chosen, chosen_gain = picked if picked is not None else (None, 0.0)
        if chosen is None:
            trace.exhausted_at = rank
            trace.notes.append(f"ran out of distinct TRAKE tuples at slot {rank}")
            break
        candidate = pool[chosen]
        remaining.discard(chosen)
        best = np.maximum(best, r_matrix[chosen])
        cumulative += chosen_gain
        trace.allocations.append(
            Allocation(
                rank=rank,
                video_id=candidate.video_id,
                frames=candidate.frames,
                gain=chosen_gain,
                cumulative=min(1.0, cumulative),
                source=candidate.source,
            )
        )
    trace.notes.append(f"E[max R] estimated over {n_samples} samples, seed={seed}")
    return trace.finalize()
