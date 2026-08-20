"""Verifying lazy greedy (CELF) — the part of the pipeline most likely to be subtly wrong.

Lazy greedy is only correct because the objective is submodular: the value popped from the heap is
an *upper bound* on the true gain, so when a recomputed gain is still no smaller than the next
element's bound, it is the global maximum. This module compares it directly against greedy that
recomputes everything (eager) on small random instances: **the gain sequences must match
element by element**.

Only the *gain* sequence is compared, not the *candidate* sequence: when several candidates have
exactly equal marginal gain — which happens constantly, since every pair of frames at least L apart
covers exactly L positions — greedy has many equally optimal solutions. What affects the score is
the gain sequence, not the identity of the candidates.
"""

import random

import pytest

from aic.core.allocator import (
    AnswerCandidate,
    FrameCandidate,
    TrakeCandidate,
    allocate_kis,
    allocate_qa,
    allocate_trake,
)
from aic.core.coverage import CoverageModel, Locus, VideoBelief
from aic.core.objective import MAX_ANSWERS, THRESHOLDS

# --------------------------------------------------------------------------
# Reference greedy: recompute every gain at every step. Slow but obviously correct.
# --------------------------------------------------------------------------


def eager_kis(model: CoverageModel, candidates, budget):
    coverage = model.normalized()
    pool = {}
    for candidate in candidates:
        if candidate.key() not in pool or candidate.score > pool[candidate.key()].score:
            pool[candidate.key()] = candidate
    order = {key: index for index, key in enumerate(pool)}
    selection: dict[str, list[int]] = {}
    gains = []
    for _ in range(min(budget, len(pool))):
        best_key, best = None, None
        for key, candidate in pool.items():
            gain = coverage.marginal_gain(selection, candidate.video_id, candidate.frame_id)
            ranked = (gain, candidate.score, -order[key])
            if best is None or ranked > best:
                best, best_key = ranked, key
        candidate = pool.pop(best_key)
        selection.setdefault(candidate.video_id, []).append(candidate.frame_id)
        gains.append(best[0])
    return gains, selection


def _instance(rng):
    beliefs, candidates = [], []
    for index in range(rng.randint(1, 4)):
        video_id = f"L21_V{index:03d}"
        loci = [
            Locus(
                start := rng.randrange(0, 300),
                start + rng.randrange(8, 80),
                0.1 + rng.random(),
            )
            for _ in range(rng.randint(1, 3))
        ]
        beliefs.append(VideoBelief(video_id, pi=0.05 + 0.9 * rng.random(), loci=loci))
        for _ in range(rng.randint(4, 10)):
            candidates.append(
                FrameCandidate(video_id, rng.randrange(0, 380), score=round(rng.random(), 3))
            )
    model = CoverageModel.from_beliefs(beliefs, answer_len=rng.choice([4, 12, 25]))
    return model, candidates


@pytest.mark.parametrize("seed", range(25))
def test_lazy_greedy_matches_eager_greedy(seed):
    rng = random.Random(seed)
    model, candidates = _instance(rng)
    budget = rng.randint(1, 20)
    trace = allocate_kis(model, candidates, budget=budget)
    expected, _ = eager_kis(model, candidates, budget)
    actual = [allocation.gain for allocation in trace.allocations]
    assert len(actual) == len(expected)
    for index, (lazy, eager) in enumerate(zip(actual, expected, strict=True)):
        assert lazy == pytest.approx(eager, abs=1e-12), f"diverged at slot {index + 1}"


@pytest.mark.parametrize("seed", range(25))
def test_cumulative_matches_the_true_probability(seed):
    """``cumulative`` is a running sum of gains; it must equal Pr(hit) recomputed from scratch."""
    rng = random.Random(1000 + seed)
    model, candidates = _instance(rng)
    trace = allocate_kis(model, candidates, budget=rng.randint(1, 25))
    selection: dict[str, list[int]] = {}
    normalised = model.normalized()
    for allocation in trace.allocations:
        selection.setdefault(allocation.video_id, []).append(allocation.frame_id)
        assert allocation.cumulative == pytest.approx(
            normalised.probability(selection), abs=1e-9
        ), f"slot {allocation.rank}"


@pytest.mark.parametrize("seed", range(15))
def test_gains_never_increase_with_rank(seed):
    """A consequence of submodularity: the greedy gain sequence must be non-increasing."""
    rng = random.Random(2000 + seed)
    model, candidates = _instance(rng)
    gains = [a.gain for a in allocate_kis(model, candidates, budget=30).allocations]
    for index in range(1, len(gains)):
        assert gains[index] <= gains[index - 1] + 1e-12, f"gain rose at slot {index + 1}"


def test_tie_break_picks_the_anchor_frame_first():
    """H3: on exactly equal gains, slot 1 must go to the highest-scoring frame.

    Locus [0, 99] with L = 25; frames 24, 49, 74 and 99 each cover exactly 25 positions, so their
    gains are *equal*. If insertion order decided, slot 1 would land on frame 24 rather than the
    anchor.
    """
    model = CoverageModel.from_beliefs(
        [VideoBelief("L21_V001", 1.0, [Locus(0, 99)])], answer_len=25
    )
    candidates = [
        FrameCandidate("L21_V001", 24, score=0.10),
        FrameCandidate("L21_V001", 49, score=0.90),  # the anchor
        FrameCandidate("L21_V001", 74, score=0.20),
        FrameCandidate("L21_V001", 99, score=0.30),
    ]
    for seed in range(6):
        shuffled = candidates[:]
        random.Random(seed).shuffle(shuffled)
        trace = allocate_kis(model, shuffled, budget=4)
        assert trace.allocations[0].frame_id == 49, [a.frame_id for a in trace.allocations]
        # The locus is fully covered after 4 slots regardless of input order.
        assert trace.allocations[-1].cumulative == pytest.approx(1.0)


def test_the_result_does_not_depend_on_candidate_order():
    """Shuffling the input must not change the objective value.

    If it does, there is a hidden order dependency somewhere in the allocator.
    """
    rng = random.Random(11)
    model, candidates = _instance(rng)
    baseline = allocate_kis(model, candidates, budget=20)
    for seed in range(5):
        shuffled = candidates[:]
        random.Random(seed).shuffle(shuffled)
        trace = allocate_kis(model, shuffled, budget=20)
        assert trace.allocations[-1].cumulative == pytest.approx(
            baseline.allocations[-1].cumulative, abs=1e-12
        )


def test_lazy_greedy_recomputes_far_less_than_eager():
    """The reason CELF exists: recomputations must be far below budget * |pool|."""
    rng = random.Random(5)
    beliefs = [
        VideoBelief(f"L21_V{index:03d}", 0.1 + 0.9 * rng.random(), [Locus(0, 999)])
        for index in range(20)
    ]
    model = CoverageModel.from_beliefs(beliefs, answer_len=25)
    candidates = [
        FrameCandidate(belief.video_id, frame, score=rng.random())
        for belief in beliefs
        for frame in range(24, 1000, 25)
    ]
    trace = allocate_kis(model, candidates, budget=MAX_ANSWERS)
    assert len(trace.allocations) == MAX_ANSWERS
    assert trace.n_reevaluations < MAX_ANSWERS * len(candidates) / 4


def test_running_out_of_candidates_is_recorded_not_silent():
    model = CoverageModel.from_beliefs(
        [VideoBelief("L21_V001", 1.0, [Locus(0, 99)])], answer_len=25
    )
    trace = allocate_kis(model, [FrameCandidate("L21_V001", 24)], budget=10)
    assert len(trace.allocations) == 1
    assert trace.exhausted_at == 2
    assert trace.notes


def test_duplicate_candidates_do_not_burn_a_slot():
    model = CoverageModel.from_beliefs(
        [VideoBelief("L21_V001", 1.0, [Locus(0, 99)])], answer_len=25
    )
    candidates = [
        FrameCandidate("L21_V001", 24, score=0.1),
        FrameCandidate("L21_V001", 24, score=0.9),
    ]
    trace = allocate_kis(model, candidates, budget=5)
    assert [a.frame_id for a in trace.allocations] == [24]
    assert trace.allocations[0].gain > 0


def test_a_budget_over_the_rules_limit_is_rejected():
    model = CoverageModel.from_beliefs([VideoBelief("v", 1.0, [Locus(0, 9)])])
    with pytest.raises(ValueError):
        allocate_kis(model, [FrameCandidate("v", 0)], budget=MAX_ANSWERS + 1)
    with pytest.raises(ValueError):
        allocate_kis(model, [FrameCandidate("v", 0)], budget=0)


def test_no_candidates_returns_an_empty_trace_without_raising():
    model = CoverageModel.from_beliefs([VideoBelief("v", 1.0, [Locus(0, 9)])])
    trace = allocate_kis(model, [], budget=5)
    assert trace.allocations == []
    assert trace.expected_final == 0.0
    assert set(trace.coverage_at_k) == set(THRESHOLDS)


def test_hedging_across_videos():
    """Once one video is fully covered, the next slot must move to another video."""
    model = CoverageModel.from_beliefs(
        [
            VideoBelief("L21_V001", 0.6, [Locus(0, 24)]),
            VideoBelief("L21_V002", 0.4, [Locus(0, 24)]),
        ],
        answer_len=25,
    )
    candidates = [FrameCandidate("L21_V001", 24), FrameCandidate("L21_V002", 24)]
    trace = allocate_kis(model, candidates, budget=2)
    assert [a.video_id for a in trace.allocations] == ["L21_V001", "L21_V002"]
    assert trace.n_distinct_videos == 2
    assert trace.allocations[-1].cumulative == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Q&A
# --------------------------------------------------------------------------


def test_qa_hedges_along_the_answer_axis():
    """H5: once the best (video, frame) is fully covered, the best gain is another answer."""
    model = CoverageModel.from_beliefs(
        [VideoBelief("L21_V001", 1.0, [Locus(0, 24)])], answer_len=25
    )
    candidates = [
        AnswerCandidate("L21_V001", 24, "hai", answer_prob=0.6, score=0.9),
        AnswerCandidate("L21_V001", 24, "ba", answer_prob=0.3, score=0.5),
        AnswerCandidate("L21_V001", 20, "hai", answer_prob=0.6, score=0.8),
    ]
    trace = allocate_qa(model, candidates, budget=2)
    assert [a.answer for a in trace.allocations] == ["hai", "ba"]
    assert trace.allocations[0].gain == pytest.approx(0.6)
    assert trace.allocations[1].gain == pytest.approx(0.3)


def test_qa_never_prefers_a_zero_probability_answer():
    model = CoverageModel.from_beliefs([VideoBelief("v", 1.0, [Locus(0, 24)])], answer_len=25)
    candidates = [
        AnswerCandidate("v", 24, "wrong", answer_prob=0.0, score=0.99),
        AnswerCandidate("v", 24, "right", answer_prob=0.5, score=0.01),
    ]
    trace = allocate_qa(model, candidates, budget=2)
    assert trace.allocations[0].answer == "right"


def test_qa_gains_never_increase():
    rng = random.Random(4)
    raw = [(f"L21_V{index:03d}", 0.1 + rng.random()) for index in range(4)]
    total = sum(pi for _, pi in raw)
    beliefs = [VideoBelief(video_id, pi / total, [Locus(0, 399)]) for video_id, pi in raw]
    model = CoverageModel.from_beliefs(beliefs, answer_len=20)
    candidates = [
        AnswerCandidate(belief.video_id, frame, answer, answer_prob=prob, score=rng.random())
        for belief in beliefs
        for frame in range(19, 400, 20)
        for answer, prob in (("a", 0.5), ("b", 0.3))
    ]
    gains = [a.gain for a in allocate_qa(model, candidates, budget=MAX_ANSWERS).allocations]
    assert len(gains) == MAX_ANSWERS
    for index in range(1, len(gains)):
        assert gains[index] <= gains[index - 1] + 1e-12


def test_qa_rejects_a_budget_over_the_rules_limit():
    model = CoverageModel.from_beliefs([VideoBelief("v", 1.0, [Locus(0, 9)])])
    with pytest.raises(ValueError):
        allocate_qa(model, [AnswerCandidate("v", 0, "x")], budget=MAX_ANSWERS + 1)


# --------------------------------------------------------------------------
# TRAKE
# --------------------------------------------------------------------------


def test_trake_preserves_the_moment_count():
    candidates = [
        TrakeCandidate("L21_V001", (10, 60, 110), score=0.9, moment_conf=(0.9, 0.5, 0.8)),
        TrakeCandidate("L21_V002", (20, 70, 120), score=0.4, moment_conf=(0.4, 0.4, 0.4)),
    ]
    trace = allocate_trake(
        candidates,
        video_pi={"L21_V001": 0.7, "L21_V002": 0.3},
        answer_len=10,
        budget=20,
        n_samples=200,
    )
    assert trace.allocations
    for allocation in trace.allocations:
        assert len(allocation.frames) == 3, allocation.frames
        assert list(allocation.frames) == sorted(allocation.frames)
    assert trace.allocations[0].video_id == "L21_V001"


def test_trake_is_deterministic_for_a_given_seed():
    candidates = [
        TrakeCandidate(
            f"L21_V{index:03d}",
            (10 + index, 60 + index, 110 + index),
            score=1.0 - index / 10,
            moment_conf=(0.6, 0.5, 0.7),
        )
        for index in range(3)
    ]
    video_pi = dict.fromkeys((c.video_id for c in candidates), 1.0 / 3)
    runs = [
        [
            allocation.frames
            for allocation in allocate_trake(
                candidates,
                video_pi=video_pi,
                answer_len=10,
                budget=15,
                n_samples=300,
                seed=0,
            ).allocations
        ]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_trake_rejects_tuples_of_differing_length():
    candidates = [
        TrakeCandidate("L21_V001", (10, 60, 110)),
        TrakeCandidate("L21_V002", (20, 70)),
    ]
    with pytest.raises(ValueError, match="same N moments"):
        allocate_trake(candidates, video_pi={"L21_V001": 0.5, "L21_V002": 0.5})


def test_trake_with_all_zero_priors_reports_retrieval_failure():
    candidates = [TrakeCandidate("L21_V001", (10, 60))]
    trace = allocate_trake(candidates, video_pi={"L21_V001": 0.0})
    assert trace.allocations == []
    assert any("retrieval layer failed" in note for note in trace.notes)
