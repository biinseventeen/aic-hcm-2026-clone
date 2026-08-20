"""The coverage model must be monotone and submodular — that is what makes greedy sound."""

import random

import pytest

from aic.core.coverage import CoverageModel, Locus, VideoBelief, interval_union_measure


def test_interval_union_merges_adjacent_intervals():
    assert interval_union_measure([]) == 0
    assert interval_union_measure([(0, 4)]) == 5
    # [0,4] and [5,9] are adjacent over the integers, so the union is [0,9], not two pieces.
    assert interval_union_measure([(0, 4), (5, 9)]) == 10
    assert interval_union_measure([(0, 4), (3, 6), (10, 10)]) == 8
    assert interval_union_measure([(0, 9), (2, 3)]) == 10


def test_an_empty_locus_is_rejected():
    with pytest.raises(ValueError):
        Locus(10, 9)
    with pytest.raises(ValueError):
        Locus(0, 10, weight=-1.0)


def test_pi_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        VideoBelief("v", 1.5, [Locus(0, 9)])


def test_kappa_closed_form_values():
    """Locus [0,99] with L=25: one frame at a window's right edge covers exactly 25/100."""
    belief = VideoBelief("L21_V001", pi=1.0, loci=[Locus(0, 99)])
    assert belief.kappa([24], 25) == pytest.approx(0.25)
    assert belief.kappa([24, 49], 25) == pytest.approx(0.50)
    assert belief.kappa([24, 49, 74, 99], 25) == pytest.approx(1.0)
    # Two frames less than L apart overlap: [0,24] union [1,25] intersect [0,99] = 26 positions.
    assert belief.kappa([24, 25], 25) == pytest.approx(0.26)
    # A repeated frame adds nothing.
    assert belief.kappa([24, 24], 25) == pytest.approx(0.25)


def test_frames_outside_the_locus_contribute_nothing():
    belief = VideoBelief("v", pi=1.0, loci=[Locus(100, 199)])
    assert belief.kappa([50], 25) == 0.0
    # Frame 124 covers start positions [100, 124] -> 25 positions.
    assert belief.kappa([124], 25) == pytest.approx(0.25)


def test_locus_weights_are_normalised():
    belief = VideoBelief("v", pi=1.0, loci=[Locus(0, 99, 3.0), Locus(1000, 1099, 1.0)])
    assert belief.kappa([24], 25) == pytest.approx(0.75 * 0.25)
    assert belief.kappa([1024], 25) == pytest.approx(0.25 * 0.25)
    assert belief.kappa([24, 49, 74, 99, 1024], 25) == pytest.approx(0.75 + 0.25 * 0.25)


def test_kappa_never_exceeds_one():
    belief = VideoBelief("v", pi=1.0, loci=[Locus(0, 9)])
    assert belief.kappa(range(10), 50) == pytest.approx(1.0)


def test_kappa_rejects_a_zero_answer_length():
    with pytest.raises(ValueError):
        VideoBelief("v", 1.0, [Locus(0, 9)]).kappa([0], 0)


def _random_model(rng: random.Random) -> tuple[CoverageModel, list[tuple[str, int]]]:
    beliefs, candidates = [], []
    for index in range(rng.randint(1, 3)):
        video_id = f"L21_V{index:03d}"
        loci = [
            Locus(
                start := rng.randrange(0, 200),
                start + rng.randrange(5, 60),
                rng.random() + 0.1,
            )
            for _ in range(rng.randint(1, 3))
        ]
        beliefs.append(VideoBelief(video_id, pi=0.1 + 0.8 * rng.random(), loci=loci))
        candidates += [(video_id, rng.randrange(0, 260)) for _ in range(6)]
    model = CoverageModel.from_beliefs(beliefs, answer_len=rng.choice([5, 13, 25]))
    return model, candidates


def test_monotone_and_submodular_over_random_instances():
    rng = random.Random(0)
    for _ in range(60):
        model, candidates = _random_model(rng)
        model = model.normalized()
        for _ in range(15):
            size = rng.randrange(0, len(candidates) + 1)
            selection: dict[str, list[int]] = {}
            for video_id, frame in rng.sample(candidates, size):
                selection.setdefault(video_id, []).append(frame)
            video_id, frame = rng.choice(candidates)
            small_gain = model.marginal_gain(selection, video_id, frame)
            # Monotone: marginal gain is never negative.
            assert small_gain >= -1e-12
            # Submodular: the marginal gain shrinks as the selected set grows.
            larger = dict(selection)
            for other_video, other_frame in rng.sample(candidates, rng.randrange(0, 4)):
                larger[other_video] = [*larger.get(other_video, []), other_frame]
            assert model.marginal_gain(larger, video_id, frame) <= small_gain + 1e-12


def test_marginal_gain_matches_the_difference_of_probabilities():
    """The closed-form marginal gain must equal recomputing the whole sum."""
    rng = random.Random(7)
    for _ in range(40):
        model, candidates = _random_model(rng)
        model = model.normalized()
        selection: dict[str, list[int]] = {}
        for video_id, frame in rng.sample(candidates, min(5, len(candidates))):
            selection.setdefault(video_id, []).append(frame)
        video_id, frame = rng.choice(candidates)
        after = {**selection, video_id: [*selection.get(video_id, []), frame]}
        assert model.marginal_gain(selection, video_id, frame) == pytest.approx(
            model.probability(after) - model.probability(selection), abs=1e-12
        )


def test_normalized_makes_pi_sum_to_one():
    model = CoverageModel.from_beliefs(
        [VideoBelief("a", 3.0 / 4, [Locus(0, 9)]), VideoBelief("b", 1.0 / 4, [Locus(0, 9)])]
    )
    assert sum(b.pi for b in model.normalized().beliefs.values()) == pytest.approx(1.0)


def test_upper_bound_gain_really_is_an_upper_bound():
    rng = random.Random(3)
    for _ in range(40):
        model, candidates = _random_model(rng)
        model = model.normalized()
        selection: dict[str, list[int]] = {}
        for video_id, frame in rng.sample(candidates, min(4, len(candidates))):
            selection.setdefault(video_id, []).append(frame)
        for video_id, frame in candidates:
            assert model.marginal_gain(selection, video_id, frame) <= (
                model.upper_bound_gain(selection, video_id) + 1e-12
            )
