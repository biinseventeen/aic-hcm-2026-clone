"""The covering guarantee: the generated frame set must cover *every* possible answer position.

This is the most important test in P10. If it breaks, every query whose locus is correct still
scores zero, because the submitted frames fall in the gaps between windows.
"""

import pytest

from aic.core.coverage import Locus
from aic.core.spread import (
    coverage_of,
    covering_frames,
    expand_locus,
    min_frames_to_cover,
    order_anchor_first,
    slots_needed_report,
    spread_frames,
)


def test_min_frames_to_cover():
    assert min_frames_to_cover(250, 10) == 25
    assert min_frames_to_cover(250, 50) == 5
    assert min_frames_to_cover(5, 50) == 1
    assert min_frames_to_cover(1, 1) == 1
    with pytest.raises(ValueError):
        min_frames_to_cover(0, 10)
    with pytest.raises(ValueError):
        min_frames_to_cover(10, 0)


@pytest.mark.parametrize("locus_len", [1, 2, 7, 25, 26, 99, 100, 251])
@pytest.mark.parametrize("answer_len", [1, 3, 10, 25, 50])
def test_covering_frames_covers_every_start_position(locus_len, answer_len):
    """For every (W, L): every start position s inside the locus must be covered."""
    locus = Locus(100, 100 + locus_len - 1)
    frames = covering_frames(locus, answer_len)
    assert len(frames) == min_frames_to_cover(locus_len, answer_len)
    for start in range(locus.start, locus.end + 1):
        # The answer interval is [s, s+L-1]; a frame f covers it when s <= f <= s+L-1.
        assert any(start <= frame <= start + answer_len - 1 for frame in frames), (
            start,
            frames,
            answer_len,
        )
    assert coverage_of(frames, locus, answer_len) == pytest.approx(1.0)


def test_covering_frames_specific_values():
    assert covering_frames(Locus(0, 99), 25) == [24, 49, 74, 99]
    assert covering_frames(Locus(0, 99), 100) == [99]
    assert covering_frames(Locus(10, 19), 5) == [14, 19]


def test_covering_frames_never_exceeds_the_last_frame():
    """Submitting a frame index that does not exist is a guaranteed miss."""
    frames = covering_frames(Locus(100, 149), 25, max_frame=120)
    assert frames == [120]
    assert all(frame <= 120 for frame in frames)


def test_covering_frames_are_never_negative():
    assert all(frame >= 0 for frame in covering_frames(Locus(0, 4), 25))


def test_covering_frames_respects_the_limit():
    frames = covering_frames(Locus(0, 999), 10, limit=5)
    assert len(frames) == 5
    # Partial coverage, not full — and it must cover the *start* of the locus.
    assert frames == [9, 19, 29, 39, 49]


def test_spread_frames_is_even_and_inside_the_locus():
    frames = spread_frames(Locus(0, 99), 25)
    assert frames == [12, 37, 62, 87]
    assert all(0 <= frame <= 99 for frame in frames)
    with pytest.raises(ValueError):
        spread_frames(Locus(0, 99), 0)


def test_spread_frames_returns_at_least_one_frame_for_a_short_locus():
    """Never return empty: an unused slot is forfeited positive expectation."""
    assert spread_frames(Locus(5, 5), 100) == [5]
    assert spread_frames(Locus(5, 5), 100, max_frame=3) == [3]


def test_order_anchor_first():
    assert order_anchor_first(50, [0, 25, 50, 75, 100]) == [50, 25, 75, 0, 100]
    # An anchor not present in the list is still placed first.
    assert order_anchor_first(50, [0, 100])[0] == 50
    # No duplicates.
    assert order_anchor_first(50, [50, 50, 25]) == [50, 25]


def test_order_anchor_first_keeps_every_element():
    frames = [3, 1, 4, 1, 5, 9, 2, 6]
    ordered = order_anchor_first(4, frames)
    assert set(ordered) == set(frames)
    assert len(ordered) == len(set(frames))
    assert ordered[0] == 4


def test_expand_locus_clamps_to_the_boundaries():
    locus = expand_locus(Locus(100, 200, 0.5, "shot#7"), 50)
    assert (locus.start, locus.end) == (50, 250)
    assert locus.weight == 0.5
    assert locus.label == "shot#7"
    # Never negative, and never past the last frame of the video.
    assert expand_locus(Locus(10, 20), 50).start == 0
    assert expand_locus(Locus(10, 20), 50, max_frame=25).end == 25


def test_expand_locus_never_produces_an_empty_locus():
    locus = expand_locus(Locus(100, 200), 0, max_frame=50)
    assert locus.end >= locus.start


def test_covering_cost_scales_as_one_over_l():
    """The numeric evidence for the claim in the module docstring."""
    width = 250
    assert min_frames_to_cover(width, 10) == 25
    assert min_frames_to_cover(width, 25) == 10
    assert min_frames_to_cover(width, 50) == 5


def test_slots_needed_report_states_the_slot_cost():
    report = slots_needed_report(250, 10, fps=25.0)
    assert "25 slots" in report
