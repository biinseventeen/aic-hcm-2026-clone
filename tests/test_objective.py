"""The scoring rules are the one specification everything else serves; they must be exact."""

import pytest

from aic.core.objective import (
    MAX_ANSWERS,
    THRESHOLDS,
    band_of,
    band_sizes,
    band_weight,
    expected_final_score,
    final_score,
    final_score_from_first_hit,
    first_hit_rank,
    r_score_kis,
    r_score_qa,
    r_score_trake,
)


def test_worked_example_from_the_rules():
    """Section 2.2: R-Scores [0.5, 0, 0.8, 0...] give 0.74."""
    assert final_score([0.5, 0.0, 0.8] + [0.0] * 97) == pytest.approx(0.74)


def test_final_score_equals_the_step_function_for_binary_r_scores():
    for rank in range(1, MAX_ANSWERS + 1):
        r_scores = [0.0] * (rank - 1) + [1.0]
        assert final_score(r_scores) == pytest.approx(final_score_from_first_hit(rank)), rank


def test_final_score_never_drops_when_a_correct_answer_moves_earlier():
    baseline = [0.0] * 100
    previous = final_score(baseline)
    for rank in range(MAX_ANSWERS, 0, -1):
        r_scores = list(baseline)
        r_scores[rank - 1] = 1.0
        score = final_score(r_scores)
        assert score >= previous - 1e-12
        previous = score


def test_more_than_100_answers_is_rejected():
    with pytest.raises(ValueError):
        final_score([1.0] * 101)


def test_submitting_fewer_slots_is_not_penalised():
    """A short list is treated as a zero tail, not as an error (H4)."""
    assert final_score([1.0]) == pytest.approx(1.0)
    assert final_score([]) == 0.0


def test_band_weight_and_band_of_agree():
    for rank in range(1, MAX_ANSWERS + 1):
        lo, hi = band_of(rank)
        assert lo <= rank <= hi
        # Every position inside a band carries the same marginal weight.
        assert band_weight(rank) == pytest.approx(band_weight(lo))
    assert sum(band_sizes().values()) == MAX_ANSWERS
    assert band_weight(MAX_ANSWERS + 1) == 0.0


def test_band_of_rejects_out_of_range_ranks():
    for rank in (0, -1, MAX_ANSWERS + 1):
        with pytest.raises(ValueError):
            band_of(rank)


def test_expected_final_score_requires_every_threshold():
    with pytest.raises(ValueError):
        expected_final_score({1: 0.5})
    assert expected_final_score(dict.fromkeys(THRESHOLDS, 1.0)) == pytest.approx(1.0)


def test_r_score_kis_is_inclusive_at_both_ends():
    assert r_score_kis("L21_V001", 500, "L21_V001", (500, 510)) == 1.0
    assert r_score_kis("L21_V001", 510, "L21_V001", (500, 510)) == 1.0
    assert r_score_kis("L21_V001", 511, "L21_V001", (500, 510)) == 0.0
    assert r_score_kis("L21_V002", 505, "L21_V001", (500, 510)) == 0.0


def test_r_score_qa_needs_all_three_conditions():
    truth = {"gt_video": "L21_V001", "gt_span": (800, 900)}
    assert r_score_qa("L21_V001", 850, answer_matches=True, **truth) == 1.0
    assert r_score_qa("L21_V001", 850, answer_matches=False, **truth) == 0.0
    assert r_score_qa("L21_V001", 950, answer_matches=True, **truth) == 0.0
    assert r_score_qa("L21_V002", 850, answer_matches=True, **truth) == 0.0


def test_r_score_trake_is_a_hard_zero_on_the_wrong_video():
    spans = [(95, 105), (145, 155)]
    assert r_score_trake("A_V1", [100, 150], "B_V1", spans) == 0.0
    assert r_score_trake("B_V1", [100, 150], "B_V1", spans) == 1.0
    assert r_score_trake("B_V1", [100, 999], "B_V1", spans) == 0.5
    with pytest.raises(ValueError):
        r_score_trake("B_V1", [100], "B_V1", spans)
    with pytest.raises(ValueError):
        r_score_trake("B_V1", [], "B_V1", [])


def test_first_hit_rank():
    assert first_hit_rank([0.0, 0.0, 1.0]) == 3
    assert first_hit_rank([0.0]) is None
