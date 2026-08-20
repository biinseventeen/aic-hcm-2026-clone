"""P1 — the frame index convention. An error here is silent score loss, so lock it with tests."""

import pytest

from aic.core.frameidx import (
    DEFAULT_CONVENTION,
    check_convention,
    frame_to_time,
    grid_alignment,
    seconds_to_frames,
    time_to_frame,
)


def test_the_default_convention_is_floor():
    """Verified on 177,321/177,321 batch 1 keyframes. Do not change this silently."""
    assert DEFAULT_CONVENTION == "floor"


def test_floor_differs_from_round_on_a_real_case():
    """11.7333 s * 30 fps = 351.999 — floor gives 351, round gives 352."""
    assert time_to_frame(11.7333, 30.0) == 351
    assert time_to_frame(11.7333, 30.0, "round") == 352
    assert time_to_frame(11.7333, 30.0, "ceil") == 352


def test_time_to_frame_reference_values():
    assert time_to_frame(0.0, 25.0) == 0
    assert time_to_frame(3.0, 30.0) == 90
    assert time_to_frame(71.0333, 30.0) == 2130


def test_time_to_frame_rejects_impossible_input():
    for bad in ((-1.0, 25.0), (1.0, 0.0), (1.0, -25.0)):
        with pytest.raises(ValueError):
            time_to_frame(*bad)
    with pytest.raises(ValueError):
        time_to_frame(1.0, 25.0, "banker")  # type: ignore[arg-type]


def test_frame_to_time_at_center_round_trips_at_every_fps():
    """The mid-frame mark is the only form that round-trips reliably through floor."""
    for fps in (23.976, 25.0, 29.97, 30.0, 59.94):
        for frame in (0, 1, 3, 97, 2130, 99999):
            assert time_to_frame(frame_to_time(frame, fps, at_center=True), fps) == frame, (
                fps,
                frame,
            )


def test_the_start_mark_can_be_off_by_one_frame():
    """Records the trap: the start mark plus floor can return f - 1 in floating point."""
    assert time_to_frame(frame_to_time(29, 25.0), 25.0) == 28
    assert time_to_frame(frame_to_time(29, 23.976), 23.976) == 28
    # And it is not rare: thousands of frames drift within the first 200,000.
    drifting = sum(
        1 for frame in range(200_000) if time_to_frame(frame_to_time(frame, 25.0), 25.0) != frame
    )
    assert drifting > 1000


def test_frame_to_time_rejects_impossible_input():
    with pytest.raises(ValueError):
        frame_to_time(-1, 25.0)
    with pytest.raises(ValueError):
        frame_to_time(0, 0.0)


def test_seconds_to_frames_rounds_up():
    """Rounding up is the safe direction: denser coverage costs slots, sparser loses points."""
    assert seconds_to_frames(1.0, 25.0) == 25
    assert seconds_to_frames(0.5, 25.0) == 13
    assert seconds_to_frames(0.001, 25.0) == 1
    assert seconds_to_frames(0.0, 25.0) == 1


def _rows(fps: float, count: int, convention: str = "floor", offset: int = 0):
    """Build ``(n, pts_time, fps, frame_idx)`` rows shaped like map-keyframes/<video>.csv."""
    rows = []
    for index in range(count):
        pts = round(index * 0.4333, 4)
        frame_idx = time_to_frame(pts, fps, convention) + offset  # type: ignore[arg-type]
        rows.append((index + 1, pts, fps, frame_idx))
    return rows


def test_check_convention_recognises_floor_data():
    report = check_convention({"L21_V001": _rows(30.0, 200), "L21_V002": _rows(25.0, 100)})
    assert report.n_checked == 300
    assert report.best_convention == "floor"
    assert report.passed
    assert not report.mismatches
    assert "floor" in report.summary()


def test_check_convention_fails_on_mismatching_data():
    report = check_convention({"L21_V001": _rows(30.0, 50, offset=3)})
    assert not report.passed
    assert len(report.mismatches) == 50
    assert "FAIL" in report.summary()


def test_check_convention_records_fps_anomalies():
    """An unusual fps, and mixed fps within one video, must surface in the report."""
    mixed = _rows(25.0, 10) + _rows(30.0, 10)
    report = check_convention({"L21_V001": mixed, "L21_V002": _rows(17.3, 5)})
    assert [video_id for video_id, _ in report.multi_fps] == ["L21_V001"]
    assert "L21_V002" in [video_id for video_id, _ in report.unusual_fps]


def test_check_convention_records_duplicate_frame_indices():
    """192 of 873 batch 1 videos have two keyframes on one frame index."""
    rows = [(1, 0.0, 30.0, 0), (2, 0.0333333, 30.0, 0), (3, 1.0, 30.0, 30)]
    report = check_convention({"L21_V006": rows})
    assert report.duplicate_frame_idx == [("L21_V006", 1)]
    assert "NOT injective" in report.summary()


def test_grid_alignment_detects_cfr_and_off_grid():
    fps = 25.0
    on_grid = [round(index / fps, 4) for index in range(0, 500, 5)]
    aligned = grid_alignment(on_grid, fps)
    assert aligned["off_grid_rate"] == pytest.approx(0.0)
    assert aligned["is_vfr"] == 0.0
    assert aligned["n"] == len(on_grid)

    shifted = [pts + 0.017 * (1 + index % 3) for index, pts in enumerate(on_grid)]
    misaligned = grid_alignment(shifted, fps)
    assert misaligned["off_grid_rate"] > 0.5
    assert misaligned["max_dev_frames"] > 0.25


def test_grid_alignment_handles_an_empty_input():
    result = grid_alignment([], 25.0)
    assert result["n"] == 0.0
    assert result["is_vfr"] == 0.0
