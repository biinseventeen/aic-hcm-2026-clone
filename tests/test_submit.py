"""Submission — the final gate. A malformed file loses points with no way to detect it later."""

import csv
import io
import zipfile

import pytest

from aic.core.objective import MAX_ANSWERS
from aic.submit.writer import (
    Answer,
    QuerySubmission,
    SubmissionNaming,
    package_submission,
    validate,
    validate_submission_dir,
    write_submission,
)


def _errors(issues):
    return [issue for issue in issues if issue.severity == "error"]


def _warnings(issues):
    return [issue for issue in issues if issue.severity == "warning"]


def _full_kis(n=MAX_ANSWERS):
    return QuerySubmission(
        "1", "kis", [Answer(f"L21_V{1 + i % 30:03d}", frame=100 + i) for i in range(n)]
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_a_full_list_of_100_has_no_issues():
    assert validate(_full_kis()) == []


def test_an_empty_list_is_an_error():
    assert len(_errors(validate(QuerySubmission("1", "kis", [])))) == 1


def test_more_than_100_rows_is_an_error():
    assert _errors(validate(_full_kis(101)))


def test_a_short_list_is_a_warning_not_an_error():
    """H4: an unused slot is forfeited expectation, but it is not a format error."""
    issues = validate(_full_kis(40))
    assert not _errors(issues)
    assert _warnings(issues)


def test_a_malformed_video_id_is_an_error():
    submission = QuerySubmission(
        "1", "kis", [Answer("L21_V001.mp4", frame=1), Answer("bad", frame=2)]
    )
    assert len(_errors(validate(submission))) == 2


def test_a_video_id_absent_from_the_corpus_is_an_error():
    submission = QuerySubmission("1", "kis", [Answer("L99_V999", frame=1)])
    assert _errors(validate(submission, known_videos={"L21_V001"}))
    assert not _errors(validate(submission, known_videos=None))


def test_a_negative_frame_id_is_an_error():
    assert _errors(validate(QuerySubmission("1", "kis", [Answer("L21_V001", frame=-1)])))


def test_a_frame_id_past_the_last_known_frame_is_a_warning():
    submission = QuerySubmission("1", "kis", [Answer("L21_V001", frame=99999)])
    issues = validate(submission, max_frames={"L21_V001": 5000})
    assert not _errors(issues)
    assert any("past the last known frame" in issue.message for issue in _warnings(issues))


def test_an_exactly_duplicated_row_is_an_error():
    submission = QuerySubmission(
        "1", "kis", [Answer("L21_V001", frame=7), Answer("L21_V001", frame=7)]
    )
    assert any("duplicate" in issue.message for issue in _errors(validate(submission)))


def test_a_missing_qa_answer_is_an_error():
    for answer in (None, "", "   "):
        submission = QuerySubmission("1", "qa", [Answer("L21_V001", frame=7, answer=answer)])
        assert _errors(validate(submission)), answer


def test_a_newline_in_an_answer_is_an_error():
    submission = QuerySubmission("1", "qa", [Answer("L21_V001", frame=7, answer="a\nb")])
    assert _errors(validate(submission))


def test_a_very_long_answer_is_a_warning():
    submission = QuerySubmission("1", "qa", [Answer("L21_V001", frame=7, answer="x" * 300)])
    issues = validate(submission)
    assert not _errors(issues)
    assert any("very long" in issue.message for issue in _warnings(issues))


def test_the_wrong_trake_moment_count_is_an_error():
    submission = QuerySubmission("1", "trake", [Answer("L21_V001", frames=(1, 2, 3))], n_moments=4)
    assert _errors(validate(submission))


def test_non_increasing_trake_frames_are_a_warning():
    submission = QuerySubmission(
        "1", "trake", [Answer("L21_V001", frames=(30, 10, 20))], n_moments=3
    )
    issues = validate(submission)
    assert not _errors(issues)
    assert any("not increasing" in issue.message for issue in _warnings(issues))


# --------------------------------------------------------------------------
# Writing and reading back
# --------------------------------------------------------------------------


def test_kis_is_written_in_the_expected_format(tmp_path):
    path, issues = write_submission(_full_kis(), tmp_path)
    assert path.name == "query-1-kis.csv"
    assert not _errors(issues)
    raw = path.read_bytes().decode("utf-8")
    assert raw.startswith("L21_V001,100\r\n")
    rows = [row for row in csv.reader(io.StringIO(raw)) if row]
    assert len(rows) == MAX_ANSWERS
    assert all(len(row) == 2 for row in rows)
    # No header row — the assumption is documented in the writer's docstring.
    assert rows[0][0] != "video_id"


def test_a_comma_inside_a_qa_answer_is_preserved(tmp_path):
    """An answer containing a comma must be quoted, or the CSV structure breaks."""
    submission = QuerySubmission("2", "qa", [Answer("L21_V001", frame=7, answer="một, hai")])
    path, _ = write_submission(submission, tmp_path, strict=False)
    rows = [row for row in csv.reader(io.StringIO(path.read_text(encoding="utf-8"))) if row]
    assert rows == [["L21_V001", "7", "một, hai"]]


def test_trake_writes_one_row_with_several_frames(tmp_path):
    submission = QuerySubmission(
        "3", "trake", [Answer("L21_V001", frames=(10, 20, 30))], n_moments=3
    )
    path, _ = write_submission(submission, tmp_path, strict=False)
    assert path.read_text(encoding="utf-8").strip() == "L21_V001,10,20,30"


def test_strict_blocks_writing_when_there_is_an_error(tmp_path):
    submission = QuerySubmission("1", "kis", [Answer("not-valid", frame=1)])
    with pytest.raises(ValueError):
        write_submission(submission, tmp_path)
    assert not list(tmp_path.glob("*.csv"))


def test_every_format_assumption_is_changeable_in_one_line(tmp_path):
    naming = SubmissionNaming(
        filename="{task}_{query_id}.txt",
        delimiter="\t",
        include_header=True,
        video_extension=".mp4",
        lineterminator="\n",
    )
    submission = QuerySubmission("7", "kis", [Answer("L21_V001", frame=42)])
    path, _ = write_submission(submission, tmp_path, naming=naming, strict=False)
    assert path.name == "kis_7.txt"
    assert path.read_text(encoding="utf-8") == "video_id\tframe_id\nL21_V001.mp4\t42\n"


def test_reading_back_from_disk_surfaces_issues(tmp_path):
    write_submission(_full_kis(3), tmp_path, strict=False)
    issues = validate_submission_dir(tmp_path)
    # It reads back, and the short-list warning still surfaces.
    assert not _errors(issues)
    assert _warnings(issues)


def test_an_empty_directory_is_an_error(tmp_path):
    assert _errors(validate_submission_dir(tmp_path))


def test_a_filename_off_the_pattern_is_a_warning(tmp_path):
    (tmp_path / "results.csv").write_text("L21_V001,1\r\n", encoding="utf-8")
    issues = validate_submission_dir(tmp_path)
    assert any("filename does not match" in issue.message for issue in _warnings(issues))


def test_reading_back_preserves_row_order(tmp_path):
    """Row order is decisive — R@k reads by position."""
    frames = [500, 12, 999, 7]
    submission = QuerySubmission("9", "kis", [Answer("L21_V001", frame=f) for f in frames])
    path, _ = write_submission(submission, tmp_path, strict=False)
    rows = [row for row in csv.reader(io.StringIO(path.read_text(encoding="utf-8"))) if row]
    assert [int(row[1]) for row in rows] == frames


def test_reading_back_strips_the_mp4_extension(tmp_path):
    naming = SubmissionNaming(video_extension=".mp4")
    submission = QuerySubmission("1", "kis", [Answer("L21_V001", frame=1)])
    write_submission(submission, tmp_path, naming=naming, strict=False)
    issues = validate_submission_dir(tmp_path)
    # The .mp4 suffix is stripped on read-back, so the video_id stays valid.
    assert not any("malformed video_id" in issue.message for issue in _errors(issues))


def test_reading_back_rejoins_a_qa_answer_containing_a_comma(tmp_path):
    submission = QuerySubmission("1", "qa", [Answer("L21_V001", frame=7, answer="một, hai")])
    write_submission(submission, tmp_path, strict=False)
    assert not _errors(validate_submission_dir(tmp_path))


# --------------------------------------------------------------------------
# Defensive read-back: a malformed row becomes a readable issue, never a crash
# --------------------------------------------------------------------------


def test_an_unexpected_header_row_does_not_crash(tmp_path):
    (tmp_path / "query-1-kis.csv").write_text(
        "video_id,frame_id\r\nL21_V001,100\r\n", encoding="utf-8"
    )
    issues = validate_submission_dir(tmp_path)
    assert any("header row" in issue.message for issue in _errors(issues))


def test_a_non_numeric_frame_id_does_not_crash(tmp_path):
    (tmp_path / "query-1-kis.csv").write_text("L21_V001,abc\r\nL21_V002,5\r\n", encoding="utf-8")
    issues = validate_submission_dir(tmp_path)
    assert any("not an integer" in issue.message for issue in _errors(issues))


def test_a_short_row_is_reported(tmp_path):
    (tmp_path / "query-1-kis.csv").write_text("L21_V001\r\n", encoding="utf-8")
    issues = _errors(validate_submission_dir(tmp_path))
    assert any("needs >= 2" in issue.message for issue in issues)
    (tmp_path / "query-2-qa.csv").write_text("L21_V001,5\r\n", encoding="utf-8")
    issues = _errors(validate_submission_dir(tmp_path))
    assert any("needs >= 3" in issue.message for issue in issues)


def test_a_file_with_no_readable_row_is_reported(tmp_path):
    (tmp_path / "query-1-kis.csv").write_text("x,y\r\n", encoding="utf-8")
    issues = validate_submission_dir(tmp_path)
    assert any("no answer rows could be read" in issue.message for issue in _errors(issues))


def test_a_wrong_encoding_does_not_crash(tmp_path):
    # 0xF9 is a valid CP1252 byte ("ù") but not valid UTF-8.
    (tmp_path / "query-1-qa.csv").write_bytes(b"L21_V001,5,m\xf9t\r\n")
    issues = validate_submission_dir(tmp_path)
    assert any("not readable as" in issue.message for issue in _errors(issues))


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------


def test_the_archive_is_flat(tmp_path):
    paths = [
        write_submission(_full_kis(), tmp_path, strict=False)[0],
        write_submission(
            QuerySubmission("2", "qa", [Answer("L21_V001", frame=1, answer="x")]),
            tmp_path,
            strict=False,
        )[0],
    ]
    archive_path = package_submission(paths, tmp_path / "submission.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert sorted(names) == ["query-1-kis.csv", "query-2-qa.csv"]
    assert all("/" not in name for name in names)


def test_packaging_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        package_submission([tmp_path / "absent.csv"], tmp_path / "s.zip")
