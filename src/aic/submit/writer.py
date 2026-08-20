"""Generating and validating submission files.

Separating specification from assumption
========================================
The document "Thông tin vòng Sơ tuyển AIC2026" specifies the *content* of an answer (section
2.1) but **not** the file format, the naming, or the packaging. What the rules state for
certain:

* at most **100** answers per query (section 2);
* Textual KIS: ``<video_id>, <frame_id>``;
* Q&A: ``<video_id>, <frame_id>, <answer>``;
* TRAKE: ``<video_id>, <frame_id_1>, ..., <frame_id_N>``;
* row **order** is decisive — R@k reads by position.

What this module *assumes*, following the convention of previous AIC seasons, and what
**must be confirmed with the organisers**:

* one CSV file per query, with **no** header row;
* the filename ``query-<id>-<task>.csv`` (``kis`` | ``qa`` | ``trake``);
* the files are packaged in a single flat .zip;
* ``video_id`` is written **without** the ``.mp4`` extension (the rules write
  ``video_abc(.mp4)``, meaning the extension is optional — the bare form matches the keyframe
  directory names and the metadata filenames, so that is the form chosen).

Pass ``naming=SubmissionNaming(...)`` to change any of the above without touching code. Run
``aic submit-selftest`` to print a complete set of sample files to send to the organisers for
confirmation before submitting for real.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..core.objective import MAX_ANSWERS
from ..data.layout import VIDEO_ID_RE

__all__ = [
    "Answer",
    "QuerySubmission",
    "SubmissionNaming",
    "ValidationIssue",
    "package_submission",
    "validate",
    "validate_submission_dir",
    "write_submission",
]

Task = Literal["kis", "qa", "trake"]

#: An answer longer than this is flagged. A long answer is more likely to be judged a
#: non-match, since the rules only require semantic equivalence with a short ground truth.
MAX_REASONABLE_ANSWER_CHARS = 200


@dataclass(slots=True)
class SubmissionNaming:
    """Every format assumption, in one place, so each is a one-line change."""

    #: filename template; available keys are query_id and task.
    filename: str = "query-{query_id}-{task}.csv"
    delimiter: str = ","
    include_header: bool = False
    #: when set, write ``L21_V001.mp4`` instead of ``L21_V001``.
    video_extension: str = ""
    #: line terminator; CRLF is the safe default for CSV readers on Windows and in Excel.
    lineterminator: str = "\r\n"
    encoding: str = "utf-8"
    zip_name: str = "submission.zip"

    def file_for(self, query_id: str, task: Task) -> str:
        return self.filename.format(query_id=query_id, task=task)

    def render_video(self, video_id: str) -> str:
        return f"{video_id}{self.video_extension}"


@dataclass(slots=True)
class Answer:
    """One answer row. ``frames`` is for TRAKE; ``frame`` for KIS and Q&A."""

    video_id: str
    frame: int | None = None
    frames: tuple[int, ...] = ()
    answer: str | None = None

    def row(self, task: Task, naming: SubmissionNaming) -> list[str]:
        video = naming.render_video(self.video_id)
        if task == "kis":
            if self.frame is None:
                raise ValueError(f"KIS needs a frame_id: {self}")
            return [video, str(self.frame)]
        if task == "qa":
            if self.frame is None or self.answer is None:
                raise ValueError(f"Q&A needs a frame_id and an answer: {self}")
            return [video, str(self.frame), self.answer]
        if task == "trake":
            if not self.frames:
                raise ValueError(f"TRAKE needs at least one frame_id: {self}")
            return [video, *(str(frame) for frame in self.frames)]
        raise ValueError(f"invalid task: {task!r}")


@dataclass(slots=True)
class QuerySubmission:
    """The *ordered* answer list for one query."""

    query_id: str
    task: Task
    answers: list[Answer] = field(default_factory=list)
    #: for TRAKE, the number of moments N the query asks for (used for validation).
    n_moments: int | None = None

    def __len__(self) -> int:
        return len(self.answers)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    query_id: str
    message: str
    row: int | None = None

    def __str__(self) -> str:
        location = f" [row {self.row}]" if self.row is not None else ""
        return f"{self.severity.upper():<7} {self.query_id}{location}: {self.message}"


def validate(
    submission: QuerySubmission,
    *,
    known_videos: set[str] | None = None,
    max_frames: dict[str, int] | None = None,
) -> list[ValidationIssue]:
    """Check one answer list before it is written to disk.

    The errors here are the *silently score-destroying* kind: the file is still syntactically
    valid, the scoring system still reads it, and the score is still zero.
    """
    issues: list[ValidationIssue] = []

    def add(severity: str, message: str, row: int | None = None) -> None:
        issues.append(ValidationIssue(severity, submission.query_id, message, row))  # type: ignore[arg-type]

    if not submission.answers:
        add("error", "empty list — submitting nothing forfeits this query entirely")
        return issues
    if len(submission.answers) > MAX_ANSWERS:
        add(
            "error",
            f"{len(submission.answers)} answers, over the limit of {MAX_ANSWERS} set by the rules",
        )
    if len(submission.answers) < MAX_ANSWERS:
        # H4: the last 50 slots are worth only 0.2 points but cost almost nothing to produce.
        add(
            "warning",
            f"only {len(submission.answers)}/{MAX_ANSWERS} slots — the remaining ones are free "
            "positive expectation, so fill them (H4)",
        )

    seen: set[tuple] = set()
    for row_number, answer in enumerate(submission.answers, start=1):
        if not VIDEO_ID_RE.match(answer.video_id):
            add("error", f"malformed video_id: {answer.video_id!r}", row_number)
        elif known_videos is not None and answer.video_id not in known_videos:
            add("error", f"video_id not present in the corpus: {answer.video_id!r}", row_number)

        frames = (
            list(answer.frames)
            if submission.task == "trake"
            else ([answer.frame] if answer.frame is not None else [])
        )
        for frame in frames:
            if frame is None or frame < 0:
                add("error", f"frame_id must be an integer >= 0, got {frame!r}", row_number)
            elif (
                max_frames is not None
                and answer.video_id in max_frames
                and frame > max_frames[answer.video_id]
            ):
                add(
                    "warning",
                    f"frame_id {frame} is past the last known frame "
                    f"({max_frames[answer.video_id]}) of {answer.video_id} — it may not exist",
                    row_number,
                )

        if submission.task == "qa":
            if answer.answer is None or not answer.answer.strip():
                add("error", "Q&A answer is missing", row_number)
            elif "\n" in answer.answer or "\r" in answer.answer:
                add("error", "answer contains a newline — it would break the CSV", row_number)
            elif len(answer.answer) > MAX_REASONABLE_ANSWER_CHARS:
                add(
                    "warning",
                    "answer is very long; long answers are easily judged a non-match",
                    row_number,
                )
        if submission.task == "trake":
            if submission.n_moments is not None and len(answer.frames) != submission.n_moments:
                add(
                    "error",
                    f"TRAKE needs exactly {submission.n_moments} frame ids, this row has "
                    f"{len(answer.frames)}",
                    row_number,
                )
            if list(answer.frames) != sorted(answer.frames):
                add(
                    "warning",
                    f"TRAKE frame ids are not increasing: {answer.frames} — the moments of an "
                    "event sequence are normally in temporal order",
                    row_number,
                )

        key = (answer.video_id, answer.frame, answer.frames, answer.answer)
        if key in seen:
            add("error", f"row {row_number} is an exact duplicate — one slot is wasted", row_number)
        seen.add(key)

    return issues


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

_CSV_HEADERS: dict[str, list[str]] = {
    "kis": ["video_id", "frame_id"],
    "qa": ["video_id", "frame_id", "answer"],
    "trake": ["video_id", "frame_ids"],
}


def _render_csv(submission: QuerySubmission, naming: SubmissionNaming) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(
        buffer,
        delimiter=naming.delimiter,
        lineterminator=naming.lineterminator,
        quoting=csv.QUOTE_MINIMAL,
    )
    if naming.include_header:
        writer.writerow(_CSV_HEADERS[submission.task])
    for answer in submission.answers:
        writer.writerow(answer.row(submission.task, naming))
    return buffer.getvalue()


def write_submission(
    submission: QuerySubmission,
    out_dir: str | Path,
    *,
    naming: SubmissionNaming | None = None,
    known_videos: set[str] | None = None,
    max_frames: dict[str, int] | None = None,
    strict: bool = True,
) -> tuple[Path, list[ValidationIssue]]:
    """Write one query's submission file. Returns ``(path, issues)``.

    ``strict=True`` raises on any ``error``-level issue. That is the right default, because a
    malformed submission loses points with no way to detect it after the fact.
    """
    naming = naming or SubmissionNaming()
    issues = validate(submission, known_videos=known_videos, max_frames=max_frames)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors and strict:
        raise ValueError(
            f"{submission.query_id}: {len(errors)} submission errors, file not written.\n"
            + "\n".join(f"  {error}" for error in errors[:10])
        )
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / naming.file_for(submission.query_id, submission.task)
    path.write_text(_render_csv(submission, naming), encoding=naming.encoding, newline="")
    return path, issues


def package_submission(files: Iterable[str | Path], out_zip: str | Path) -> Path:
    """Package submission files into a **flat** .zip, with no nested directory.

    A flat structure is the safest form: if the scoring system expects a subdirectory it will
    usually still find the files, whereas the reverse does not hold.
    """
    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(file) for file in files]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing files to package: {missing[:5]}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, arcname=path.name)
    return out


_FILENAME_RE = re.compile(r"^query-(?P<query_id>[^-]+)-(?P<task>kis|qa|trake)\.csv$")

#: Minimum column count per task, used by the defensive re-read parser.
_MIN_COLUMNS: dict[str, int] = {"kis": 2, "qa": 3, "trake": 2}


def _parse_submission_file(
    path: Path, query_id: str, task: str, naming: SubmissionNaming
) -> tuple[list[Answer], int | None, list[ValidationIssue]]:
    """Parse one written submission file *defensively*.

    This is the last gate before submitting, so a malformed row must surface as a readable issue
    rather than killing the process. Handles a stray header row, a non-integer frame id, a short
    row, and a wrong file encoding.
    """
    issues: list[ValidationIssue] = []
    try:
        text = path.read_text(encoding=naming.encoding)
    except UnicodeDecodeError as exc:
        issues.append(
            ValidationIssue(
                "error",
                query_id,
                f"file is not readable as {naming.encoding}: {exc}. A wrong encoding turns "
                "Vietnamese answers into mojibake during scoring.",
            )
        )
        return [], None, issues

    rows = [row for row in csv.reader(io.StringIO(text), delimiter=naming.delimiter) if row]
    answers: list[Answer] = []
    n_moments: int | None = None
    minimum_columns = _MIN_COLUMNS[task]
    for row_number, row in enumerate(rows, start=1):
        if len(row) < minimum_columns:
            issues.append(
                ValidationIssue(
                    "error",
                    query_id,
                    f"row has only {len(row)} columns, needs >= {minimum_columns}",
                    row_number,
                )
            )
            continue
        video_id = row[0].removesuffix(".mp4")
        numeric_fields = row[1:2] if task in ("kis", "qa") else row[1:]
        try:
            numbers = [int(field.strip()) for field in numeric_fields]
        except ValueError:
            hint = " — is this a header row?" if row_number == 1 else ""
            issues.append(
                ValidationIssue(
                    "error",
                    query_id,
                    f"frame_id is not an integer: {numeric_fields!r}{hint}",
                    row_number,
                )
            )
            continue
        if task == "kis":
            answers.append(Answer(video_id, frame=numbers[0]))
        elif task == "qa":
            answers.append(
                Answer(video_id, frame=numbers[0], answer=naming.delimiter.join(row[2:]))
            )
        else:
            frames = tuple(numbers)
            n_moments = n_moments or len(frames)
            answers.append(Answer(video_id, frames=frames))
    return answers, n_moments, issues


def validate_submission_dir(
    directory: str | Path,
    *,
    naming: SubmissionNaming | None = None,
    known_videos: set[str] | None = None,
) -> list[ValidationIssue]:
    """Re-read every written submission file and validate it — the final check before sending.

    Reading back from disk (rather than validating the in-memory objects) is deliberate: it is
    the only way to catch encoding errors, stray newlines, and filename errors.
    """
    naming = naming or SubmissionNaming()
    issues: list[ValidationIssue] = []
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        return [ValidationIssue("error", "-", f"no .csv files found in {directory}")]
    for path in files:
        match = _FILENAME_RE.match(path.name)
        if not match:
            issues.append(
                ValidationIssue(
                    "warning",
                    path.name,
                    "filename does not match query-<id>-<task>.csv (a naming assumption — "
                    "needs confirmation from the organisers)",
                )
            )
            continue
        query_id, task = match.group("query_id"), match.group("task")
        answers, n_moments, parse_issues = _parse_submission_file(path, query_id, task, naming)
        issues.extend(parse_issues)
        if not answers:
            issues.append(
                ValidationIssue("error", query_id, f"{path.name}: no answer rows could be read")
            )
            continue
        issues.extend(
            validate(
                QuerySubmission(query_id, task, answers, n_moments=n_moments),  # type: ignore[arg-type]
                known_videos=known_videos,
            )
        )
    return issues
