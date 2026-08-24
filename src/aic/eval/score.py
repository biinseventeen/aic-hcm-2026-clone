"""Offline scoring against the internal evaluation set.

This is an *executable replica* of the scoring system described in section 2 of the rules. It
exists because the organisers **do not supply queries with answers** (constraint R3): every
number measured during development comes from a self-built set, so the quality of that set is
the upper bound on the quality of every tuning decision.

Semantic answer matching
------------------------
The rules say an answer is correct when it "matches the ground truth semantically", without
defining the matching. The internal scorer uses three tiers and records which one matched, so it
stays visible which assumption a result rests on:

1. ``exact``   — equal after normalisation (punctuation removed, lowercased, whitespace
                 collapsed).
2. ``numeric`` — the same numeric value after reading both digits and Vietnamese number words
                 ("5" == "năm"). Required: the rules give exactly this example.
3. ``alias``   — matched through a synonym list supplied by the annotator.

Automatic fuzzy matching is deliberately *not* used for answers: "màu xanh" and "màu xám" are
close in characters but are two different answers, and an over-permissive scorer makes every
internal number optimistic in a way that looks correct — the worst failure mode in a tuning loop.

Which tier the real scoring system uses is **not settled**. The result specification says the
answer is compared "chính xác về mặt ngữ nghĩa" (semantically) in its Q&A section and "dưới dạng
chuỗi chính xác" (as an exact string) in its closing notes. The three tiers here implement the
*generous* reading, so every Q&A number this module reports is an **upper bound**: under the strict
reading only the ``exact`` tier scores, and the ``numeric`` and ``alias`` hits become zero. The
tier is recorded per query for exactly that reason — subtracting the non-``exact`` hits gives the
pessimistic number without re-running anything. See ``docs/SUBMISSION.md`` §2, question 1.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..core.objective import (
    MAX_ANSWERS,
    THRESHOLDS,
    final_score,
    first_hit_rank,
    r_score_kis,
    r_score_qa,
    r_score_trake,
)
from ..submit.writer import QuerySubmission

__all__ = ["EvalReport", "GroundTruth", "QueryScore", "answers_match", "score_submission"]


#: Vietnamese (and English) number words -> value. Needed by the ``numeric`` matching tier.
#: These stay in Vietnamese because they are the answer language, not prose.
_NUMBER_WORDS: dict[str, int] = {
    "không": 0,
    "một": 1,
    "mot": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "bon": 4,
    "tư": 4,
    "năm": 5,
    "nam": 5,
    "sáu": 6,
    "sau": 6,
    "bảy": 7,
    "bay": 7,
    "tám": 8,
    "tam": 8,
    "chín": 9,
    "chin": 9,
    "mười": 10,
    "muoi": 10,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

#: Vietnamese quantifiers and classifiers stripped before reading a number, so that
#: "có 5 người" compares equal to "5".
_QUANTIFIER_RE = re.compile(r"\b(có|khoảng|chừng|tất cả|gồm|là|người|cái|chiếc)\b")


def _normalise(text: str) -> str:
    normalised = unicodedata.normalize("NFC", text or "").lower().strip()
    normalised = re.sub(r"[.,;:!?\"'()\[\]{}]", " ", normalised)
    return re.sub(r"\s+", " ", normalised).strip()


def _as_number(text: str) -> float | None:
    stripped = _QUANTIFIER_RE.sub(" ", _normalise(text)).strip()
    digits = re.fullmatch(r"-?\d+(?:[.,]\d+)?", stripped.replace(" ", ""))
    if digits:
        return float(digits.group(0).replace(",", "."))
    if stripped in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[stripped])
    words = stripped.split()
    if len(words) == 1 and words[0] in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[words[0]])
    return None


def answers_match(got: str, expected: str, *, aliases: list[str] | None = None) -> tuple[bool, str]:
    """Return ``(matched, tier)``. The tier is for diagnostics, not for scoring."""
    if _normalise(got) == _normalise(expected):
        return True, "exact"
    got_number, expected_number = _as_number(got), _as_number(expected)
    if got_number is not None and got_number == expected_number:
        return True, "numeric"
    for alias in aliases or ():
        if _normalise(got) == _normalise(alias):
            return True, "alias"
        alias_number = _as_number(alias)
        if alias_number is not None and alias_number == got_number:
            return True, "alias-numeric"
    return False, "none"


@dataclass(slots=True)
class GroundTruth:
    """The ground truth for one query in the internal set."""

    query_id: str
    task: str
    video_id: str
    #: KIS and Q&A: a single span [s, e]. TRAKE: leave empty and use ``spans``.
    span: tuple[int, int] | None = None
    #: TRAKE: N spans, one per moment.
    spans: list[tuple[int, int]] = field(default_factory=list)
    #: Q&A: the correct answer, plus phrasings considered equivalent.
    answer: str = ""
    answer_aliases: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if self.task in ("kis", "qa") and self.span is None:
            raise ValueError(f"{self.query_id}: task {self.task} needs a span [s, e]")
        if self.task == "trake" and not self.spans:
            raise ValueError(f"{self.query_id}: TRAKE needs a list of spans")
        if self.task == "qa" and not self.answer:
            raise ValueError(f"{self.query_id}: Q&A needs an answer")

    @property
    def n_moments(self) -> int:
        return len(self.spans)


@dataclass(slots=True)
class QueryScore:
    query_id: str
    task: str
    final: float = 0.0
    r_at_k: dict[int, float] = field(default_factory=dict)
    #: 1-based position of the first correct answer; None if there is none.
    first_hit: int | None = None
    n_submitted: int = 0
    #: the highest R-Score reached at any position.
    best_r: float = 0.0
    #: for Q&A: the answer matching tier used by the highest-scoring row.
    match_tier: str = ""
    #: right video but wrong frame — the single most informative diagnostic.
    right_video_wrong_frame: bool = False
    notes: list[str] = field(default_factory=list)


def _r_scores_for(submission: QuerySubmission, truth: GroundTruth) -> tuple[list[float], str]:
    """R-Score per submitted row, plus the answer matching tier that first scored."""
    scores: list[float] = []
    tier = ""
    for answer in submission.answers[:MAX_ANSWERS]:
        if truth.task == "kis":
            score = r_score_kis(
                answer.video_id,
                answer.frame or -1,
                truth.video_id,
                truth.span,  # type: ignore[arg-type]
            )
        elif truth.task == "qa":
            matched, matched_tier = answers_match(
                answer.answer or "", truth.answer, aliases=truth.answer_aliases
            )
            score = r_score_qa(
                answer.video_id,
                answer.frame or -1,
                truth.video_id,
                truth.span,  # type: ignore[arg-type]
                matched,
            )
            if score > 0 and not tier:
                tier = matched_tier
        else:
            try:
                score = r_score_trake(answer.video_id, answer.frames, truth.video_id, truth.spans)
            except ValueError:
                # The wrong number of moments is an invalid answer: score 0, do not crash.
                score = 0.0
        scores.append(score)
    return scores, tier


def score_submission(submission: QuerySubmission, truth: GroundTruth) -> QueryScore:
    """Score one answer list using exactly the formula from section 2 of the rules."""
    if submission.task != truth.task:
        raise ValueError(
            f"{submission.query_id}: submitted task ({submission.task}) differs from the "
            f"ground truth task ({truth.task})"
        )
    r_scores, tier = _r_scores_for(submission, truth)

    score = QueryScore(
        query_id=submission.query_id,
        task=submission.task,
        final=final_score(r_scores),
        n_submitted=len(submission.answers),
        best_r=max(r_scores) if r_scores else 0.0,
        match_tier=tier,
    )
    running = 0.0
    for position, value in enumerate(r_scores, start=1):
        running = max(running, value)
        if position in THRESHOLDS:
            score.r_at_k[position] = running
    for k in THRESHOLDS:
        score.r_at_k.setdefault(k, running)
    score.first_hit = first_hit_rank(r_scores, threshold=1.0)

    # Diagnostic: hitting the right video without scoring means the frame localisation layer
    # failed, not the retrieval layer. Those two failures call for different investments.
    hit_video = any(answer.video_id == truth.video_id for answer in submission.answers)
    if hit_video and score.best_r == 0.0:
        score.right_video_wrong_frame = True
        score.notes.append(
            "right video, but no frame landed inside the answer span — a failure in P10 "
            "(localisation and frame spread), not in P8 (retrieval)"
        )
    if not hit_video:
        score.notes.append(
            "no row hit the right video — a failure in the retrieval layer (P8); no later "
            "stage can recover from it"
        )
    if score.n_submitted < MAX_ANSWERS:
        score.notes.append(f"only {score.n_submitted}/{MAX_ANSWERS} slots submitted (H4)")
    return score


#: Marker used to recognise a retrieval-layer failure when aggregating notes.
_RETRIEVAL_FAILURE_MARKER = "failure in the retrieval layer"


@dataclass
class EvalReport:
    scores: list[QueryScore] = field(default_factory=list)

    def by_task(self) -> dict[str, list[QueryScore]]:
        grouped: dict[str, list[QueryScore]] = {}
        for score in self.scores:
            grouped.setdefault(score.task, []).append(score)
        return grouped

    @property
    def mean_final(self) -> float:
        if not self.scores:
            return 0.0
        return sum(score.final for score in self.scores) / len(self.scores)

    def summary(self) -> str:
        if not self.scores:
            return "no queries scored yet"
        lines = [
            f"Internal evaluation: {len(self.scores)} queries, "
            f"mean Final Score = {self.mean_final:.4f}",
            "",
            f"  {'task':<8} {'n':>3} {'Final':>7} "
            + " ".join(f"{'R@' + str(k):>6}" for k in THRESHOLDS),
        ]
        for task, task_scores in sorted(self.by_task().items()):
            count = len(task_scores)
            mean_final = sum(score.final for score in task_scores) / count
            per_k = [sum(s.r_at_k[k] for s in task_scores) / count for k in THRESHOLDS]
            lines.append(
                f"  {task:<8} {count:>3} {mean_final:>7.4f} "
                + " ".join(f"{value:>6.3f}" for value in per_k)
            )
        lines.append("")
        lines.extend(self._failure_breakdown())
        return "\n".join(lines)

    def _failure_breakdown(self) -> list[str]:
        """Where the losses come from. This table decides the next engineering investment."""
        total = len(self.scores)
        missed_video = sum(
            1
            for score in self.scores
            if any(_RETRIEVAL_FAILURE_MARKER in note for note in score.notes)
        )
        wrong_frame = sum(1 for score in self.scores if score.right_video_wrong_frame)
        scored_anywhere = sum(1 for score in self.scores if score.best_r > 0)
        hit_at_1 = sum(1 for score in self.scores if score.first_hit == 1)
        hit_at_5 = sum(
            1 for score in self.scores if score.first_hit is not None and score.first_hit <= 5
        )
        lines = [
            "  Failure breakdown:",
            f"    retrieval missed entirely     : {missed_video:>3}/{total}",
            f"    right video, wrong frame      : {wrong_frame:>3}/{total}",
            f"    scored somewhere in the 100   : {scored_anywhere:>3}/{total}",
            f"    correct at slot 1             : {hit_at_1:>3}/{total}",
            f"    correct within the top 5      : {hit_at_5:>3}/{total}",
        ]
        tiers: dict[str, int] = {}
        for score in self.scores:
            if score.task == "qa" and score.match_tier:
                tiers[score.match_tier] = tiers.get(score.match_tier, 0) + 1
        if tiers:
            lines.append(f"    answer matching tiers used    : {tiers}")
        return lines
