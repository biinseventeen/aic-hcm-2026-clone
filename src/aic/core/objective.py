"""P0 — the scoring function of the AIC 2026 preliminary round.

Source: "Thong tin vong So tuyen AIC2026.pdf", section 2.

    R@k        = max_{1<=i<=k} R-Score(r_i)
    FinalScore = (1/5) * sum_{k in K} R@k,      K = {1, 5, 20, 50, 100}

This module is the *executable* specification of the scoring rules. Every other module
optimises only through the functions defined here; no other module is allowed to
redefine what a score is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Rank cut-offs that are scored (rules, section 2.2).
THRESHOLDS: tuple[int, ...] = (1, 5, 20, 50, 100)

#: Maximum number of answers per query (rules, section 2: "tối đa 100 câu trả lời").
MAX_ANSWERS: int = 100


def band_weight(rank: int) -> float:
    """Marginal weight of position ``rank`` (1-based).

    w(j) = |{k in K : k >= j}| / |K| — the number of cut-offs that position j can still
    influence.

    >>> [band_weight(j) for j in (1, 2, 5, 6, 20, 21, 50, 51, 100, 101)]
    [1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0.0]
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return sum(1 for k in THRESHOLDS if k >= rank) / len(THRESHOLDS)


def band_of(rank: int) -> tuple[int, int]:
    """The band (lo, hi) containing ``rank``. Every position in a band scores the same.

    >>> band_of(1), band_of(3), band_of(20), band_of(77)
    ((1, 1), (2, 5), (6, 20), (51, 100))
    """
    if rank < 1 or rank > MAX_ANSWERS:
        raise ValueError(f"rank outside [1, {MAX_ANSWERS}]: {rank}")
    lo = 1
    for k in THRESHOLDS:
        if rank <= k:
            return (lo, k)
        lo = k + 1
    raise AssertionError("unreachable")


def band_sizes() -> dict[tuple[int, int], int]:
    """Slots per band: {(1,1):1, (2,5):4, (6,20):15, (21,50):30, (51,100):50}."""
    out: dict[tuple[int, int], int] = {}
    lo = 1
    for k in THRESHOLDS:
        out[(lo, k)] = k - lo + 1
        lo = k + 1
    return out


def final_score(r_scores: Sequence[float]) -> float:
    """Final Score for one query, from R-Scores in submission order.

    A list shorter than 100 is treated as having R-Score 0 for the remaining positions:
    submitting fewer slots is not penalised, it only forfeits positive expectation
    (see H4 in DESIGN.md).

    >>> final_score([0.5, 0.0, 0.8] + [0.0] * 97)   # worked example from the rules
    0.74
    """
    if len(r_scores) > MAX_ANSWERS:
        raise ValueError(
            f"got {len(r_scores)} answers, over the limit of {MAX_ANSWERS} set by the rules"
        )
    running = 0.0
    total = 0.0
    next_threshold = 0
    for i in range(1, MAX_ANSWERS + 1):
        if i <= len(r_scores):
            running = max(running, float(r_scores[i - 1]))
        if next_threshold < len(THRESHOLDS) and i == THRESHOLDS[next_threshold]:
            total += running
            next_threshold += 1
    return total / len(THRESHOLDS)


def final_score_from_first_hit(rank: int | None) -> float:
    """Final Score when R-Score is binary and the first correct answer is at ``rank``.

    This is the step function of DESIGN.md section 1.2. ``None`` means no correct answer.

    >>> [final_score_from_first_hit(r) for r in (1, 2, 5, 6, 21, 51, 101, None)]
    [1.0, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0]
    """
    if rank is None or rank > MAX_ANSWERS:
        return 0.0
    return band_weight(rank)


def expected_final_score(coverage_at_k: dict[int, float]) -> float:
    """E[Final] = (1/5) * sum_k P_k, where P_k = Pr(a correct answer is in the top k)."""
    missing = set(THRESHOLDS) - coverage_at_k.keys()
    if missing:
        raise ValueError(f"missing P_k for k = {sorted(missing)}")
    return sum(coverage_at_k[k] for k in THRESHOLDS) / len(THRESHOLDS)


# --------------------------------------------------------------------------
# R-Score per query type (rules, section 2.1)
# --------------------------------------------------------------------------


def r_score_kis(video_id: str, frame_id: int, gt_video: str, gt_span: tuple[int, int]) -> float:
    """Section 2.1.1 — I(v = GT_v AND frame_id in [s, e])."""
    s, e = gt_span
    return float(video_id == gt_video and s <= frame_id <= e)


def r_score_qa(
    video_id: str,
    frame_id: int,
    gt_video: str,
    gt_span: tuple[int, int],
    answer_matches: bool,
) -> float:
    """Section 2.1.2 — the conjunction of three conditions.

    ``answer_matches`` is decided by an external semantic matcher (see :mod:`aic.eval.score`);
    the rules only say the answer must match "về mặt ngữ nghĩa" (semantically). The answer
    text itself is deliberately not a parameter here: this function scores a *decision* that
    has already been made, and taking the text would suggest it does the matching too.
    """
    return float(r_score_kis(video_id, frame_id, gt_video, gt_span) > 0 and answer_matches)


def r_score_trake(
    video_id: str,
    frame_ids: Sequence[int],
    gt_video: str,
    gt_spans: Sequence[tuple[int, int]],
) -> float:
    """Section 2.1.3 — hard zero on the wrong video, else the fraction of moments hit.

    >>> r_score_trake("L10_V010", [101, 156, 203, 251], "L10_V010",
    ...               [(95, 105), (145, 155), (195, 205), (245, 255)])
    0.75
    """
    if video_id != gt_video:
        return 0.0
    n = len(gt_spans)
    if n == 0:
        raise ValueError("TRAKE needs at least one moment")
    if len(frame_ids) != n:
        raise ValueError(
            f"TRAKE needs exactly {n} frame ids, got {len(frame_ids)} — "
            "an answer with the wrong number of moments is an invalid answer"
        )
    hits = sum(1 for f, (s, e) in zip(frame_ids, gt_spans, strict=True) if s <= f <= e)
    return hits / n


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def marginal_value_report() -> str:
    """Marginal value of each slot band — used to justify the engineering budget."""
    lines = ["slot band       | slots | w(j) | max contribution"]
    lines.append("-" * 52)
    previous = 0.0
    for (lo, hi), n in band_sizes().items():
        w = band_weight(lo)
        lines.append(f"{lo:>4}–{hi:<10} | {n:>5} | {w:.1f}  | {w - previous:+.1f}")
        previous = w
    return "\n".join(lines)


def first_hit_rank(r_scores: Iterable[float], threshold: float = 1.0) -> int | None:
    """The first 1-based position with R-Score >= ``threshold``, or None."""
    for i, r in enumerate(r_scores, start=1):
        if r >= threshold:
            return i
    return None
