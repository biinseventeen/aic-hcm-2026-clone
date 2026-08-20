"""P12 — event sequence alignment for TRAKE (monotonic dynamic programming).

Given a candidate video and N *ordered* moment descriptions, pick N frame indices
``t_1 < t_2 < ... < t_N`` maximising the total similarity:

    D[j][t] = S[j][t] + max_{t' <= t - delta} D[j-1][t']

With a cumulative prefix maximum, the recursion runs in O(N*T).

Basis for the monotonicity constraint
-------------------------------------
If the athlete performs the action ten times, choosing an independent argmax per moment can
pair the "approach" of the first repetition with the "take-off" of the fifth — a meaningless
tuple with a high total score. Dynamic programming with an ordering constraint and a minimum
gap ``delta`` cannot produce that combination.

The number of moments and their content are defined by the judges *at query time*, so no
training labels exist. The solution must be zero-shot and text-conditioned: ``S`` comes from
cosine similarity between the moment description embedding and the frame embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AlignedPath",
    "alternative_paths",
    "detect_cycles",
    "moment_confidence",
    "monotonic_align",
]

#: Sentinel for "unreachable" cells in the DP table. Large enough that adding real scores
#: cannot lift a blocked cell above a reachable one.
UNREACHABLE = -1.0e30


@dataclass(slots=True)
class AlignedPath:
    """One aligned moment sequence, indexed in the *column space of S*."""

    indices: tuple[int, ...]
    total: float
    #: the S[j][t_j] score of each moment.
    per_moment: tuple[float, ...] = ()
    #: normalised [0, 1] confidence of each moment.
    confidence: tuple[float, ...] = ()

    @property
    def n(self) -> int:
        return len(self.indices)

    def weakest_moment(self) -> int:
        """Index of the least confident moment — the one to vary when hedging."""
        if not self.confidence:
            return int(np.argmin(self.per_moment)) if self.per_moment else 0
        return int(np.argmin(self.confidence))


def monotonic_align(
    similarity: np.ndarray,
    *,
    delta: int = 1,
    forbid: np.ndarray | None = None,
) -> AlignedPath:
    """Optimal monotonic alignment. ``similarity`` has shape (N, T).

    ``delta`` — minimum gap between consecutive moments (>= 1).
    ``forbid`` — boolean mask (N, T); True forbids choosing that cell.

    >>> S = np.array([[9.0, 1, 1, 1], [1, 1, 9.0, 1]])
    >>> monotonic_align(S, delta=1).indices
    (0, 2)
    >>> S2 = np.array([[1.0, 9, 1, 1], [1, 9.0, 1, 1]])   # both moments want t=1
    >>> monotonic_align(S2, delta=1).indices               # the earlier one yields
    (0, 1)
    """
    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"similarity must have shape (N, T), got {scores.shape}")
    n_moments, n_frames = scores.shape
    if n_moments == 0 or n_frames == 0:
        raise ValueError(f"similarity is empty: shape {scores.shape}")
    if delta < 1:
        raise ValueError(f"delta must be >= 1, got {delta}")
    if n_frames < (n_moments - 1) * delta + 1:
        raise ValueError(
            f"window too short: T={n_frames} cannot fit N={n_moments} moments "
            f"separated by delta={delta} (needs >= {(n_moments - 1) * delta + 1})"
        )

    work = scores.copy()
    if forbid is not None:
        work[np.asarray(forbid, dtype=bool)] = UNREACHABLE

    table = np.full((n_moments, n_frames), UNREACHABLE)
    backpointer = np.full((n_moments, n_frames), -1, dtype=np.int64)
    table[0] = work[0]

    for j in range(1, n_moments):
        # running_max[x] = max_{t' <= x} table[j-1][t'];  argmax_at[x] = the matching argmax.
        previous = table[j - 1]
        running_max = np.maximum.accumulate(previous)
        argmax_at = np.zeros(n_frames, dtype=np.int64)
        best = 0
        for x in range(n_frames):
            if previous[x] > previous[best]:
                best = x
            argmax_at[x] = best
        # t is admissible once t - delta >= 0.
        if delta < n_frames:
            source = running_max[: n_frames - delta]
            table[j, delta:] = work[j, delta:] + source
            backpointer[j, delta:] = argmax_at[: n_frames - delta]
            table[j, delta:][source <= UNREACHABLE / 2] = UNREACHABLE

    end = int(np.argmax(table[n_moments - 1]))
    if table[n_moments - 1, end] <= UNREACHABLE / 2:
        raise ValueError("no feasible monotonic path exists")

    chosen = [0] * n_moments
    chosen[n_moments - 1] = end
    for j in range(n_moments - 1, 0, -1):
        chosen[j - 1] = int(backpointer[j, chosen[j]])
    indices = tuple(chosen)
    per_moment = tuple(float(scores[j, indices[j]]) for j in range(n_moments))
    return AlignedPath(
        indices=indices,
        total=float(sum(per_moment)),
        per_moment=per_moment,
        confidence=moment_confidence(scores, indices),
    )


def moment_confidence(
    similarity: np.ndarray, indices: tuple[int, ...], *, temperature: float = 0.05
) -> tuple[float, ...]:
    """Per-moment confidence in [0, 1]: the softmax of S[j] evaluated at t_j.

    A moment where every frame matches equally well (occluded, or described too abstractly)
    yields a near-uniform distribution and low confidence — the flag that tells P13 this is
    the moment to vary across hedge tuples.
    """
    scores = np.asarray(similarity, dtype=np.float64)
    out = []
    for j, t in enumerate(indices):
        row = scores[j]
        logits = (row - row.max()) / max(temperature, 1e-9)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        out.append(float(probabilities[t]))
    return tuple(out)


def alternative_paths(
    similarity: np.ndarray,
    *,
    delta: int = 1,
    n_paths: int = 8,
    min_shift: int = 1,
) -> list[AlignedPath]:
    """Generate hedge tuples by *varying the weakest moment*.

    This is the correct strategy for TRAKE (a consequence of H6): keep the high-confidence
    moments fixed and vary only the highest-entropy moment across successive tuples. Because
    ``E[max_i R_i]`` does not decompose per moment, varying several moments at once dilutes
    probability mass without raising the ``max``.

    Returns a list in decreasing ``total``, with the optimal path first.
    """
    scores = np.asarray(similarity, dtype=np.float64)
    best = monotonic_align(scores, delta=delta)
    paths = [best]
    if n_paths <= 1:
        return paths

    weakest_first = np.argsort(best.confidence)
    blocked = np.zeros(scores.shape, dtype=bool)
    for moment in weakest_first:
        moment = int(moment)
        used = {best.indices[moment]}
        while len(paths) < n_paths:
            mask = blocked.copy()
            for index in used:
                lo = max(0, index - min_shift + 1)
                hi = min(scores.shape[1], index + min_shift)
                mask[moment, lo:hi] = True
            try:
                path = monotonic_align(scores, delta=delta, forbid=mask)
            except ValueError:
                break
            if path.indices in {p.indices for p in paths}:
                break
            paths.append(path)
            used.add(path.indices[moment])
        if len(paths) >= n_paths:
            break
    paths.sort(key=lambda p: -p.total)
    return paths[:n_paths]


def detect_cycles(
    embeddings: np.ndarray, *, min_period: int = 8, max_period: int | None = None
) -> list[tuple[int, int]]:
    """Detect repeating cycles in a window; returns (start, end) pairs in column space.

    When an action repeats many times (an athlete jumping ten times), aligning over the whole
    window can mix cycles. Cycles are detected from the peak of the autocorrelation of the
    self-similarity matrix, then each cycle is aligned *independently*. Later cycles become
    hedge candidates for the lower slot bands.

    ``embeddings`` has shape (T, D) and must already be L2-normalised.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must have shape (T, D), got {matrix.shape}")
    n_frames = matrix.shape[0]
    if n_frames < 2 * min_period:
        return [(0, n_frames - 1)]
    max_period = max_period or n_frames // 2

    self_similarity = matrix @ matrix.T
    # Autocorrelation by lag: the mean of the diagonal offset by k.
    lags = np.arange(min_period, min(max_period, n_frames - 1) + 1)
    if lags.size == 0:
        return [(0, n_frames - 1)]
    autocorrelation = np.array(
        [float(np.mean(np.diagonal(self_similarity, offset=int(k)))) for k in lags]
    )
    period = int(lags[int(np.argmax(autocorrelation))])

    cycles: list[tuple[int, int]] = []
    start = 0
    while start < n_frames:
        end = min(n_frames - 1, start + period - 1)
        if end - start + 1 >= min_period // 2:
            cycles.append((start, end))
        start = end + 1
    return cycles or [(0, n_frames - 1)]
