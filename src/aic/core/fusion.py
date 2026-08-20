"""P8 — fusing multiple retrieval channels with Reciprocal Rank Fusion.

RRF consumes *ranks* only, never scores. That is why it was chosen: each channel's score
scale shifts as the corpus grows (batch 2 — constraint R5), but ranks are invariant under
rescaling, so weights tuned on batch 1 still mean something on batch 1+2.

    RRF(d) = sum_c w_c / (eta + rank_c(d))
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

__all__ = [
    "DEFAULT_ETA",
    "ChannelResult",
    "FusedItem",
    "reciprocal_rank_fusion",
    "series_recall",
    "union_recall",
    "weights_for_query",
]

T = TypeVar("T", bound=Hashable)

#: The standard RRF smoothing constant (Cormack et al. 2009).
DEFAULT_ETA: int = 60


@dataclass(slots=True)
class ChannelResult:
    """The ranked output of one retrieval channel.

    ``ranked`` is already in decreasing order of relevance. ``scores`` exists only for
    logging and diagnostics — RRF does not read it.
    """

    name: str
    ranked: list[T]
    scores: dict[T, float] = field(default_factory=dict)
    weight: float = 1.0

    def rank_of(self, item: T) -> int | None:
        """1-based rank, or None if this channel did not return ``item``."""
        try:
            return self.ranked.index(item) + 1
        except ValueError:
            return None


@dataclass(slots=True)
class FusedItem:
    item: T
    score: float
    #: 1-based rank per channel; a missing channel did not return the item.
    ranks: dict[str, int]
    #: raw per-channel scores, for diagnostics only.
    raw: dict[str, float] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.ranks)


def reciprocal_rank_fusion(
    channels: Sequence[ChannelResult],
    *,
    eta: int = DEFAULT_ETA,
    top_k: int | None = None,
    depth: int | None = None,
) -> list[FusedItem]:
    """Fuse channels with RRF. Returns a list in decreasing fused score.

    ``depth`` — consider only the first ``depth`` results of each channel, bounding the
    noisy tail. ``top_k`` — truncate the output list.

    A channel returning an empty list is handled naturally, with no special case.
    """
    if eta <= 0:
        raise ValueError(f"eta must be > 0, got {eta}")

    accumulated: dict[T, float] = {}
    ranks: dict[T, dict[str, int]] = {}
    raw_scores: dict[T, dict[str, float]] = {}

    for channel in channels:
        if channel.weight == 0.0:
            continue
        ranked = channel.ranked if depth is None else channel.ranked[:depth]
        seen: set[T] = set()
        for rank, item in enumerate(ranked, start=1):
            if item in seen:  # a channel repeating itself: count the first occurrence only
                continue
            seen.add(item)
            accumulated[item] = accumulated.get(item, 0.0) + channel.weight / (eta + rank)
            ranks.setdefault(item, {})[channel.name] = rank
            if item in channel.scores:
                raw_scores.setdefault(item, {})[channel.name] = channel.scores[item]

    fused = [
        FusedItem(item=item, score=score, ranks=ranks[item], raw=raw_scores.get(item, {}))
        for item, score in accumulated.items()
    ]
    # Tie-break: agreement across more channels wins; then by id, for reproducibility.
    fused.sort(key=lambda f: (-f.score, -f.n_channels, str(f.item)))
    return fused[:top_k] if top_k else fused


def union_recall(channel_recalls: Iterable[float]) -> float:
    """Recall of the union under an independence assumption: 1 - prod(1 - r_i).

    This is the number that justifies running channels in *parallel* rather than in series.

    >>> round(union_recall([0.6, 0.6, 0.6, 0.6]), 4)
    0.9744
    """
    product = 1.0
    for recall in channel_recalls:
        if not 0.0 <= recall <= 1.0:
            raise ValueError(f"recall must lie in [0, 1], got {recall}")
        product *= 1.0 - recall
    return 1.0 - product


def series_recall(stage_recalls: Iterable[float]) -> float:
    """Recall of a series chain: prod(r_i). The counterpart to :func:`union_recall`.

    >>> round(series_recall([0.6, 0.6, 0.6, 0.6]), 4)
    0.1296
    """
    product = 1.0
    for recall in stage_recalls:
        if not 0.0 <= recall <= 1.0:
            raise ValueError(f"recall must lie in [0, 1], got {recall}")
        product *= recall
    return product


def weights_for_query(
    kind: str,
    *,
    has_named_entity: bool = False,
    is_visual_only: bool = False,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Channel weights per query type — the only hyperparameter of P8.

    The defaults are *uncalibrated priors*. They must be tuned on the internal evaluation
    set (:mod:`aic.eval.devset`) before being trusted.
    """
    weights = {
        "dense_translated": 1.0,
        "dense_original": 0.6,
        "sparse_text": 0.8,
        "entity_fuzzy": 0.7,
    }
    if has_named_entity:
        # A string match on a rare proper noun is stronger evidence than visual similarity.
        weights["entity_fuzzy"] *= 2.0
        weights["sparse_text"] *= 1.25
    if is_visual_only:
        # No textual cue available: the dense channel has to carry the query.
        weights["dense_translated"] *= 1.5
        weights["sparse_text"] *= 0.5
        weights["entity_fuzzy"] *= 0.25
    if kind == "trake":
        # TRAKE is dominated by Pr(correct video); favour the broadest-coverage channel.
        weights["dense_translated"] *= 1.25
    if overrides:
        weights.update(overrides)
    return weights
