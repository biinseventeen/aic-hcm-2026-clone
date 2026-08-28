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
    has_translation: bool = False,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Channel weights per query type — the only hyperparameter of P8.

    ``has_translation`` switches to the weights measured on the eight labelled queries of round 1,
    where the translated dense channel ranked the correct video **1st to 3rd on every one of them**
    while the other four channels ranked it outside the top hundred on seven of the eight. Treating
    them as near-equals turned a correct first place into a fourth-to-thirteenth, and the
    allocator
    then spent its early slots elsewhere: the correct answer ended up at row 51, 71, or off the
    list entirely.
    Weighting the translated channel six times the others, and halving the sparse channels, moved
    mean Final Score on those eight from 0.125 to 0.575 and R@100 from 0.375 to 1.000.

    Without a translation the old, uncalibrated priors stand: the down-weighted sparse channels
    would otherwise remove signal with nothing put in its place.
    """
    weights = {
        "dense_translated": 1.0,
        "dense_multilingual": 1.0,
        "dense_original": 0.6,
        "sparse_text": 0.8,
        "objects": 1.0,
        # A title match is curated text with no boilerplate, so it earns the highest weight of
        # any sparse channel: the mock set has queries whose subject is literally the title.
        "sparse_title": 1.4,
        "entity_fuzzy": 0.7,
    }
    if has_translation:
        # Calibrated on 8 labelled queries — enough to see a 4x effect, not enough to separate
        # neighbouring values. `pi_sharpness` (aic.config) is the other half of this change.
        weights["dense_translated"] = 6.0
        weights["sparse_title"] = 0.4
        weights["sparse_text"] = 0.3
    if has_named_entity:
        # A string match on a rare proper noun is stronger evidence than visual similarity.
        weights["entity_fuzzy"] *= 2.0
        weights["sparse_text"] *= 1.25
        weights["sparse_title"] *= 1.25
    if is_visual_only:
        # No textual cue available: the dense channels have to carry the query.
        weights["dense_translated"] *= 1.5
        weights["dense_multilingual"] *= 1.5
        weights["sparse_text"] *= 0.5
        weights["sparse_title"] *= 0.75
        weights["entity_fuzzy"] *= 0.25
    if kind == "trake":
        # TRAKE is dominated by Pr(correct video); favour the broadest-coverage channels.
        weights["dense_translated"] *= 1.25
        weights["dense_multilingual"] *= 1.25
    if overrides:
        weights.update(overrides)
    return weights
