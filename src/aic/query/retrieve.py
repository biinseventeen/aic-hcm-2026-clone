"""P8 — retrieval and fusion: reducing the whole corpus to 300–500 candidate shots.

This stage sets the **ceiling on P_100**: an answer that does not reach the candidate set is
lost permanently, and no later stage can recover it.

Four channels run in parallel over the whole corpus and are fused with RRF (a **union**, not an
intersection — no channel holds a veto):

=====================  =========================  =====================
channel                input                      index unit
=====================  =========================  =====================
dense_translated       English translation        keyframe
dense_original         raw Vietnamese query       keyframe
sparse_text            keywords (BM25)            video / shot
entity_fuzzy           entities (character n-gram) video / shot
=====================  =========================  =====================

With four channels each at 0.6 recall, the union reaches 0.974 while the intersection gives
0.13. That is the entire reason for this architecture.

Two mechanisms this stage requires
----------------------------------
**Deduplicate *within* the candidate list, not afterwards.** A query matching a studio
backdrop will fill all 500 candidate slots with near-identical frames.

**But keep every cluster member when content is rebroadcast.** The same report is replayed
across several bulletins, and episodes of one series share a set and title graphics; many
videos match legitimately but only one is the answer. Deduplicating *across* videos is harmful
if it removes the copy that happens to be the answer. Hence: cluster for ranking, keep every
member as a cheap hedge for the lower slot bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.fusion import ChannelResult, reciprocal_rank_fusion, weights_for_query
from ..data.features import DenseIndex
from ..index.priors import DomainPrior
from ..index.shots import ShotTable
from ..index.text import TextIndex
from .parse import ParsedQuery

__all__ = ["Candidate", "RetrievalResult", "Retriever"]


@dataclass(slots=True)
class Candidate:
    """One candidate shot, with a per-channel trail so failures stay diagnosable."""

    video_id: str
    #: frame range of the candidate shot.
    start: int
    end: int
    #: the anchor frame — where the evidence is strongest, usually the best-matching keyframe.
    anchor: int
    fused_score: float = 0.0
    #: rank per channel; a missing channel was silent about this candidate.
    ranks: dict[str, int] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)
    #: cluster id for visually near-duplicate shots; -1 means not yet clustered.
    cluster: int = -1
    shot_id: int = -1

    @property
    def key(self) -> tuple[str, int]:
        return (self.video_id, self.shot_id if self.shot_id >= 0 else self.start)

    @property
    def n_channels(self) -> int:
        return len(self.ranks)


@dataclass
class RetrievalResult:
    candidates: list[Candidate] = field(default_factory=list)
    #: normalised video-level scores — this is the uncalibrated pi_v.
    video_scores: dict[str, float] = field(default_factory=dict)
    channel_sizes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def top_videos(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.video_scores.items(), key=lambda kv: -kv[1])[:n]

    def diagnostics(self) -> str:
        distinct_videos = len({c.video_id for c in self.candidates})
        lines = [
            f"candidates: {len(self.candidates)}  distinct videos: {distinct_videos}",
            f"channel sizes: {self.channel_sizes}",
        ]
        per_channel: dict[str, int] = {}
        for candidate in self.candidates:
            for channel in candidate.ranks:
                per_channel[channel] = per_channel.get(channel, 0) + 1
        lines.append(f"candidates contributed per channel: {per_channel}")
        agreed = sum(1 for c in self.candidates if c.n_channels >= 2)
        lines.append(f"candidates with >=2 channels agreeing: {agreed}/{len(self.candidates)}")
        lines.append(
            "top videos: "
            + ", ".join(f"{video}={score:.3f}" for video, score in self.top_videos(8))
        )
        lines.extend(f"  [i] {note}" for note in self.notes)
        return "\n".join(lines)

#Add helper for eng --> vie objects.
OBJECT_ALIASES = {
    "phụ nữ": ["woman", "person"],
    "đàn ông": ["man", "person"],
    "người": ["person"],

    "xe tải": ["truck"],
    "xe hơi": ["car"],
    "ô tô": ["car"],
    "xe": ["vehicle"],

    "thuyền": ["boat"],

    "đĩa": ["plate"],
    "dĩa": ["plate"],
    "tô": ["bowl"],
    "bát": ["bowl"],
    "chảo": ["frying pan", "pan"],
    "nồi": ["pot"],

    "bánh": ["cake"],
    "cua": ["crab"],
    "mực": ["squid"],

    "túi": ["bag"],
    "gói": ["bag", "package"],

    "chai": ["bottle"],
    "cốc": ["cup"],
    "ly": ["cup"],
}


def object_query_text(query: ParsedQuery) -> str:
    """Translate Vietnamese object mentions into OpenImages-style English labels."""
    raw = query.raw.lower()

    labels: list[str] = []

    # Longest phrases first so "xe tải" is preferred over generic "xe".
    for phrase, aliases in sorted(
        OBJECT_ALIASES.items(),
        key=lambda item: -len(item[0]),
    ):
        if phrase in raw:
            for alias in aliases:
                if alias not in labels:
                    labels.append(alias)

    return " ".join(labels)

@dataclass
class Retriever:
    """Orchestrates the four retrieval channels and fuses them with RRF."""

    dense: DenseIndex
    shots: ShotTable
    text: TextIndex | None = None
    #: Title-only index — the same BM25 machinery over documents that contain nothing but the
    #: video title, so a hit can be trusted far more than one in a YouTube description.
    title: TextIndex | None = None
    objects: TextIndex | None = None
    encoder: object | None = None
    #: Second text tower in the same image space, covering Vietnamese directly. Optional: when
    #: absent the multilingual channel is simply empty and RRF ignores it.
    encoder_multilingual: object | None = None
    prior: DomainPrior = field(default_factory=DomainPrior)

    #: parameters (defaults mirror aic.config.RetrievalConfig)
    channel_depth: int = 2000
    rrf_eta: int = 60
    n_candidates: int = 500
    dedup_cosine: float = 0.92
    max_shots_per_video: int = 12
    #: highest-IDF query terms kept for BM25; see aic.config.RetrievalConfig.sparse_max_terms.
    sparse_max_terms: int = 8
    #: exponent sharpening pi_v; see aic.config.RetrievalConfig.pi_sharpness.
    pi_sharpness: float = 1.0

    # ------------------------------------------------------------------
    # individual channels
    # ------------------------------------------------------------------

    def _dense_channel(
        self, text: str, name: str, *, encoder: object | None = None
    ) -> ChannelResult:
        """One dense search. Returns a ranked list of (video, shot_id) keys."""
        encoder = encoder if encoder is not None else self.encoder
        if encoder is None:
            return ChannelResult(name=name, ranked=[])
        query_vector = encoder.encode([text])  # type: ignore[union-attr]
        scores, rows = self.dense.search(query_vector, top_k=self.channel_depth)
        ranked: list[tuple[str, int]] = []
        raw: dict[tuple[str, int], float] = {}
        for score, row in zip(scores[0], rows[0], strict=True):
            video_id, _n, frame = self.dense.decode(int(row))
            shot = self.shots.find(video_id, frame)
            key = (video_id, shot.shot_id if shot else frame)
            if key in raw:
                continue
            ranked.append(key)
            raw[key] = float(score)
        return ChannelResult(name=name, ranked=ranked, scores=raw)

    def _sparse_channel(
        self,
        query: ParsedQuery,
        name: str,
        *,
        index: TextIndex | None = None,
        query_vector: np.ndarray | None = None,
    ) -> ChannelResult:
        index = index if index is not None else self.text
        if index is None or not query.keywords:
            return ChannelResult(name=name, ranked=[])
        text = " ".join(query.keywords) or query.raw
        hits = index.search_bm25(text, top_k=self.channel_depth, max_terms=self.sparse_max_terms)
        return self._text_hits_to_channel(hits, name, index=index, query_vector=query_vector)

    def _entity_channel(
        self, query: ParsedQuery, name: str, *, query_vector: np.ndarray | None = None
    ) -> ChannelResult:
        if self.text is None or not query.entities:
            return ChannelResult(name=name, ranked=[])
        hits = self.text.search_fuzzy(query.entities, top_k=self.channel_depth)
        return self._text_hits_to_channel(hits, name, query_vector=query_vector)

    def _shots_for_video_level_hit(
        self, video_id: str, query_vector: np.ndarray | None
    ) -> list[int]:
        """Which shots of a video a *video-level* text hit should stand for.

        This function exists because of a measured, catastrophic failure. The first version took
        ``shots.of(video_id)[:max_shots_per_video]`` — the **first twelve shots**, which in this
        corpus is the series title sequence. Every episode of a series shares that sequence, so a
        BM25 hit on any episode contributed a handful of interchangeable intro shots, and the
        allocator, seeing high-scoring candidates in many different videos, spread its slots
        across them. Counted on the first scored submission: **37 % of all 2,400 submitted rows
        fell within the first five seconds of their video**, and the most frequent frame ids were
        177, 89, 94, 43, 32 — credits, not content. The correct video for one query sat at rank 3
        with its frame inside the intro, which scores exactly zero.

        A text hit says *which video*; the dense index says *where inside it*. So the shots are
        ranked by the best cosine any of their keyframes reaches against the query vector. With no
        query vector (no encoder), the fallback is an **even spread** across the whole video —
        still wrong, but wrong in a way that does not concentrate on the one shot every video
        shares.
        """
        shots = self.shots.of(video_id)
        if not shots:
            return []
        limit = self.max_shots_per_video
        if query_vector is None:
            step = max(1, len(shots) // limit)
            return [shot.shot_id for shot in shots[::step]][:limit]

        rows = self.dense.video_rows(video_id)
        if rows.size == 0:
            step = max(1, len(shots) // limit)
            return [shot.shot_id for shot in shots[::step]][:limit]
        vectors = np.asarray(self.dense.vectors[rows], dtype=np.float32)
        sims = vectors @ np.asarray(query_vector, dtype=np.float32).ravel()
        best: dict[int, float] = {}
        for row, sim in zip(rows.tolist(), sims.tolist(), strict=True):
            _video, _n, frame = self.dense.decode(int(row))
            shot = self.shots.find(video_id, frame)
            shot_id = shot.shot_id if shot else frame
            if sim > best.get(shot_id, -2.0):
                best[shot_id] = sim
        return [shot_id for shot_id, _ in sorted(best.items(), key=lambda kv: -kv[1])[:limit]]

    def _text_hits_to_channel(
        self,
        hits: list[tuple[int, float]],
        name: str,
        *,
        index: TextIndex | None = None,
        query_vector: np.ndarray | None = None,
    ) -> ChannelResult:
        """Convert document-level hits into (video, shot_id) keys.

        ``media-info`` documents are *video*-level: they do not say which shot holds the event.
        Which shots the hit stands for is decided by :meth:`_shots_for_video_level_hit`.
        """
        index = index if index is not None else self.text
        if index is None:
            return ChannelResult(name=name, ranked=[])
        ranked: list[tuple[str, int]] = []
        raw: dict[tuple[str, int], float] = {}
        for doc_id, score in hits:
            doc = index.docs[doc_id]
            video_id = doc.video_id
            if doc.is_video_level:
                shot_ids = self._shots_for_video_level_hit(video_id, query_vector)
                if not shot_ids:
                    continue
                for shot_id in shot_ids:
                    key = (video_id, shot_id)
                    if key not in raw:
                        ranked.append(key)
                        raw[key] = float(score)
            else:
                shot = self.shots.find(video_id, (doc.start + doc.end) // 2)
                key = (video_id, shot.shot_id if shot else doc.start)
                if key not in raw:
                    ranked.append(key)
                    raw[key] = float(score)
            if len(ranked) >= self.channel_depth:
                break
        return ChannelResult(name=name, ranked=ranked, scores=raw)

    # ------------------------------------------------------------------
    # fusion
    # ------------------------------------------------------------------

    def retrieve(
        self, query: ParsedQuery, *, weights: dict[str, float] | None = None
    ) -> RetrievalResult:
        dense_texts = query.dense_texts()
        channels: list[ChannelResult] = []

        # The raw query is ALWAYS its own channel: if P7 mangles the translation, this one is
        # still intact. Cost is about 5 ms.
        if len(dense_texts) == 2:
            channels.append(self._dense_channel(dense_texts[0], "dense_translated"))
            channels.append(self._dense_channel(dense_texts[1], "dense_original"))
        else:
            channels.append(self._dense_channel(dense_texts[0], "dense_original"))
        # A second dense channel over the *same* raw Vietnamese text, through a text tower that
        # was distilled to land in this image space from any of 50 languages. It is not a
        # replacement for the channel above: on the mock set each tower finds videos the other
        # misses entirely, which is exactly the case RRF exists for.
        if self.encoder_multilingual is not None:
            channels.append(
                self._dense_channel(
                    query.raw, "dense_multilingual", encoder=self.encoder_multilingual
                )
            )
        # One query vector, reused by every text channel to place its video-level hits.
        query_vector = None
        if self.encoder is not None:
            query_vector = np.asarray(self.encoder.encode([query.raw]), dtype=np.float32).ravel()
        channels.append(self._sparse_channel(query, "sparse_text", query_vector=query_vector))
        if self.title is not None:
            channels.append(
                self._sparse_channel(
                    query, "sparse_title",
                    index=self.title, query_vector=query_vector
                )
            )
        if self.objects is not None:
            object_text = object_query_text(query)

            if object_text:
                hits = self.objects.search_bm25(
                    object_text,
                    top_k=self.channel_depth,
                    max_terms=self.sparse_max_terms,
                )

                channels.append(
                    self._text_hits_to_channel(
                        hits,
                        "objects",
                        index=self.objects,
                        query_vector=query_vector,
                    )
                )
            else:
                channels.append(
                    ChannelResult(
                        name="objects",
                        ranked=[],
                    )
                )
        
        channels.append(self._entity_channel(query, "entity_fuzzy", query_vector=query_vector))

        channel_weights = weights or weights_for_query(
            query.task,
            has_named_entity=bool(query.entities),
            is_visual_only=not (query.entities or query.keywords),
            has_translation=bool(query.english),
        )
        for channel in channels:
            channel.weight = channel_weights.get(channel.name, 1.0)

        fused = reciprocal_rank_fusion(channels, eta=self.rrf_eta, depth=self.channel_depth)

        result = RetrievalResult(
            channel_sizes={channel.name: len(channel.ranked) for channel in channels}
        )
        if not fused:
            result.notes.append(
                "every channel came back empty — check the text encoder and the text index"
            )
            return result

        # Build candidates, capping shots per video so one video cannot fill the whole list.
        shots_taken: dict[str, int] = {}
        for item in fused:
            video_id, shot_id = item.item  # type: ignore[misc]
            if shots_taken.get(video_id, 0) >= self.max_shots_per_video:
                continue
            shot = next((s for s in self.shots.of(video_id) if s.shot_id == shot_id), None)
            if shot is None:
                continue
            shots_taken[video_id] = shots_taken.get(video_id, 0) + 1
            result.candidates.append(
                Candidate(
                    video_id=video_id,
                    start=shot.start,
                    end=shot.end,
                    anchor=(shot.start + shot.end) // 2,
                    fused_score=item.score,
                    ranks=dict(item.ranks),
                    raw=dict(item.raw),
                    shot_id=shot.shot_id,
                )
            )
            if len(result.candidates) >= self.n_candidates:
                break

        n_capped = sum(1 for count in shots_taken.values() if count >= self.max_shots_per_video)
        if n_capped:
            result.notes.append(
                f"{n_capped} videos capped at {self.max_shots_per_video} shots "
                "(prevents one video filling the candidate list)"
            )

        result.video_scores = self._video_scores(result.candidates, query)
        return result

    def _video_scores(self, candidates: list[Candidate], query: ParsedQuery) -> dict[str, float]:
        """Aggregate shot scores into video-level ``pi_v``, then apply the domain prior.

        Uses a rank-*discounted* sum within each video rather than a plain sum: a video with ten
        mediocre shots should not outrank a video with one strong shot, because the answer is
        **one** moment, not a topic.
        """
        by_video: dict[str, list[float]] = {}
        for candidate in candidates:
            by_video.setdefault(candidate.video_id, []).append(candidate.fused_score)
        aggregated = {
            video_id: sum(
                score / (rank + 1) for rank, score in enumerate(sorted(scores, reverse=True))
            )
            for video_id, scores in by_video.items()
        }
        if self.pi_sharpness != 1.0:
            # Monotone, so no ranking changes: this only decides how much mass the leaders hold,
            # and therefore how many slots the allocator commits before hedging.
            aggregated = {
                video_id: score**self.pi_sharpness for video_id, score in aggregated.items()
            }
        total = sum(aggregated.values())
        normalised = (
            {video_id: score / total for video_id, score in aggregated.items()}
            if total > 0
            else aggregated
        )
        return self.prior.apply(normalised, task=query.task, domain_hints=query.domain_hints)

    # ------------------------------------------------------------------
    # deduplication
    # ------------------------------------------------------------------

    def cluster_near_duplicates(
        self, candidates: list[Candidate], *, threshold: float | None = None
    ) -> list[Candidate]:
        """Assign a ``cluster`` to visually near-duplicate candidates.

        **Nothing is removed.** Rebroadcast content is the case where removal hurts: several
        copies of the same report all match, but only one is the answer. Clustering tells the
        allocation layer that they are redundant with each other and lets it decide — the
        submodular coverage function will lower the marginal gain of the second member of a
        cluster on its own.
        """
        cosine_threshold = self.dedup_cosine if threshold is None else threshold
        if len(candidates) < 2:
            for candidate in candidates:
                candidate.cluster = 0
            return candidates

        # Representative vector: the keyframe closest to the anchor frame.
        vectors = []
        for candidate in candidates:
            rows = self.dense.video_rows(candidate.video_id)
            if len(rows) == 0:
                vectors.append(np.zeros(self.dense.dim, dtype=np.float32))
                continue
            frames = self.dense.rows[rows, 2]
            nearest = int(np.argmin(np.abs(frames - candidate.anchor)))
            vectors.append(np.asarray(self.dense.vectors[rows[nearest]], dtype=np.float32))
        matrix = np.stack(vectors)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

        # Greedy clustering in decreasing score order. O(n^2) over 500 candidates is 0.25 M
        # comparisons, which is negligible.
        order = sorted(range(len(candidates)), key=lambda i: -candidates[i].fused_score)
        centers: list[int] = []
        for index in order:
            placed = False
            for cluster_id, center in enumerate(centers):
                if float(matrix[index] @ matrix[center]) >= cosine_threshold:
                    candidates[index].cluster = cluster_id
                    placed = True
                    break
            if not placed:
                candidates[index].cluster = len(centers)
                centers.append(index)
        return candidates
