"""Human-in-the-loop review shortlist.

The automatic solver maximises submission score.  This module maximises a
different quantity: how quickly a human can eliminate most of the corpus and
land on the right video / temporal neighbourhood.

Important rules:
- retrieval scores are never changed;
- P10/P13 are never called;
- a bad candidate anchor is NOT trusted when it lies outside its own locus;
- broad loci are probed at several positions instead of being represented by
  one arbitrary anchor;
- display order is round-robin: first show one frame from every video, then
  show temporal extras.  This prevents rank-12 from appearing only after a
  human has inspected 40 thumbnails from ranks 1-10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..query.retrieve import Candidate, RetrievalResult

__all__ = [
    "ReviewFrame",
    "ReviewVideo",
    "ReviewShortlist",
    "build_review_shortlist",
]


@dataclass(slots=True, frozen=True)
class ReviewFrame:
    frame: int
    timestamp_seconds: float
    score: float
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    shot_id: int = -1
    channels: tuple[str, ...] = ()
    cluster: int = -1
    kind: str = "anchor"

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "score": float(self.score),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "shot_id": self.shot_id,
            "channels": list(self.channels),
            "cluster": self.cluster,
            "kind": self.kind,
        }


@dataclass(slots=True, frozen=True)
class ReviewVideo:
    video_id: str
    rank: int
    video_score: float
    fps: float
    frames: tuple[ReviewFrame, ...]
    selection_reason: str = "global"

    @property
    def rescued(self) -> bool:
        return self.selection_reason != "global"

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "rank": self.rank,
            "video_score": float(self.video_score),
            "fps": float(self.fps),
            "selection_reason": self.selection_reason,
            "rescued": self.rescued,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(slots=True)
class ReviewShortlist:
    videos: list[ReviewVideo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return sum(len(video.frames) for video in self.videos)

    @property
    def n_rescued_videos(self) -> int:
        return sum(video.rescued for video in self.videos)

    def scan_order(self) -> list[tuple[ReviewVideo, ReviewFrame]]:
        """Human scan order.

        Pass 1 shows exactly one thumbnail from every video.  Only after that do
        we spend attention on second/third/fourth temporal hypotheses.  Extras
        are also round-robin so one video cannot monopolise the screen.
        """
        ordered: list[tuple[ReviewVideo, ReviewFrame]] = []
        max_depth = max((len(video.frames) for video in self.videos), default=0)
        for depth in range(max_depth):
            for video in self.videos:
                if depth < len(video.frames):
                    ordered.append((video, video.frames[depth]))
        return ordered

    def to_dict(self) -> dict:
        scan = self.scan_order()
        return {
            "videos": [video.to_dict() for video in self.videos],
            "scan_order": [
                {
                    "scan_rank": index,
                    "video_id": video.video_id,
                    "video_rank": video.rank,
                    "frame": frame.to_dict(),
                }
                for index, (video, frame) in enumerate(scan, 1)
            ],
            "n_videos": len(self.videos),
            "n_frames": self.n_frames,
            "n_rescued_videos": self.n_rescued_videos,
            "notes": list(self.notes),
        }

    def report(self) -> str:
        lines = [
            (
                f"review shortlist: {len(self.videos)} videos, "
                f"{self.n_frames} frames, "
                f"{self.n_rescued_videos} channel-rescued videos"
            ),
            "",
            "PASS 1 — VIDEO SCAN (one frame per video)",
        ]

        for scan_rank, video in enumerate(self.videos, 1):
            if not video.frames:
                continue
            frame = video.frames[0]
            reason = (
                ""
                if video.selection_reason == "global"
                else f"  [{video.selection_reason}]"
            )
            lines.append(
                f"{scan_rank:>3}. #{video.rank:<3} {video.video_id:<10} "
                f"t={frame.timestamp_seconds:8.2f}s "
                f"frame={frame.frame:<7} "
                f"score={frame.score:.5f}{reason}"
            )

        extras = [
            (video, frame, depth)
            for depth in range(1, max((len(v.frames) for v in self.videos), default=0))
            for video in self.videos
            if depth < len(video.frames)
            for frame in (video.frames[depth],)
        ]
        if extras:
            lines.extend(["", "PASS 2 — TEMPORAL EXTRAS"])
            for video, frame, depth in extras:
                channels = ",".join(frame.channels) if frame.channels else "-"
                lines.append(
                    f"  #{video.rank:<3} {video.video_id:<10} "
                    f"[{depth + 1}] t={frame.timestamp_seconds:8.2f}s "
                    f"frame={frame.frame:<7} "
                    f"kind={frame.kind:<10} "
                    f"window={frame.start_seconds:.2f}-{frame.end_seconds:.2f}s "
                    f"channels={channels}"
                )

        lines.extend(["", *[f"  [i] {note}" for note in self.notes]])
        return "\n".join(lines)


@dataclass(slots=True, frozen=True)
class _Probe:
    candidate: Candidate
    frame: int
    kind: str
    priority: tuple


def _seconds(frame: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return max(0, frame) / fps


def _raw_anchor(candidate: Candidate) -> int:
    """Keep retrieval's anchor unchanged.

    Review v3 replaced anchors outside their locus with the midpoint.  That
    improved internal consistency but regressed a previously useful KL29
    anchor.  Human review can afford to keep *both* signals instead: raw anchor
    as one hypothesis and locus representatives as separate hypotheses.
    """
    return int(candidate.anchor)


def _locus_midpoint(candidate: Candidate) -> int:
    return (int(candidate.start) + int(candidate.end)) // 2


def _anchor_inside_locus(candidate: Candidate) -> bool:
    return int(candidate.start) <= int(candidate.anchor) <= int(candidate.end)


def _is_temporal_channel(channel: str) -> bool:
    lowered = channel.lower()
    return "temporal" in lowered or "localized" in lowered


def _broad_locus(candidate: Candidate, fps: float) -> bool:
    return int(candidate.end) - int(candidate.start) >= int(round(12.0 * fps))


def _probe_to_review(probe: _Probe, fps: float) -> ReviewFrame:
    candidate = probe.candidate
    return ReviewFrame(
        frame=int(probe.frame),
        timestamp_seconds=_seconds(int(probe.frame), fps),
        score=float(candidate.fused_score),
        start_frame=int(candidate.start),
        end_frame=int(candidate.end),
        start_seconds=_seconds(int(candidate.start), fps),
        end_seconds=_seconds(int(candidate.end), fps),
        shot_id=int(candidate.shot_id),
        channels=tuple(
            channel
            for channel, _rank in sorted(
                candidate.ranks.items(),
                key=lambda item: item[1],
            )
        ),
        cluster=int(candidate.cluster),
        kind=probe.kind,
    )


def _candidate_probe_pool(
    candidates: list[Candidate],
    *,
    fps: float,
) -> dict[str, list[_Probe]]:
    """Build evidence families instead of one flat score-sorted pool.

    A flat pool caused the strongest fused anchors to consume every frame
    budget before channel-specific or broad-locus evidence could be shown.
    Families let :func:`_review_frames` allocate slots deliberately.
    """
    ranked = sorted(
        candidates,
        key=lambda c: (
            -float(c.fused_score),
            min(c.ranks.values()) if c.ranks else 10**9,
            int(c.anchor),
        ),
    )

    fused: list[_Probe] = []
    temporal: list[_Probe] = []
    channels: list[_Probe] = []
    locus: list[_Probe] = []

    # Raw fused anchors: preserve retrieval behaviour as evidence, even if the
    # anchor lies outside the candidate locus.
    for candidate in ranked:
        fused.append(
            _Probe(
                candidate=candidate,
                frame=_raw_anchor(candidate),
                kind=(
                    "anchor"
                    if _anchor_inside_locus(candidate)
                    else "anchor_out"
                ),
                priority=(
                    -float(candidate.fused_score),
                    min(candidate.ranks.values()) if candidate.ranks else 10**9,
                    int(candidate.anchor),
                ),
            )
        )

    channel_names = sorted(
        {
            channel
            for candidate in candidates
            for channel in candidate.ranks
        }
    )

    for channel in channel_names:
        rows = [candidate for candidate in candidates if channel in candidate.ranks]
        if not rows:
            continue
        candidate = min(
            rows,
            key=lambda c: (
                int(c.ranks[channel]),
                -float(c.fused_score),
                int(c.anchor),
            ),
        )

        # For temporal/localized channels, the locus itself is meaningful.  If
        # the locus is broad, midpoint is a safer human cue than trusting one
        # refined anchor; the raw anchor remains available in fused evidence.
        if _is_temporal_channel(channel):
            frame = (
                _locus_midpoint(candidate)
                if _broad_locus(candidate, fps) or not _anchor_inside_locus(candidate)
                else _raw_anchor(candidate)
            )
            temporal.append(
                _Probe(
                    candidate=candidate,
                    frame=frame,
                    kind=f"temporal:{channel}",
                    priority=(
                        int(candidate.ranks[channel]),
                        -float(candidate.fused_score),
                        frame,
                    ),
                )
            )
        else:
            channels.append(
                _Probe(
                    candidate=candidate,
                    frame=_raw_anchor(candidate),
                    kind=f"channel:{channel}",
                    priority=(
                        int(candidate.ranks[channel]),
                        -float(candidate.fused_score),
                        int(candidate.anchor),
                    ),
                )
            )

    # Broad loci get explicit interior probes. Quarter points are particularly
    # useful when the temporal span is much wider than the true event.
    broad = sorted(
        (candidate for candidate in candidates if _broad_locus(candidate, fps)),
        key=lambda c: (
            0 if any(_is_temporal_channel(ch) for ch in c.ranks) else 1,
            -float(c.fused_score),
            min(c.ranks.values()) if c.ranks else 10**9,
        ),
    )
    for candidate in broad[:4]:
        start = int(candidate.start)
        end = int(candidate.end)
        width = end - start
        for q_index, numerator in enumerate((1, 2, 3), 1):
            frame = start + (width * numerator) // 4
            locus.append(
                _Probe(
                    candidate=candidate,
                    frame=frame,
                    kind=f"locus_q{q_index}",
                    priority=(
                        0 if any(_is_temporal_channel(ch) for ch in candidate.ranks) else 1,
                        -float(candidate.fused_score),
                        q_index,
                        frame,
                    ),
                )
            )

    return {
        "fused": fused,
        "temporal": sorted(temporal, key=lambda p: p.priority),
        "channels": sorted(channels, key=lambda p: p.priority),
        "locus": sorted(locus, key=lambda p: p.priority),
    }


def _review_frames(
    candidates: list[Candidate],
    *,
    fps: float,
    limit: int,
    min_gap_seconds: float,
) -> list[ReviewFrame]:
    """Allocate a small frame budget across *different evidence types*.

    Slot plan:
    1. best raw fused anchor (keeps known-good retrieval behaviour);
    2. best temporal/localized locus representative;
    3. best independent non-temporal channel;
    4. broad-locus probe;
    then repeat remaining evidence by strength.

    Rank >20 videos normally get two slots in v4: one fused cue + one temporal
    alternative.  This is intentionally more expensive than v3 because a
    single wrong thumbnail can make a human discard the correct video.
    """
    if limit <= 0 or not candidates:
        return []
    if min_gap_seconds < 0:
        raise ValueError(
            f"min_gap_seconds must be >= 0, got {min_gap_seconds}"
        )

    min_gap_frames = int(round(min_gap_seconds * fps))
    families = _candidate_probe_pool(candidates, fps=fps)

    selected: list[_Probe] = []
    seen_frames: set[int] = set()

    def try_add(probe: _Probe) -> bool:
        if len(selected) >= limit:
            return False
        frame = int(probe.frame)
        if frame in seen_frames:
            return False
        if any(
            abs(frame - int(previous.frame)) < min_gap_frames
            for previous in selected
        ):
            return False
        selected.append(probe)
        seen_frames.add(frame)
        return True

    def add_first_available(name: str) -> None:
        for probe in families[name]:
            if try_add(probe):
                return

    # Deliberate diversity before score-based filling.
    add_first_available("fused")
    if len(selected) < limit:
        add_first_available("temporal")
    if len(selected) < limit:
        add_first_available("channels")
    if len(selected) < limit:
        add_first_available("locus")

    # Remaining slots: temporal/locus first, then other channel evidence, then
    # more fused anchors.  This prevents four near-identical fused peaks.
    if len(selected) < limit:
        for family_name in ("temporal", "locus", "channels", "fused"):
            for probe in families[family_name]:
                try_add(probe)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

    return [_probe_to_review(probe, fps) for probe in selected]


def _select_videos(
    result: RetrievalResult,
    *,
    top_videos: int,
    max_videos: int,
    rescue_per_channel: int,
) -> list[tuple[str, float, int, str]]:
    review_scores = result.recall_video_scores or result.video_scores
    ranked_all = sorted(
        review_scores.items(),
        key=lambda item: -float(item[1]),
    )
    global_rank = {
        video_id: rank
        for rank, (video_id, _score) in enumerate(ranked_all, 1)
    }

    selected: dict[str, str] = {}
    for video_id, _score in ranked_all[:top_videos]:
        selected[video_id] = "global"

    if rescue_per_channel > 0 and len(selected) < max_videos:
        best_by_channel: dict[str, dict[str, Candidate]] = {}
        review_candidates = [*result.candidates, *result.recall_candidates]
        for candidate in review_candidates:
            for channel, channel_rank in candidate.ranks.items():
                current = best_by_channel.setdefault(channel, {}).get(
                    candidate.video_id
                )
                if current is None:
                    best_by_channel[channel][candidate.video_id] = candidate
                    continue
                if (
                    int(channel_rank),
                    -float(candidate.fused_score),
                ) < (
                    int(current.ranks.get(channel, 10**9)),
                    -float(current.fused_score),
                ):
                    best_by_channel[channel][candidate.video_id] = candidate

        channel_lists: dict[str, list[Candidate]] = {}
        for channel, per_video in best_by_channel.items():
            channel_lists[channel] = sorted(
                per_video.values(),
                key=lambda c: (
                    int(c.ranks[channel]),
                    -float(c.fused_score),
                    global_rank.get(c.video_id, 10**9),
                ),
            )

        channels = sorted(channel_lists)
        used = {channel: 0 for channel in channels}
        cursor = {channel: 0 for channel in channels}
        progressed = True

        while len(selected) < max_videos and progressed:
            progressed = False
            for channel in channels:
                if len(selected) >= max_videos:
                    break
                if used[channel] >= rescue_per_channel:
                    continue

                rows = channel_lists[channel]
                i = cursor[channel]
                while i < len(rows) and rows[i].video_id in selected:
                    i += 1
                cursor[channel] = i
                if i >= len(rows):
                    continue

                candidate = rows[i]
                cursor[channel] += 1
                selected[candidate.video_id] = f"channel:{channel}"
                used[channel] += 1
                progressed = True

    rows = [
        (
            video_id,
            float(review_scores.get(video_id, 0.0)),
            global_rank.get(video_id, len(ranked_all) + 1),
            reason,
        )
        for video_id, reason in selected.items()
    ]
    rows.sort(
        key=lambda row: (
            0 if row[3] == "global" else 1,
            row[2],
            row[0],
        )
    )
    return rows


def build_review_shortlist(
    result: RetrievalResult,
    *,
    fps_by_video: dict[str, float],
    top_videos: int = 50,
    max_videos: int = 60,
    rescue_per_channel: int = 2,
    tier1_videos: int = 10,
    tier1_frames: int = 4,
    tier2_videos: int = 20,
    tier2_frames: int = 3,
    later_frames: int = 2,
    min_gap_seconds: float = 8.0,
    default_fps: float = 25.0,
) -> ReviewShortlist:
    if top_videos < 1:
        raise ValueError(f"top_videos must be >= 1, got {top_videos}")
    if max_videos < top_videos:
        raise ValueError(
            f"max_videos ({max_videos}) must be >= top_videos ({top_videos})"
        )
    if rescue_per_channel < 0:
        raise ValueError("rescue_per_channel must be >= 0")
    if not 0 <= tier1_videos <= tier2_videos:
        raise ValueError("require 0 <= tier1_videos <= tier2_videos")
    if min(tier1_frames, tier2_frames, later_frames) < 0:
        raise ValueError("frame budgets must be >= 0")
    if default_fps <= 0:
        raise ValueError(f"default_fps must be > 0, got {default_fps}")

    by_video: dict[str, list[Candidate]] = {}
    seen_candidate_keys: set[tuple[str, int, int]] = set()

    # Automatic candidates first: they carry temporal/localised evidence and
    # should win when the recall reserve points at the same shot.
    for candidate in [*result.candidates, *result.recall_candidates]:
        key = (candidate.video_id, int(candidate.shot_id), int(candidate.anchor))
        if key in seen_candidate_keys:
            continue
        seen_candidate_keys.add(key)
        by_video.setdefault(candidate.video_id, []).append(candidate)

    selected_videos = _select_videos(
        result,
        top_videos=top_videos,
        max_videos=max_videos,
        rescue_per_channel=rescue_per_channel,
    )

    videos: list[ReviewVideo] = []
    missing_candidates: list[str] = []

    for video_id, video_score, rank, reason in selected_videos:
        candidates = by_video.get(video_id, [])
        if not candidates:
            missing_candidates.append(video_id)
            continue

        fps = float(fps_by_video.get(video_id, default_fps))
        if fps <= 0:
            fps = default_fps

        if reason != "global":
            frame_budget = later_frames
        elif rank <= tier1_videos:
            frame_budget = tier1_frames
        elif rank <= tier2_videos:
            frame_budget = tier2_frames
        else:
            frame_budget = later_frames

        frames = tuple(
            _review_frames(
                candidates,
                fps=fps,
                limit=frame_budget,
                min_gap_seconds=min_gap_seconds,
            )
        )
        if not frames:
            continue

        videos.append(
            ReviewVideo(
                video_id=video_id,
                rank=rank,
                video_score=video_score,
                fps=fps,
                frames=frames,
                selection_reason=reason,
            )
        )

    notes = [
        (
            f"global top {top_videos}; up to {max_videos} videos after "
            f"per-channel rescue ({rescue_per_channel}/channel)"
        ),
        (
            f"frame budget: ranks 1-{tier1_videos} -> {tier1_frames}, "
            f"{tier1_videos + 1}-{tier2_videos} -> {tier2_frames}, "
            f"later/rescued -> {later_frames}; "
            f"minimum temporal gap={min_gap_seconds:g}s"
        ),
        "global video ranking uses the pre-truncation core recall pool when available",
        "display order is video-first round-robin: one frame per video before temporal extras",
        "raw retrieval anchors are preserved; temporal/locus alternatives are shown separately",
        "broad loci (>=12s) contribute quarter-position probes",
        "timestamps are for human review only; submission frame ids are unchanged",
    ]
    if missing_candidates:
        notes.append(
            "video-level hypotheses with no frame candidate were skipped: "
            + ", ".join(missing_candidates[:10])
            + (" ..." if len(missing_candidates) > 10 else "")
        )

    return ReviewShortlist(videos=videos, notes=notes)
