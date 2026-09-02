"""Human-in-the-loop review shortlist.

The automatic solver tries to maximise submission score.  This module has a
different objective: maximise the chance that a human sees the correct video
and at least one useful temporal clue with a review budget of roughly 100
thumbnails.

It does not modify retrieval scores and it never calls P10/P13.
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
        }


@dataclass(slots=True, frozen=True)
class ReviewVideo:
    video_id: str
    # Rank in the complete video_scores ordering, not rank inside the shortlist.
    rank: int
    video_score: float
    fps: float
    frames: tuple[ReviewFrame, ...]
    # "global" means the video was already in the normal top-N video ranking.
    # "channel:<name>" means a single retrieval channel rescued it.
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

    def to_dict(self) -> dict:
        return {
            "videos": [video.to_dict() for video in self.videos],
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
            )
        ]

        for video in self.videos:
            reason = (
                ""
                if video.selection_reason == "global"
                else f"  [{video.selection_reason}]"
            )
            lines.append(
                f"#{video.rank:<3} {video.video_id:<10} "
                f"video_score={video.video_score:.4f}  "
                f"fps={video.fps:g}{reason}"
            )
            for index, frame in enumerate(video.frames, 1):
                channels = ",".join(frame.channels) if frame.channels else "-"
                lines.append(
                    f"    [{index}] frame={frame.frame:<7} "
                    f"t={frame.timestamp_seconds:8.2f}s  "
                    f"score={frame.score:.5f}  channels={channels}"
                )

        lines.extend(f"  [i] {note}" for note in self.notes)
        return "\n".join(lines)


def _seconds(frame: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return max(0, frame) / fps


def _candidate_to_review(candidate: Candidate, fps: float) -> ReviewFrame:
    return ReviewFrame(
        frame=int(candidate.anchor),
        timestamp_seconds=_seconds(int(candidate.anchor), fps),
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
    )


def _far_enough(
    candidate: Candidate,
    selected: list[Candidate],
    *,
    min_gap_frames: int,
) -> bool:
    anchor = int(candidate.anchor)
    return all(
        abs(anchor - int(previous.anchor)) >= min_gap_frames
        for previous in selected
    )


def _review_frames(
    candidates: list[Candidate],
    *,
    fps: float,
    limit: int,
    min_gap_seconds: float,
) -> list[Candidate]:
    """Pick temporal evidence with both channel and time diversity.

    Order of preference:
    1. strongest fused candidate;
    2. strongest candidate from each individual retrieval channel;
    3. remaining candidates by fused score.

    This matters for human review: a dense-parts peak and an object peak can be
    more useful than four almost-identical fused peaks.
    """
    if limit <= 0 or not candidates:
        return []
    if min_gap_seconds < 0:
        raise ValueError(
            f"min_gap_seconds must be >= 0, got {min_gap_seconds}"
        )

    min_gap_frames = int(round(min_gap_seconds * fps))
    ranked = sorted(
        candidates,
        key=lambda c: (
            -float(c.fused_score),
            min(c.ranks.values()) if c.ranks else 10**9,
            int(c.anchor),
        ),
    )

    selected: list[Candidate] = []
    selected_keys: set[tuple[int, int, int]] = set()

    def try_add(candidate: Candidate) -> None:
        if len(selected) >= limit:
            return
        key = (int(candidate.start), int(candidate.end), int(candidate.anchor))
        if key in selected_keys:
            return
        if not _far_enough(
            candidate,
            selected,
            min_gap_frames=min_gap_frames,
        ):
            return
        selected.append(candidate)
        selected_keys.add(key)

    # 1) Best overall evidence.
    try_add(ranked[0])

    # 2) Let each channel nominate one distinct temporal peak.
    channels = sorted(
        {
            channel
            for candidate in candidates
            for channel in candidate.ranks
        }
    )
    channel_nominees: list[tuple[int, float, str, Candidate]] = []
    for channel in channels:
        channel_candidates = [
            candidate
            for candidate in candidates
            if channel in candidate.ranks
        ]
        if not channel_candidates:
            continue
        best = min(
            channel_candidates,
            key=lambda c: (
                c.ranks[channel],
                -float(c.fused_score),
                int(c.anchor),
            ),
        )
        channel_nominees.append(
            (
                int(best.ranks[channel]),
                -float(best.fused_score),
                channel,
                best,
            )
        )

    for _rank, _neg_score, _channel, candidate in sorted(channel_nominees):
        try_add(candidate)
        if len(selected) >= limit:
            return selected

    # 3) Fill any remaining slots with the best fused evidence.
    for candidate in ranked:
        try_add(candidate)
        if len(selected) >= limit:
            break

    return selected


def _select_videos(
    result: RetrievalResult,
    *,
    top_videos: int,
    max_videos: int,
    rescue_per_channel: int,
) -> tuple[list[tuple[str, float, int, str]], dict[str, int]]:
    """Top video ranking plus a small per-channel rescue union.

    The automatic fused ranking can bury a true video even when one individual
    channel has strong evidence for it.  Human review can cheaply keep a few of
    those disagreements instead of forcing fusion to choose a single winner.
    """
    ranked_all = sorted(
        result.video_scores.items(),
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
        # For every channel, keep its best candidate per video.
        best_by_channel: dict[str, dict[str, Candidate]] = {}
        for candidate in result.candidates:
            for channel, channel_rank in candidate.ranks.items():
                current = best_by_channel.setdefault(channel, {}).get(
                    candidate.video_id
                )
                if current is None:
                    best_by_channel[channel][candidate.video_id] = candidate
                    continue
                current_rank = current.ranks.get(channel, 10**9)
                if (
                    int(channel_rank),
                    -float(candidate.fused_score),
                ) < (
                    int(current_rank),
                    -float(current.fused_score),
                ):
                    best_by_channel[channel][candidate.video_id] = candidate

        # Round-robin over channels so one noisy channel cannot consume the
        # entire rescue budget.
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
        used_per_channel = {channel: 0 for channel in channels}
        cursors = {channel: 0 for channel in channels}

        progressed = True
        while len(selected) < max_videos and progressed:
            progressed = False
            for channel in channels:
                if len(selected) >= max_videos:
                    break
                if used_per_channel[channel] >= rescue_per_channel:
                    continue

                rows = channel_lists[channel]
                cursor = cursors[channel]
                while cursor < len(rows) and rows[cursor].video_id in selected:
                    cursor += 1
                cursors[channel] = cursor
                if cursor >= len(rows):
                    continue

                candidate = rows[cursor]
                cursors[channel] += 1
                selected[candidate.video_id] = f"channel:{channel}"
                used_per_channel[channel] += 1
                progressed = True

    rows: list[tuple[str, float, int, str]] = []
    for video_id, reason in selected.items():
        score = float(result.video_scores.get(video_id, 0.0))
        rank = global_rank.get(video_id, len(ranked_all) + 1)
        rows.append((video_id, score, rank, reason))

    # Keep the normal global top-N first. Rescued videos follow, ordered by
    # their global rank so the board remains predictable for a human.
    rows.sort(
        key=lambda row: (
            0 if row[3] == "global" else 1,
            row[2],
            row[0],
        )
    )
    return rows, global_rank


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
    tier2_frames: int = 2,
    later_frames: int = 1,
    min_gap_seconds: float = 8.0,
    default_fps: float = 25.0,
) -> ReviewShortlist:
    """Build a high-recall human review board.

    Default theoretical maximum:
      ranks 1..10   : 10 * 4 = 40 thumbnails
      ranks 11..20 : 10 * 2 = 20
      ranks 21..50 : 30 * 1 = 30
      channel rescue: up to 10 * 1 = 10
      ------------------------------------------------
      total: about 100 thumbnails

    ``top_videos`` is the normal fused-ranking baseline. ``max_videos`` leaves
    room for videos rescued by individual retrieval channels.
    """
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
    for candidate in result.candidates:
        by_video.setdefault(candidate.video_id, []).append(candidate)

    selected_videos, _global_rank = _select_videos(
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

        selected = _review_frames(
            candidates,
            fps=fps,
            limit=frame_budget,
            min_gap_seconds=min_gap_seconds,
        )
        frames = tuple(
            _candidate_to_review(candidate, fps)
            for candidate in selected
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
        "timestamps are for human review only; submission frame ids are unchanged",
    ]
    if missing_candidates:
        notes.append(
            "video-level hypotheses with no frame candidate were skipped: "
            + ", ".join(missing_candidates[:10])
            + (" ..." if len(missing_candidates) > 10 else "")
        )

    return ReviewShortlist(videos=videos, notes=notes)
