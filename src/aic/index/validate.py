"""P1 plus data consistency checks — **a blocking item, run before anything else**.

Acceptance condition (DESIGN.md section 12, Phase 0): exactly **zero** absolute mismatches
over 100 % of batch 1 keyframes. If this step does not pass, no other component may be built:
every internal metric will still look normal while the competition score is zero.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from ..core.frameidx import FrameIndexReport, check_convention, grid_alignment
from ..data.keyframes import load_all_keyframe_tables
from ..data.layout import DataRoot, parse_video_id

__all__ = ["ConsistencyReport", "validate_all"]


@dataclass(slots=True)
class ConsistencyReport:
    """Cross-check across the organiser-supplied data sources."""

    n_videos: int = 0
    n_keyframes: int = 0
    total_hours: float = 0.0
    #: videos present in map-keyframes but missing from another source.
    missing_features: list[str] = field(default_factory=list)
    missing_media_info: list[str] = field(default_factory=list)
    #: (video, csv_rows, npy_rows) where the keyframe count does not match the vector count.
    feature_count_mismatch: list[tuple[str, int, int]] = field(default_factory=list)
    #: (dim, dtype) -> number of videos.
    feature_shapes: dict[tuple[int, str], int] = field(default_factory=dict)
    #: videos whose vectors are not L2-normalised.
    unnormalized_features: list[tuple[str, float]] = field(default_factory=list)
    #: videos where most pts_time values are off the CFR grid of the nominal fps.
    vfr_suspects: list[tuple[str, float]] = field(default_factory=list)
    #: keyframe gap statistics.
    gap_stats: dict[str, float] = field(default_factory=dict)
    bad_video_ids: list[str] = field(default_factory=list)
    #: whether the feature arrays were actually opened. In fast mode they are not, so the
    #: feature-shape check has nothing to assert and must not count as a failure.
    features_checked: bool = True

    @property
    def passed(self) -> bool:
        feature_checks_ok = (
            not self.feature_count_mismatch and len(self.feature_shapes) == 1
            if self.features_checked
            else True
        )
        return (
            self.n_videos > 0
            and not self.missing_features
            and not self.bad_video_ids
            and feature_checks_ok
        )

    def summary(self) -> str:
        lines = [
            "Cross-checking data sources",
            "",
            f"  videos              : {self.n_videos:,}",
            f"  keyframes           : {self.n_keyframes:,}",
            f"  total duration      : {self.total_hours:.1f} hours",
            "  feature shapes      : "
            + (str(self.feature_shapes) if self.features_checked else "not checked (fast mode)"),
        ]
        gaps = self.gap_stats
        if gaps:
            lines.append(
                f"  keyframe gaps       : mean={gaps['mean']:.1f}f median={gaps['median']:.0f}f "
                f"p99={gaps['p99']:.0f}f max={gaps['max']:.0f}f"
            )
            lines.append(
                f"  -> TRAKE coverage ceiling using ONLY existing keyframes: "
                f"~{100 * gaps['trake_hit_rate']:.1f} % per moment "
                f"(10-frame answer window) => full-fps decoding is mandatory"
            )
        for label, items in (
            ("missing CLIP features", self.missing_features),
            ("missing media-info", self.missing_media_info),
            ("vector count mismatch", self.feature_count_mismatch),
            ("malformed video_id", self.bad_video_ids),
            ("unnormalised vectors", self.unnormalized_features),
        ):
            if items:
                lines.append(f"  [!] {label}: {len(items)} -> {items[:5]}")
        if self.vfr_suspects:
            lines.append(
                f"  [i] pts_time off the CFR grid on most keyframes: "
                f"{len(self.vfr_suspects)}/{self.n_videos} videos "
                f"(off-grid rate) -> {self.vfr_suspects[:4]}"
            )
            lines.append(
                "      This follows from rounding pts_time to 4 decimals when 1/fps is a "
                "repeating decimal (29.97). It is NOT evidence of VFR. Genuine VFR "
                "validation needs ffprobe -show_frames — see docs/CONSTRAINTS.md."
            )
        lines.append("")
        lines.append(f"  VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def validate_all(
    root: DataRoot,
    *,
    check_features: bool = True,
    trake_window: int = 10,
) -> tuple[FrameIndexReport, ConsistencyReport]:
    """Run every Phase 0 blocking check.

    Returns ``(P1 report, consistency report)``. Both must be ``passed`` before the index layer
    is built.
    """
    tables = load_all_keyframe_tables(root)
    frame_report = check_convention(
        {video_id: list(table.rows()) for video_id, table in tables.items()}
    )

    report = ConsistencyReport(n_videos=len(tables), features_checked=check_features)
    gaps: list[int] = []

    feature_keys = set(root.clip_features.keys())
    media_info_keys = set(root.media_info.keys())

    for video_id, table in tables.items():
        try:
            parse_video_id(video_id)
        except ValueError:
            report.bad_video_ids.append(video_id)
        report.n_keyframes += len(table)
        gaps.extend(table.gaps())

        alignment = grid_alignment(table.pts_time, table.fps)
        if alignment["is_vfr"]:
            report.vfr_suspects.append((video_id, round(alignment["off_grid_rate"], 3)))

        if f"{video_id}.json" not in media_info_keys:
            report.missing_media_info.append(video_id)
        if f"{video_id}.npy" not in feature_keys:
            report.missing_features.append(video_id)
        elif check_features:
            vectors = np.load(io.BytesIO(root.clip_features.read(f"{video_id}.npy")))
            shape_key = (int(vectors.shape[1]), str(vectors.dtype))
            report.feature_shapes[shape_key] = report.feature_shapes.get(shape_key, 0) + 1
            if vectors.shape[0] != len(table):
                report.feature_count_mismatch.append((video_id, len(table), int(vectors.shape[0])))
            norm = float(np.linalg.norm(vectors[0].astype(np.float32)))
            if abs(norm - 1.0) > 1e-2:
                report.unnormalized_features.append((video_id, round(norm, 4)))

    # Duration from media-info (the `length` field, in seconds).
    total_seconds = 0.0
    for video_id in tables:
        key = f"{video_id}.json"
        if key in media_info_keys:
            info = root.media_info.read_json(key)
            if isinstance(info.get("length"), int | float):
                total_seconds += float(info["length"])
    report.total_hours = total_seconds / 3600.0

    if gaps:
        gap_array = np.asarray(gaps, dtype=np.float64)
        report.gap_stats = {
            "mean": float(gap_array.mean()),
            "median": float(np.median(gap_array)),
            "p90": float(np.percentile(gap_array, 90)),
            "p99": float(np.percentile(gap_array, 99)),
            "max": float(gap_array.max()),
            # Probability that a narrow answer window contains at least one existing keyframe.
            "trake_hit_rate": float(min(1.0, trake_window / gap_array.mean())),
        }
    return frame_report, report
