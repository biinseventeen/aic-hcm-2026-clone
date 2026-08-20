"""Building the internal evaluation set — an **infrastructure** item, not a side task.

The organisers do not supply queries with answers (R3). Without an internal set, no number in
DESIGN.md is verifiable and **every hyperparameter is a guess**: the RRF weights, the assumed
answer span length, the probability calibration function.

The required procedure, and why its order matters
-------------------------------------------------
**Sample a video segment at random FIRST, then write a query describing it.**

The reverse order — think of a query, then go looking for a matching segment — produces a set
biased towards what the system *already* does well (weakness E6). The annotator unconsciously
picks queries they know the system can answer, and the internal score rises while the
competition score does not move. This module *enforces* the correct order: :func:`sample_targets`
samples blind, before a human sees anything.

Minimum targets (DESIGN.md section 12): 40 KIS, 20 Q&A, 15 TRAKE. For TRAKE, the answer windows
must be annotated by hand to **under 10 frames** of precision.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.keyframes import KeyframeTable
from ..index.priors import GROUP_DOMAIN, group_of
from .score import GroundTruth

__all__ = ["MINIMUM_COUNTS", "DevSet", "SampleTarget", "sample_targets"]

#: Minimum internal-set size per task.
MINIMUM_COUNTS: dict[str, int] = {"kis": 40, "qa": 20, "trake": 15}

#: Below this many queries for a task, a difference between two configurations is not
#: statistically meaningful and the report says so explicitly.
MIN_QUERIES_FOR_RESOLUTION = 30


@dataclass(slots=True)
class SampleTarget:
    """A **blindly** sampled video segment for an annotator to write a query about."""

    target_id: str
    video_id: str
    #: the frame range the annotator has to watch.
    start: int
    end: int
    fps: float
    task: str = "kis"
    domain: str = ""
    #: nearby keyframes, so the images can be opened quickly from Keyframes_*.zip.
    keyframes: tuple[int, ...] = ()

    @property
    def start_seconds(self) -> float:
        return self.start / self.fps if self.fps else 0.0

    @property
    def end_seconds(self) -> float:
        return self.end / self.fps if self.fps else 0.0

    def timecode(self) -> str:
        def format_seconds(seconds: float) -> str:
            return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"

        return f"{format_seconds(self.start_seconds)}–{format_seconds(self.end_seconds)}"

    def instruction(self) -> str:
        """Annotator instructions, printed with each target so the order cannot be mistaken."""
        header = (
            f"[{self.target_id}] {self.video_id} frames {self.start}–{self.end} "
            f"({self.timecode()}, {self.domain})\n"
            f"  1. Watch this segment first. Do not imagine a query before watching.\n"
        )
        if self.task == "kis":
            return header + (
                "  2. Write a natural-language description of the event observed.\n"
                "  3. Record the frame range [s, e] in which EVERY frame satisfies it."
            )
        if self.task == "qa":
            return header + (
                "  2. Write an event description PLUS a question about information in it.\n"
                "  3. Record the frame range [s, e] and the correct answer, together with\n"
                '     equivalent phrasings (for example "5" and "năm").'
            )
        return header + (
            "  2. Identify the N semantic moments of a structured event sequence.\n"
            "  3. For EACH moment, record a frame range [s_j, e_j] — the width must be\n"
            "     UNDER 10 FRAMES. This is the most time-consuming part and cannot be skipped.\n"
            "  4. If the segment contains no structured event sequence, mark it SKIPPED\n"
            "     and record why, instead of forcing a fit."
        )


def sample_targets(
    tables: dict[str, KeyframeTable],
    *,
    counts: dict[str, int] | None = None,
    seed: int = 0,
    min_seconds: float = 2.0,
    max_seconds: float = 12.0,
    trake_groups: tuple[str, ...] = ("L23", "L24"),
    stratify: bool = True,
) -> list[SampleTarget]:
    """Blindly sample video segments for annotation.

    ``stratify`` allocates KIS and Q&A samples **by share of hours** per content domain rather
    than by video count. This is necessary because the corpus is strongly skewed: L26 (cooking)
    is 57 % of videos but only 33 % of duration, while L25 is 10 % of videos and 28 % of
    duration. Sampling by video count would make the internal set almost entirely cooking.

    TRAKE samples are **restricted** to ``trake_groups``: in batch 1 only cycling (L23) and lion
    dance (L24) contain event sequences structured enough to define semantic moments. Sampling
    TRAKE over cooking videos produces labels the annotators themselves cannot agree on.
    """
    requested = {**MINIMUM_COUNTS, **(counts or {})}
    rng = random.Random(seed)
    targets: list[SampleTarget] = []

    def pick(video_pool: list[str], task: str, count: int, prefix: str) -> None:
        if not video_pool:
            return
        for index in range(count):
            for _attempt in range(50):
                video_id = rng.choice(video_pool)
                table = tables[video_id]
                if len(table) < 3:
                    continue
                fps = table.fps or 25.0
                span_frames = int(rng.uniform(min_seconds, max_seconds) * fps)
                last_frame = table.duration_frames
                if last_frame <= span_frames + 1:
                    continue
                start = rng.randint(0, last_frame - span_frames)
                end = start + span_frames
                domain, _ = GROUP_DOMAIN.get(group_of(video_id), ("unknown", "?"))
                targets.append(
                    SampleTarget(
                        target_id=f"{prefix}{index + 1:03d}",
                        video_id=video_id,
                        start=start,
                        end=end,
                        fps=fps,
                        task=task,
                        domain=domain,
                        keyframes=tuple(table.keyframes_in_range(start, end)),
                    )
                )
                break

    all_videos = sorted(tables)
    if stratify:
        # Weight by the known total duration of each group.
        hours_per_group: dict[str, float] = {}
        for video_id, table in tables.items():
            group = group_of(video_id)
            duration = table.duration_frames / max(table.fps or 25.0, 1e-9)
            hours_per_group[group] = hours_per_group.get(group, 0.0) + duration
        groups = sorted(hours_per_group)
        total_hours = sum(hours_per_group.values()) or 1.0
        for task, prefix in (("kis", "K"), ("qa", "Q")):
            wanted = requested.get(task, 0)
            allocated = 0
            for position, group in enumerate(groups):
                share = hours_per_group[group] / total_hours
                for_group = round(share * wanted)
                if position == len(groups) - 1:
                    for_group = max(0, wanted - allocated)
                allocated += for_group
                if for_group:
                    pick(
                        [v for v in all_videos if group_of(v) == group],
                        task,
                        for_group,
                        f"{prefix}{group}-",
                    )
    else:
        pick(all_videos, "kis", requested.get("kis", 0), "K")
        pick(all_videos, "qa", requested.get("qa", 0), "Q")

    trake_pool = [v for v in all_videos if group_of(v) in trake_groups]
    pick(trake_pool or all_videos, "trake", requested.get("trake", 0), "T")
    return targets


@dataclass
class DevSet:
    """The internal evaluation set: queries plus annotated ground truth."""

    queries: dict[str, str] = field(default_factory=dict)
    truths: dict[str, GroundTruth] = field(default_factory=dict)
    #: targets that were sampled but skipped by the annotator, with the reason.
    skipped: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.truths)

    def by_task(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for query_id, truth in self.truths.items():
            grouped.setdefault(truth.task, []).append(query_id)
        return grouped

    def coverage_report(self) -> str:
        counts = {task: len(ids) for task, ids in self.by_task().items()}
        lines = ["Internal evaluation set", ""]
        complete = True
        for task, needed in MINIMUM_COUNTS.items():
            have = counts.get(task, 0)
            status = "OK" if have >= needed else f"SHORT BY {needed - have}"
            if have < needed:
                complete = False
            lines.append(f"  {task:<8} {have:>3} / {needed:<3} minimum   {status}")
        if self.skipped:
            lines.append(f"  skipped: {len(self.skipped)} targets")
        # Statistical resolution warning — matters so the measurements are not over-trusted.
        n_trake = counts.get("trake", 0)
        if 0 < n_trake < MIN_QUERIES_FOR_RESOLUTION:
            lines.append(
                f"  [i] {n_trake} TRAKE queries give low statistical resolution: one query is "
                f"{100 / n_trake:.1f} percentage points. A small difference between two "
                "configurations is NOT meaningful (weakness N3)."
            )
        lines.append("")
        lines.append(
            f"  VERDICT: {'minimum targets met' if complete else 'minimum targets NOT met'}"
        )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "queries": self.queries,
                    "truths": {key: asdict(value) for key, value in self.truths.items()},
                    "skipped": self.skipped,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> DevSet:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        truths = {}
        for query_id, raw in payload.get("truths", {}).items():
            fields = dict(raw)
            if fields.get("span") is not None:
                fields["span"] = tuple(fields["span"])
            fields["spans"] = [tuple(span) for span in fields.get("spans", [])]
            truths[query_id] = GroundTruth(**fields)
        return cls(
            queries=payload.get("queries", {}),
            truths=truths,
            skipped=payload.get("skipped", {}),
        )

    @classmethod
    def template(cls, targets: list[SampleTarget], path: str | Path) -> Path:
        """Write the blank file for an annotator to fill in.

        The file pre-fills ``video_id`` and initialises ``span`` to exactly the sampled segment;
        the annotator *narrows* the span and writes the query. Pre-filling ``video_id`` removes
        one data-entry error source, but the span must be corrected: the sampled segment is 2–12
        seconds long, whereas the true answer span is usually far narrower.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_instructions": [
                "Watch the video segment FIRST, then write the query. Never the reverse (E6).",
                "Narrow 'span' to the range in which every frame satisfies the description.",
                "TRAKE: each moment needs a span UNDER 10 FRAMES wide.",
                "Cannot write a natural query? Move the target to 'skipped' with a reason.",
            ],
            "queries": {target.target_id: "" for target in targets},
            "truths": {
                target.target_id: {
                    "query_id": target.target_id,
                    "task": target.task,
                    "video_id": target.video_id,
                    "span": ([target.start, target.end] if target.task in ("kis", "qa") else None),
                    "spans": ([[target.start, target.start + 9]] if target.task == "trake" else []),
                    "answer": "",
                    "answer_aliases": [],
                    "note": f"blind sample {target.timecode()} ({target.domain})",
                }
                for target in targets
            },
            "skipped": {},
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
