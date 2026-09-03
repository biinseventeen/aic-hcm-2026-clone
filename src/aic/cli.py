"""Command line interface: ``python -m aic.cli <command>``.

The command order mirrors the mandatory implementation order in DESIGN.md section 12:

    inventory         list the data, confirm it is present
    validate          P1 plus data cross-checks            <-- BLOCKING ITEM
    build-index       build the dense, shot and text indexes
    devset            blind sampling for the internal evaluation set
    query             run one query, print diagnostics
    review            human-in-the-loop video/frame shortlist
    run               run a whole query set, produce submission files
    evaluate          score submissions against the internal set
    check-submission  validate submission files before sending
    serve             run the HTTP backend

Do not run ``build-index`` before ``validate`` reports PASS. A wrong frame index convention causes
silent score loss: every internal metric still looks normal while the competition score is zero.

This module is a thin presentation layer over :mod:`aic.service`; all logic lives there.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# The Windows console defaults to cp1252 and raises UnicodeEncodeError on Vietnamese text.
# Wrap stdout and stderr before printing anything.
from .console import enable_utf8_stdio

enable_utf8_stdio()

from .config import Config, load_config  # noqa: E402
from .log import setup_cli_logging  # noqa: E402
from .service import Engine, ReviewResult, SolveResult  # noqa: E402

#: Diagnostic rows printed per command. Anything longer belongs in the JSON report.
_MAX_ERRORS_SHOWN = 30
_MAX_WARNINGS_SHOWN = 15


# --------------------------------------------------------------------------
# Commands
#
# Every command takes (args, cfg) so main() can dispatch uniformly; some ignore one of them.
# --------------------------------------------------------------------------


def cmd_inventory(args, cfg: Config) -> int:
    from .data.layout import DataRoot

    print(DataRoot(cfg.paths.data_root).inventory(count_all=args.count_all))
    return 0


def cmd_validate(args, cfg: Config) -> int:
    from .data.layout import DataRoot
    from .index.validate import validate_all

    root = DataRoot(cfg.paths.data_root)
    started = time.time()
    frame_report, consistency = validate_all(root, check_features=not args.fast)
    print(frame_report.summary())
    print()
    print(consistency.summary())
    print(f"\n({time.time() - started:.1f}s)")

    out = Path(cfg.paths.report_dir) / "validate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "frame_index": {
                    "n_checked": frame_report.n_checked,
                    "matches": frame_report.matches,
                    "passed": frame_report.passed,
                    "n_mismatches": len(frame_report.mismatches),
                    "unusual_fps": frame_report.unusual_fps,
                    "n_duplicate_frame_idx": len(frame_report.duplicate_frame_idx),
                },
                "consistency": {
                    "n_videos": consistency.n_videos,
                    "n_keyframes": consistency.n_keyframes,
                    "total_hours": consistency.total_hours,
                    "gap_stats": consistency.gap_stats,
                    "features_checked": consistency.features_checked,
                    "passed": consistency.passed,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report: {out}")
    if not (frame_report.passed and consistency.passed):
        print("\nFAIL — do not build the index before the mismatch is traced to its cause.")
        return 1
    return 0


def cmd_build_index(args, cfg: Config) -> int:
    from .data.features import build_dense_index
    from .data.keyframes import load_all_keyframe_tables
    from .data.layout import DataRoot
    from .index.shots import build_shot_table
    from .index.text import (
    TextIndex,
    docs_from_media_info,
    docs_from_objects,
    docs_from_titles,
)

    root = DataRoot(cfg.paths.data_root)
    out = Path(cfg.paths.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = load_all_keyframe_tables(root)
    n_keyframes = sum(len(table) for table in tables.values())
    print(f"{len(tables)} videos, {n_keyframes:,} keyframes")

    only = getattr(args, "only", "all")
    if only in ("all", "dense"):
        print("\n[1/3] dense visual index (P3)")
        build_dense_index(root, tables=tables).save(out)
    else:
        print(f"\n[1/3] dense visual index: skipped (--only {only})")

    print("\n[2/3] shot table (P2)")
    media_info = {video_id: root.media_info.read_json(f"{video_id}.json") for video_id in tables}
    last_frames = {
        video_id: int(info["length"] * (tables[video_id].fps or cfg.fps_default))
        for video_id, info in media_info.items()
        if isinstance(info.get("length"), int | float)
    }
    shot_table = build_shot_table(tables, fps_default=cfg.fps_default, last_frames=last_frames)
    if only in ("all", "shots"):
        shot_table.save(out / "shots.json")
    stats = shot_table.stats(tables)
    print(
        f"  {len(shot_table):,} shots; median length {stats['median_s']:.1f}s, "
        f"p99 {stats['p99_s']:.1f}s, max {stats['max_s']:.1f}s"
    )
    if stats["over_20s_pct"] < 0.01:
        print("  [i] no shot exceeds 20s: long-shot splitting does not trigger on batch 1")

    if only not in ("all", "text"):
        print(f"\nindex written to {out}")
        return 0

    print("\n[3/3] sparse text index (P4/P5 — currently media-info only)")
    text_index = TextIndex(docs=docs_from_media_info(media_info, last_frames=last_frames)).build(
        verbose=True
    )
    text_index.save(out / "text_media.json")
    title_index = TextIndex(docs=docs_from_titles(media_info)).build(verbose=False)
    title_index.save(out / "text_title.json")
    print("  building object index...")

    object_docs = docs_from_objects(
        root.objects,
        tables,
        min_score=0.20,
        max_labels=20,
    )

    if object_docs:
        object_index = TextIndex(
            docs=object_docs
        ).build(
            with_fuzzy=False,
            verbose=True,
        )

        object_index.save(
            out / "text_objects.json"
        )

        print(
            f"  object index: {len(object_index):,} frame-level documents"
        )
    else:
        print(
            "  [!] object data present but produced no indexable documents"
        )
    print(f"  title-only index: {len(title_index)} documents (boilerplate-free channel)")
    print(
        "  [!] no OCR and no speech channel yet. Both text retrieval channels are missing — "
        "see docs/CONSTRAINTS.md, items G2 and G3."
    )
    print(f"\nindex written to {out}")
    return 0


def cmd_devset(args, cfg: Config) -> int:
    from .data.keyframes import load_all_keyframe_tables
    from .data.layout import DataRoot
    from .eval.devset import DevSet, sample_targets

    root = DataRoot(cfg.paths.data_root)
    tables = load_all_keyframe_tables(root)
    counts = {"kis": args.kis, "qa": args.qa, "trake": args.trake}
    targets = sample_targets(tables, counts=counts, seed=args.seed)
    devset_dir = Path(cfg.paths.devset_dir)
    template_path = DevSet.template(targets, devset_dir / "devset.json")
    instructions_path = devset_dir / "annotation_instructions.txt"
    instructions_path.write_text(
        "BLIND SAMPLING — watch the video FIRST, write the query AFTER (weakness E6)\n"
        + "=" * 72
        + "\n\n"
        + "\n\n".join(target.instruction() for target in targets),
        encoding="utf-8",
    )
    by_task: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for target in targets:
        by_task[target.task] = by_task.get(target.task, 0) + 1
        by_domain[target.domain] = by_domain.get(target.domain, 0) + 1
    print(f"sampled {len(targets)} segments:")
    print(f"  by task   : {by_task}")
    print(f"  by domain : {by_domain}")
    print(f"\ntemplate to fill in : {template_path}")
    print(f"instructions        : {instructions_path}")
    print("\nOnce filled in, run: python -m aic.cli evaluate")
    return 0


def _solve_and_print(
    engine: Engine,
    query_id: str,
    text: str,
    *,
    task_hint=None,
    verbose: bool = True,
    hedge_answers: bool = True,
    answers: list[tuple[str, float]] | None = None,
    pins: list[tuple[str, int]] | None = None,
    english: str = "",
) -> SolveResult | None:
    """Call the engine and print diagnostics. Returns None when the query itself is invalid."""
    try:
        result = engine.solve(
            text,
            query_id=query_id,
            task_hint=task_hint,
            hedge_answers=hedge_answers,
            answers=answers,
            pins=pins,
            english=english,
        )
    except ValueError as exc:
        print(f"  [!] {exc}")
        return None
    if verbose:
        print(result.query.summary())
        print(f"  candidates: {result.n_candidates}  channels: {result.channel_sizes}")
        print(result.report())
    return result


def cmd_query(args, cfg: Config) -> int:
    engine = Engine.load(cfg, allow_stub=args.allow_stub)
    print()
    result = _solve_and_print(
        engine,
        args.id,
        args.text,
        task_hint=args.task,
        hedge_answers=not args.no_hedge_answers,
    )
    if result is None:
        return 1
    if args.json:
        print(json.dumps(result.to_dict(top=args.top), ensure_ascii=False, indent=2))
    if args.write:
        path, issues = engine.write(result)
        errors = [issue for issue in issues if issue.severity == "error"]
        print(f"\nwrote {path} ({len(errors)} errors, {len(issues) - len(errors)} warnings)")
        for error in errors[:5]:
            print(f"  {error}")
    return 0


def cmd_review(args, cfg: Config) -> int:
    """Retrieve a human-review shortlist without running the solver/allocator."""
    engine = Engine.load(cfg, allow_stub=args.allow_stub)
    print()
    result: ReviewResult = engine.review(
        args.text,
        task_hint=args.task,
        top_videos=args.top_videos,
        max_videos=args.max_videos,
        rescue_per_channel=args.rescue_per_channel,
        tier1_videos=args.tier1_videos,
        tier1_frames=args.tier1_frames,
        tier2_videos=args.tier2_videos,
        tier2_frames=args.tier2_frames,
        later_frames=args.later_frames,
        min_gap_seconds=args.min_gap_seconds,
    )

    print(result.query.summary())
    print(
        f"  candidates: {result.n_candidates}  "
        f"channels: {result.channel_sizes}"
    )
    print(result.report())

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args, cfg: Config) -> int:
    from .submit.writer import package_submission

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    if not isinstance(queries, dict):
        print("the query file must be a JSON object {query_id: text}")
        return 2

    # Human-supplied Q&A answers. Until a VQA model exists this is the only way the answer axis
    # can score at all, so it is worth a file of its own rather than a code change per round.
    supplied: dict[str, list[tuple[str, float]]] = {}
    if getattr(args, "answers", None):
        payload = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        for query_id, entries in payload.items():
            if query_id.startswith("_"):
                continue
            supplied[query_id] = [
                (entry["text"], float(entry.get("prob", 1.0)))
                if isinstance(entry, dict)
                else (str(entry), 1.0)
                for entry in entries
            ]
        print(f"answer hypotheses supplied for {len(supplied)} queries: {sorted(supplied)}")

    english: dict[str, str] = {}
    if getattr(args, "english", None):
        payload = json.loads(Path(args.english).read_text(encoding="utf-8"))
        english = {k: v for k, v in payload.items() if not k.startswith("_") and isinstance(v, str)}
        print(f"English renderings supplied for {len(english)} queries")

    pins: dict[str, list[tuple[str, int]]] = {}
    if getattr(args, "pins", None):
        payload = json.loads(Path(args.pins).read_text(encoding="utf-8"))
        for query_id, entries in payload.items():
            if query_id.startswith("_"):
                continue
            pins[query_id] = [(entry["video_id"], int(entry["frame"])) for entry in entries]
        print(f"verified rows pinned for {len(pins)} queries: {sorted(pins)}")
    engine = Engine.load(cfg, allow_stub=args.allow_stub)
    if engine.encoder_is_stub and not args.allow_stub_submission:
        print(
            "\nRefusing to write submission files: the encoder in use is a stub, so retrieval "
            "results are noise.\nInstall a real model (uv sync --extra encoder), or pass "
            "--allow-stub-submission to exercise the pipeline."
        )
        return 2

    out_dir = Path(cfg.paths.submission_dir)
    if getattr(args, "out", None):
        # Two published query sets number their queries p1-1..p1-25 while assigning them
        # different tasks, so one directory per set is the only way not to overwrite half of it.
        candidate = Path(args.out)
        out_dir = candidate if candidate.is_absolute() else out_dir / candidate
    files, total_errors = [], 0
    for position, (query_id, text) in enumerate(queries.items(), 1):
        print(f"\n{'=' * 72}\n[{position}/{len(queries)}] {query_id}: {text[:70]}")
        result = _solve_and_print(
            engine,
            query_id,
            text,
            verbose=args.verbose,
            hedge_answers=not args.no_hedge_answers,
            answers=supplied.get(query_id),
            pins=pins.get(query_id),
            english=english.get(query_id, ""),
        )
        if result is None:
            continue
        path, issues = engine.write(result, out_dir)
        errors = [issue for issue in issues if issue.severity == "error"]
        total_errors += len(errors)
        files.append(path)
        print(
            f"  -> {path.name}  {result.n_answers} rows, {len(errors)} errors "
            f"({result.elapsed_s:.1f}s)"
        )
        for error in errors[:3]:
            print(f"     {error}")

    if files and not total_errors:
        archive = package_submission(files, out_dir / "submission.zip")
        print("  the archive holds a submission/ directory, as the result specification requires")
        print(f"\npackaged: {archive}")
    elif total_errors:
        print(f"\nNOT packaged: {total_errors} errors remain. Fix them before submitting.")
        return 1
    return 0


def cmd_serve(args, cfg: Config) -> int:  # noqa: ARG001 — uniform dispatch signature
    """Run the HTTP backend. Requires ``uv sync --extra backend``."""
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn/fastapi are missing. Install them into the workspace environment with:\n"
            "  uv sync --extra backend"
        )
        return 2

    # Load the engine once at startup: a missing index becomes an immediately visible error
    # rather than a 500 on the first user request.
    os.environ.setdefault("AIC_EAGER_LOAD", "1")
    if args.allow_stub:
        os.environ["AIC_ALLOW_STUB"] = "1"
    os.environ.setdefault("AIC_MAX_CONCURRENCY", str(args.concurrency))
    print(f"http://{args.host}:{args.port}/docs  (Ctrl+C to stop)")
    uvicorn.run("aic.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_evaluate_review(args, cfg: Config) -> int:
    """Evaluate the human-review objective, not the competition submission score."""
    from statistics import median

    from .review import build_review_shortlist

    devset_path = Path(args.devset or Path(cfg.paths.devset_dir) / "devset.json")
    payload = json.loads(devset_path.read_text(encoding="utf-8"))
    queries = {
        query_id: text
        for query_id, text in payload.get("queries", {}).items()
        if isinstance(text, str) and text.strip()
    }
    truths = payload.get("truths", {})
    if not queries:
        print("no labelled queries in the devset")
        return 1

    engine = Engine.load(cfg, allow_stub=args.allow_stub)
    rows = []
    try:
        for query_id, text in queries.items():
            truth = truths.get(query_id)
            if not truth:
                continue

            query, retrieval = engine.retrieve(
                text,
                task_hint=truth.get("task") or None,
            )
            shortlist = build_review_shortlist(
                retrieval,
                fps_by_video=engine.fps,
                top_videos=args.top_videos,
                max_videos=args.max_videos,
                rescue_per_channel=args.rescue_per_channel,
                tier1_videos=args.tier1_videos,
                tier1_frames=args.tier1_frames,
                tier2_videos=args.tier2_videos,
                tier2_frames=args.tier2_frames,
                later_frames=args.later_frames,
                min_gap_seconds=args.min_gap_seconds,
                default_fps=cfg.fps_default,
            )

            truth_video = str(truth.get("video_id") or "")
            auto_ranked = sorted(
                retrieval.video_scores.items(),
                key=lambda item: -float(item[1]),
            )
            auto_video_rank = next(
                (
                    rank
                    for rank, (video_id, _score) in enumerate(auto_ranked, 1)
                    if video_id == truth_video
                ),
                None,
            )

            recall_scores = retrieval.recall_video_scores or retrieval.video_scores
            recall_ranked = sorted(
                recall_scores.items(),
                key=lambda item: -float(item[1]),
            )
            video_rank = next(
                (
                    rank
                    for rank, (video_id, _score) in enumerate(recall_ranked, 1)
                    if video_id == truth_video
                ),
                None,
            )

            shown = False
            rescued = False
            first_thumbnail = None
            frame_hit = False
            nearest_gap = None

            span_raw = truth.get("span") or []
            span = None
            if isinstance(span_raw, list) and len(span_raw) == 2:
                span = (int(span_raw[0]), int(span_raw[1]))

            for video in shortlist.videos:
                if video.video_id == truth_video:
                    shown = True
                    rescued = video.rescued

            # Measure the actual human viewing order: one thumbnail from every
            # video first, then second/third/fourth temporal hypotheses.
            for thumbnail_index, (video, frame) in enumerate(
                shortlist.scan_order(),
                1,
            ):
                if video.video_id != truth_video:
                    continue
                if first_thumbnail is None:
                    first_thumbnail = thumbnail_index
                if span is None:
                    continue
                if span[0] <= frame.frame <= span[1]:
                    gap = 0
                    frame_hit = True
                elif frame.frame < span[0]:
                    gap = span[0] - frame.frame
                else:
                    gap = frame.frame - span[1]
                nearest_gap = gap if nearest_gap is None else min(nearest_gap, gap)

            truth_fps = float(engine.fps.get(truth_video, cfg.fps_default))
            gap_seconds = (
                (float(nearest_gap) / truth_fps)
                if nearest_gap is not None and truth_fps > 0
                else None
            )
            row = {
                "query_id": query_id,
                "video_rank": video_rank,
                "auto_video_rank": auto_video_rank,
                "shown": shown,
                "rescued": rescued,
                "first_thumbnail": first_thumbnail,
                "frame_hit": frame_hit,
                "nearest_frame_gap": nearest_gap,
                "nearest_gap_seconds": gap_seconds,
                "near_5s": gap_seconds is not None and gap_seconds <= 5.0,
                "near_15s": gap_seconds is not None and gap_seconds <= 15.0,
                "near_30s": gap_seconds is not None and gap_seconds <= 30.0,
                "near_60s": gap_seconds is not None and gap_seconds <= 60.0,
                "n_thumbnails": shortlist.n_frames,
            }
            rows.append(row)
            gap_s_text = "None" if gap_seconds is None else f"{gap_seconds:.1f}s"
            print(
                f"{query_id:<10} "
                f"recall_rank={str(video_rank):<4} "
                f"auto_rank={str(auto_video_rank):<4} "
                f"shown={str(shown):<5} "
                f"rescued={str(rescued):<5} "
                f"thumb={str(first_thumbnail):<4} "
                f"frame_hit={str(frame_hit):<5} "
                f"gap={str(nearest_gap):<5} ({gap_s_text})"
            )
    finally:
        engine.close()

    if not rows:
        print("no query had ground truth")
        return 1

    def recall_at(k: int) -> float:
        return sum(
            row["video_rank"] is not None and row["video_rank"] <= k
            for row in rows
        ) / len(rows)

    shown_count = sum(bool(row["shown"]) for row in rows)
    frame_hit_count = sum(bool(row["frame_hit"]) for row in rows)
    thumbnails = [
        int(row["first_thumbnail"])
        for row in rows
        if row["first_thumbnail"] is not None
    ]
    shown_gaps = [
        float(row["nearest_gap_seconds"])
        for row in rows
        if row["shown"] and row["nearest_gap_seconds"] is not None
    ]

    print("\n=== HUMAN REVIEW EVALUATION ===")
    for k in (5, 10, 20, 50, 100):
        print(f"Recall-pool Video R@{k:<3}: {recall_at(k):.3f}")
    print(
        f"truth video shown       : {shown_count}/{len(rows)} "
        f"({shown_count / len(rows):.1%})"
    )
    print(
        f"selected frame inside GT: {frame_hit_count}/{len(rows)} "
        f"({frame_hit_count / len(rows):.1%})"
    )
    for seconds, key in ((5, "near_5s"), (15, "near_15s"), (30, "near_30s"), (60, "near_60s")):
        count = sum(bool(row[key]) for row in rows)
        print(
            f"temporal cue <= {seconds:>2}s    : {count}/{len(rows)} "
            f"({count / len(rows):.1%})"
        )
    print(
        "median nearest gap (shown): "
        + (f"{median(shown_gaps):.1f}s" if shown_gaps else "n/a")
    )
    print(
        "median thumbnails to truth: "
        + (f"{median(thumbnails):g}" if thumbnails else "n/a")
    )

    out = Path(cfg.paths.report_dir) / "evaluate_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n": len(rows),
                "video_recall": {
                    str(k): recall_at(k) for k in (5, 10, 20, 50, 100)
                },
                "truth_video_shown": shown_count,
                "frame_hits": frame_hit_count,
                "temporal_cue_within_seconds": {
                    "5": sum(bool(row["near_5s"]) for row in rows),
                    "15": sum(bool(row["near_15s"]) for row in rows),
                    "30": sum(bool(row["near_30s"]) for row in rows),
                    "60": sum(bool(row["near_60s"]) for row in rows),
                },
                "median_nearest_gap_seconds_shown": (
                    median(shown_gaps) if shown_gaps else None
                ),
                "median_thumbnails_to_truth": (
                    median(thumbnails) if thumbnails else None
                ),
                "per_query": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report: {out}")
    return 0


def cmd_evaluate(args, cfg: Config) -> int:
    from .eval.devset import DevSet
    from .eval.score import EvalReport, score_submission

    devset = DevSet.load(args.devset or Path(cfg.paths.devset_dir) / "devset.json")
    print(devset.coverage_report())
    labelled = {query_id: text for query_id, text in devset.queries.items() if text.strip()}
    if not labelled:
        print("\nno query has been annotated yet — run `devset`, then fill in the file.")
        return 1

    supplied: dict[str, list[tuple[str, float]]] = {}
    if getattr(args, "answers", None):
        payload = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        for query_id, entries in payload.items():
            if query_id.startswith("_"):
                continue
            supplied[query_id] = [
                (entry["text"], float(entry.get("prob", 1.0)))
                if isinstance(entry, dict)
                else (str(entry), 1.0)
                for entry in entries
            ]

    pinned: dict[str, list[tuple[str, int]]] = {}
    if getattr(args, "pins", None):
        payload = json.loads(Path(args.pins).read_text(encoding="utf-8"))
        for query_id, entries in payload.items():
            if not query_id.startswith("_"):
                pinned[query_id] = [(e["video_id"], int(e["frame"])) for e in entries]

    engine = Engine.load(cfg, allow_stub=args.allow_stub)
    report = EvalReport()
    print()
    for query_id, text in labelled.items():
        truth = devset.truths.get(query_id)
        if truth is None:
            continue
        result = _solve_and_print(
            engine,
            query_id,
            text,
            task_hint=truth.task,
            verbose=False,
            answers=supplied.get(query_id),
            pins=pinned.get(query_id),
        )
        if result is None:
            continue
        score = score_submission(result.submission, truth)
        report.scores.append(score)
        first_note = "; ".join(score.notes[:1])
        print(
            f"  {query_id:<10} {truth.task:<6} Final={score.final:.3f} "
            f"first_hit={score.first_hit} {first_note}"
        )
    print()
    print(report.summary())
    out = Path(cfg.paths.report_dir) / "evaluate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "mean_final": report.mean_final,
                "n": len(report.scores),
                "per_query": [
                    {
                        "query_id": score.query_id,
                        "task": score.task,
                        "final": score.final,
                        "first_hit": score.first_hit,
                        "best_r": score.best_r,
                        "right_video_wrong_frame": score.right_video_wrong_frame,
                    }
                    for score in report.scores
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out}")
    return 0


def cmd_check_submission(args, cfg: Config) -> int:
    from .data.layout import DataRoot
    from .submit.writer import validate_submission_dir

    known_videos = None
    try:
        known_videos = {key[:-4] for key in DataRoot(cfg.paths.data_root).map_keyframes}
    except Exception as exc:
        print(f"  [!] could not read the video list ({exc}); skipping the video_id check")
    directory = args.dir or cfg.paths.submission_dir
    issues = validate_submission_dir(directory, known_videos=known_videos)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    print(f"validating {directory}: {len(errors)} errors, {len(warnings)} warnings")
    for issue in errors[:_MAX_ERRORS_SHOWN]:
        print(f"  {issue}")
    for issue in warnings[:_MAX_WARNINGS_SHOWN]:
        print(f"  {issue}")
    if errors:
        print("\nERRORS PRESENT — do not submit until every one is fixed.")
        return 1
    print("\nno blocking errors.")
    print(
        "Note: the file format follows the published result specification (one .csv per query, "
        "no header, UTF-8, archive holding a submission/ directory). What is still unconfirmed "
        "is how a Q&A answer is compared — see docs/SUBMISSION.md §2."
    )
    return 0


def cmd_selftest(args, cfg: Config) -> int:  # noqa: ARG001 — uniform dispatch signature
    """Print a set of sample submission files to send the organisers for confirmation."""
    from .submit.writer import (
        MAX_ANSWER_CHARS,
        Answer,
        QuerySubmission,
        SubmissionNaming,
        write_submission,
    )

    naming = SubmissionNaming()
    out_dir = Path(cfg.paths.submission_dir) / "_selftest"
    samples = [
        QuerySubmission(
            "1",
            "kis",
            [Answer("L21_V001", frame=1500), Answer("L21_V001", frame=1525)],
        ),
        QuerySubmission(
            "2",
            "qa",
            [
                Answer("L26_V001", frame=3450, answer="5"),
                Answer("L26_V001", frame=3450, answer="năm"),
            ],
        ),
        QuerySubmission(
            "3", "trake", [Answer("L23_V001", frames=(101, 156, 203, 251))], n_moments=4
        ),
    ]
    print("Sample submission files (to ask the organisers to confirm the format):\n")
    for sample in samples:
        path, _ = write_submission(sample, out_dir, naming=naming, strict=False)
        print(f"--- {path.name} ---")
        print(path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip())
        print()
    print("Format in use — all of it stated by the result specification:")
    print(f"  - filename            : {naming.filename}")
    print(f"  - delimiter           : {naming.delimiter!r}   header row: {naming.include_header}")
    print(f"  - video_id extension  : {naming.video_extension!r} (empty = no .mp4)")
    print(f"  - archive             : {naming.zip_name} containing {naming.zip_dir}/")
    print(f"  - Q&A answer limit    : {MAX_ANSWER_CHARS} characters")
    print("Still to confirm with the organisers:")
    print("  - Q&A: is the answer compared semantically or as an exact string? The published")
    print("    specification says both, in two different places.")
    print("  - Q&A: may several rows share a (video_id, frame_id) with different answers?")
    print(f"\nfiles at: {out_dir}")
    return 0


# --------------------------------------------------------------------------
# Argument parsing and dispatch
# --------------------------------------------------------------------------

_HEDGE_HELP = (
    "Q&A: keep only one answer per (video_id, frame_id). Use this if the organisers "
    "deduplicate on that pair — see docs/SUBMISSION.md, question 4."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aic", description="AIC 2026 video retrieval — preliminary round"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON configuration file (defaults to configs/default.json at the project root)",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    inventory = subparsers.add_parser("inventory", help="list the organiser-supplied data")
    inventory.add_argument(
        "--count-all",
        action="store_true",
        help="count every leaf file of the keyframes and objects families (slow: 177,321 paths)",
    )

    validate = subparsers.add_parser("validate", help="P1 plus cross-checks (BLOCKING ITEM)")
    validate.add_argument("--fast", action="store_true", help="skip the CLIP feature checks")

    build_index = subparsers.add_parser(
        "build-index", help="build the dense, shot and text indexes"
    )
    build_index.add_argument(
        "--only",
        choices=("all", "dense", "shots", "text"),
        default="all",
        help="rebuild one stage only. The dense index is 173 MiB and takes minutes; iterating "
        "on the sparse channel does not need it.",
    )

    devset = subparsers.add_parser("devset", help="blind sampling for the internal eval set")
    devset.add_argument("--kis", type=int, default=40)
    devset.add_argument("--qa", type=int, default=20)
    devset.add_argument("--trake", type=int, default=15)
    devset.add_argument("--seed", type=int, default=0)

    query = subparsers.add_parser("query", help="run one query")
    query.add_argument("text")
    query.add_argument("--id", default="1")
    query.add_argument("--task", choices=["kis", "qa", "trake"], default=None)
    query.add_argument("--write", action="store_true", help="write the submission file")
    query.add_argument(
        "--json",
        action="store_true",
        help="print the result as JSON — the same shape the backend's /solve returns",
    )
    query.add_argument("--top", type=int, default=None, help="with --json: cap answers printed")
    query.add_argument(
        "--allow-stub",
        action="store_true",
        help="allow the stub encoder (pipeline testing only)",
    )
    query.add_argument("--no-hedge-answers", action="store_true", help=_HEDGE_HELP)

    review = subparsers.add_parser(
        "review",
        help="human-in-the-loop shortlist from retrieval; does not run P10/P13",
    )
    review.add_argument("text")
    review.add_argument("--task", choices=["kis", "qa", "trake"], default=None)
    review.add_argument(
        "--top-videos",
        type=int,
        default=50,
        help="normal fused-ranking videos kept before channel rescue",
    )
    review.add_argument(
        "--max-videos",
        type=int,
        default=60,
        help="maximum videos after adding per-channel rescue hypotheses",
    )
    review.add_argument(
        "--rescue-per-channel",
        type=int,
        default=2,
        help="extra video hypotheses each retrieval channel may nominate",
    )
    review.add_argument("--tier1-videos", type=int, default=10)
    review.add_argument("--tier1-frames", type=int, default=4)
    review.add_argument("--tier2-videos", type=int, default=20)
    review.add_argument("--tier2-frames", type=int, default=3)
    review.add_argument(
        "--later-frames",
        type=int,
        default=2,
        help="frames for rank > tier2 and channel-rescued videos",
    )
    review.add_argument(
        "--min-gap-seconds",
        type=float,
        default=8.0,
        help="minimum time gap between shown frames from the same video",
    )
    review.add_argument(
        "--json",
        action="store_true",
        help="also print the review result as JSON",
    )
    review.add_argument(
        "--allow-stub",
        action="store_true",
        help="allow the stub encoder (pipeline testing only)",
    )

    run = subparsers.add_parser("run", help="run a whole query set and write submissions")
    run.add_argument("queries", help="JSON {query_id: text}")
    run.add_argument(
        "--answers",
        default=None,
        help="JSON {query_id: [{text, prob}, ...]} of Q&A answer hypotheses. Without it the "
        "answer axis of every Q&A row scores zero, because no VQA model is built (G5).",
    )
    run.add_argument(
        "--pins",
        default=None,
        help="JSON {query_id: [{video_id, frame}, ...]} of rows a human has verified by looking "
        "at the frame. They go to the head of the list, where slot 1 is worth a fifth of the "
        "query's score.",
    )
    run.add_argument(
        "--english",
        default=None,
        help="JSON {query_id: english text} feeding the dense_translated channel. CLIP's text "
        "tower is trained on English; the raw Vietnamese query keeps its own channel regardless.",
    )
    run.add_argument(
        "--out",
        default=None,
        help="subdirectory of the submission directory to write into, e.g. --out phase1",
    )
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--allow-stub", action="store_true")
    run.add_argument(
        "--allow-stub-submission",
        action="store_true",
        help="allow writing submissions from the stub encoder (NEVER for a real submission)",
    )
    run.add_argument("--no-hedge-answers", action="store_true", help=_HEDGE_HELP)

    evaluate_review = subparsers.add_parser(
        "evaluate-review",
        help="measure video/thumbnail recall for the human-review pipeline",
    )
    evaluate_review.add_argument("--devset", default=None)
    evaluate_review.add_argument("--top-videos", type=int, default=50)
    evaluate_review.add_argument("--max-videos", type=int, default=60)
    evaluate_review.add_argument("--rescue-per-channel", type=int, default=2)
    evaluate_review.add_argument("--tier1-videos", type=int, default=10)
    evaluate_review.add_argument("--tier1-frames", type=int, default=4)
    evaluate_review.add_argument("--tier2-videos", type=int, default=20)
    evaluate_review.add_argument("--tier2-frames", type=int, default=3)
    evaluate_review.add_argument("--later-frames", type=int, default=2)
    evaluate_review.add_argument("--min-gap-seconds", type=float, default=8.0)
    evaluate_review.add_argument("--allow-stub", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="score against the internal eval set")
    evaluate.add_argument("--devset", default=None)
    evaluate.add_argument(
        "--answers",
        default=None,
        help="the same Q&A answer hypotheses file passed to `run` — scoring the pipeline as it "
        "would actually be submitted, rather than with an empty answer axis",
    )
    evaluate.add_argument("--pins", default=None, help="the same verified-row file passed to `run`")
    evaluate.add_argument("--allow-stub", action="store_true")

    check = subparsers.add_parser("check-submission", help="validate submission files")
    check.add_argument("--dir", default=None)

    subparsers.add_parser(
        "submit-selftest", help="print sample submission files to ask the organisers"
    )

    serve = subparsers.add_parser("serve", help="run the HTTP backend (needs .[backend])")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="reload on code change (dev)")
    serve.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="queries solved in parallel; solve is CPU-bound, so keep this low",
    )
    serve.add_argument(
        "--allow-stub",
        action="store_true",
        help="allow the stub encoder — /health will report degraded",
    )
    return parser


_COMMANDS = {
    "inventory": cmd_inventory,
    "validate": cmd_validate,
    "build-index": cmd_build_index,
    "devset": cmd_devset,
    "query": cmd_query,
    "review": cmd_review,
    "evaluate-review": cmd_evaluate_review,
    "run": cmd_run,
    "evaluate": cmd_evaluate,
    "check-submission": cmd_check_submission,
    "submit-selftest": cmd_selftest,
    "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The library logs rather than prints; the CLI decides to show that on stderr.
    setup_cli_logging()
    cfg = load_config(args.config)
    try:
        return _COMMANDS[args.cmd](args, cfg)
    except FileNotFoundError as exc:
        print(f"\nMISSING FILE: {exc}")
        print("Run `validate` and then `build-index` first.")
        return 2
    except Exception as exc:
        print(f"\nERROR {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
