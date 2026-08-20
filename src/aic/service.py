"""The service layer — the only façade a backend needs to import.

Basis
-----
Index loading and query solving used to live inside ``cli.py``, mixed with ``argparse`` and
``print``. That locked the whole system into one shape of use: one process, one query, one run. A
backend needs exactly the properties that shape lacks:

* **Load once, use many times.** The dense index is 173 MiB and the text encoder takes seconds to
  initialise. That cost must be paid **once at startup** and amortised across every request, not
  repaid per request. :class:`Engine` is the object holding that state.
* **No printing to stdout.** See :mod:`aic.log`. Warnings raised while loading are **collected**
  into :attr:`Engine.warnings` so a ``/health`` endpoint can return them, rather than disappearing
  into the server's log.
* **Serialisable results.** :meth:`SolveResult.to_dict` returns plain JSON structures containing
  no numpy arrays and no dataclasses, so they drop straight into a response body.
* **Independent of the CWD.** Paths are made absolute against the project root
  (:func:`aic.config.find_project_root`), not the process working directory.

``cli.py`` is now a thin presentation layer over this module; the two entry points share exactly
one code path.

Thread safety
-------------
:meth:`Engine.solve` only **reads** the loaded state, so calling it concurrently from several
threads is safe *provided* the underlying text encoder is. ``sentence-transformers`` calls into
PyTorch, which is safe for inference but not cheap under contention; a backend should put the
engine behind a queue or a small pool rather than allowing unbounded concurrency. See
:mod:`aic.api.app`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .config import Config, load_config
from .log import log

__all__ = ["Engine", "EngineStatus", "SolveResult", "TaskName"]

TaskName = Literal["kis", "qa", "trake"]

#: Candidate videos whose windows are encoded for the TRAKE branch. Beyond this the alignment
#: cost grows without improving Pr(correct video), which is what dominates the TRAKE score.
TRAKE_MAX_VIDEOS = 20


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class SolveResult:
    """The result of solving one query: enough to submit **and** enough to diagnose.

    Keeps the original objects (``submission``, ``trace``, ``query``) for in-process callers, and
    offers :meth:`to_dict` for callers over HTTP.
    """

    query_id: str
    task: TaskName
    submission: Any  # aic.submit.writer.QuerySubmission
    trace: Any  # aic.core.allocator.AllocationTrace
    query: Any  # aic.query.parse.ParsedQuery
    #: the task layer's own solution object (KisSolution | QaSolution | TrakeSolution). Kept so
    #: the CLI can print a human-readable report without duplicating the formatting logic.
    solution: Any = None
    n_candidates: int = 0
    channel_sizes: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    #: True when the result is **not** usable for a submission (stub encoder, TRAKE without
    #: ffmpeg, Q&A without a VQA model).
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def n_answers(self) -> int:
        return len(self.submission.answers)

    @property
    def expected_final(self) -> float:
        return float(self.trace.expected_final)

    def report(self) -> str:
        """Human-readable report, formatted by the task layer. Used by the CLI."""
        return self.solution.report() if self.solution is not None else ""

    def rows(self) -> list[list[str]]:
        """The rows exactly as they would be written to the submission file, as plain strings."""
        from .submit.writer import SubmissionNaming

        naming = SubmissionNaming()
        return [answer.row(self.task, naming) for answer in self.submission.answers]

    def to_dict(self, *, top: int | None = None) -> dict:
        """Plain JSON structure. ``top`` limits how many answers are returned (None = all 100)."""
        answers = []
        for answer, allocation in zip(self.submission.answers, self.trace.allocations, strict=True):
            row: dict[str, Any] = {
                "rank": allocation.rank,
                "video_id": answer.video_id,
                "gain": round(float(allocation.gain), 6),
                "cumulative": round(float(allocation.cumulative), 6),
                "source": allocation.source,
            }
            if self.task == "trake":
                row["frame_ids"] = list(answer.frames)
            else:
                row["frame_id"] = answer.frame
            if self.task == "qa":
                row["answer"] = answer.answer
            answers.append(row)
            if top is not None and len(answers) >= top:
                break
        return {
            "query_id": self.query_id,
            "task": self.task,
            "degraded": self.degraded,
            "n_answers": self.n_answers,
            "elapsed_s": round(self.elapsed_s, 3),
            "query": {
                "raw": self.query.raw,
                "task_confidence": round(float(self.query.task_confidence), 3),
                "parser": self.query.parser,
                "keywords": list(self.query.keywords),
                "entities": list(self.query.entities),
                "moments": list(self.query.moments),
                "domain_hints": list(self.query.domain_hints),
            },
            "retrieval": {
                "n_candidates": self.n_candidates,
                "channel_sizes": dict(self.channel_sizes),
            },
            "allocation": {
                "expected_final": round(self.expected_final, 6),
                "coverage_at_k": {
                    str(k): round(float(v), 6) for k, v in sorted(self.trace.coverage_at_k.items())
                },
                "n_distinct_videos": self.trace.n_distinct_videos,
                "exhausted_at": self.trace.exhausted_at,
            },
            "answers": answers,
            "notes": [*self.trace.notes, *self.notes],
        }


@dataclass
class EngineStatus:
    """A snapshot of engine state, for a health endpoint.

    ``ready`` answers exactly one question: *can this engine produce a real submission*. It is
    False when the encoder is a stub, because retrieval is then noise — and a backend returning
    noise while reporting 200 OK is worse than one reporting an error.
    """

    ready: bool
    data_root: str
    index_dir: str
    n_videos: int
    n_keyframes: int
    dense_dim: int
    encoder: str | None
    encoder_is_stub: bool
    has_text_index: bool
    n_shots: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "data_root": self.data_root,
            "index_dir": self.index_dir,
            "n_videos": self.n_videos,
            "n_keyframes": self.n_keyframes,
            "n_shots": self.n_shots,
            "dense_dim": self.dense_dim,
            "encoder": self.encoder,
            "encoder_is_stub": self.encoder_is_stub,
            "has_text_index": self.has_text_index,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class Engine:
    """The loaded index plus the query orchestrator. Build once, reuse forever.

    Construct through :meth:`load`, not through ``__init__``: ``load`` is where configuration is
    read, the index is loaded, and warnings are collected.
    """

    cfg: Config
    root: Any  # aic.data.layout.DataRoot
    tables: dict
    dense: Any
    shots: Any
    text: Any | None
    encoder: Any | None
    retriever: Any
    fps: dict[str, float]
    max_frames: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        cfg: Config | None = None,
        *,
        config_path: str | Path | None = None,
        allow_stub: bool = False,
        need_text: bool = True,
        need_encoder: bool = True,
    ) -> Engine:
        """Load everything one query needs. Takes seconds to tens of seconds — call once.

        ``allow_stub=True`` permits running without a real encoder; the results are then **noise**
        and :attr:`EngineStatus.ready` will be False.
        """
        from .data.features import load_dense_index
        from .data.keyframes import load_all_keyframe_tables
        from .data.layout import DataRoot
        from .index.priors import DomainPrior
        from .index.shots import ShotTable
        from .index.text import TextIndex
        from .query.retrieve import Retriever
        from .query.text_encoder import is_stub, load_text_encoder

        cfg = cfg or load_config(config_path)
        warnings: list[str] = []

        index_dir = Path(cfg.paths.index_dir)
        if not index_dir.is_dir():
            raise FileNotFoundError(f"no index at {index_dir}. Run `aic build-index` first.")
        root = DataRoot(cfg.paths.data_root)
        tables = load_all_keyframe_tables(root)
        dense = load_dense_index(index_dir).prepare()
        shots = ShotTable.load(index_dir / "shots.json")

        text = None
        if need_text:
            text_path = index_dir / "text_media.json"
            if text_path.exists():
                text = TextIndex.load(text_path)
            else:
                warnings.append(
                    f"{text_path.name} is missing — the sparse and entity channels will be silent"
                )
        encoder = None
        if need_encoder:
            encoder = load_text_encoder(
                cfg.text_encoder, expect_dim=dense.dim, allow_stub=allow_stub
            )
            if is_stub(encoder):
                warnings.append(
                    "the text encoder is a STUB — retrieval results are noise, not usable for a "
                    "submission"
                )

        retrieval = cfg.retrieval
        engine = cls(
            cfg=cfg,
            root=root,
            tables=tables,
            dense=dense,
            shots=shots,
            text=text,
            encoder=encoder,
            fps={video_id: table.fps for video_id, table in tables.items()},
            max_frames={video_id: table.duration_frames for video_id, table in tables.items()},
            retriever=Retriever(
                dense=dense,
                shots=shots,
                text=text,
                encoder=encoder,
                prior=DomainPrior(),
                channel_depth=retrieval.channel_depth,
                rrf_eta=retrieval.rrf_eta,
                n_candidates=retrieval.n_candidates,
                dedup_cosine=retrieval.dedup_cosine,
                max_shots_per_video=retrieval.max_shots_per_video,
            ),
            warnings=warnings,
        )
        for warning in warnings:
            log.warning("  [!] %s", warning)
        return engine

    # -- status ------------------------------------------------------------

    @property
    def encoder_is_stub(self) -> bool:
        from .query.text_encoder import is_stub

        return is_stub(self.encoder)

    def status(self) -> EngineStatus:
        return EngineStatus(
            ready=self.encoder is not None and not self.encoder_is_stub,
            data_root=str(self.cfg.paths.data_root),
            index_dir=str(self.cfg.paths.index_dir),
            n_videos=len(self.tables),
            n_keyframes=sum(len(table) for table in self.tables.values()),
            n_shots=len(self.shots),
            dense_dim=int(self.dense.dim),
            encoder=getattr(self.encoder, "name", None),
            encoder_is_stub=self.encoder_is_stub,
            has_text_index=self.text is not None,
            warnings=list(self.warnings),
        )

    @property
    def known_videos(self) -> set[str]:
        return set(self.tables)

    # -- solving -----------------------------------------------------------

    def parse(self, text: str, *, task_hint: TaskName | None = None):
        from .query.parse import parse_query

        return parse_query(text, task_hint=task_hint)

    def solve(
        self,
        text: str,
        *,
        query_id: str = "1",
        task_hint: TaskName | None = None,
        hedge_answers: bool = True,
    ) -> SolveResult:
        """Solve one query, from raw text to 100 ordered answers.

        Raises :class:`ValueError` when a TRAKE query yields no moments — that is an input error,
        not a system error, so a backend should map it to a 4xx response.
        """
        from .tasks.kis import solve_kis
        from .tasks.qa import AnswerHypothesis, solve_qa
        from .tasks.trake import solve_trake, windows_from_keyframes

        started = time.perf_counter()
        query = self.parse(text, task_hint=task_hint)
        retrieval = self.retriever.retrieve(query)
        retrieval.candidates = self.retriever.cluster_near_duplicates(retrieval.candidates)

        answer_len = self.cfg.answer_span.for_task(query.task)
        pad_seconds = self.cfg.retrieval.locus_pad_seconds
        notes: list[str] = []
        degraded = self.encoder is None or self.encoder_is_stub

        if query.task == "kis":
            solution = solve_kis(
                query_id,
                retrieval,
                fps_by_video=self.fps,
                answer_len=answer_len,
                pad_seconds=pad_seconds,
                max_frames=self.max_frames,
            )
        elif query.task == "qa":
            # No VQA model yet: a single neutral hypothesis keeps the pipeline running. The
            # answer score will be zero — a GPU-dependent item, see docs/CONSTRAINTS.md G5.
            hypotheses = [AnswerHypothesis("(no VQA model available)", 1.0, "placeholder")]
            notes.append("no VQA model: the Q&A answer axis always scores zero (G5)")
            degraded = True
            solution = solve_qa(
                query_id,
                retrieval,
                fps_by_video=self.fps,
                hypotheses=hypotheses,
                answer_len=answer_len,
                pad_seconds=pad_seconds,
                max_frames=self.max_frames,
                hedge_answers=hedge_answers,
            )
        else:
            if not query.moments:
                raise ValueError(
                    "the TRAKE query yielded no moments — list the stages of the event "
                    'sequence, for example "(1) take-off, (2) clearance"'
                )
            windows = windows_from_keyframes(
                retrieval, self.dense, fps_by_video=self.fps, max_videos=TRAKE_MAX_VIDEOS
            )
            moment_embeddings = self.encoder.encode(query.moments)
            solution = solve_trake(
                query_id,
                retrieval,
                windows,
                moment_embeddings,
                fps_by_video=self.fps,
                answer_len=answer_len,
                seed=self.cfg.seed,
                degraded=True,
            )
            notes.append(
                "TRAKE ran in keyframe-only mode: coverage ceiling ~14 % per moment, ffmpeg is "
                "needed to decode at full fps (C5)"
            )
            degraded = True

        return SolveResult(
            query_id=query_id,
            task=query.task,
            submission=solution.submission,
            trace=solution.trace,
            query=query,
            solution=solution,
            n_candidates=len(retrieval.candidates),
            channel_sizes=dict(getattr(retrieval, "channel_sizes", {}) or {}),
            elapsed_s=time.perf_counter() - started,
            degraded=degraded,
            notes=notes,
        )

    # -- writing submissions -----------------------------------------------

    def write(
        self,
        result: SolveResult,
        out_dir: str | Path | None = None,
        *,
        strict: bool = False,
    ) -> tuple[Path, list]:
        """Write one submission file, with validation. Returns ``(path, issues)``."""
        from .submit.writer import write_submission

        return write_submission(
            result.submission,
            out_dir or self.cfg.paths.submission_dir,
            known_videos=self.known_videos,
            max_frames=self.max_frames,
            strict=strict,
        )

    def close(self) -> None:
        """Close archive handles. Safe to call more than once."""
        for name in ("map_keyframes", "media_info", "clip_features", "objects", "keyframes"):
            source = self.root.__dict__.get(name)
            if source is not None:
                source.close()
