"""Central configuration. Every hyperparameter that affects the score lives here.

Principle: no magic constants scattered through the code. A hyperparameter that cannot be read
from one place is a hyperparameter that cannot be tuned.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

__all__ = [
    "AnswerSpanConfig",
    "Config",
    "PathConfig",
    "RetrievalConfig",
    "default_config_path",
    "find_project_root",
    "load_config",
]

#: Markers used to recognise the project root when walking up the directory tree.
_ROOT_MARKERS = ("pyproject.toml", "configs/default.json")


@lru_cache(maxsize=1)
def find_project_root() -> Path:
    """The project root, found by walking up from this file to a known marker.

    ``Path.cwd()`` is deliberately not used: when the library runs inside a backend process, the
    working directory has nothing to do with where the index and the data live. The
    ``AIC_PROJECT_ROOT`` environment variable overrides the search — needed for deployments where
    the code and the data sit in different places (containers, installed wheels).
    """
    override = os.environ.get("AIC_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for directory in (here.parent, *here.parents):
        if any((directory / marker).exists() for marker in _ROOT_MARKERS):
            return directory
    # No marker found (for instance an installed wheel without configs): fall back to the working
    # directory so the caller still has a way to point at things explicitly, rather than raising
    # at import time.
    return Path.cwd()


def default_config_path() -> Path:
    return find_project_root() / "configs" / "default.json"


@dataclass
class AnswerSpanConfig:
    """Assumed length of the answer span ``[s, e]``, in **frames**.

    This is the most sensitive hyperparameter of the slot allocation layer, and the rules give
    **no** figure for Textual KIS or Q&A. The available evidence:

    ==========  ==========================  =====================
    Task        Example in the rules        Implied width
    ==========  ==========================  =====================
    KIS         frames 500..510              11 frames
    Q&A         frames 800..900             101 frames
    TRAKE       "usually under 10"          < 10 frames
    ==========  ==========================  =====================

    The safe direction of error is to **underestimate**. Assuming an L smaller than reality
    spreads frames more densely than needed — coverage still holds, only slots are spent.
    Assuming an L larger than reality spreads too thinly and *misses* the answer span. The
    defaults are therefore set deliberately low, not at the expected value.
    """

    kis: int = 25
    qa: int = 25
    trake: int = 10

    def for_task(self, task: str) -> int:
        try:
            return {"kis": self.kis, "qa": self.qa, "trake": self.trake}[task]
        except KeyError as exc:
            raise ValueError(f"invalid task: {task!r}") from exc


@dataclass
class RetrievalConfig:
    """Parameters of the retrieval and fusion layer (P8) and the verification layer (P9)."""

    #: results taken from each channel before fusion.
    channel_depth: int = 2000
    #: the RRF smoothing constant.
    rrf_eta: int = 60
    #: candidate shots kept after fusion — the ceiling on P_100.
    n_candidates: int = 500
    #: candidates passed to the expensive verification step.
    n_verify: int = 50
    #: cosine threshold above which two frames count as near-duplicates when clustering.
    dedup_cosine: float = 0.92
    #: maximum shots kept from one video (stops one video filling the whole list).
    max_shots_per_video: int = 12
    #: locus padding on each side when spreading frames, in seconds (events cross boundaries).
    locus_pad_seconds: float = 0.5
    #: Exponent applied to the video-level score before it is normalised into ``pi_v``.
    #: 1.0 is the linear normalisation of the RRF scores, which is nearly uniform and makes the
    #: allocator hedge across ~30 videos; larger values concentrate the mass on the leaders, which
    #: is what the band weights reward when the leading channel is usually right. Calibrated on
    #: the eight labelled queries of round 1 — a sample far too small to trust beyond one
    #: significant figure. See DESIGN.md P13.
    pi_sharpness: float = 3.0
    #: highest-IDF query terms kept for the BM25 channel. The published queries are paragraphs
    #: wrapped in instruction phrasing, and running BM25 over all ~24 of their terms hands the
    #: top slots to whichever video is topically broadest: measured on the mock set, one
    #: travel-show video held slot 1 of four queries while the video whose title contained the
    #: query's subject sat at slot 26. Pruning to 8 terms puts that video at BM25 rank 1.
    sparse_max_terms: int = 8


@dataclass
class PathConfig:
    """Paths. Relative values are joined to the **project root**, not to the CWD.

    That distinction is mandatory once the library is embedded in a backend: a server process has
    an arbitrary working directory (often ``/`` or a systemd unit's directory), so a path relative
    to the CWD would point at nothing. :func:`load_config` calls :meth:`Config.resolve` on behalf
    of every caller.
    """

    #: Read-only input: the ``data/batch1`` link pointing at the organiser's corpus; see
    #: ``scripts/link_data.py``. ``AIC_DATA_ROOT`` overrides this value.
    data_root: str = "data/batch1"

    # Everything derived from the raw data lives under data/processed. It is all reproducible
    # from `aic build-index` and `aic run`, and none of it belongs in version control — the
    # dense index alone is 173 MiB. `.gitignore` therefore excludes all of `data/`.
    index_dir: str = "data/processed/index"
    submission_dir: str = "data/processed/submissions"
    report_dir: str = "data/processed/reports"
    #: where videos are extracted temporarily for the TRAKE branch; cleaned up after use.
    video_cache_dir: str = "data/processed/video_cache"

    #: The internal evaluation set. Deliberately **outside** ``data/``: it is hand-annotated
    #: ground truth, as much authored content as the code is, and it must be version-controlled.
    #: Putting it under a fully ignored ``data/`` is how that gets lost.
    devset_dir: str = "devset"

    #: Fields carrying path semantics — shared by resolution and by the tests.
    FIELDS: ClassVar[tuple[str, ...]] = (
        "data_root",
        "index_dir",
        "submission_dir",
        "report_dir",
        "devset_dir",
        "video_cache_dir",
    )

    def resolved(self, base: str | Path | None = None) -> PathConfig:
        """A copy with every path made absolute. Absolute paths are left untouched."""
        root = Path(base) if base is not None else find_project_root()
        return PathConfig(
            **{
                name: str(
                    path
                    if (path := Path(getattr(self, name))).is_absolute()
                    else (root / path).resolve()
                )
                for name in self.FIELDS
            }
        )


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    answer_span: AnswerSpanConfig = field(default_factory=AnswerSpanConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    #: fps used when it cannot be read from map-keyframes.
    fps_default: float = 25.0
    #: seed for every stochastic component (the Monte Carlo estimate in TRAKE).
    seed: int = 0
    #: the text encoder must be in the *same space* as the indexed features.
    text_encoder: str = "clip-ViT-B-32"
    #: Second text tower, distilled to embed 50+ languages into the *same* image space as
    #: ``clip-ViT-B-32``. It runs as an independent dense channel rather than replacing the
    #: first, because neither dominates: measured on the mock set, the monolingual tower puts
    #: the panna-cotta video at video-rank 5 where this one puts it at 564, while this one puts
    #: the FANA charity video at rank 1 where the monolingual tower does not find it at all.
    #: Loan words favour the English tower; Vietnamese prose favours this one. Empty disables it.
    text_encoder_multilingual: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    def resolve(self, base: str | Path | None = None) -> Config:
        """A copy with every path made absolute. The original is not modified."""
        return replace(self, paths=self.paths.resolved(base))


def load_config(
    path: str | Path | None = None,
    *,
    base: str | Path | None = None,
    resolve: bool = True,
) -> Config:
    """Load configuration from JSON over the defaults, then make paths absolute.

    Precedence, later winning over earlier:

    1. the dataclass defaults;
    2. the JSON file at ``path`` (by default ``configs/default.json`` at the project root);
    3. the ``AIC_DATA_ROOT`` environment variable.

    ``resolve=False`` keeps paths relative — used only when *writing* a configuration file back
    out, because baking one machine's absolute paths into a shared file is wrong.

    YAML is avoided to keep the dependency list short; JSON is sufficient for a flat file.
    """
    config = Config()
    config_path = Path(path) if path is not None else default_config_path()
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if "paths" in payload:
            config.paths = PathConfig(**{**asdict(config.paths), **payload["paths"]})
        if "answer_span" in payload:
            config.answer_span = AnswerSpanConfig(
                **{**asdict(config.answer_span), **payload["answer_span"]}
            )
        if "retrieval" in payload:
            config.retrieval = RetrievalConfig(
                **{**asdict(config.retrieval), **payload["retrieval"]}
            )
        for key in ("fps_default", "seed", "text_encoder", "text_encoder_multilingual"):
            if key in payload:
                setattr(config, key, payload[key])
    elif path is not None:
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    data_root_override = os.environ.get("AIC_DATA_ROOT")
    if data_root_override:
        config.paths.data_root = data_root_override
    return config.resolve(base) if resolve else config
