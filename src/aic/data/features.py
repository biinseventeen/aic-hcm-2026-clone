"""P3 — the dense visual index: merging 873 .npy files into one searchable index.

The organiser-supplied features (``clip-ViT-B-32``, 512 dimensions, fp16, already
L2-normalised) are a **free baseline**: 177,321 vectors = 181 MiB, loads entirely into RAM,
and search is a single matrix multiplication. At this size, exhaustive search is *faster and
more accurate* than approximate search; an approximate index structure only becomes necessary
past roughly 20 million vectors.

A limitation worth knowing
--------------------------
These features exist **only at keyframes** — a median of 55 frames between consecutive
keyframes. That is sufficient for shot-level recall (Textual KIS, Q&A) but *not* for TRAKE,
where the answer window is under 10 frames. The TRAKE branch has to encode frames itself at
full temporal resolution (see :mod:`aic.tasks.trake`).

ViT-B-32 is also a generation behind SigLIP and EVA-CLIP. The architecture allows a swap:
write out a different :class:`DenseIndex` with the same ``rows`` schema.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..log import log
from .keyframes import KeyframeTable, load_all_keyframe_tables
from .layout import DataRoot

__all__ = ["DenseIndex", "build_dense_index", "load_dense_index"]

#: Filenames inside the index directory.
VECTORS_FILE = "dense_vectors.f16.npy"
META_FILE = "dense_meta.json"
ROWS_FILE = "dense_rows.npy"


@dataclass
class DenseIndex:
    """A flat vector index plus the row -> (video, keyframe, frame) mapping table.

    ``vectors``   (M, D) fp16, L2-normalised — cosine equals the dot product.
    ``rows``      (M, 3) int32: [video_ordinal, n, frame_idx].
    ``video_ids`` list indexed by ``video_ordinal``.
    """

    vectors: np.ndarray
    rows: np.ndarray
    video_ids: list[str]
    model: str = "clip-ViT-B-32"
    source: str = "organizer"
    #: contiguous fp32 copy, built lazily. See :meth:`prepare`.
    _float32: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-dimensional, got {self.vectors.shape}")
        if self.rows.shape[0] != self.vectors.shape[0]:
            raise ValueError(
                f"row count mismatch: vectors={self.vectors.shape[0]} rows={self.rows.shape[0]}"
            )

    @property
    def n_vectors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.vectors.nbytes + self.rows.nbytes)

    def video_of(self, row: int) -> str:
        return self.video_ids[int(self.rows[row, 0])]

    def decode(self, row: int) -> tuple[str, int, int]:
        """row -> ``(video_id, n, frame_idx)``."""
        video_ordinal, n, frame_idx = self.rows[row]
        return (self.video_ids[int(video_ordinal)], int(n), int(frame_idx))

    # -- search ----------------------------------------------------------

    def prepare(self) -> DenseIndex:
        """Materialise a contiguous fp32 copy in RAM. Call once before batch querying.

        Measured on the batch 1 index (177,321 x 512):

            fp16 + mmap, cast on every query : ~430 ms per query
            fp32 resident in RAM, one query  : ~27 ms
            fp32 resident in RAM, batch of 32: ~5 ms per query

        The multiplication itself is only 181 MFLOP, so FLOPs are *not* the bottleneck — the
        fp16 -> fp32 cast over 175 MiB on every call is. The trade is 346 MiB of RAM. The
        173 MiB index lives in system RAM and search runs on the CPU; see ``DESIGN.md``
        section 2.2 for why device memory does not enter this decision.
        """
        if self._float32 is None:
            self._float32 = np.ascontiguousarray(self.vectors, dtype=np.float32)
        return self

    def release(self) -> None:
        """Drop the fp32 copy, to hand RAM back to another stage."""
        self._float32 = None

    def search(
        self, query: np.ndarray, *, top_k: int = 1000, batch: int = 65536
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exhaustive search. ``query`` is (D,) or (Q, D), normalised.

        Returns ``(scores, rows)`` of shape (Q, top_k), in decreasing score.

        Uses the fp32 copy when :meth:`prepare` has been called; otherwise casts in batches so
        that peak memory does not scale with corpus size (about 40x slower).
        """
        queries = np.atleast_2d(np.asarray(query, dtype=np.float32))
        if queries.shape[1] != self.dim:
            raise ValueError(f"query dim {queries.shape[1]} != index dim {self.dim}")
        # Re-normalise in case the caller forgot.
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-12)

        k = min(top_k, self.n_vectors)
        best_scores = np.full((queries.shape[0], 0), -np.inf, dtype=np.float32)
        best_rows = np.zeros((queries.shape[0], 0), dtype=np.int64)
        source = self._float32 if self._float32 is not None else self.vectors
        step = self.n_vectors if self._float32 is not None else batch
        for start in range(0, self.n_vectors, step):
            end = min(start + step, self.n_vectors)
            block = np.asarray(source[start:end], dtype=np.float32)
            similarities = queries @ block.T  # (Q, end-start)
            block_k = min(k, similarities.shape[1])
            partition = np.argpartition(-similarities, block_k - 1, axis=1)[:, :block_k]
            partition_scores = np.take_along_axis(similarities, partition, axis=1)
            best_scores = np.concatenate([best_scores, partition_scores], axis=1)
            best_rows = np.concatenate([best_rows, partition + start], axis=1)
            if best_scores.shape[1] > 4 * k:  # compact periodically
                keep = np.argpartition(-best_scores, k - 1, axis=1)[:, :k]
                best_scores = np.take_along_axis(best_scores, keep, axis=1)
                best_rows = np.take_along_axis(best_rows, keep, axis=1)
        order = np.argsort(-best_scores, axis=1)[:, :k]
        return (
            np.take_along_axis(best_scores, order, axis=1),
            np.take_along_axis(best_rows, order, axis=1),
        )

    def video_rows(self, video_id: str) -> np.ndarray:
        """Every row index belonging to one video, in increasing ``n`` order."""
        try:
            ordinal = self.video_ids.index(video_id)
        except ValueError as exc:
            raise KeyError(f"video {video_id} is not in the index") from exc
        return np.nonzero(self.rows[:, 0] == ordinal)[0]

    # -- save / load -----------------------------------------------------

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / VECTORS_FILE, self.vectors)
        np.save(out / ROWS_FILE, self.rows)
        (out / META_FILE).write_text(
            json.dumps(
                {
                    "video_ids": self.video_ids,
                    "model": self.model,
                    "source": self.source,
                    "n_vectors": self.n_vectors,
                    "dim": self.dim,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return out


def load_dense_index(index_dir: str | Path, *, mmap: bool = True) -> DenseIndex:
    """Load a saved index. ``mmap`` keeps the vectors on disk, mapped page by page.

    Loading 181 MiB entirely into RAM is trivial, but mmap helps once the index grows — for
    instance after encoding at 2 fps instead of only at keyframes.
    """
    directory = Path(index_dir)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    vectors = np.load(directory / VECTORS_FILE, mmap_mode="r" if mmap else None)
    rows = np.load(directory / ROWS_FILE)
    return DenseIndex(
        vectors=vectors,
        rows=rows,
        video_ids=list(meta["video_ids"]),
        model=meta.get("model", "?"),
        source=meta.get("source", "?"),
    )


def build_dense_index(
    root: DataRoot,
    *,
    tables: dict[str, KeyframeTable] | None = None,
    video_ids: list[str] | None = None,
    verbose: bool = True,
) -> DenseIndex:
    """Merge ``clip-features-32/*.npy`` into a single :class:`DenseIndex`.

    This step *re-checks* the vector count against the ``map-keyframes`` row count for every
    video and raises on a mismatch: a row-shifted index produces the wrong ``frame_idx`` for
    *every* query, and that is another silent failure mode.
    """
    keyframe_tables = (
        tables if tables is not None else load_all_keyframe_tables(root, video_ids=video_ids)
    )
    ids = sorted(keyframe_tables)
    started = time.time()

    vector_chunks: list[np.ndarray] = []
    row_chunks: list[np.ndarray] = []
    for ordinal, video_id in enumerate(ids):
        table = keyframe_tables[video_id]
        vectors = np.load(io.BytesIO(root.clip_features.read(f"{video_id}.npy")))
        if vectors.shape[0] != len(table):
            raise ValueError(
                f"{video_id}: {vectors.shape[0]} vectors but {len(table)} map-keyframes rows. "
                "A row-shifted index means a wrong frame_idx for every query. Stopping."
            )
        if vectors.dtype != np.float16:
            vectors = vectors.astype(np.float16)
        vector_chunks.append(vectors)
        row_chunks.append(
            np.stack(
                [
                    np.full(len(table), ordinal, dtype=np.int32),
                    np.asarray(table.n, dtype=np.int32),
                    np.asarray(table.frame_idx, dtype=np.int32),
                ],
                axis=1,
            )
        )
        if verbose and (ordinal + 1) % 200 == 0:
            log.info("  ... %d/%d videos", ordinal + 1, len(ids))

    vectors = np.concatenate(vector_chunks, axis=0)
    rows = np.concatenate(row_chunks, axis=0)

    # Confirm normalisation: cosine equals the dot product only when ||v|| = 1.
    sample = vectors[:: max(1, len(vectors) // 2000)].astype(np.float32)
    norms = np.linalg.norm(sample, axis=1)
    if np.abs(norms - 1.0).max() > 2e-2:
        raise ValueError(
            f"vectors are not L2-normalised (||v|| in [{norms.min():.3f}, {norms.max():.3f}]). "
            "Normalise before saving, otherwise the dot product is not a cosine."
        )

    index = DenseIndex(vectors=vectors, rows=rows, video_ids=ids)
    if verbose:
        log.info(
            "  dense index: %s vectors x %d dims = %.0f MiB, %.1fs",
            f"{index.n_vectors:,}",
            index.dim,
            index.nbytes / 2**20,
            time.time() - started,
        )
    return index
