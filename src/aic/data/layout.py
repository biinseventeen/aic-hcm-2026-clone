"""Locating and opening the organiser-supplied data — from an extracted directory *or* a zip.

Two sources, one interface
--------------------------
Every data family is opened through an object exposing the same ``read(key)`` interface,
whether the bytes live inside a zip or in an extracted directory. ``DataRoot`` **prefers the
extracted directory** ``<root>/extracted/<family>/`` when it exists, and falls back to the
archives otherwise. Extracting is therefore a disk decision, not an architectural one, and
no layer above needs to know which source is in use.

Why the direct-from-archive path is kept
----------------------------------------
All of batch 1 is ~107 GiB compressed. The four small derived families
(``map-keyframes``, ``media-info``, ``clip-features-32``, ``objects``) come to only ~1.85 GiB
extracted, and ``Keyframes_*`` to 28.7 GiB — both fit on the current development machine (see
``scripts/extract_data.py``). ``Videos_*`` is 77.3 GiB and does **not** fit, so the video
branch still reads from archives.

``Videos_*.zip`` has a second, independent reason: decoding pixels at full temporal
resolution (mandatory for TRAKE) requires seeking within the video stream, and ffmpeg cannot
seek inside a zip. Video files must be extracted — but only *on demand*, for the videos that
actually reach the candidate set, and then deleted.

The cost of reading from a zip is indexing the central directory once per process start: the
177,321 entries of ``Keyframes_*`` take tens of seconds. An extracted directory needs no such
step — :class:`DirSet` indexes lazily, and only if somebody calls :meth:`DirSet.keys`.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

__all__ = [
    "VIDEO_ID_RE",
    "ArchiveSet",
    "DataRoot",
    "DirSet",
    "Source",
    "parse_video_id",
]

#: The organiser's video identifier: L<group>_V<number>, e.g. L21_V001.
VIDEO_ID_RE = re.compile(r"^(L\d+)_V(\d+)$")


def parse_video_id(video_id: str) -> tuple[str, int]:
    """``"L21_V001"`` -> ``("L21", 1)``. Raises if the format is wrong.

    >>> parse_video_id("L26_V444")
    ('L26', 444)
    """
    match = VIDEO_ID_RE.match(video_id)
    if not match:
        raise ValueError(
            f"video_id does not match L<group>_V<number>: {video_id!r}. "
            "Submitting a wrong video identifier forfeits the whole query."
        )
    return match.group(1), int(match.group(2))


@dataclass(slots=True)
class ArchiveSet:
    """A family of same-kind archives, e.g. ``Keyframes_L21.zip`` .. ``Keyframes_L30.zip``.

    The ``entry -> (zip path, entry name)`` mapping is built once and cached, so subsequent
    lookups are O(1) and never reopen an archive.
    """

    name: str
    paths: list[Path] = field(default_factory=list)
    _index: dict[str, tuple[Path, str]] = field(default_factory=dict, repr=False)
    _handles: dict[Path, zipfile.ZipFile] = field(default_factory=dict, repr=False)

    def build_index(self, *, suffix: str | None = None, strip_dir: bool = True) -> ArchiveSet:
        """Scan every archive and index entry -> (zip, entry).

        ``strip_dir`` drops the leading directory component (``map-keyframes/L21_V001.csv``
        becomes ``L21_V001.csv``) so lookup keys do not depend on how the data was packaged.
        """
        self._index.clear()
        for path in self.paths:
            with zipfile.ZipFile(path) as archive:
                for entry in archive.namelist():
                    if entry.endswith("/"):
                        continue
                    if suffix and not entry.endswith(suffix):
                        continue
                    key = entry.split("/", 1)[1] if (strip_dir and "/" in entry) else entry
                    self._index[key] = (path, entry)
        return self

    def _archive(self, path: Path) -> zipfile.ZipFile:
        handle = self._handles.get(path)
        if handle is None:
            handle = zipfile.ZipFile(path)
            self._handles[path] = handle
        return handle

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    def keys(self) -> Iterator[str]:
        return iter(sorted(self._index))

    #: Iterating a source yields its keys, so ``for key in source`` reads naturally.
    __iter__ = keys

    def read(self, key: str) -> bytes:
        location = self._index.get(key)
        if location is None:
            raise KeyError(f"{self.name}: no entry {key!r} ({len(self._index)} entries indexed)")
        path, entry = location
        return self._archive(path).read(entry)

    def read_text(self, key: str, encoding: str = "utf-8") -> str:
        # utf-8-sig: some organiser CSV files carry a BOM.
        return self.read(key).decode("utf-8-sig" if encoding == "utf-8" else encoding)

    def read_json(self, key: str):
        return json.loads(self.read_text(key))

    def local_path(self, key: str) -> Path | None:  # noqa: ARG002 — shared Source interface
        """``None``: the entry lives inside a zip and has no filesystem path."""
        return None

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


@dataclass(slots=True)
class DirSet:
    """An extracted data family, exposing the same interface as :class:`ArchiveSet`.

    Lookup keys are **identical** to those of the corresponding ``ArchiveSet`` so the two are
    interchangeable: ``key_prefix`` restores the directory component that ``strip_dir=False``
    preserves (the keyframes family uses keys like ``keyframes/L21_V001/079.jpg``).

    Indexing is **lazy**: :meth:`read` only joins a path, so no scan is needed, and walking
    all 177,321 files happens only if somebody actually calls :meth:`keys` or ``len()``.
    """

    name: str
    root: Path
    suffix: str | None = None
    key_prefix: str = ""
    _keys: list[str] | None = field(default=None, repr=False)

    def _path_for(self, key: str) -> Path:
        relative = (
            key[len(self.key_prefix) :]
            if self.key_prefix and key.startswith(self.key_prefix)
            else key
        )
        return self.root / relative

    def _scan(self) -> list[str]:
        if self._keys is None:
            pattern = f"*{self.suffix}" if self.suffix else "*"
            self._keys = sorted(
                self.key_prefix + path.relative_to(self.root).as_posix()
                for path in self.root.rglob(pattern)
                if path.is_file()
            )
        return self._keys

    def __contains__(self, key: str) -> bool:
        return self._path_for(key).is_file()

    def __len__(self) -> int:
        return len(self._scan())

    def keys(self) -> Iterator[str]:
        return iter(self._scan())

    __iter__ = keys

    def read(self, key: str) -> bytes:
        path = self._path_for(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(f"{self.name}: no {key!r} at {path}") from exc

    def read_text(self, key: str, encoding: str = "utf-8") -> str:
        # utf-8-sig: some organiser CSV files carry a BOM.
        return self.read(key).decode("utf-8-sig" if encoding == "utf-8" else encoding)

    def read_json(self, key: str):
        return json.loads(self.read_text(key))

    def local_path(self, key: str) -> Path | None:
        """A real path — lets ffmpeg, PIL or any external tool open the file without a copy."""
        path = self._path_for(key)
        return path if path.is_file() else None

    def close(self) -> None:
        return None


#: The shared interface of the two sources. No layer above may rely on more than this.
Source = ArchiveSet | DirSet


# slots=True is deliberately omitted: cached_property needs __dict__ to store results.
@dataclass
class DataRoot:
    """The directory holding all organiser-supplied data.

    Defaults to the ``AIC_DATA_ROOT`` environment variable. In this repository the configured
    default is the ``data/raw`` link created by ``scripts/link_data.py``.
    """

    root: Path
    #: Set False to force reading from archives — used to cross-check the two sources.
    prefer_extracted: bool = True
    #: Subdirectory holding extracted data, relative to ``root``.
    extracted_dirname: str = "extracted"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"data directory not found: {self.root}. "
                "Set AIC_DATA_ROOT or fix configs/default.json."
            )

    @classmethod
    def from_env(cls, default: str = "data/raw", **kwargs) -> DataRoot:
        return cls(Path(os.environ.get("AIC_DATA_ROOT", default)), **kwargs)

    def _glob(self, pattern: str) -> list[Path]:
        return sorted(self.root.glob(pattern))

    @property
    def extracted_root(self) -> Path:
        """Where ``scripts/extract_data.py`` puts extracted data."""
        return self.root / self.extracted_dirname

    def _open(
        self,
        name: str,
        zip_pattern: str,
        *,
        suffix: str,
        strip_dir: bool = True,
    ) -> Source:
        """Open one data family: the extracted directory if present, else the archives.

        The lookup keys of both branches coincide by design — see :class:`DirSet`.
        """
        directory = self.extracted_root / name
        if self.prefer_extracted and directory.is_dir() and any(directory.iterdir()):
            return DirSet(
                name, directory, suffix=suffix, key_prefix="" if strip_dir else f"{name}/"
            )
        archives = self._glob(zip_pattern)
        if not archives:
            raise FileNotFoundError(
                f"no data found for {name!r}: there is no directory {directory} and no "
                f"archive matching {zip_pattern!r} in {self.root}"
            )
        return ArchiveSet(name, archives).build_index(suffix=suffix, strip_dir=strip_dir)

    # -- the four small derived families ----------------------------------

    @cached_property
    def map_keyframes(self) -> Source:
        return self._open("map-keyframes", "map-keyframes*.zip", suffix=".csv")

    @cached_property
    def media_info(self) -> Source:
        return self._open("media-info", "media-info*.zip", suffix=".json")

    @cached_property
    def clip_features(self) -> Source:
        return self._open("clip-features", "clip-features*.zip", suffix=".npy")

    @cached_property
    def objects(self) -> Source:
        # objects/<video>/<nnn>.json — both levels are kept in the key.
        return self._open("objects", "objects*.zip", suffix=".json")

    @cached_property
    def keyframes(self) -> Source:
        # Keys keep the ``keyframes/`` prefix because map-keyframes refers to them that way.
        return self._open("keyframes", "Keyframes_*.zip", suffix=".jpg", strip_dir=False)

    # -- source videos: listed only, extracted on demand -------------------

    @cached_property
    def video_archives(self) -> list[Path]:
        return self._glob("Videos_*.zip")

    def video_ids(self) -> list[str]:
        """Video ids taken from ``map-keyframes`` — the source of truth for the video set."""
        return [key[:-4] for key in self.map_keyframes]

    def locate_video(self, video_id: str) -> tuple[Path, str] | None:
        """Find ``<video_id>.mp4`` in the ``Videos_*.zip`` family. Returns (zip, entry).

        Scanned lazily because the total size is large, and the result is not cached between
        calls so no handle stays open on a multi-gigabyte archive.
        """
        target = f"{video_id}.mp4"
        for path in self.video_archives:
            with zipfile.ZipFile(path) as archive:
                for entry in archive.namelist():
                    # Compare the basename, not a suffix: a suffix test would also accept
                    # an entry whose name merely ends with the target.
                    if entry.rsplit("/", 1)[-1] == target:
                        return (path, entry)
        return None

    def extract_video(self, video_id: str, dest_dir: str | Path) -> Path:
        """Extract one video into ``dest_dir`` so ffmpeg can seek it. Returns the path.

        Used by the TRAKE branch (P12) and anything else needing pixels at full temporal
        resolution. The caller is responsible for deleting the file afterwards.
        """
        location = self.locate_video(video_id)
        if location is None:
            raise FileNotFoundError(
                f"{video_id}.mp4 not found in {len(self.video_archives)} "
                f"Videos_*.zip archives under {self.root}"
            )
        path, entry = location
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{video_id}.mp4"
        if out.exists() and out.stat().st_size > 0:
            return out
        with zipfile.ZipFile(path) as archive, archive.open(entry) as src, out.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        return out

    def inventory(self, *, count_all: bool = False) -> str:
        """Data inventory — run before anything else to confirm the data is present.

        ``count_all=False`` skips counting leaf files for the two large families (keyframes,
        objects): against a directory source that means an ``rglob`` over 177,321 files and
        takes tens of seconds, while the inventory only needs to answer "is the data here".
        Those families are reported by their per-video directory count instead, which costs a
        single listdir — a dash would be indistinguishable from "empty", which is the one thing
        this command exists to rule out.
        """
        lines = [f"AIC data root: {self.root}", ""]
        large = {"keyframes", "objects"}
        for label in ("map-keyframes", "media-info", "clip-features", "objects", "keyframes"):
            try:
                source = getattr(self, label.replace("-", "_"))
            except FileNotFoundError as exc:
                lines.append(f"  {label:<16} MISSING — {exc}")
                continue
            kind = (
                "extracted directory" if isinstance(source, DirSet) else f"{len(source.paths)} zip"
            )
            if label in large and not count_all and isinstance(source, DirSet):
                count = sum(1 for entry in source.root.iterdir() if entry.is_dir())
                unit, note = "videos", ", leaf files not counted"
            else:
                count, unit, note = len(source), "entries", ""
            lines.append(f"  {label:<16} {count:>9,} {unit:<7} ({kind}{note})")

        video_bytes = sum(path.stat().st_size for path in self.video_archives)
        lines.append(
            f"  {'Videos_*.zip':<16} {len(self.video_archives):>9,} archives, "
            f"{video_bytes / 2**30:.1f} GiB (read from zip, extracted on demand)"
        )
        return "\n".join(lines)
