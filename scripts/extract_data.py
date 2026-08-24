"""Extract the organiser's data into flat directories, with a disk-space check first.

Basis
-----
``aic.data.layout`` can read straight from the archives, so extracting is not required. It changes
two things: the time spent indexing the central directory of ``Keyframes_*`` (177,321 entries) on
every process start, and the ability to open a keyframe with an external tool for human
verification. When disk space allows, paying that cost once is cheaper than paying it every time.

``Videos_*.zip`` is out of scope: the total size exceeds free space, and the TRAKE branch only
needs one video at a time through ``DataRoot.extract_video``.

Manifest
--------
Once verification passes, the source archives can be deleted (``scripts/prune_zips.py``). But the
expected entry counts and byte totals are read *from the archives*, so deleting them also removes
the ability to verify again. To let that ability survive, verification records its result in
``extracted/.manifest.json`` together with the provenance of the numbers:

* ``source: "archive"`` — verified against the archives; this is evidence the extraction is correct.
* ``source: "snapshot"`` — a snapshot of what exists after the archives were deleted; useful only
  as a tripwire for later corruption, not as evidence the extraction was correct.

    python scripts/extract_data.py --dry-run
    python scripts/extract_data.py            # extract, then write the manifest
    python scripts/extract_data.py --verify   # verify (archives if present, else manifest)
    python scripts/extract_data.py --snapshot # write a manifest from what is on disk now
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aic.console import enable_utf8_stdio

GiB = 1 << 30

#: (archive glob, destination directory name)
GROUPS: list[tuple[str, str]] = [
    ("map-keyframes*.zip", "map-keyframes"),
    ("media-info*.zip", "media-info"),
    ("clip-features*.zip", "clip-features"),
    ("objects*.zip", "objects"),
    ("Keyframes_*.zip", "keyframes"),
]

#: Reserved for index artifacts, the TRAKE video cache, and filesystem slack.
HEADROOM_GIB = 12.0

#: Manifest filename, placed inside the destination directory.
MANIFEST_NAME = ".manifest.json"

#: Progress is logged every this many files, so a long extraction shows movement.
_PROGRESS_EVERY = 5000


@dataclass
class Group:
    dest: str
    archives: list[Path]
    n_entries: int
    n_bytes: int

    @property
    def gib(self) -> float:
        return self.n_bytes / GiB


def plan(root: Path, patterns: list[tuple[str, str]] | None = None) -> list[Group]:
    groups: list[Group] = []
    for pattern, dest in patterns if patterns is not None else GROUPS:
        archives = sorted(root.glob(pattern))
        if not archives:
            # From here alone there is no way to tell "not downloaded" from "deleted after
            # verification", so the message names both and lets the caller conclude.
            print(
                f"  [i] no archive matches {pattern} (not downloaded, or deleted after "
                "verification)",
                file=sys.stderr,
            )
            continue
        n_entries = n_bytes = 0
        for archive in archives:
            with zipfile.ZipFile(archive) as handle:
                for info in handle.infolist():
                    if info.is_dir():
                        continue
                    n_entries += 1
                    n_bytes += info.file_size
        groups.append(Group(dest, archives, n_entries, n_bytes))
    return groups


def scan_extracted(dest_root: Path, dest: str) -> tuple[int, int]:
    """(file count, total bytes) actually present in one family's extracted directory."""
    directory = dest_root / dest
    if not directory.is_dir():
        return (0, 0)
    files = [path for path in directory.rglob("*") if path.is_file()]
    return (len(files), sum(path.stat().st_size for path in files))


def load_manifest(dest_root: Path) -> dict:
    path = dest_root / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_manifest(dest_root: Path, entries: dict[str, dict], *, source: str) -> Path:
    """Write the manifest, keeping families this run did not touch.

    ``source`` is only ever upgraded to ``"archive"``, never downgraded to ``"snapshot"``: one
    verification against the archives is evidence, and it does not lose value because a snapshot
    was taken afterwards.
    """
    document = load_manifest(dest_root)
    document.setdefault("families", {}).update(entries)
    if source == "archive" or "source" not in document:
        document["source"] = source
    dest_root.mkdir(parents=True, exist_ok=True)
    path = dest_root / MANIFEST_NAME
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _target_path(dest_dir: Path, entry_name: str) -> Path:
    """Drop the leading directory component (``keyframes/L21_V001/079.jpg``)."""
    return dest_dir / (entry_name.split("/", 1)[1] if "/" in entry_name else entry_name)


def extract(group: Group, dest_root: Path, *, force: bool = False) -> tuple[int, int]:
    """Extract one archive family. Skips files already present at the right size (re-runnable)."""
    dest_dir = dest_root / group.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    started = time.time()
    for archive_path in group.archives:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                out = _target_path(dest_dir, info.filename)
                if not force and out.exists() and out.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, out.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1 << 20)
                written += 1
                if written % _PROGRESS_EVERY == 0:
                    print(
                        f"    {group.dest}: {written:,} files, {time.time() - started:.0f}s",
                        flush=True,
                    )
    print(
        f"  {group.dest}: {written:,} extracted, {skipped:,} skipped, {time.time() - started:.0f}s",
        flush=True,
    )
    return written, skipped


def _report_line(
    dest: str, n_files: int, n_bytes: int, expected_files: int, expected_bytes: int
) -> bool:
    ok = n_files == expected_files and n_bytes == expected_bytes
    print(
        f"  {dest:<14} {n_files:>7,}/{expected_files:,} files, "
        f"{n_bytes / GiB:.2f}/{expected_bytes / GiB:.2f} GiB  {'OK' if ok else 'MISMATCH'}"
    )
    return ok


def verify(group: Group, dest_root: Path) -> bool:
    """Verify the file count *and* the byte total — catches missing files and truncated ones."""
    if not (dest_root / group.dest).is_dir():
        print(f"  {group.dest:<14} MISSING (not extracted)")
        return False
    n_files, n_bytes = scan_extracted(dest_root, group.dest)
    return _report_line(group.dest, n_files, n_bytes, group.n_entries, group.n_bytes)


def verify_from_manifest(dest_root: Path, only: set[str] | None = None) -> bool:
    """Verify against the manifest — the only route left once the archives are gone."""
    document = load_manifest(dest_root)
    families = document.get("families", {})
    if not families:
        print(
            f"  no archives and no {MANIFEST_NAME} available to verify against.",
            file=sys.stderr,
        )
        return False
    print(f"  (expected values from {MANIFEST_NAME}, source={document.get('source', '?')})")
    all_ok = True
    for dest, expected in sorted(families.items()):
        if only and dest not in only:
            continue
        n_files, n_bytes = scan_extracted(dest_root, dest)
        all_ok &= _report_line(
            dest,
            n_files,
            n_bytes,
            int(expected.get("n_entries", 0)),
            int(expected.get("n_bytes", 0)),
        )
    return all_ok


def _manifest_of(groups: list[Group]) -> dict[str, dict]:
    return {
        group.dest: {"n_entries": group.n_entries, "n_bytes": group.n_bytes} for group in groups
    }


def _snapshot(dest_root: Path, only: set[str] | None) -> int:
    entries = {}
    for _pattern, dest in GROUPS:
        if only and dest not in only:
            continue
        n_files, n_bytes = scan_extracted(dest_root, dest)
        if n_files:
            entries[dest] = {"n_entries": n_files, "n_bytes": n_bytes}
    if not entries:
        print("no extracted family found to snapshot.", file=sys.stderr)
        return 1
    path = write_manifest(dest_root, entries, source="snapshot")
    print(f"\nwrote {path} ({len(entries)} families, source=snapshot)")
    for dest, entry in sorted(entries.items()):
        print(f"  {dest:<14} {entry['n_entries']:>7,} files, {entry['n_bytes'] / GiB:6.2f} GiB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default="data/batch1", help="the organiser's data directory")
    parser.add_argument("--dest", default=None, help="defaults to <root>/extracted")
    parser.add_argument("--only", action="append", help="restrict to this family (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="verify only, do not extract")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="write a manifest from what is on disk (use once the archives are deleted)",
    )
    parser.add_argument("--force", action="store_true", help="rewrite files already present")
    args = parser.parse_args(argv)
    enable_utf8_stdio()

    root = Path(args.root)
    if not root.is_dir():
        print(f"data directory not found: {root}", file=sys.stderr)
        return 2
    dest_root = Path(args.dest) if args.dest else root / "extracted"
    only = set(args.only) if args.only else None

    print(f"source     : {root}")
    print(f"destination: {dest_root}\n")

    if args.snapshot:
        return _snapshot(dest_root, only)

    groups = [group for group in plan(root) if not only or group.dest in only]

    print("plan:")
    for group in groups:
        print(
            f"  {group.dest:<14} {len(group.archives)} archives, "
            f"{group.n_entries:>8,} entries, {group.gib:>7.2f} GiB"
        )
    needed = sum(group.gib for group in groups)
    video_archives = sorted(root.glob("Videos_*.zip"))
    video_gib = sum(path.stat().st_size for path in video_archives) / GiB
    free = shutil.disk_usage(dest_root if dest_root.exists() else root).free / GiB
    print(f"\n  needs {needed:.1f} GiB + {HEADROOM_GIB:.0f} GiB reserved, {free:.1f} GiB free")
    print(
        f"  Videos_*.zip: {len(video_archives)} archives, {video_gib:.1f} GiB — NOT extracted "
        "(does not fit; extracted on demand for TRAKE)"
    )

    if args.verify:
        print("\nverification:")
        if not groups:
            return 0 if verify_from_manifest(dest_root, only) else 1
        ok = all(verify(group, dest_root) for group in groups)
        if ok:
            write_manifest(dest_root, _manifest_of(groups), source="archive")
        return 0 if ok else 1
    if args.dry_run:
        return 0
    if free < needed + HEADROOM_GIB:
        print(
            f"\nSTOPPING: not enough space ({free:.1f} < {needed + HEADROOM_GIB:.1f} GiB).",
            file=sys.stderr,
        )
        return 3

    print("\nextracting:")
    for group in groups:
        extract(group, dest_root, force=args.force)
    print("\nverification:")
    ok = all(verify(group, dest_root) for group in groups)
    if ok:
        write_manifest(dest_root, _manifest_of(groups), source="archive")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
