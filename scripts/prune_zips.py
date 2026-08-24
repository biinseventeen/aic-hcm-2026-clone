"""Delete the source archives of families that are **extracted and verified**.

Principle: an archive is deleted only once its extracted copy has been confirmed complete by both
file count and byte total. The verification is re-run immediately before deleting; the result of an
earlier run is not trusted.

``Videos_*.zip`` is out of scope by default. The rules (section 3) designate the video as the
official competition data, and there is currently no extracted copy of it; deleting the archives
would mean unrecoverable data loss. The ``--include-videos`` flag only takes effect once the
extracted copy of the video family has itself been verified.

    python scripts/prune_zips.py              # list, delete nothing
    python scripts/prune_zips.py --yes        # delete the verified families
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aic.config import find_project_root
from aic.console import enable_utf8_stdio
from extract_data import GROUPS, GiB, plan, scan_extracted

#: The video family, which is not part of GROUPS because it is never extracted wholesale.
VIDEO_GROUP = ("Videos_*.zip", "video")


@dataclass
class Verdict:
    dest: str
    archives: list[Path]
    n_entries: int
    n_bytes: int
    found_files: int
    found_bytes: int

    @property
    def ok(self) -> bool:
        return self.found_files == self.n_entries and self.found_bytes == self.n_bytes

    @property
    def zip_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.archives)

    def line(self) -> str:
        mark = "COMPLETE" if self.ok else "MISMATCH"
        return (
            f"  {self.dest:<14} {self.found_files:>7,}/{self.n_entries:<7,} files  "
            f"{self.found_bytes / GiB:6.2f}/{self.n_bytes / GiB:<6.2f} GiB  {mark:<9} "
            f"zip {self.zip_bytes / GiB:6.2f} GiB ({len(self.archives)})"
        )


def verify(group, dest_root: Path) -> Verdict:
    found_files, found_bytes = scan_extracted(dest_root, group.dest)
    return Verdict(
        dest=group.dest,
        archives=list(group.archives),
        n_entries=group.n_entries,
        n_bytes=group.n_bytes,
        found_files=found_files,
        found_bytes=found_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=None, help="defaults to <project root>/data/batch1")
    parser.add_argument("--dest", default=None, help="defaults to <root>/extracted")
    parser.add_argument("--yes", action="store_true", help="actually delete")
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="allow deleting Videos_*.zip — only if its extracted copy verifies COMPLETE",
    )
    parser.add_argument("--only", action="append", help="restrict to this family (repeatable)")
    args = parser.parse_args(argv)
    enable_utf8_stdio()

    root = Path(args.root) if args.root else find_project_root() / "data" / "batch1"
    if not root.is_dir():
        print(f"data directory not found: {root}", file=sys.stderr)
        return 2
    dest_root = Path(args.dest) if args.dest else root / "extracted"

    patterns = list(GROUPS) + ([VIDEO_GROUP] if args.include_videos else [])
    only = set(args.only) if args.only else None
    groups = [group for group in plan(root, patterns=patterns) if not only or group.dest in only]
    if not groups:
        print("no family matches.", file=sys.stderr)
        return 2

    print(f"archives : {root}")
    print(f"extracted: {dest_root}\n")
    print("re-verifying before deleting:")
    verdicts = [verify(group, dest_root) for group in groups]
    for verdict in verdicts:
        print(verdict.line())

    deletable = [verdict for verdict in verdicts if verdict.ok and verdict.archives]
    blocked = [verdict for verdict in verdicts if not verdict.ok]
    freed = sum(verdict.zip_bytes for verdict in deletable)

    if blocked:
        print("\nKEEPING (the extracted copy is not complete):")
        for verdict in blocked:
            print(
                f"  {verdict.dest}: {verdict.n_entries - verdict.found_files:,} files short, "
                f"{(verdict.n_bytes - verdict.found_bytes) / GiB:.2f} GiB short"
            )

    if not args.include_videos:
        videos = sorted(root.glob("Videos_*.zip"))
        if videos:
            video_gib = sum(path.stat().st_size for path in videos) / GiB
            print(f"\nOUT OF SCOPE: Videos_*.zip — {len(videos)} archives, {video_gib:.1f} GiB.")
            print("  Video is the official competition data and has no extracted copy. Deleting")
            print("  it is unrecoverable. Extract and verify first, then use --include-videos.")

    free_before = shutil.disk_usage(root).free / GiB
    print(
        f"\nwould free {freed / GiB:.1f} GiB "
        f"({sum(len(v.archives) for v in deletable)} archives); "
        f"free {free_before:.1f} -> {free_before + freed / GiB:.1f} GiB"
    )

    if not args.yes:
        print("\nnothing deleted. Add --yes to proceed.")
        return 0
    if not deletable:
        print("\nnothing is deletable.", file=sys.stderr)
        return 1

    print("\ndeleting:")
    deleted = 0
    for verdict in deletable:
        for path in verdict.archives:
            size = path.stat().st_size
            path.unlink()
            deleted += 1
            print(f"  {path.name:<32} {size / GiB:6.2f} GiB")
    print(
        f"\ndeleted {deleted} archives, freed {freed / GiB:.1f} GiB; "
        f"{shutil.disk_usage(root).free / GiB:.1f} GiB free"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
