"""Point ``data/batch1`` at the organiser's corpus with a link, not a copy.

Basis
-----
The corpus is ~107 GiB compressed plus ~30.5 GiB extracted; it cannot live inside the source tree
and it cannot be copied. But every path in ``configs/default.json`` should be *relative to the
project root* — that is the condition for one configuration to work on a development machine, in a
container, and inside a backend process. A link at ``data/batch1`` satisfies both: the configuration
says ``data/batch1``, and the link is the single place that knows where the data actually is.

    python scripts/link_data.py                     # use the default paths
    python scripts/link_data.py --target E:/AIC     # corpus stored elsewhere
    python scripts/link_data.py --check             # check only, create nothing
    python scripts/link_data.py --force             # replace an existing link

On Windows this creates a **directory junction** rather than a symlink: a junction needs neither
administrator rights nor Developer Mode, while ``os.symlink`` needs one of the two. On POSIX it
creates an ordinary symlink. Both are traversed transparently by ``pathlib`` and ``zipfile``, so the
rest of the system never knows a link is involved.

One junction constraint: the target must be an *absolute* path. This script converts it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aic.config import find_project_root
from aic.console import enable_utf8_stdio

DEFAULT_TARGET = "D:/workspace/AI-Challenge-HCM"

#: Families that satisfy the check when *either* the archive is present *or* the extracted copy
#: is. After ``scripts/prune_zips.py`` runs, the archives are gone by design and only the
#: extracted directory remains, so requiring the archive would raise a false alarm.
EXPECTED_EITHER: tuple[tuple[str, str], ...] = (
    ("map-keyframes*.zip", "map-keyframes"),
    ("media-info*.zip", "media-info"),
    ("clip-features*.zip", "clip-features"),
    ("objects*.zip", "objects"),
    ("Keyframes_*.zip", "keyframes"),
)

#: Families that must be present as archives: video is never extracted wholesale.
EXPECTED_ARCHIVE_ONLY: tuple[str, ...] = ("Videos_*.zip",)

#: Windows file attribute marking a reparse point (junctions and symlinks both set it).
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def link_kind(path: Path) -> str | None:
    """``"junction"`` | ``"symlink"`` | ``"dir"`` | ``None`` (does not exist)."""
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        return "symlink"
    # A junction on Windows looks like a directory but carries the reparse-point flag.
    try:
        if os.name == "nt" and bool(
            path.lstat().st_file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT  # type: ignore[attr-defined]
        ):
            return "junction"
    except (AttributeError, OSError):
        pass
    return "dir"


def resolve_link(path: Path) -> Path | None:
    if link_kind(path) is None:
        return None
    try:
        return path.resolve()
    except OSError:
        return None


def create_link(link: Path, target: Path) -> str:
    """Create the link and return which kind was made. Raises with a usable message on failure."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # mklink /J makes a junction: no administrator rights needed, unlike /D for a symlink.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise OSError(f"mklink /J failed (code {result.returncode}): {message}")
        return "junction"
    link.symlink_to(target, target_is_directory=True)
    return "symlink"


def audit(link: Path) -> tuple[bool, list[str]]:
    """Check what is visible through the link. Returns (ok, report lines).

    A family counts as present when either its archive or its extracted copy exists. That is the
    right criterion because ``scripts/prune_zips.py`` deliberately removes archives once their
    extracted copy has been verified.
    """
    lines: list[str] = []
    ok = True
    extracted_root = link / "extracted"
    for pattern, family in EXPECTED_EITHER:
        n_archives = len(list(link.glob(pattern)))
        directory = extracted_root / family
        n_extracted = sum(1 for _ in directory.iterdir()) if directory.is_dir() else 0
        if n_archives and n_extracted:
            state = f"{n_archives} archives + extracted"
        elif n_archives:
            state = f"{n_archives} archives (not extracted)"
        elif n_extracted:
            state = "extracted (archives pruned)"
        else:
            state = "MISSING"
            ok = False
        lines.append(f"  {family:<16} {state}")

    for pattern in EXPECTED_ARCHIVE_ONLY:
        n_archives = len(list(link.glob(pattern)))
        lines.append(
            f"  {pattern:<16} {n_archives} archives" + ("" if n_archives else "   MISSING")
        )
        ok &= n_archives > 0
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("AIC_DATA_TARGET", DEFAULT_TARGET),
        help=f"the real corpus directory (default {DEFAULT_TARGET})",
    )
    parser.add_argument("--link", default=None, help="defaults to <project root>/data/batch1")
    parser.add_argument("--check", action="store_true", help="check only, create nothing")
    parser.add_argument("--force", action="store_true", help="replace an existing link")
    args = parser.parse_args(argv)
    enable_utf8_stdio()

    root = find_project_root()
    link = Path(args.link) if args.link else root / "data" / "batch1"
    target = Path(args.target).resolve()

    print(f"project root: {root}")
    print(f"link        : {link}")
    print(f"target      : {target}\n")

    kind = link_kind(link)
    if kind is not None:
        current = resolve_link(link)
        same_target = current is not None and current == target
        state = "correct target" if same_target else "DIFFERENT TARGET"
        print(f"already exists: {kind} -> {current}  ({state})")
        if args.check or (same_target and not args.force):
            ok, lines = audit(link)
            print("\nvisible through the link:")
            print("\n".join(lines))
            return 0 if (same_target and ok) else 1
        if kind == "dir":
            print(
                f"STOPPING: {link} is a REAL directory, not a link. A directory that may hold "
                "data is never removed automatically — inspect it and delete it yourself if you "
                "are sure.",
                file=sys.stderr,
            )
            return 3
        # Only the link is removed, never a recursive delete: rmdir on a junction or symlink
        # detaches the link and leaves the data at the target untouched.
        print("detaching the old link")
        if kind == "symlink":
            link.unlink()
        else:
            link.rmdir()
    elif args.check:
        print("no link yet. Run again without --check to create it.", file=sys.stderr)
        return 1

    if not target.is_dir():
        print(f"STOPPING: the target does not exist: {target}", file=sys.stderr)
        return 2

    created = create_link(link, target)
    print(f"created {created}: {link} -> {target}")
    ok, lines = audit(link)
    print("\nvisible through the link:")
    print("\n".join(lines))
    if not ok:
        print("\nWARNING: the expected archives were not all found at the target.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
