"""Repository hygiene checks used as pre-commit hooks.

These are deliberately implemented here rather than pulled from ``pre-commit-hooks``: that repo's
hooks hang on this machine when handed more than a handful of files, and every check below needs
nothing beyond the standard library, so an isolated hook environment buys nothing.

    python scripts/check_repo_hygiene.py --max-kb 2048 FILE...

Checks applied to each file given:

* size ceiling — the dense index is 173 MiB and submission archives are generated output; neither
  belongs in history. ``.gitignore`` already excludes them, so this is the backstop.
* parseable TOML and JSON — a broken ``pyproject.toml`` or ``configs/default.json`` breaks the build
  and the config loader respectively.
* no trailing whitespace, and a final newline — ruff-format enforces both for Python; this extends
  them to Markdown, TOML, YAML and JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

#: Extensions whose whitespace hygiene is checked here. Python is left to ruff-format.
TEXT_SUFFIXES = frozenset({".md", ".toml", ".yaml", ".yml", ".json", ".cfg", ".txt"})


def check_file(path: Path, max_bytes: int) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return problems

    size = path.stat().st_size
    if size > max_bytes:
        problems.append(f"{path}: {size / 1024:.0f} kB exceeds the {max_bytes / 1024:.0f} kB limit")

    if path.suffix == ".toml":
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            problems.append(f"{path}: invalid TOML: {exc}")
    elif path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            problems.append(f"{path}: invalid JSON: {exc}")

    if path.suffix in TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{path}: not valid UTF-8: {exc}")
            return problems
        for number, line in enumerate(text.splitlines(), 1):
            if line != line.rstrip():
                problems.append(f"{path}:{number}: trailing whitespace")
        if text and not text.endswith("\n"):
            problems.append(f"{path}: no newline at end of file")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-kb", type=int, default=2048, help="per-file size ceiling in kB")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)

    problems: list[str] = []
    for name in args.files:
        problems.extend(check_file(Path(name), args.max_kb * 1024))
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
