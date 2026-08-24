"""Build a query set and its answer-key template from the organisers' ``.txt`` files.

The organisers publish one plain-text file per query, named ``query-<id>-<task>.txt``. That
filename is the **authoritative** source of two things the pipeline would otherwise have to
guess: the query id used in the submission filename, and the task. Deriving them here instead of
re-typing the queries removes the two error sources that cost a whole query each — a task
classified wrong (the answer goes into the wrong branch and the wrong file format) and a query
text that differs from the one the organisers scored against.

The generated files:

* ``queries.json``      — ``{query_id: text}``, the input to ``aic run``.
* ``ground-truth.json`` — the answer key in the :class:`aic.eval.devset.DevSet` schema, input to
  ``aic evaluate --devset``. Regenerating **preserves** everything a human has already
  annotated; only missing entries are added.
* ``queries.md``        — the query set as a readable document, text quoted verbatim.

    python scripts/build_query_set.py                          # phase1 -> devset/phase1
    python scripts/build_query_set.py --check                  # generated files still in sync?
    python scripts/build_query_set.py --source data/query/demo --out devset/demo

One directory per published set. The two sets released so far both number their queries
p1-1..p1-25 while assigning them different tasks — demo p1-17 is KIS, phase1 p1-17 is Q&A — so
merging them would silently overwrite half of either.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aic.config import find_project_root
from aic.console import enable_utf8_stdio

#: ``query-p1-18-trake.txt`` -> id ``p1-18``, task ``trake``. The id may contain hyphens, so the
#: task suffix is anchored at the end rather than the id being restricted to one segment.
_FILENAME_RE = re.compile(r"^query-(?P<query_id>.+)-(?P<task>kis|qa|trake)\.txt$")

#: A TRAKE moment line: ``E1: ...``. Counting these fixes N, which the rules say is set by the
#: query — a submission with the wrong number of frames is an invalid answer.
_MOMENT_RE = re.compile(r"^[ 	]*E[ 	]*(\d+)[ 	]*(?:[:.)]|[ 	])", re.MULTILINE)

#: Sort key: ``p1-10`` must follow ``p1-9``, which a plain string sort gets wrong.
_NUMBER_RE = re.compile(r"(\d+)")

TASK_ORDER = {"kis": 0, "qa": 1, "trake": 2}


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    task: str
    text: str
    #: TRAKE only: the moment labels as written in the file ("E1", "E2", ...).
    moment_labels: tuple[str, ...] = ()
    source: str = ""

    @property
    def n_moments(self) -> int:
        return len(self.moment_labels)

    @property
    def submission_file(self) -> str:
        return f"query-{self.query_id}-{self.task}.csv"


def _sort_key(query: Query) -> tuple[int, list[int | str]]:
    parts: list[int | str] = [
        int(piece) if piece.isdigit() else piece
        for piece in _NUMBER_RE.split(query.query_id)
        if piece
    ]
    return TASK_ORDER.get(query.task, 9), parts


def read_queries(source: Path) -> tuple[list[Query], list[str]]:
    """Read every ``query-<id>-<task>.txt`` in *source*. Returns the queries and any problems."""
    problems: list[str] = []
    queries: list[Query] = []
    files = sorted(source.glob("*.txt"))
    if not files:
        problems.append(f"no .txt query files found in {source}")
    for path in files:
        match = _FILENAME_RE.match(path.name)
        if not match:
            problems.append(f"{path.name}: filename is not query-<id>-<kis|qa|trake>.txt — skipped")
            continue
        # utf-8-sig: a BOM written by a Windows editor would otherwise become part of the first
        # word of the query and reach the encoder.
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
        if not text:
            problems.append(f"{path.name}: file is empty — skipped")
            continue
        task = match.group("task")
        labels = tuple(f"E{number}" for number in _MOMENT_RE.findall(text))
        if task == "trake" and not labels:
            problems.append(f"{path.name}: TRAKE query with no 'E<n>:' moment line — N is unknown")
        if task != "trake" and labels:
            problems.append(
                f"{path.name}: task is {task} but the text has {len(labels)} 'E<n>:' lines"
            )
        queries.append(
            Query(
                query_id=match.group("query_id"),
                task=task,
                text=text,
                moment_labels=labels if task == "trake" else (),
                source=path.name,
            )
        )
    seen: dict[str, str] = {}
    for query in queries:
        if query.query_id in seen:
            problems.append(
                f"{query.source}: query id {query.query_id} already came from "
                f"{seen[query.query_id]} — one of the two files will be lost"
            )
        seen[query.query_id] = query.source
    return sorted(queries, key=_sort_key), problems


def build_queries_json(queries: list[Query]) -> dict[str, str]:
    return {query.query_id: query.text for query in queries}


def build_kq(queries: list[Query], existing: dict | None) -> dict:
    """The answer key, **merging** over anything already annotated.

    A regeneration must never overwrite hand-annotated ground truth: the annotation is hours of
    frame-accurate work, whereas the text and the task can be re-derived in a second.
    """
    previous = (existing or {}).get("truths", {})
    truths: dict[str, dict] = {}
    for query in queries:
        old = previous.get(query.query_id)
        if old is not None:
            entry = dict(old)
            # The filename is authoritative for the task; the annotator may not override it.
            entry["task"] = query.task
            entry.setdefault("video_id", "")
            entry.setdefault("answer_aliases", [])
            if query.task == "trake":
                spans = [list(span) for span in entry.get("spans") or []]
                # Keep annotated spans, pad or flag against the N the query itself sets.
                while len(spans) < query.n_moments:
                    spans.append([0, 0])
                entry["spans"] = spans
                entry["span"] = None
            else:
                entry["spans"] = []
                entry["span"] = list(entry.get("span") or [0, 0])
        else:
            entry = {
                "query_id": query.query_id,
                "task": query.task,
                "video_id": "",
                "span": [0, 0] if query.task in ("kis", "qa") else None,
                "spans": [[0, 0] for _ in range(query.n_moments)],
                "answer": "TODO" if query.task == "qa" else "",
                "answer_aliases": [],
                "note": "",
            }
        truths[query.query_id] = entry

    # Nothing but the schema DevSet.load reads: an empty video_id is what marks an entry
    # unannotated, and the workflow belongs in README.md rather than in every regenerated copy.
    return {
        "queries": build_queries_json(queries),
        "truths": truths,
        "skipped": (existing or {}).get("skipped", {}),
    }


def build_markdown(queries: list[Query], source: Path) -> str:
    """The readable paper. Text is quoted verbatim, so this file never disagrees with the txt."""
    by_task: dict[str, list[Query]] = {}
    for query in queries:
        by_task.setdefault(query.task, []).append(query)
    counts = ", ".join(
        f"**{len(by_task[task])} {label}**"
        for task, label in (("kis", "Textual KIS"), ("qa", "Q&A"), ("trake", "TRAKE"))
        if task in by_task
    )

    lines = [
        "# Query set — verbatim",
        "",
        f"<!-- Generated by scripts/build_query_set.py from {source.as_posix()}. Do not edit:",
        "     a regeneration overwrites this file. Team notes live in README.md. -->",
        "",
        f"{len(queries)} queries: {counts}. The query text below is quoted **verbatim** from the",
        "organisers' files, including their spelling and numbering mistakes.",
        "",
        "| ID | Task | Submission file | Source | N |",
        "|---|---|---|---|---|",
    ]
    for query in queries:
        n = str(query.n_moments) if query.task == "trake" else "—"
        lines.append(
            f"| `{query.query_id}` | {query.task.upper()} | `{query.submission_file}` "
            f"| `{query.source}` | {n} |"
        )

    headings = {"kis": "Textual KIS", "qa": "Question Answering", "trake": "TRAKE"}
    for task in ("kis", "qa", "trake"):
        if task not in by_task:
            continue
        lines += ["", "---", "", f"## {headings[task]}"]
        for query in by_task[task]:
            suffix = f" — N = {query.n_moments}" if query.task == "trake" else ""
            lines += ["", f"### {query.source[:-4]}{suffix}", ""]
            lines += ["```text", query.text, "```"]
    return "\n".join(lines) + "\n"


def write_if_changed(path: Path, content: str, *, check: bool) -> bool:
    """Write *content* to *path*. Returns True when the file on disk differed."""
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    enable_utf8_stdio()
    root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="data/query/phase1", help="directory of the .txt query files"
    )
    parser.add_argument("--out", default="devset/phase1", help="directory for the generated files")
    parser.add_argument(
        "--check", action="store_true", help="write nothing; exit 1 if a file is out of date"
    )
    args = parser.parse_args()

    source = (root / args.source).resolve()
    out = (root / args.out).resolve()
    if not source.is_dir():
        print(f"source directory does not exist: {source}", file=sys.stderr)
        return 2

    queries, problems = read_queries(source)
    if not queries:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    truth_path = out / "ground-truth.json"
    existing = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else None
    payloads = {
        out / "queries.json": json.dumps(build_queries_json(queries), ensure_ascii=False, indent=2)
        + "\n",
        truth_path: json.dumps(build_kq(queries, existing), ensure_ascii=False, indent=2) + "\n",
        out / "queries.md": build_markdown(queries, Path(args.source)),
    }
    stale = [
        path
        for path, content in payloads.items()
        if write_if_changed(path, content, check=args.check)
    ]

    counts: dict[str, int] = {}
    for query in queries:
        counts[query.task] = counts.get(query.task, 0) + 1
    print(
        f"{len(queries)} queries from {source}: " + ", ".join(f"{n} {t}" for t, n in counts.items())
    )
    for query in queries:
        if query.task == "trake":
            labels = ", ".join(query.moment_labels)
            print(f"  {query.query_id:<8} TRAKE N={query.n_moments} ({labels})")
    annotated = sum(
        1 for truth in build_kq(queries, existing)["truths"].values() if truth.get("video_id")
    )
    print(f"  answer key: {annotated}/{len(queries)} entries annotated")
    for problem in problems:
        print(f"  [!] {problem}")

    if args.check:
        for path in stale:
            print(f"  OUT OF DATE: {path.relative_to(root).as_posix()}", file=sys.stderr)
        return 1 if stale else 0
    for path in payloads:
        marker = "wrote" if path in stale else "unchanged"
        print(f"  {marker:<10} {path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
