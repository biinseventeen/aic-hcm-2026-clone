# AIC 2026 — multi-task video retrieval (preliminary round)

A system for the three tasks of the AI Challenge HCM 2026 preliminary round: **Textual KIS**,
**Q&A** and **TRAKE**. The design starts from one observation about the scoring function: the
correct answer is an *interval* of frames, and each query may be answered with up to 100 rows — so
the output layer is not a ranker but a **coverage maximisation** problem over a submodular
function.

## Documentation

| Document | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Full design: objective formalisation, the 13 sub-problems, cost analysis |
| [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) | **Input** — the supplied data, schemas, measurements on the real corpus, anomalies |
| [`docs/PIPELINE_IO.md`](docs/PIPELINE_IO.md) | **The contract** — every file the pipeline reads and writes, and where each rule is enforced |
| [`docs/SUBMISSION.md`](docs/SUBMISSION.md) | **Output** — submission format, what the rules state vs. what we assume, checklist |
| [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md) | **Constraints**, blocking items, gaps, open risks |
| [`docs/BACKEND.md`](docs/BACKEND.md) | Using `aic` as a module for a backend: `Engine`, HTTP endpoints, operations |
| `docs/Thong tin vong So tuyen AIC2026.pdf` | The rules, as published |
| `docs/AIC_2026_Baseline_v1.ipynb` | The organisers' FiftyOne baseline, kept verbatim; what it confirms is in `DATA_AUDIT.md` §11 |

---

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). The environment lives **inside the
workspace** at `.venv/`, and `uv.lock` pins every version, so two machines resolve to the same
dependency set.

```bash
uv sync --extra backend      # creates .venv/ and installs the core + dev + backend dependencies
```

That is all that is needed to run the tests, the linter and the HTTP backend. Two extras are
deliberately left out of the default install because they are large and only needed for specific
branches:

```bash
uv sync --extra encoder      # sentence-transformers + torch — REQUIRED for real queries
uv sync --all-extras         # everything, including pillow for the TRAKE branch
```

Run commands through the workspace environment, never through the machine's Python:

```bash
uv run pytest -q             # or: .venv/Scripts/python -m pytest -q   (Windows)
uv run ruff check .
uv run aic --help
uv run pre-commit install    # once, so the hooks run on every commit
```

> **Do not `pip install` into the system Python.** Anything installed there is invisible to
> `uv.lock`, so it works on one machine and fails on the next. If a dependency is missing, add it to
> `pyproject.toml` and re-run `uv sync`.

### Dependency layout

| Group | Contents | Installed by default |
|---|---|---|
| `dependencies` | `numpy` — index building, slot allocation, submission writing | yes |
| `dependency-groups.dev` | `pytest`, `httpx`, `ruff`, `pre-commit` | yes (`uv sync`) |
| `optional-dependencies.backend` | `fastapi`, `uvicorn[standard]` | no |
| `optional-dependencies.encoder` | `sentence-transformers`, `torch` | no |
| `optional-dependencies.video` | `pillow` | no |

The core package depends on **no** web framework and **no** deep learning framework; that is what
lets `aic validate` and the whole test suite run in a minimal environment.

> **TLS note.** This machine terminates TLS with a certificate uv's bundled roots do not trust, so
> `[tool.uv] native-tls = true` is set in `pyproject.toml`. Without it every uv command fails with
> `invalid peer certificate: UnknownIssuer`. The setting is harmless on machines that do not need it.

---

## Blocking items

The text encoder is **installed and working** (`clip-ViT-B-32`, 512 dimensions, matching the
index), together with a second multilingual tower in the same image space that carries the
Vietnamese queries. Three things had to be true at once and two are easy to miss — install every
extra in one `uv sync`, add `pillow`, and give Python a certificate store it trusts before fetching
the weights: [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md) item C4.

What still blocks parts of the system: `torch` resolves to a CPU build although the machine has an
RTX 4060 8 GiB (C6), `ffmpeg` is absent so TRAKE runs keyframe-only at a ~14.5 % ceiling per moment
(C5), and there is no OCR, speech or VQA channel (G2, G3, G5) — which is why Q&A answers currently
enter through `--answers` rather than being produced by the pipeline.

---

## Pointing at the data

```bash
uv run python scripts/link_data.py            # create the data/batch1 link to the corpus
uv run python scripts/link_data.py --check    # verify the link and what is visible through it
```

`configs/default.json` says `data/batch1`, and `data/batch1` is a **directory junction** pointing
at the real corpus (107 GiB, which cannot live in the repository). The configuration is therefore
identical on every machine, and the link is the single place that knows where the data actually
is. For a corpus stored elsewhere use `--target E:/AIC`, or set `AIC_DATA_ROOT` (which overrides
the config file).

The name says **which release** the link points at: everything measured so far comes from batch 1
(L21–L30, 873 videos). A second batch arrives as its own link and its own `data_root`, so the two
never end up mixed inside one index — and a number in the docs can always be traced to the batch
it was measured on.

### Extracting and pruning archives

```bash
uv run python scripts/extract_data.py --dry-run    # show the plan, check free space
uv run python scripts/extract_data.py              # extract (~30.5 GiB, ~15 minutes)
uv run python scripts/extract_data.py --verify     # verify file counts and byte totals
uv run python scripts/prune_zips.py --yes          # delete archives of verified families
```

`prune_zips.py` re-runs verification immediately before deleting and keeps `Videos_*.zip` out of
scope by default: video is the official competition data and has no extracted copy, so deleting it
would be unrecoverable. On this machine the step freed 29.4 GiB.

Verification results are recorded in `extracted/.manifest.json` so the ability to check survives the
archives being deleted — see [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) §1.2.

---

## Workflow

```bash
uv run aic inventory          # list the data, confirm what is present and from which source
uv run aic validate           # BLOCKING ITEM — frame index convention + cross-checks
uv run aic build-index        # build the dense, shot and text indexes
uv run aic build-index --only text   # rebuild one stage; the dense index is 173 MiB
uv run aic query "<text>"     # run one query, print the full trace
uv run aic run queries.json   # run a whole set, write submissions, package the zip
uv run aic run queries.json --answers a.json --pins p.json   # see below
uv run aic check-submission   # final gate: read back from disk and validate
uv run aic serve              # HTTP backend — see docs/BACKEND.md
```

Two of those flags exist because parts of the pipeline are not built yet, and the score cannot
wait for them. `--answers` supplies Q&A answer hypotheses (the answer axis is one of three conjuncts
a Q&A row is scored on, and no VQA model exists), and `--pins` puts rows a human has verified by
looking at the frame at the head of the list, where slot 1 is worth a fifth of a query's score. Both
take a JSON file; [`devset/demo/`](devset/demo/README.md) holds a worked example of each.

Every hyperparameter that affects the score lives in
[`configs/default.json`](configs/default.json), generated from `aic.config.Config` so the file and
the dataclass cannot drift apart. Override with `aic --config configs/mine.json <command>`.

`aic validate` **must pass before** the index is built. It is a blocking item because a wrong frame
index convention causes silent score loss: the system runs normally, produces valid indices, and
scores zero.

Current state on batch 1:

```
P1 — frame index convention checked over 177,321 keyframes
  floor   177,321 / 177,321  = 100.00 %  <-- use this one
  round   154,399 / 177,321  =  87.07 %
  ceil    138,655 / 177,321  =  78.19 %
  VERDICT: PASS (0 mismatches)
```

---

## Repository layout

```
src/aic/          the library (see Architecture below)
tests/            321 tests, including doctests in src
scripts/          data plumbing: link, extract, prune, build query set
configs/          default.json — every score-affecting hyperparameter
docs/             design, data audit, submission spec, constraints, backend
devset/           evaluation sets — TRACKED: blind-sampled + the organisers' mock exam
data/batch1/      link to the organiser's corpus, batch 1  (git-ignored)
data/processed/   index, reports, submissions, video cache  (git-ignored)
```

`data/` is git-ignored in full: it holds either a link to the corpus or output reproducible from
`aic build-index`. The ignore pattern is `/data/` with a leading slash — without it the pattern also
matches `src/aic/data/` and silently drops that entire package from version control.

`devset/` sits deliberately outside `data/`: it is hand-annotated ground truth, authored work that
must be version-controlled.

---

## Architecture

The two entry points — CLI and HTTP — call into **one** façade, `aic.service.Engine`; no logic
exists on only one path.

```
                    ┌─────────────────────┐
   aic <command> ──► │                     │
                    │  aic.service.Engine │ ──► core / index / query / tasks / submit
   POST /solve   ──► │                     │
                    └─────────────────────┘

src/aic/
  config.py        every score-affecting hyperparameter; paths relative to the project root
  service.py       Engine + SolveResult — the only façade a backend imports
  log.py           the library logs, it does NOT print to stdout
  console.py       makes Vietnamese output printable on a Windows console
  cli.py           command line interface (a thin presentation layer over service.py)
  api/
    schemas.py     request/response contract in standard dataclasses, not pydantic
    app.py         the FastAPI application (optional, lazily imported)
  core/
    objective.py   P0  — the scoring rules, executable. No other module may redefine a score.
    frameidx.py    P1  — the frame index convention (BLOCKING ITEM)
    coverage.py         Pr(hit | S) = Σ π_v · κ_v(S_v) — the objective
    allocator.py   P13 — allocating the 100 slots with lazy greedy (CELF)
    spread.py      P10 — generating a frame set that certainly covers a locus
    fusion.py      P8  — fusing channels with RRF
    align.py       P12 — event sequence alignment
  data/
    layout.py           two sources (zip / directory), one interface
    keyframes.py        the keyframe table from map-keyframes
    features.py    P3  — the dense visual index
  index/
    validate.py         cross-checks across every data source
    shots.py       P2  — temporal segmentation
    text.py        P4/P5 — BM25 + fuzzy character n-grams
    priors.py           the content-domain prior
  query/
    parse.py       P7  — query understanding
    text_encoder.py     the text encoder, with an embedding-space check
    retrieve.py    P8  — channel orchestration and fusion
  tasks/
    kis.py  qa.py  trake.py
  submit/
    writer.py           generating, validating and packaging submissions
  eval/
    devset.py  score.py the internal evaluation set
```

### Seven properties locked by tests

```bash
uv run pytest -q          # 321 tests, including doctests in src
```

1. **Lazy greedy is correct.** `tests/test_allocator.py` compares CELF against greedy that
   recomputes everything at every step, over 25 random instances: **the gain sequences must match
   element by element**. Plus: `cumulative` equals `Pr(hit)` recomputed from scratch, and marginal
   gains never increase with rank (a consequence of submodularity).
2. **Coverage is guaranteed.** `tests/test_spread.py` checks, for every `(W, L)` pair, that **every**
   possible answer start position inside a locus is covered by some submitted frame.
3. **The two data sources are equivalent.** `tests/test_layout.py` checks that the zip and the
   extracted directory yield the **same keys and the same bytes** — if they diverged, extracting the
   data would silently change the pipeline's behaviour.
4. **The core package pulls in no web framework.** `tests/test_api.py` runs a subprocess that
   imports `aic.service` and asserts `fastapi` is absent from `sys.modules` — the condition for
   `aic validate` to run where no web package is installed.
5. **Paths are independent of the CWD.** `tests/test_config.py` asserts that changing directory does
   not change the project root, and that `configs/default.json` contains no machine-specific
   absolute path.
6. **Only real names reach the entity channel.** `tests/test_parse.py` asserts that an
   all-lowercase Vietnamese query yields no entities. The regex class `[A-ZĐÀ-Ỹ]` looks like
   "uppercase Latin plus Vietnamese", but that range spans blocks where the two cases
   interleave, so it also matched every lowercase accented letter: the query "áo đỏ đang nấu
   ăn" produced the entity "áo đỏ đang", which fed the fuzzy OCR channel and flipped the
   `has_named_entity` channel weighting. Both failures are silent.
7. **Domain labels agree in both directions.** `tests/test_priors.py` asserts the label set used by
   query understanding matches the L-group table — if they diverge, the prior silently stops working
   because a hint inferred from a query matches no video.

Plus: the scoring rules reproduce all four worked examples from the rules document
(`tests/test_objective.py`), and the submission validator turns a malformed file into readable
issues instead of a crash (`tests/test_submit.py`).

---

## What to do next, by value over cost

1. `uv sync --extra encoder` — blocks **every** query
2. Ask the organisers about the submission format and the answer-span length `L`
3. The `objects` channel — 1.68 GiB of unused signal, needs no GPU
4. A CUDA `torch` build plus `ffmpeg` — unblocks TRAKE, OCR and speech
5. A labelled internal evaluation set — every hyperparameter is currently a *reasoned* value, not a
   *calibrated* one

The full list with reasoning: [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md).
