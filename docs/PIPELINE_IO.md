# Pipeline input and output — the exact contract

Everything the pipeline reads, everything it writes, and where each requirement of the organisers'
result specification is enforced. `SUBMISSION.md` argues about the format; this document *states* it.

---

## 1. Inputs

### 1.1. The corpus — read-only, git-ignored

`data/batch1/` is a directory junction to the organisers' corpus (`scripts/link_data.py`). Five
families are read, either from the `.zip` or from `extracted/`, whichever is present:

| Family | Path inside the corpus | What the pipeline takes from it |
|---|---|---|
| `map-keyframes` | `map-keyframes/<video>.csv` | `n`, `pts_time`, `fps`, `frame_idx` — **the frame index system every answer is expressed in** |
| `clip-features` | `clip-features/<video>.npy` | `(n_keyframes, 512)` float16, L2-normalised, row `i` ↔ keyframe `n = i+1` |
| `media-info` | `media-info/<video>.json` | title, author, publish date, description, keywords → the two sparse channels |
| `keyframes` | `keyframes/<video>/<nnn>.jpg` | images, for human verification only; no module embeds them |
| `objects` | `objects/<video>/<nnn>.json` | **read by nothing yet** (CONSTRAINTS G1) |
| `Videos_*.zip` | archives | **read by nothing yet**; needed for TRAKE at full fps (C5) |

Built once by `aic build-index` into `data/processed/index/` (git-ignored, reproducible):

| File | Contents |
|---|---|
| `dense_vectors.npy` + `dense_meta.json` | 177,321 × 512 float16, plus the row → (video, keyframe, frame) table |
| `shots.json` | 176,707 shots, derived from keyframe spacing |
| `text_media.json` | BM25 index over title + author + date + description + keywords |
| `text_title.json` | BM25 index over **titles only** — the boilerplate-free channel |

`aic build-index --only {dense,shots,text}` rebuilds one stage; the dense stage is minutes, the rest
seconds.

### 1.2. The query set — authored, version-controlled

The organisers publish one plain-text file per query, `data/query/<set>/query-<id>-<task>.txt`. The
filename is authoritative for two things the pipeline must not guess: the **query id** used in the
submission filename, and the **task**. `scripts/build_query_set.py` reads them and writes one
directory per set:

| Set | Source | Generated | Submission |
|---|---|---|---|
| mock round | `data/query/demo/` (24 queries) | `devset/demo/` | `data/processed/submissions/demo/` |
| round 1 | `data/query/phase1/` (25 queries) | `devset/phase1/` | `data/processed/submissions/phase1/` |

**The sets must never share a directory.** Both number their queries `p1-1 … p1-25` while assigning
them *different tasks* — `p1-17` is KIS in the mock set and Q&A in round 1, `p1-3` does not exist in
the mock set and is Q&A in round 1 — so a merged directory would overwrite half of either, and the
overwritten half would still validate.

```
devset/<set>/queries.json        {query_id: text}                  -> aic run
devset/<set>/ground-truth.json   DevSet schema, hand-annotated     -> aic evaluate
devset/<set>/queries.md          the paper, verbatim, for humans
devset/<set>/answers.json        Q&A hypotheses, hand-written      -> aic run --answers
devset/<set>/pins.json           verified rows, hand-written       -> aic run --pins
```

`--check` exits 1 when a generated file has drifted from the `.txt` source. The two moment-label
forms seen so far are both accepted: `E1:` with a colon (mock) and bare `E1 ` (round 1).

### 1.3. Human input — the two files that exist because parts of the pipeline do not

```jsonc
// devset/demo/answers.json  ->  aic run --answers
{
  "p1-15": [ { "text": "Giang Ly", "prob": 0.5 },
             { "text": "Xã Giang Ly", "prob": 0.35 } ]
}
```
Q&A answer hypotheses. The Q&A R-Score is a conjunction of *three* conditions, so with no VQA model
(G5) a Q&A row cannot score at all without this file. Probabilities need not sum to 1; they order the
hedge. Keys starting with `_` are ignored, which is where the evidence for each answer is recorded.

```jsonc
// devset/demo/pins.json  ->  aic run --pins
{
  "p1-20": [ { "video_id": "L26_V004", "frame": 476,
               "why": "kf 014 - finished plate, three glasses, two edible flowers" } ]
}
```
Rows a human verified by opening the keyframe. They are placed at the **head** of the answer list,
before anything the model proposed, and carry `source="pinned"` with gain 0 in the trace. Reason:
slot 1 is worth a fifth of a query's score, and a verified answer sitting at slot 29 — which is where
one of these was found — keeps only 0.4 of the 1.0 it had earned. Ignored for TRAKE, whose rows need
N frames rather than one.

### 1.4. Configuration

`configs/default.json`, every score-affecting value in one place. The ones that change results most:

| Key | Value | Effect |
|---|---|---|
| `answer_span.kis` / `.qa` / `.trake` | 25 / 25 / 10 | assumed width `L` of the organisers' answer interval (R3, **unconfirmed**) |
| `retrieval.n_candidates` | 500 | ceiling on `P@100` |
| `retrieval.max_shots_per_video` | 12 | stops one video filling the list |
| `retrieval.sparse_max_terms` | 8 | highest-IDF query terms kept for BM25 |
| `text_encoder` | `clip-ViT-B-32` | must be the tower of the checkpoint the supplied features come from |
| `text_encoder_multilingual` | `clip-ViT-B-32-multilingual-v1` | second dense channel; empty disables it |

`AIC_DATA_ROOT` overrides the corpus path; `HF_HUB_OFFLINE=1` keeps the encoders off the network
once their weights are cached.

---

## 2. Output

### 2.1. One CSV per query

`data/processed/submissions/<set>/query-<query_id>-<task>.csv` (`aic run --out <set>`), at most 100
rows, no header, UTF-8, CRLF, comma-separated, quoting only where a field needs it. Real rows:

```text
# query-p1-1-kis.csv        <video_id>,<frame_id>
L27_V013,2314
L27_V012,2265

# query-p1-15-qa.csv        <video_id>,<frame_id>,<answer>
L30_V072,676,Giang Ly
L30_V072,1776,Xã Giang Ly

# query-p1-19-qa.csv        an answer containing a comma is quoted
L27_V010,5535,"Hỏa hồng Nhật Tảo oanh thiên địa, Kiếm bạt Kiên Giang khấp quỷ thần"

# query-p1-18-trake.csv     <video_id>,<frame_1>,...,<frame_N>   (N = 4 here)
L26_V469,372,480,496,542
```

`frame_id` is the **frame index used for scoring**, i.e. `frame_idx` from `map-keyframes`
(`floor(pts_time × fps)`, verified against all 177,321 keyframes). It is *not* the keyframe ordinal
that the organisers' own baseline notebook displays — submitting that scores zero on every row while
the file still validates (`DATA_AUDIT.md` §11).

### 2.2. The archive

`data/processed/submissions/<set>/submission.zip`, containing a `submission/` **directory** with the
CSV files inside it. Zipping the files directly is explicitly wrong per the specification. Rename the zip
freely; the layout inside is what matters.

```
submission.zip
└── submission/
    ├── query-p1-1-kis.csv
    ├── … round 1: 20 KIS, 4 Q&A, 1 TRAKE   (mock: 18 KIS, 3 Q&A, 3 TRAKE)
    └── query-p1-16-trake.csv
```

### 2.3. Diagnostics, not submitted

| File | Written by | Contents |
|---|---|---|
| `data/processed/reports/validate.json` | `aic validate` | frame-convention check over every keyframe, corpus cross-checks |
| `data/processed/reports/evaluate.json` | `aic evaluate` | per-query Final Score, first-hit rank, failure breakdown |
| `data/processed/submissions/_selftest/` | `aic submit-selftest` | three sample files to send the organisers |

### 2.4. Where each requirement is enforced

| Requirement | Enforced in | Locked by |
|---|---|---|
| ≤ 100 rows, order decisive | `objective.final_score`, `allocator` | `test_objective.py`, `test_allocator.py` |
| Column count per task | `Answer.row` | `test_submit.py` |
| `video_id` matches `L<group>_V<number>` and exists | `writer.validate` | `test_submit.py` |
| `frame_id` integer ≥ 0, inside the video | `writer.validate` | `test_submit.py` |
| Q&A answer present, ≤ 100 characters, no newline | `writer.validate` (errors) | `test_submit.py` |
| Q&A answer without stray whitespace | `writer.validate` (warning) | `test_submit.py` |
| TRAKE frame count = N, increasing | `writer.validate` | `test_submit.py` |
| No duplicate rows | `writer.validate` | `test_submit.py` |
| UTF-8, CRLF, no header, minimal quoting | `SubmissionNaming` | `test_submit.py` |
| Archive holds `submission/` | `package_submission` | `test_submit.py` |
| Read-back of what was actually written | `validate_submission_dir` | `aic check-submission` |

---

## 3. End to end

```bash
uv run python scripts/link_data.py                 # point at the corpus
uv run aic validate                                # BLOCKING: frame convention
uv run aic build-index                             # dense + shots + text + titles
uv run python scripts/build_query_set.py           # .txt -> queries.json + ground-truth.json

# round 1 — the set being scored
HF_HUB_OFFLINE=1 uv run aic run devset/phase1/queries.json --out phase1
uv run aic check-submission --dir data/processed/submissions/phase1

# the mock set, where the answers of four queries are known and pinned
ANS=devset/demo/answers.json PINS=devset/demo/pins.json
HF_HUB_OFFLINE=1 uv run aic run devset/demo/queries.json --out demo --answers $ANS --pins $PINS
uv run aic check-submission --dir data/processed/submissions/demo
uv run aic evaluate --devset devset/demo/ground-truth.json --answers $ANS --pins $PINS
```

Between `run` and `check-submission` nothing is edited by hand: a file that needs hand-editing is a
bug in the writer, and hand-edits are exactly what `check-submission` cannot see.
