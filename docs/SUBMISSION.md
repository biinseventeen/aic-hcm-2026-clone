# Submission — the output specification

This document is the **output specification** of the system. It separates two things: what the
rules **state for certain**, and what we **assume**. Confusing the two is the fastest way to lose
points with no warning at all — the file is still syntactically valid, the scoring system still
reads it, and the score is still zero.

---

## 1. What the rules state for certain

From "Thông tin vòng Sơ tuyển AIC2026", section 2:

| Item | Content |
|---|---|
| Answer count | At most **100** per query |
| Order | **Decisive** — `R@k` is the maximum R-Score over the **first** `k` rows |
| Textual KIS | `<video_id>, <frame_id>` |
| Q&A | `<video_id>, <frame_id>, <answer>` |
| TRAKE | `<video_id>, <frame_id_1>, ..., <frame_id_N>` |
| Score | `FinalScore = (1/5) · Σ_{k ∈ {1,5,20,50,100}} R@k` |
| `video_id` | Written in the rules as `video_abc(.mp4)` — the extension is **optional** |
| Q&A `answer` | May be **Vietnamese or English** |
| TRAKE | Exactly **one** frame per stage; `N` is fixed by the query |
| TRAKE, wrong video | **Hard zero**, with no per-moment credit |

From "Yêu cầu kết quả" (the result specification), which settles the **file format** as well:

| Item | Content |
|---|---|
| File | one `.csv` **text** file per query — never `.xlsx`/`.xls` |
| Rows | at most 100, no header row, data starts on line 1 |
| Encoding / delimiter | UTF-8 / comma |
| Line ending | CRLF **or** LF, both accepted |
| Quoting | only when needed: comma, quote (doubled `""`), or newline in an answer |
| Whitespace | **preserved, not trimmed** — a stray space changes the answer |
| `video_id` | **without** `.mp4` |
| `frame_id` | compared as an integer |
| Q&A `answer` | at most **100 characters** |
| TRAKE | frame count matches N exactly, in temporal order |
| Archive | `.zip` containing a **`submission/` directory** with the CSV files inside |
| Zip name | letters and digits recommended, e.g. `team_ABC_round1.zip` |

The four worked examples from the rules, used as test cases (`tests/test_objective.py`):

```
KIS   : GT = L01_V001, [500, 510]   ->  L01_V001, 505 -> 1.0 ; L01_V001, 600 -> 0.0
Q&A   : GT = L05_V005, [800, 900], "màu xanh"  ->  L05_V005, 888, màu xanh -> 1.0
TRAKE : GT = L10_V010, [95,105] [145,155] [195,205] [245,255]
        submitted L10_V010, 101, 156, 203, 251 -> 3/4 = 0.75
Final : R-Scores [0.5, 0, 0.8, 0...] -> (0.5+0.8+0.8+0.8+0.8)/5 = 0.74
```

All four are reproduced exactly by `aic.core.objective` and locked by tests.

---

## 2. Where each format decision comes from

Most of this table used to read "convention of previous AIC seasons". The result specification has
since settled nearly all of it — including one item where **the previous assumption was wrong**: the
archive needs a `submission/` directory, not a flat layout.

| Decision | Value | Basis |
|---|---|---|
| One file per query | yes | **Stated** by the result specification, and matched by the organisers' own query files (`query-p1-1-kis.txt`) |
| Filename | `query-<id>-<task>.csv` | **Stated**: their examples are `query-1-kis.csv`, `query-2-qa.csv`, `query-3-trake.csv` |
| Header row | **no** | **Stated** |
| Delimiter | `,` | **Stated** |
| Line terminator | CRLF | **Stated** that CRLF or LF are both accepted; CRLF chosen |
| Encoding | UTF-8 | **Stated** |
| `video_id` | **without** `.mp4` | **Stated** |
| Packaging | `.zip` holding `submission/` | **Stated**, and the opposite of what this project assumed before: a flat archive is explicitly wrong |
| Q&A answer matching | semantic, with aliases | **Contradicted inside the specification itself** — see below |
| Q&A rows sharing one `(video_id, frame_id)` | allowed | Not addressed either way |

Every one of these lives in `aic.submit.writer.SubmissionNaming`, changeable in a single line.
`package_submission` writes the `submission/` directory by default; `SubmissionNaming(zip_dir="")`
restores a flat archive.

> **The one real contradiction.** The Q&A section of the specification says the answer is compared
> "chính xác **về mặt ngữ nghĩa**" (semantically), while the closing notes say "Answer (Q&A) sẽ được
> so sánh dưới dạng **chuỗi chính xác**" (as an exact string). The two readings call for different
> strategies:
>
> * *semantic* — one well-formed answer per frame is enough, and `aic.eval.score`'s numeric/alias
>   tiers model the scoring correctly;
> * *exact string* — the wording of the answer becomes a lottery, which makes the answer-axis hedge
>   of §4.3 **more** valuable, not less: several phrasings of the same answer on the same frame are
>   then the only defence, and every internal number produced by the semantic scorer is optimistic.
>
> Until the organisers answer, `--hedge-answers` (the default) is the strategy that survives both
> readings. This is question 1 below.

### Questions to ask the organisers before submitting

1. **Is the Q&A answer compared semantically or as an exact string?** The specification states
   both, in two different places. This decides whether the answer-axis hedge is optional or
   essential.
2. **May several rows share a `(video_id, frame_id)` with different `answer` values?** Not addressed
   by the specification. See §4.3.
3. Roughly how many frames wide is the answer span `[s, e]` for KIS and Q&A?
   (`CONSTRAINTS.md` R3: the most sensitive hyperparameter in the system.)
4. For TRAKE, is `N` taken from the labels as written? The mock set numbers one query
   `E1, E2, E2, E4` — four moments, three distinct labels. We submit four frames.
5. Is a `frame_id` the **video frame index** (`floor(pts_time * fps)` from `map-keyframes`) rather
   than the keyframe ordinal? The published baseline notebook displays the ordinal, so a team
   following it submits a different number entirely (`DATA_AUDIT.md` §11).

Generate a sample set to attach to those questions:

```bash
uv run aic submit-selftest
```

It prints three complete sample files (KIS, Q&A, TRAKE) plus the format in use and what is
still open, into `data/processed/submissions/_selftest/`.

---

## 3. Constraints enforced in code

`aic.submit.writer.validate()` separates **errors** (which block writing) from **warnings** (which
do not, but must be read). The dividing line is one criterion: an error makes the score
**certainly** zero; a warning reduces expectation.

### Errors — writing is blocked (`strict=True`, the default)

| Error | Why it is an error |
|---|---|
| Empty list | Forfeits the whole query |
| More than 100 rows | Over the limit set by the rules |
| `video_id` not matching `L<group>_V<number>` | Matches no video, so scores zero |
| `video_id` absent from the corpus | Same reason; checkable because all 873 ids are known |
| Negative or non-integer `frame_id` | A frame index that does not exist |
| Missing Q&A `answer` | One of the three conditions is absent |
| `answer` containing a newline | Breaks the CSV, shifting every following row |
| `answer` over 100 characters | Over the limit the specification states |
| Wrong TRAKE moment count `N` | An invalid answer |
| Two exactly identical rows | One slot wasted for nothing |

### Warnings — must be read

| Warning | Meaning |
|---|---|
| Fewer than 100 rows | The last 50 slots are still worth 0.2 points and cost almost nothing. **Always fill them.** |
| `frame_id` past the last known frame of the video | Possibly a nonexistent index |
| `answer` with leading or trailing whitespace | Whitespace is preserved, so it changes the answer |
| TRAKE frames not increasing | Moments of an event sequence are normally in temporal order |
| Filename off the pattern | Not the naming the specification shows; the file may be ignored |

### Reading back from disk, not validating objects in memory

`validate_submission_dir()` **re-reads the written files** and validates those. This is deliberate:
it is the only way to catch encoding errors, stray newlines and filename errors. The parser there is
**defensive** — a malformed row (a short row, a non-numeric `frame_id`, a wrongly encoded file, or a
stray header row) becomes a readable issue and **never** kills the process. Locked by
`tests/test_submit.py`.

---

## 4. Three submission strategies the scoring function forces

### 4.1. Cover, do not rank (KIS)

The answer is an *interval* `[s, e]`, not a point. For a locus of width `W` and an answer interval
of length `L`, submitting **one** best frame per candidate shot is dominated while slots remain
free: `ceil(W / L)` frames spaced exactly `L` apart are needed for guaranteed coverage.

The real numbers on batch 1: the median shot is 55 frames, `L = 25`, so **3 frames per shot**. One
hundred slots therefore guarantee coverage of about 30 median-length shots.

### 4.2. Hedge across videos, and hedge within one video

`Pr(hit | S) = Σ_v π_v · κ_v(S_v)` separates the two sources of uncertainty: picking the wrong
video, and picking the wrong frame inside the right video. `aic.core.allocator` allocates slots
between the two kinds of hedge by greedy maximisation of a submodular function, so the trade-off is
**quantitative** rather than a matter of taste. `max_shots_per_video = 12` stops one video from
filling the list.

### 4.3. Hedge along the answer axis (Q&A)

Q&A has **three** failure axes: video, frame, answer. Once the best `(video, frame)` pair is fully
covered, the largest remaining marginal gain is to attach a **different answer** to that same frame,
because it opens a new term whose `κ` starts again from zero. For example, submitting both `"5"` and
`"năm"` for one frame — exactly the two answer forms the rules give in their Q&A example.

> This strategy depends on **question 4 in §2**. If the organisers accept only one answer per
> `(video_id, frame_id)`, it must be switched off:
>
> ```bash
> uv run aic run queries.json --no-hedge-answers
> ```
>
> The flag passes `hedge_answers=False` down to `solve_qa`, which keeps exactly the
> highest-probability answer hypothesis per pair and records the reason in `trace.notes`.

---

## 5. The submission workflow

```bash
uv run aic validate           # BLOCKING: frame convention + cross-checks. Must pass first.
uv run aic build-index        # build the index (once, offline)
uv run aic run queries.json   # run the query set, write submissions, package the zip
uv run aic check-submission   # final gate: read back from disk and validate
```

`queries.json` is a JSON object `{query_id: "query text"}`.

**`aic run` blocks two cases by itself:**

* A **stub** encoder in use, so it **refuses** to write submissions (retrieval results are noise).
  Exercising the pipeline requires passing `--allow-stub-submission` explicitly.
* Any remaining `error`-level issue, so it does **not** package the `.zip` and exits with status 1.

---

## 6. Pre-submission checklist

- [ ] `aic validate` passes (0 frame-convention mismatches)
- [ ] The text encoder is a **real model** in the `clip-ViT-B-32` space — not the stub
- [ ] Every query has **exactly 100 rows**
- [ ] `aic check-submission` reports **0 errors**
- [ ] File count matches query count, filenames follow the pattern confirmed with the organisers
- [ ] The six questions in §2 have been **asked and answered** — question 4 especially
- [ ] Open one file in an editor and **look at it**: no header row, no stray BOM, Vietnamese
      diacritics rendering correctly
- [ ] For TRAKE: every row has exactly `N` frames, in increasing order
- [ ] The `.zip` contains a `submission/` directory holding the CSV files
