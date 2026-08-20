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

## 2. What the rules do **not** say, and what we assume

The rules specify the *content* of an answer but **not** the filename, the file format, or the
packaging. Every assumption lives in **one place**: `aic.submit.writer.SubmissionNaming`, changeable
in a single line without touching code.

| Assumption | Default | Basis |
|---|---|---|
| One file per query | yes | Convention of previous AIC seasons |
| Filename | `query-<id>-<task>.csv` | Convention of previous AIC seasons |
| Header row | **no** | The scoring system reads by position; a header is easily counted as row 1 |
| Delimiter | `,` | The rules write answers with commas |
| Line terminator | `\r\n` | Safe for every CSV reader |
| Encoding | UTF-8 | Q&A permits Vietnamese |
| `video_id` | **without** `.mp4` | Matches the keyframe directory names and metadata filenames |
| Packaging | one **flat** `.zip` | If the scoring system expects a subdirectory it will usually still find the files; the reverse does not hold |

> Dropping the extension is a reasoned choice: the rules write `video_abc(.mp4)`, meaning the
> extension is optional, and **every** other identifier in the data (keyframe directories,
> `media-info` filenames, `clip-features` filenames) appears without it. If the organisers require
> the extension: `SubmissionNaming(video_extension=".mp4")`.

### Questions to ask the organisers before submitting

1. What is the submission file format and naming? One file per query, or one combined file?
2. Is there a header row?
3. Does `video_id` need the `.mp4` extension?
4. **Q&A: may several rows share a `(video_id, frame_id)` with different `answer` values?** This is
   the most consequential question for strategy. See §4.
5. Roughly how many frames wide is the answer span `[s, e]` for KIS and Q&A?
   (`CONSTRAINTS.md` R3: this is the most sensitive hyperparameter in the system.)
6. By what mechanism is "matches semantically" decided for Q&A? (`CONSTRAINTS.md` R4)

Generate a sample set to attach to those questions:

```bash
uv run aic submit-selftest
```

It prints three complete sample files (KIS, Q&A, TRAKE) along with the list of assumptions, into
`data/processed/submissions/_selftest/`.

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
| Wrong TRAKE moment count `N` | An invalid answer |
| Two exactly identical rows | One slot wasted for nothing |

### Warnings — must be read

| Warning | Meaning |
|---|---|
| Fewer than 100 rows | The last 50 slots are still worth 0.2 points and cost almost nothing. **Always fill them.** |
| `frame_id` past the last known frame of the video | Possibly a nonexistent index |
| `answer` longer than 200 characters | Long answers are easily judged a non-match (R4) |
| TRAKE frames not increasing | Moments of an event sequence are normally in temporal order |
| Filename off the pattern | A naming assumption that needs confirmation |

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
- [ ] The `.zip` is flat, with no nested directory
