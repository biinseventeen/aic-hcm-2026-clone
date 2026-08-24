# `devset/demo` — the organisers' mock exam

The query set the organisers published for the **mock round** ("thi thử"): 24 queries, ids prefixed
`p1-`. It is not the blind-sampled internal set described in [`../README.md`](../README.md), and the
two are not interchangeable:

| | `devset/devset.json` (internal) | `devset/demo/` (mock exam) |
|---|---|---|
| Where the queries come from | sample a segment blind, *then* describe it | written by the organisers |
| What it is for | **tuning** hyperparameters (RRF weights, span `L`, calibration) | checking the **submission format** and the pipeline end to end |
| Bias | avoids weakness E6 by construction | carries exactly the bias E6 describes |

Because these queries were written first and matched to a clip afterwards, tuning against them
optimises for what the system already does well. Tuning decisions still come from the internal set.

## Files

| File | Generated | Contents |
|---|---|---|
| [`queries.md`](queries.md) | yes | The queries verbatim, plus id → task → submission filename → N |
| [`queries.json`](queries.json) | yes | `{query_id: text}` — the input to `aic run` |
| [`ground-truth.json`](ground-truth.json) | yes (merged) | Answer key in the `DevSet` schema — the input to `aic evaluate` |
| `README.md` | no | This file |

The first three come from [`scripts/build_query_set.py`](../../scripts/build_query_set.py), which
reads the organisers' 24 `.txt` files in `data/query/demo/`. The filename carries the two things the
pipeline would otherwise guess — the query id and the task — and `N` for TRAKE is the number of
`E<n>:` lines. **Do not hand-edit `queries.md` or `queries.json`**; edit the `.txt` and regenerate.
`ground-truth.json` is *merged* on regeneration, so annotated entries survive.

```bash
uv run python scripts/build_query_set.py            # regenerate from data/query
uv run python scripts/build_query_set.py --check     # exit 1 if a generated file is out of date
uv run aic run devset/demo/queries.json              # 24 .csv files + submission.zip
uv run aic check-submission                          # read the files back and validate them
uv run aic evaluate --devset devset/demo/ground-truth.json   # only once the key is annotated
```

> `data/query/demo/` is inside the git-ignored `/data/` tree, so the organisers' `.txt` files are **not**
> version-controlled. The text itself survives in `queries.json`, which is tracked.

## Filling in the answer key

`ground-truth.json` holds nothing but the `DevSet` schema — `queries`, `truths`, `skipped`. An entry
with `video_id: ""` is unannotated; right now all 24 are.

| Task | Fields to fill |
|---|---|
| KIS | `video_id`, `span = [s, e]` — the range in which **every** frame satisfies the description |
| Q&A | the above, plus `answer` and `answer_aliases` (phrasings that count as equivalent) |
| TRAKE | `video_id` and `spans` — exactly N ranges, each **under 10 frames** wide, in `E1 → EN` order |

`frame_id` is the **frame index used for scoring** (`frame_idx` in `map-keyframes`), never the
keyframe ordinal. The organisers' own baseline notebook shows `frameid` derived from the image
filename (`001.jpg` → `1`), which is the ordinal; submitting that scores zero on every row while the
file still validates. See `DATA_AUDIT.md` §3.1 and §11.

A query whose answer cannot be established belongs in `skipped` with a reason. Do not leave a
placeholder `[0, 0]`: it scores 0 and looks exactly like a system failure.

## Notes on the exam paper itself

Properties of the published set, not transcription errors — `queries.json` is verbatim:

1. **`query-p1-3` does not exist.** Ids run 1, 2, 4, 5, …, 25: 24 queries, no number 3.
2. **`p1-18` labels two moments `E2`** (water chestnut, then tofu) and has no `E3`. Four labels, so
   **N = 4**. If the organisers score it as N = 3, every 4-frame row is an invalid answer — **ask
   them before submitting**.
3. `p1-4` writes both "miến" and "miếng" măng tây, and "Khoảng khắc" in E4.
4. `p1-21` writes "vể" for "về".
5. `p1-1`, `p1-2`, `p1-10`, `p1-14` have no full stop at the end.
6. `p1-12`, `p1-13`, `p1-14` are KIS but describe **several consecutive scenes** ("Phân cảnh bắt
   đầu…", "Tiếp theo…"). One frame still has to answer them, so aim at the opening scene the query
   names and spread the remaining slots over the later scenes in case the organisers' span sits
   elsewhere.

## What this set found in the code

All fixed, with regression tests:

* `aic run` classified all three TRAKE queries as KIS. `_MOMENT_SPLIT` in `aic/query/parse.py`
  recognised `(1)`, `1.` and `bước 1:` but not the `E1:` form the organisers actually use, so the
  moment count came out below two and the query fell through to the KIS branch — the wrong task,
  the wrong file, the wrong column count. Labelled moments are now parsed directly, and the label
  *values* are ignored (see note 2 above).
* `aic check-submission` skipped every file. `_FILENAME_RE` in `aic/submit/writer.py` allowed no
  hyphen in the query id, so `query-p1-1-kis.csv` failed to match, warned, and was skipped: the
  last gate before submitting validated nothing at all.
* The archive was built **flat**, while the result specification requires the CSV files to sit
  inside a `submission/` directory and calls zipping them directly wrong. `package_submission` now
  writes that directory (`SubmissionNaming.zip_dir`).
* A Q&A answer over 100 characters was a warning; the specification makes it a hard limit, so it is
  now an error. Leading or trailing whitespace in an answer is a new warning — whitespace is
  preserved, not trimmed.

## Files, extended

Two hand-written inputs joined the set after the first real score came back (0.2 from the
organisers' system):

| File | Written by | Purpose |
|---|---|---|
| [`answers.json`](answers.json) | hand | Q&A answer hypotheses, read off the frames. `aic run --answers` |
| [`pins.json`](pins.json) | hand | `(video, frame)` rows verified by opening the keyframe. `aic run --pins` |

```bash
ANS=devset/demo/answers.json PINS=devset/demo/pins.json
HF_HUB_OFFLINE=1 uv run aic run devset/demo/queries.json --answers $ANS --pins $PINS
uv run aic check-submission
uv run aic evaluate --devset devset/demo/ground-truth.json --answers $ANS --pins $PINS
```

Both files record *why* each entry is there, down to the keyframe it was read from. Neither is a
model prediction: a pin claims only what was visible in that frame.

## What the first real score changed

The organisers scored the first submission **0.2**. Reading that number against the slot weights
(`objective.band_weight`: slot 1 is worth 1.0, slots 51–100 are worth 0.2 between them) and against
the two components that could not score at all — Q&A had a placeholder answer, TRAKE runs
keyframe-only — puts roughly 0.26 mean on the 18 KIS queries and ~0 on the other six. Four fixes
followed, in order of measured leverage.

**1. `allocate_kis` never terminated at 100 slots.** Same candidate set, budget the only variable:
70 slots finished in 0.08 s, 100 slots ran past 78 s and 12.3 million interval-union evaluations
without returning. One key, `('L26_V329', 45)`, was popped **399,645** times: its recomputed gain
was the largest by 3e-18 — inside the tie tolerance — but its score lost the tie-break, so it was
pushed back unchanged, returned to the top of the heap, and popped again forever. The lazy-greedy
loop now accepts an entry only once its gain has been recomputed against the current selection
(textbook CELF), bounding a slot at `2 * len(heap)` pops. `p1-6`: never finishes → 0.41 s.

**2. The sparse channel handed the top slots to whichever video was topically broadest.** Counted
over the 24 submitted files: `L27_V013` held slot 1 of **4** queries and appeared in the top 5 of
**9**; the six worst offenders were all L27 travel-show episodes. Two causes, both fixed:

* BM25 ran on all ~24 terms of a query, most of them instruction phrasing. Terms are now pruned to
  the 8 highest-IDF ones (`retrieval.sparse_max_terms`). Corpus IDF alone is not enough — "đoạn",
  "clip", "phân" are *rare in the corpus* precisely because they are query boilerplate, so they
  scored a high IDF; `QUERY_FRAME_WORDS` removes them on the query side only.
* Title and description were one document, so a hit on a curated title was indistinguishable from
  a hit in a subscribe-link wall. Titles are now their own index (`text_title.json`) and their own
  channel, weighted 1.4 against 0.8 for the mixed one.

**3. The dense channel read Vietnamese through an English-only text tower.** A second tower
(`clip-ViT-B-32-multilingual-v1`, same image space) now runs as an independent channel. Neither
dominates, which is why both are kept: the monolingual tower puts the panna-cotta video at
video-rank 5 where the multilingual one puts it at 564, and the multilingual tower puts the FANA
video at rank 1 where the monolingual one does not find it in 4,000 vectors.

Effect of 2 and 3 together, on the videos whose identity is verifiable:

| Query | Correct video | Before | After |
|---|---|---|---|
| `p1-20` | `L26_V004` | slot 26 | **slot 3** |
| `p1-19` | `L24_V011` (a wrong guess — see below) | slot 10 | slot 3 |
| `p1-15` | `L30_V072` | slot 1 | slot 1 |
| worst magnet in a top 5 | — | 9 of 24 queries | **4 of 24** |

**4. The three Q&A answers were read off the frames.** All three are on-screen text, so a contact
sheet of each candidate video plus a crop was enough:

| Query | Video / keyframe | Answer | Evidence |
|---|---|---|---|
| `p1-15` | `L30_V072` kf 009, 035 | **Giang Ly** | banner: "CÙNG EM ĐẾN TRƯỜNG — Xã Giang Ly, huyện Khánh Vĩnh, tỉnh Khánh Hoà — 04.08.2024" |
| `p1-19` | `L27_V010` kf 146 | **Hỏa hồng Nhật Tảo oanh thiên địa, Kiếm bạt Kiên Giang khấp quỷ thần** | the couplet beside the bust inside the temple |
| `p1-22` | `L30_V078` kf 031 | **Nhân bánh cuốn** | recipe sheet: "Nạc dăm xay 200 gr (ướp bột nêm)" |

`p1-19` is the cautionary one: the metadata candidate was `L24_V011`, a lion-dance act *named*
after the temple, and the answer is actually in `L27_V010`, an episode titled "Rạch Giá — Phố biển
cổ xưa và nay" that never mentions the temple in its metadata. A title match is evidence, not proof.

These three are now ground truth in `ground-truth.json` (3 of 24 annotated), which makes
`aic evaluate` a real measurement for the first time:

| Task | n | Final before | Final after |
|---|---|---|---|
| Q&A | 3 | 0.400, 0.000, 0.000 | **1.000, 1.000, 1.000** |

The remaining 21 entries are still placeholders, so the aggregate line `aic evaluate` prints is not
a system score — only the Q&A row is measured.

## What is still broken

* **TRAKE, 3 queries.** Keyframe-only mode: ~14.5 % coverage per moment (`DATA_AUDIT.md` §3.5),
  so E[Final] sits at 0.035–0.046. `ffmpeg` is the blocker (C5).
* **18 KIS queries have no ground truth**, so nothing about them is measured offline. The portal
  score is still the only oracle, at one number per submission.
* **No OCR (P4), no ASR (P5), no VQA (G5), and the `objects` family is read by nothing (G1).**
  Every Q&A answer above was on-screen text — exactly what OCR would have found automatically.
* **`exclude` and `spatial` constraints are extracted and then dropped**: no verification stage
  consumes them (P9).
* **The posterior is uncalibrated.** The leading video holds ~6 % of the mass on average, which is
  why the allocator spreads across ~10 videos in the top 20. Whether that is correct hedging or
  simply an unconfident model cannot be told apart without labels.
