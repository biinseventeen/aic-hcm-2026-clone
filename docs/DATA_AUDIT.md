# Input data audit — batch 1 (`aic25-b1`)

This document is the **input specification** of the system: what the organisers supply, the schema of
each family, and the figures measured on the real data. Every number below comes from reading the
whole corpus, not from an estimate.

To reproduce:

```bash
uv run python scripts/extract_data.py --verify   # file counts and byte totals
uv run aic validate                              # P1 plus cross-checks (blocking item)
```

---

## 1. Data inventory

The organisers supply six data families, 106.7 GiB compressed, under `AIC_DATA_ROOT` (the `data/raw`
link, see `scripts/link_data.py`).

| Family | Entries | Compressed | Extracted | Purpose |
|---|---|---|---|---|
| `map-keyframes` | 873 CSV | 1.5 MB | 1.5 MB | The source of truth for frame indices |
| `media-info` | 873 JSON | 1.1 MB | 1.1 MB | YouTube metadata; the text channel |
| `clip-features-32` | 873 `.npy` | 0.16 GiB | 0.17 GiB | The dense retrieval channel |
| `objects` | 177,321 JSON | 0.60 GiB | 1.68 GiB | Unused; see §7 |
| `Keyframes_L21..L30` | 177,321 JPEG | 28.69 GiB | 28.66 GiB | Verification, OCR, re-encoding |
| `Videos_L21..L30` | 873 MP4 | 77.3 GiB | — | TRAKE only; extracted on demand |

### 1.1. State on disk

The first five families are extracted into `$AIC_DATA_ROOT/extracted/` (30.5 GiB) and have been
**verified complete** by both file count and byte total. After that verification, the source archives
of those five families were deleted (`scripts/prune_zips.py`), freeing 29.4 GiB.

`Videos_*.zip` is kept compressed, for two independent reasons:

1. The rules (section 3) designate video as the official competition data, and there is currently no
   extracted copy of it. Deleting the archives would mean unrecoverable data loss.
2. Extracting all of it needs 77.3 GiB, while the TRAKE branch only ever needs one video at a time;
   the on-demand mechanism into `data/processed/video_cache/` covers that with a few GiB.

`scripts/prune_zips.py` enforces this rule: it re-runs verification immediately before deleting, and
keeps `Videos_*.zip` out of scope by default.

`aic.data.layout.DataRoot` selects its source in the order extracted-directory-then-archive, with the
same interface and the same lookup keys, so the state of the disk does not affect any layer above.
That equivalence is locked by `tests/test_layout.py`.

### 1.2. The manifest — keeping verification possible after the archives are gone

The expected entry counts and byte totals are read *from the archives*. Deleting them therefore also
removes the ability to verify again, that is, the means of detecting later corruption. To let that
ability survive, verification results are written to `extracted/.manifest.json` along with the
provenance of the numbers:

| `source` | Meaning |
|---|---|
| `archive` | Verified against the archives; evidence that the extraction is correct and complete. |
| `snapshot` | A snapshot taken after the archives were deleted; only a tripwire for later corruption. |

The current manifest carries `source: "snapshot"`, because it was written after the pruning step. It
does not prove the extraction was correct — that was proved before deletion, by comparison against
the archives — but it detects any deviation arising from now on.

```bash
uv run python scripts/extract_data.py --verify     # archives if present, else the manifest
uv run python scripts/extract_data.py --snapshot   # record a new baseline
```

---

## 2. Corpus scale

| Quantity | Value |
|---|---|
| Videos | **873** |
| Keyframes | **177,321** |
| Total duration | **130.7 hours** (from `media-info.length`; 129.8 h from `max(pts_time)`) |
| Keyframes per video | min 24 · median 163 · mean 203 · p95 463 · max 632 |
| Keyframes per hour of video | ~1,366 |

### 2.1. Composition by genre

Read from the `author` and `title` of all 873 `media-info` files. Each L group corresponds to exactly
one YouTube channel, and each channel publishes one kind of content, so the L group determines the
genre with no inference over the content.

| Group | Channel | Videos | % videos | Hours | % hours | Genre |
|---|---|---|---|---|---|---|
| L21 | 60 Giây Official | 29 | 3.3 | 8.9 | 6.8 | television news bulletin |
| L22 | 60 Giây Official | 31 | 3.6 | 10.2 | 7.8 | television news bulletin |
| L23 | HTV Sports | 25 | 2.9 | 2.7 | 2.1 | road cycling race |
| L24 | HTV Sports | 43 | 4.9 | 6.1 | 4.7 | lion dance competition |
| L25 | Báo Thanh Niên | 88 | 10.1 | 36.2 | 27.7 | exam revision lessons |
| L26 | ViVU TV | **498** | **57.0** | 43.8 | **33.5** | cooking show |
| L27 | HTV Giải Trí | 16 | 1.8 | 2.6 | 2.0 | travel programme |
| L28 | HTV Entertainment | 24 | 2.7 | 7.6 | 5.8 | documentary |
| L29 | HTV Entertainment | 23 | 2.6 | 6.8 | 5.2 | documentary |
| L30 | Báo Tuổi Trẻ | 96 | 11.0 | 5.7 | 4.4 | talk show |

Uniformity within a group is near absolute: 88 of 88 L25 videos have titles beginning "BÍ QUYẾT ÔN
THI" (exam revision), and 95 of 96 L30 videos belong to the series "Lan tỏa năng lượng tích cực".

> **The corpus is not news.** The first version of `DESIGN.md` assumed a corpus of "Vietnamese news
> video" and filed the possibility of it being otherwise as a batch 2 risk (N4). The measurements
> refute that assumption already in batch 1: television news is **6.9 % of videos and 14.6 % of
> hours**. The two genres dominating by duration are cooking (33.5 %) and exam revision (27.7 %).
>
> Consequence: the argument "OCR is the highest-precision channel because on-screen text carries
> names and job titles" holds over 14.6 % of duration. `DESIGN.md` R4 and N4 have been corrected
> against these measurements.

Three consequences that affect the score:

1. **Weighting by video and weighting by hour give different pictures.** L26 is 498 short videos
   (5.3 minutes on average); L25 is 88 long ones (24.7 minutes). Since `pi_v` and
   `max_shots_per_video` are per video, video count is the right unit for the allocation layer.
   `retrieval.max_shots_per_video = 12` exists so one video cannot fill all 100 slots.
2. **Genre is a cheap prior.** It follows from the `video_id` prefix with no inference. Implemented in
   `aic.index.priors`; the label set is locked against query understanding by `tests/test_priors.py`.
3. **TRAKE will almost certainly come from L23 and L24.** Only those two groups carry an event
   sequence whose semantic moments can be defined, in the same shape as the high-jump example in the
   rules. Because a wrong video multiplies the TRAKE score by zero, shifting probability mass towards
   those 68 videos (7.8 % of the corpus) is the highest-leverage intervention for that branch.

---

## 3. `map-keyframes/<video>.csv` — the source of truth for frames

Four columns, **with** a header row:

```
n,pts_time,fps,frame_idx
1,0.0,30.0,0
2,3.0,30.0,90
3,8.7,30.0,261
```

| Column | Meaning |
|---|---|
| `n` | Keyframe ordinal, **1-based**. It is the filename in `keyframes/<video>/<n:03d>.jpg`. |
| `pts_time` | Presentation timestamp in seconds, rounded to 4–7 decimal places. |
| `fps` | Frames per second of the video. |
| `frame_idx` | The frame index **the organisers score against**. |

### 3.1. The frame index convention — SETTLED

Three candidate formulas compared over **all 177,321 keyframes**:

| Formula | Matches | Rate |
|---|---|---|
| `floor(pts_time * fps)` | 177,321 / 177,321 | **100.00 %** |
| `round(pts_time * fps)` | 154,399 / 177,321 | 87.07 % |
| `ceil(pts_time * fps)` | 138,655 / 177,321 | 78.19 % |

**The convention is `floor`.** Using `round` is off by one frame on 12.93 % of keyframes. With TRAKE
answer windows "usually under 10 frames", a one-frame offset is real score loss, and it **never**
surfaces as a runtime error. The constant lives at `aic.core.frameidx.DEFAULT_CONVENTION` and is
locked by tests.

### 3.2. Supporting evidence: a rounded CFR grid

`frac(pts_time * fps)` **never** lands near 0.5 across the corpus — it always sits close to 0
(83.6 %) or close to 1 (12.9 %). That is the signature of a CFR grid rounded at output, and it
*proves* `floor`: the true value of `pts*fps` always lies in `[frame_idx, frame_idx + 1)`. The
`fps = 29.97` group has a mean `frac` of 0.835 because `1/29.97` is a repeating decimal — which is
exactly why `round()` is wrong on nearly that entire group.

### 3.3. fps distribution

| fps | Videos | Note |
|---|---|---|
| 25.00 | 781 | PAL standard, 89.5 % of the corpus |
| 30.00 | 61 | |
| 29.97 | 30 | NTSC drop-frame — the group `round()` fails on hardest |
| **26.44** | **1** | `L24_V044` — an anomaly, see §6 |

No video carries more than one `fps` value in `map-keyframes`, so there is **no evidence of VFR** from
the supplied data. Genuinely validating VFR needs `ffprobe -show_frames`; that is an open risk, see
`CONSTRAINTS.md` item R2.

### 3.4. Keyframe density — this sets the cost of coverage

Gaps between consecutive keyframes, in frames:

| min | p5 | median | p95 | max |
|---|---|---|---|---|
| 0 | 2 | **55** | 150 | 211 |

(`aic validate` reports mean 69.0 · median 55 · p99 183 · max 211.)

A median of 55 frames is about **2.2 seconds** at 25 fps. This is the number that sets P10's slot
budget: a median-length shot with `answer_span.kis = 25` needs `ceil(55/25) = 3` frames for
**guaranteed coverage**. The longest shot (211 frames) needs 9 slots. Put differently, 100 slots are
enough to guarantee coverage of about 30 median-length shots — that is the real budget
`aic.core.allocator` allocates.

### 3.5. A blocking consequence for TRAKE

The same density figure puts a **hard ceiling** on TRAKE. Submitting only frames that *exist as
keyframes* gives a probability of about `10 / 69 ≈ 14.5 %` of covering one moment with a 10-frame
answer window — and since the TRAKE score is the *fraction* of moments hit, that ceiling applies to
each moment independently:

```
aic validate
  -> TRAKE coverage ceiling using ONLY existing keyframes:
     ~14.5 % per moment (10-frame answer window) => full-fps decoding is mandatory
```

So **TRAKE requires decoding video at full fps**; the supplied keyframes are not sufficient. This is
why `Videos_*.zip` cannot be dropped from scope, and why the absence of `ffmpeg` is a blocking item —
see `CONSTRAINTS.md` item C5.

---

## 4. `clip-features-32/<video>.npy`

An array of shape `(n_keyframes, 512)` in `float16`, where row `i` corresponds to keyframe `n = i + 1`.
The rows are **already L2-normalised** (checked: `norm(row) = 1.0`), so the dot product *is* the
cosine and no renormalisation is needed.

Total: 177,321 × 512 × 2 bytes = **173 MiB**. The entire dense index fits in RAM; FAISS is not needed
and search is one `numpy` matrix multiplication.

> **A non-negotiable constraint.** These features come from the `clip-ViT-B-32` checkpoint. A text
> query **must** use the text tower of **that exact** checkpoint. A different model — even a stronger
> one — produces meaningless cosines, and the failure is *silent*: scores stay within `[-1, 1]`, ranks
> still exist, and the list still looks valid.
> `aic.query.text_encoder.load_text_encoder` checks the dimensionality and refuses to run on a
> mismatch.

---

## 5. `media-info/<video>.json`

YouTube metadata, ten keys:

| Key | Example / type | Used for |
|---|---|---|
| `title` | "60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024" | BM25 channel, genre prior |
| `description` | a long block, containing links | BM25 channel |
| `keywords` | `list[str]` | entity matching channel |
| `author`, `channel_id`, `channel_url` | strings | genre prior |
| `length` | `int`, seconds | cross-check against `pts_time` |
| `publish_date` | `"01/08/2024"` — **DD/MM/YYYY** | time filtering |
| `thumbnail_url`, `watch_url` | strings | manual tracing |

This is **video-level** text, not shot-level. A BM25 hit therefore says "this video is relevant", not
"this shot is". `aic.query.retrieve._text_hits_to_channel` has to spread a document-level hit down to
shot level, and that is a known noise source.

`publish_date` is DD/MM/YYYY — parsing it as MM/DD either raises or, worse, silently succeeds for days
up to 12 and is wrong beyond.

---

## 6. Recorded anomalies

| # | Anomaly | Extent | Assessment |
|---|---|---|---|
| A1 | Duplicate `frame_idx` within one video | **192/873 videos**, 1,228/177,321 keyframes (0.69 %) | Benign, but rules out one usage — see below |
| A2 | `fps = 26.44` in `L24_V044` | 1 video, 42 keyframes | A non-standard value; `floor` still matches 100 %, so no special handling |
| A3 | Group L26 is 57 % of the corpus | 498/873 videos | Affects the prior and `max_shots_per_video` |
| A4 | Non-monotonic `pts_time` | **0 videos** | None found |
| A5 | Video-set mismatch between the three metadata families | **0** — all three carry exactly the same 873 videos | None |
| A6 | The rules warn "some videos may have no metadata" | **0 missing** in batch 1 | Has not happened yet, but batch 2 might; the code must not assume metadata exists |

**A1 in detail.** The cause: two consecutive keyframes exactly `1/fps` seconds apart, which `floor`
collapses onto one index. For example `L21_V006`: `n=1` at `pts=0.0` and `n=2` at `pts=0.0333333`,
both `frame_idx = 0` at 30 fps. There are 614 such pairs.

> **Implementation consequence:** the `keyframe -> frame_idx` mapping is **not injective**. Looking up
> a keyframe by `frame_idx` is wrong; it must be looked up by `n`. The reverse direction (`frame_idx`
> for submission) remains entirely correct.

---

## 7. `objects/<video>/<nnn>.json` — data lying unused

One file per **keyframe** (177,321 files, matching the keyframe count exactly). Each holds five
**parallel lists of length 100**, with every value a **string**, including the numbers:

| Key | Example | Note |
|---|---|---|
| `detection_class_entities` | `"Lantern"`, `"Skyscraper"` | Human-readable labels, in English |
| `detection_class_names` | `"/m/01jfsr"` | Open Images MIDs |
| `detection_class_labels` | `"84"` | Class index |
| `detection_scores` | `"0.79673874"` | Descending |
| `detection_boxes` | `["0.468", "0.366", "0.636", "0.467"]` | Four values normalised to `[0, 1]` |

The rules (section 3) state the provenance outright: **Faster R-CNN pretrained on OpenImages V4**,
referring directly to the TensorFlow documentation for the output format. The box order is therefore
`[ymin, xmin, ymax, xmax]`, normalised to `[0, 1]` — **confirmed by the rules**, no longer an
assumption. It matters when using spatial relations (above, below, left, right); counting objects and
testing presence do not depend on the order.

> **The largest remaining gap in the system.** These 1.68 GiB of signal are **read by no module**:
> `grep -rn detection_ src/` returns nothing. It is the only remaining channel that needs **no GPU**,
> whereas both P4 (OCR) and P5 (speech) need one and neither is built. See `CONSTRAINTS.md` item G1.

---

## 8. `Keyframes_L*.zip` and `Videos_L*.zip`

`keyframes/<video>/<n:03d>.jpg` — the filename is the `n` from `map-keyframes` (1-based), **not**
`frame_idx`.

Two divergences between the rules' examples and the real data, recorded together because both are the
kind that makes a lookup fail without raising:

| Item | The rules (section 3) | The real data |
|---|---|---|
| Keyframe filename | `L01_V001/0000.jpg` — 4 digits, 0-based | `001.jpg` — 3 digits, 1-based |
| CLIP features | "a single `.npy` file" | one file per video |

Both are harmless once the real data is taken as the truth. The implementation consequence: filenames
are derived from the contents of `map-keyframes`, never from a formula in the rules.

The keyframes are extracted into `extracted/keyframes/`, so `DirSet.local_path()` returns a real path
and the file can be handed straight to PIL or ffmpeg with no temporary copy.

`video/<video>.mp4` — 873 files across 14 archives, with an **uneven** mapping from L group to archive
(L26 is split into `_a`..`_e`). `DataRoot.locate_video` scans lazily because the total size is large.
This is the **only** family that must still be read from `.zip`; `DataRoot.extract_video` extracts one
video into `data/processed/video_cache/` when the TRAKE branch needs it, and the caller is responsible
for deleting it afterwards.

---

## 9. Cross-checks already run

`aic validate` checks the following and **blocks** on failure:

1. **The frame convention** — mismatches must be exactly zero across all 177,321 keyframes.
2. **The video sets agree** across `map-keyframes`, `media-info` and `clip-features`.
3. **The feature row count** of each video matches that video's keyframe count.
4. **`media-info.length`** agrees with `max(pts_time)` within a reasonable tolerance.
5. The anomalies in §6 are listed in the report rather than passed over.

The machine-readable report is written to `data/processed/reports/validate.json`.

---

## 10. Data scope according to the rules

Section 3 of the rules contains two sentences that determine scope:

> "Dữ liệu thi chính thức là **Video**; các thành phần còn lại (Keyframes, Objects, CLIP features,
> Metadata) chỉ nhằm mục đích cung cấp thêm thông tin hoặc hỗ trợ xây dựng giải pháp mẫu."
>
> ("The official competition data is the **Video**; the remaining components — Keyframes, Objects,
> CLIP features, Metadata — are provided only as additional information or to help build a sample
> solution.")

> "Đây cũng là dữ liệu **batch 1** của AIC 2025. Dữ liệu đầy đủ của vòng sơ tuyển AIC 2026 sẽ bao gồm
> thêm **batch 2**."
>
> ("This is also the **batch 1** data of AIC 2025. The full preliminary-round data for AIC 2026 will
> additionally include **batch 2**.")

Three consequences:

1. **The video is the truth.** The answer span `[s, e]` is defined on the frame index system of the
   source video. `map-keyframes` is *one way* of reading that system, and §3.1 confirms it is 100 %
   consistent. But everything derived is a convenience, not a specification.
2. **The supplied features and keyframes are not a guarantee.** Batch 2 may be packaged differently,
   or be missing pieces. `aic build-index` must be able to rebuild from scratch, and `DataRoot` must
   report a clear error when a family is absent — `tests/test_layout.py` locks that property.
3. **The corpus will grow.** 130.7 hours is batch 1. Every cost estimate in `DESIGN.md` section 11
   scales when batch 2 arrives, and `objects` (§7) grows in proportion.
