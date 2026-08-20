# Constraints, blocking items and open risks

*Updated: 2026-08-20. Corpus: batch 1 (`aic25-b1`).*

This document assigns concrete values to the parameters defined in `DESIGN.md` section 2.2, plus a
register of what is not built and what is not known. `DESIGN.md` states architectural conclusions as
functions of those parameters; this document supplies the arguments, with a date, and is the **only**
place holding environment-dependent values.

Three kinds of entry, distinguished because they are handled differently:

* **C — Constraint.** A condition the architecture must satisfy. Each is classified further as
  *invariant* (derived from the rules or from a measured property of the data; it does not go away)
  or *phase parameter* (a property of the current execution environment; it goes away when the
  environment changes). That classification decides whether a constraint should be designed around
  or designed out.
* **G — Gap.** A component with a place in the design that is not built yet.
* **R — Open risk.** Something unknown, needing either information from the organisers or a
  measurement that cannot currently be taken.

State at this update: `aic validate` passes (0 mismatches over 177,321 keyframes), 309 tests pass.

| Item | Content | Kind |
|---|---|---|
| C1 | 100-answer ceiling; the score is a step function | invariant |
| C2 | The answer is an interval, so the task is coverage | invariant |
| C3 | The answer-span length `L` is not published | invariant (missing information) |
| C4 | No text encoder in the shared embedding space | phase parameter |
| C5 | No `ffmpeg`; TRAKE needs full fps | phase parameter / invariant |
| C6 | Compute budget `B` and device memory `V` | phase parameter |
| C7 | Disk capacity `D` | phase parameter |
| C8 | Video is the official competition data; batch 2 is coming | invariant |
| C9 | The dense index fits in RAM | invariant at the current scale |

---

## C — Constraints

### C1. The 100-answer ceiling, and the score is a step function

The rules (section 2): at most **100** answers per query, scored as
`FinalScore = (1/5) · Σ_{k ∈ {1,5,20,50,100}} R@k`.

The non-obvious consequence: the marginal value of a slot is **not** uniform. The marginal weight of
position `j` is `w(j) = |{k ∈ K : k ≥ j}| / 5`:

| Slot band | Slots | `w(j)` | Max contribution |
|---|---|---|---|
| 1 | 1 | 1.0 | +1.0 |
| 2–5 | 4 | 0.8 | −0.2 |
| 6–20 | 15 | 0.6 | −0.2 |
| 21–50 | 30 | 0.4 | −0.2 |
| 51–100 | 50 | 0.2 | −0.2 |

Slot 1 alone is worth as much as the last 50 combined. But those last 50 are still worth **0.2
points** and cost almost nothing to produce, so a submission should **always carry 100 rows**.
Leaving them empty forfeits positive expectation. This is enforced: `validate()` warns below 100.

The executable specification is `aic.core.objective`. No other module may redefine a score.

### C2. The answer is an *interval*, so the task is coverage rather than ranking

The KIS R-Score is `I(right video ∧ frame_id ∈ [s, e])`. We do not know `[s, e]`, only that it is an
interval. Submitting one "best" frame per candidate shot is **dominated** while slots remain free:
for a locus of width `W` and answer length `L`, the minimum for **guaranteed coverage** is
`ceil(W / L)` frames, and the optimal sampling step is exactly `L`.

This is why the output layer is submodular maximisation (`aic.core.coverage` plus
`aic.core.allocator`) rather than a ranker.

### C3. `L`, the answer-span length, is not published

The only available evidence is the examples in the rules:

| Task | Example in the rules | Implied `L` |
|---|---|---|
| KIS | frames 500..510 | 11 frames |
| Q&A | frames 800..900 | 101 frames |
| TRAKE | "usually under 10" | < 10 frames |

The safe direction of error is to **underestimate**: assuming `L` smaller than reality spreads
frames more densely than needed, which only costs slots; assuming `L` larger than reality spreads too
thinly and **misses** the answer span. The defaults are therefore deliberately low:
`answer_span = {kis: 25, qa: 25, trake: 10}` in `configs/default.json`.

This is the **most sensitive** hyperparameter in the system. See R3.

### C4. The text encoder must share the embedding space — and is **not installed**

The organiser-supplied image features come from `clip-ViT-B-32`. The text tower must be the text
tower of **that exact** checkpoint. A different model produces meaningless cosine similarities, and
the failure is **silent**.

> **State: BLOCKING.** `sentence-transformers` is not part of the default install. Without it the
> dense retrieval channel is silent and every query scores zero.
>
> ```bash
> uv sync --extra encoder
> ```
>
> `load_text_encoder` tries three routes (`sentence-transformers` -> `open_clip` ->
> `transformers`) and checks the dimensionality; `StubTextEncoder` runs only when explicitly
> requested with `allow_stub=True` and must **never** produce a submission.

### C5. TRAKE needs full fps (invariant); `ffmpeg` is absent (phase parameter)

Measured on the real data (`DATA_AUDIT.md` §3.5): the mean gap between consecutive keyframes is 69
frames, so submitting only existing keyframes gives roughly a **14.5 %** chance of covering one TRAKE
moment with a 10-frame answer window. The TRAKE score is the *fraction* of moments hit, so this
ceiling applies to each moment independently.

> **State: BLOCKS the TRAKE branch.** `ffmpeg` and `ffprobe` are not on PATH. `aic.tasks.trake` still
> runs in a degraded, keyframe-only mode so the pipeline can be developed, but that mode is **not for
> submissions**.

### C6. Compute budget `B` and device memory `V`

The hardware available is an RTX 4060 Laptop, `V` = 8 GiB. But the resolved `torch` on this platform
is a CPU build, so `torch.cuda.is_available()` returns `False` and the effective `B` is a **CPU
budget**. Every stage requiring GPU inference (text encoding, OCR, speech, semantic description)
currently runs on the CPU or does not run at all.

Installing a CUDA build means overriding the index for torch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

The two parameters do different work, and `DESIGN.md` section 2.2 keeps them apart:

* `B` enters the stage-allocation rule `N_s · c_s ≤ B` directly. It is the parameter that decides
  where P6 lives.
* `V` does **not** enter that rule. It decides which model sizes are available (a 4-bit quantised 2B
  model instead of a 7B at fp16) and forces stages to run sequentially. At `V` = 8 GiB, replacing 7B
  with 2B lowers `c_s`, but not enough to bring the semantic-description stage's `N_s · c_s` below
  `B`; see `DESIGN.md` section 11.3.

The reversal condition for P6 is quantified: `B ≳ 123` GPU-hours.

**Environment location.** The project environment lives inside the workspace at `.venv/`, created by
`uv sync`, and `uv.lock` pins every version. Nothing is installed into the machine's Python: a
package installed there is invisible to the lock file, so it works on one machine and fails on the
next. Run everything through `uv run` or `.venv/Scripts/python`.

### C7. Disk capacity `D`

| Item | GiB |
|---|---|
| Extracted and verified complete (5 families) | 30.5 |
| `Videos_*.zip` still held compressed | 77.3 |
| Source archives deleted after verification | −29.4 |
| Free | **100.2** |

The applied procedure: extract, verify by both file count and byte total, then delete the source
archives of the families that verified complete (`scripts/prune_zips.py`). `Videos_*.zip` is out of
scope for deletion because it is the official competition data and has no extracted copy; the TRAKE
branch uses on-demand extraction into `data/processed/video_cache/`.

`scripts/extract_data.py` checks free space before running and stops rather than filling the disk
(`HEADROOM_GIB` = 12).

The architecture does not depend on the value of `D`: `DataRoot` reads from the extracted directory
when present and from the archives otherwise, with the same interface and the same key set.
`tests/test_layout.py` locks that equivalence.

### C8. Video is the official competition data; everything else is a convenience

The rules, section 3:

> "Dữ liệu thi chính thức là Video; các thành phần còn lại (Keyframes, Objects, CLIP features,
> Metadata) chỉ nhằm mục đích cung cấp thêm thông tin hoặc hỗ trợ xây dựng giải pháp mẫu."
>
> ("The official competition data is the Video; the remaining components — Keyframes, Objects, CLIP
> features, Metadata — are provided only as additional information or to help build a sample
> solution.")

The architectural consequence: the supplied keyframes and CLIP features are a **convenient input, not
a contract**. `aic build-index` must be able to rebuild the whole index from video if needed, and
`DataRoot` must fail loudly when a family is absent rather than silently running with a channel
missing.

The same section announces that **batch 2** will follow, and that "some videos may have no metadata
file". Batch 1 has metadata for all 873, but the code must not rely on that.

### C9. The dense index fits in RAM, so no ANN structure is needed

177,321 × 512 × 2 bytes = **173 MiB** of fp16. One `numpy` matrix multiplication suffices; FAISS need
not be installed. This is a *favourable* constraint: it removes all ANN approximation error from the
system, so the dense channel's `recall@k` is exact rather than an unmeasured approximation.

The figure grows with batch 2 (C8); even at ~350 GiB of raw data it stays under 1 GiB of features.

---

## G — Known gaps

### G1. `objects/` — 1.68 GiB of signal nothing reads

**Highest priority.** The organisers supply object detections for **every** keyframe (177,321 files,
100 detections each, Open Images labels). `grep -rn detection_ src/` returns nothing — no module uses
it.

The priority follows from the budget: this is the only remaining channel whose `N_s · c_s` is zero —
the data is precomputed and only needs indexing, with no GPU inference at all. G2 and G3 have higher
semantic coverage but both need a `B` that does not currently exist. This channel serves both
retrieval (queries naming countable objects) and verification (eliminating candidates that do not
contain an object the query names).

`TextIndex` already has an extensible `Source` axis, so the channel fits without an architectural
change.

### G2. P4 — on-screen text (OCR): not built

The channel with the highest precision over the news portion of the corpus: inserted titles, tickers,
speaker names. `aic.index.text` already has `Source = "ocr"` and the fuzzy character n-gram layer
(needed because OCR drops diacritics: "Nguyễn" becomes "Nguyên" or "Nguyen"). What is missing is
running OCR over 177,321 keyframes. Needs GPU (C6).

That portion is 14.6 % of duration (`DATA_AUDIT.md` §2.1), so usefulness across the whole corpus is
not yet measured — see N4 in `DESIGN.md`.

### G3. P5 — speech: not built

A retrieval channel for abstract semantics that images do not carry, over 130.7 hours of audio. Needs
budget `B` (C6). Its cost unit is hours of audio rather than keyframe count, so this channel's
`N_s · c_s` is lower than OCR's despite higher semantic coverage.

`aic build-index` prints an explicit warning while these two are missing:

```
[3/3] sparse text index (P4/P5 — currently media-info only)
  [!] no OCR and no speech channel yet. Both text retrieval channels are missing.
```

### G4. The text channel is video-level, not shot-level

`media-info` is YouTube metadata for the **whole video**. A BM25 hit says "this video is relevant",
not "this shot is". `retrieve._text_hits_to_channel` has to spread a document-level hit across shots,
which is a known noise source, and it only goes away once G2 and G3 exist.

### G5. Semantic verification (P9) and answer generation (P11) have no real model

`aic.tasks.qa` generates answer candidates, but `answer_prob` is not calibrated by any model. The
answer score is therefore zero and the third Q&A failure axis is wide open. Needs GPU.

---

## R — Open risks

### R1. The submission format is NOT confirmed

The rules specify the *content* of an answer but **not** the filename, file format or packaging.
Every assumption is in one place (`SubmissionNaming`) and changeable in a single line. **This must be
asked before submitting.** Details and the list of questions: `SUBMISSION.md`.

### R2. VFR has not genuinely been ruled out

No video in `map-keyframes` carries two `fps` values, and `frac(pts·fps)` never lands near 0.5 — both
are evidence of CFR. But that is evidence *computed from the organiser's own derived data*, not from
the video stream. Genuinely ruling out VFR needs `ffprobe -show_frames` per video, comparing per-frame
`pts` against the organiser's indices. Currently impossible because of C5.

If any video is VFR, `frame_idx = floor(pts · fps)` is wrong on exactly those videos, and wrong
silently. TRAKE is affected worst.

An earlier version of this measurement was **removed** as conceptually wrong: it fitted a straight
line to `pts_time` against *keyframe index*, but keyframes are sampled at shot boundaries rather than
at even time intervals, so the residual measured editing irregularity rather than VFR. It raised a
false alarm on 873/873 videos. `grid_alignment` replaces it.

### R3. The true `L` for KIS and Q&A

See C3. The current guess is 25 frames for both, while the rules' examples suggest 11 (KIS) and 101
(Q&A). If the true Q&A `L` is around 101, the current setting spreads four times denser than needed
and **burns slots**: 4 slots per locus where 1 would do, which costs three quarters of the loci that
could have been covered.

This is a question worth asking the organisers. Failing that, it is a hyperparameter to calibrate on
the internal evaluation set (`aic devset` plus `aic evaluate`).

### R4. How "matches semantically" is decided for Q&A

The rules say the answer must match the ground truth semantically, but not by what mechanism: exact
string match, match after normalisation, or a semantic judge. This decides how answers should be
generated: under exact matching an answer must be short and in canonical form; under a semantic judge
there is more latitude in phrasing.

`validate()` currently warns when an answer exceeds 200 characters, on the assumption that a long
answer is more likely to be judged a non-match.

### R5. ~~Coordinate order in `detection_boxes`~~ — CLOSED

The rules, section 3, state it outright: `objects` is the output of **Faster R-CNN pretrained on
OpenImages V4**, referring to the TensorFlow documentation for the format. The order is
`[ymin, xmin, ymax, xmax]`, normalised to `[0, 1]`. No longer a risk.

### R6. There is no labelled internal evaluation set yet

`aic devset` samples blind and `aic evaluate` can score, but no query has ground truth yet. Without
it, every hyperparameter (`L`, `rrf_eta`, channel weights, `max_shots_per_video`) is a *reasoned*
value rather than a *calibrated* one. This is a process risk rather than a code one, and it blocks
answering R3 empirically. See `devset/README.md`.

---

## Order of work, by value over cost

| # | Task | What it unblocks | Cost |
|---|---|---|---|
| 1 | `uv sync --extra encoder` (C4) | **Every** query | Minutes |
| 2 | Ask the organisers about the format (R1) and `L` (R3) | Correctness of the submission | One email |
| 3 | The `objects` channel (G1) | Retrieval + verification, no GPU needed | Hours |
| 4 | A CUDA `torch` build (C6) plus `ffmpeg` (C5) | TRAKE, OCR, speech | Hours |
| 5 | A labelled internal evaluation set (R6) | Calibrating every hyperparameter | Days |
| 6 | OCR (G2) then speech (G3) | The two highest-coverage text channels | Days, needs `B` |
