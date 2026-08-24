# Design of a multi-task video retrieval system for AIC 2026 — preliminary round

**Scope.** The three tasks Textual KIS, Q&A and TRAKE over the preliminary-round Vietnamese video
corpus (873 videos, 130.7 hours, seven YouTube channels across six genres — see `DATA_AUDIT.md`
section 2.1).

**How to read this document.** It separates two kinds of premise. The first is *invariant*: derived
from the scoring rules and from measured properties of the data, and unchanged when the execution
environment changes. The second consists of *parameters of the current deployment phase*: the compute
budget, disk capacity, installed software. The second kind is written as named parameters
(section 2.2) and assigned concrete values in exactly one place, `CONSTRAINTS.md`, with a date. Every
architectural conclusion here is stated as a function of those parameters, together with its reversal
condition — so that when a parameter value changes, it is readable which conclusions still hold
without rewriting the design.

---

## 0. Summary

The objective function of the preliminary round is not ranking but **coverage at five discrete
cut-offs** under a budget of 100 answers per query (section 1.2). This is an invariant premise, and it
determines most of the architecture: the output problem is maximisation of a submodular set function,
not sorting a list by score.

The second invariant premise comes from the data: the answer span `[s, e]` is defined on the frame
index system of the source video, while the organiser-supplied keyframes have a median gap of 55
frames (section 2.1, R1). With TRAKE answer windows under 10 frames, the available keyframes are not
sufficient to cover; the TRAKE branch requires decoding at full temporal resolution.

### The stage-allocation rule, by compute budget

Whether an analysis runs at index time or at query time is not an aesthetic choice but the result of a
quantitative condition. Let $N_s$ be the number of units stage $s$ must process if run over the whole
corpus, $c_s$ the cost per unit, and $B$ the compute budget of the preparation phase. Stage $s$ may
live at index time if and only if

$$N_s \cdot c_s \le B .$$

A stage failing that condition can still be kept, but only by reducing $N_s$: calling it on the
filtered candidate set instead of the whole corpus, which moves it from the index phase to the query
phase. On the batch 1 corpus the ratio
$N_{\text{corpus}} / N_{\text{candidates}} \approx 176\,700 / 50 \approx 3\,500$ is the margin that
move buys (section 11.2).

Applying the rule with the throughput figures of section 11.1 gives: visual encoding, OCR and speech
satisfy the condition and live at index time; semantic description with a vision-language model does
not, and moves to on-demand verification (P6). That split is a consequence of $c_s$ — the per-unit
costs differ by two orders of magnitude — and **not** a consequence of VRAM capacity: all three
retained stages need GPU inference, so device memory does not distinguish between them. The role of
VRAM is a different and weaker constraint, acting through two indirect paths; see section 2.2.

### Three architectural decisions

1. **Retrieval channels run in parallel and are fused by union, not intersection.** With $m$ series
   stages of recall $r_i$, end-to-end recall is $\prod_i r_i$; with $m$ independent parallel channels
   it is $1 - \prod_i (1 - r_i)$. The design therefore minimises series depth and maximises parallel
   width.
2. **The resident index is at coarse temporal resolution; full-resolution pixels are re-decoded on
   demand.** This is a necessary condition for solving TRAKE.
3. **The 100-slot allocator is an independent algorithmic module, not a post-processing step.** Its
   compute cost is negligible against the layers before it, while its contribution to the score is of
   the same order as model quality (section 10.2).

---

## 1. Problem specification

### 1.1. The three tasks

| | Answer format | Correctness condition | Score structure |
|---|---|---|---|
| **Textual KIS** | `⟨video_id, frame_id⟩` | $v = GT_v \wedge id \in [s,e]$ | Binary |
| **Q&A** | `⟨video_id, frame_id, answer⟩` | $v = GT_v \wedge id \in [s,e] \wedge a \equiv GT_a$ | Binary, three ANDed conditions |
| **TRAKE** | `⟨video_id, f_1, \dots, f_N⟩` | Wrong video ⇒ hard 0 | Continuous: $\frac{1}{N}\sum_j \mathbb{I}(f_j \in [s_j, e_j])$ |

The three tasks share a video retrieval layer and diverge at temporal localisation. Q&A adds an
independent failure axis (the answer content). TRAKE adds a sequence constraint and demands temporal
resolution an order of magnitude tighter than the other two: the query defines an answer window for
each semantic moment that is **usually under 10 frames**.

### 1.2. Formalising the objective

For an ordered answer list $r_1, \dots, r_{100}$ and the cut-off set $K = \{1, 5, 20, 50, 100\}$:

$$\text{Final} = \frac{1}{5}\sum_{k \in K} \max_{i \le k} R(r_i)$$

Because $R@k$ is a running maximum, with binary $R$ the final score is a **step function of the rank
of the first correct answer**:

| Rank of the first correct answer | Final Score |
|---|---|
| 1 | 1.00 |
| 2–5 | 0.80 |
| 6–20 | 0.60 |
| 21–50 | 0.40 |
| 51–100 | 0.20 |
| > 100 | 0.00 |

Taking the expectation:

$$\mathbb{E}[\text{Final}] = \frac{1}{5}\sum_{k \in K} P_k, \qquad P_k = \Pr(\exists\, i \le k : r_i \text{ correct})$$

Let $S_j$ be the first $j$ answers. The marginal weight of position $j$ is the number of cut-offs it
can still influence:

$$w(j) = \frac{|\{k \in K : k \ge j\}|}{5} = \begin{cases} 1.0 & j = 1\ 0.8 & 2 \le j \le 5\ 0.6 & 6 \le j \le 20\ 0.4 & 21 \le j \le 50\ 0.2 & 51 \le j \le 100\end{cases}$$

The gain from placing candidate $c$ at position $j$:

$$\Delta(c, j) = w(j)\cdot\Big[\Pr(\text{hit} \mid S_{j-1} \cup \{c\}) - \Pr(\text{hit} \mid S_{j-1})\Big]$$

The function $\Pr(\text{hit} \mid \cdot)$ is monotone and **submodular**, and $w(j)$ is non-increasing
in $j$. Those two properties are what permit greedy with a $(1 - 1/e)$ approximation guarantee.

### 1.3. Six design consequences of the objective

**H1 — Order within a band carries no value.** A correct answer at rank 21 and at rank 50 score
identically. Any effort spent refining rank within a band produces no points. The engineering budget
must go into **pushing an answer across a band boundary**, not nudging it up a few places.

**H2 — The problem is coverage, not ranking.** The objective is to maximise five coverage
probabilities at five cut-offs. This gives diversification a theoretical justification and yields a
concrete algorithm rather than an intuition.

**H3 — There is no trade-off between diversification and top-end precision.** Position 1 is always
chosen by argmax of probability, entirely independently of how the later positions are filled.
Diversification only takes effect from position 2 onwards.

**H4 — Always fill all 100 slots.** The last fifty slots are worth 0.2 points in total, but they cost
almost nothing to produce. That is free positive expectation.

**H5 — Q&A permits hedging along the answer axis at near-zero cost.** When confidence in
`(video, frame)` is high but the answer is uncertain (counts, colours, proper nouns), submit the same
`(video, frame)` pair with several different `answer` values in consecutive slots. Three variants
occupy three slots in the 2–5 band (which has four) and move almost all of the answer axis's
probability mass into $P_5$.

**H6 — TRAKE has a separable multiplicative structure.**

$$\mathbb{E}[R] = \Pr(v = GT_v)\cdot\mathbb{E}\big[\tfrac{1}{N}\textstyle\sum_j \mathbb{I}(f_j \in [s_j,e_j]) \,\big|\, v = GT_v\big]$$

The first factor is a gating condition. Within a *single* tuple, because $R$ is the mean of indicators
that are **independent across $j$**, the optimal tuple is the product of per-moment argmaxes. But
$\mathbb{E}[\max]$ across several tuples does **not** separate — the maximum of an average is smaller
than the average of maxima. The correct strategy: hold the high-confidence moments fixed and vary only
the highest-entropy moment across successive tuples.

### 1.4. A hierarchical probability model for coverage

A candidate is a pair $c = (v, f)$. Let $\pi_v$ be the probability that video $v$ is the answer, and
$[s,e]$ the random answer span inside $v$. For an answer set $S$, write $S_v$ for the frames of $S$
belonging to video $v$:

$$\Pr(\text{hit} \mid S) = \sum_v \pi_v \cdot \kappa_v(S_v), \qquad \kappa_v(S_v) = \Pr\big(S_v \cap [s,e] \ne \emptyset \,\big|\, v = GT_v\big)$$

This factorisation separates the two sources of uncertainty — **picking the wrong video** and
**picking the wrong frame inside the right video** — and makes it possible to allocate slots between
the two kinds of hedge quantitatively. The marginal gain of adding a frame to a video already present
in the list is bounded by $\pi_v \cdot (1 - \kappa_v)$; once $\kappa_v$ approaches 1, every further
slot must move to a different video.

---

## 2. The constraint space

### 2.1. Data constraints

**R1 — Video is the official competition data; the supplied keyframes are auxiliary.** Keyframes are
sampled at shot boundaries, with gaps of tens to hundreds of frames between consecutive ones. With
TRAKE answer windows under 10 frames, the probability that an existing keyframe lands inside the
window is close to zero. **TRAKE requires decoding video at full temporal resolution.** The supplied
CLIP features inherit the same limitation.

**R2 — The frame index convention is a systematic error source.** Source video from YouTube often has
a variable frame rate. The formula `frame_idx = round(pts × fps)` applied to VFR video accumulates
drift with duration, enough to destroy the TRAKE score even when semantic localisation is perfectly
correct. The error is silent: the system runs, the indices are valid, and the score is zero.

**R3 — There is no validation set.** The organisers do not supply queries with answers. Every number
measured during development comes from a self-built evaluation set, so the quality of that set becomes
the upper bound on the quality of every tuning decision.

**R4 — The corpus is not uniform in genre, and news is a minority.** The first version of this design
assumed a corpus of "Vietnamese news video". The `media-info` audit refutes that: the corpus is seven
channels across six genres, of which television news is 6.9 % of videos and 14.6 % of duration
(`DATA_AUDIT.md` section 2.1). The two genres dominating by duration are cooking (33.5 %) and
secondary-school exam revision (27.7 %).

The consequence for the arguments in P4 and P5: the density of identifying information in on-screen
text and in narration is higher than in purely visual content, but that has only been verified over
the news portion. Across the cooking and exam-revision portions, the text and speech characteristics
differ in kind, and the usefulness of those two channels has not been measured. The priority ordering
among retrieval channels therefore stands, but its basis is narrower than originally stated.

A second consequence: a genre prior is a high-leverage intervention, because the genre follows from
the L group with no inference at all. See `aic.index.priors`.

**R5 — The data will grow.** Batch 2 has been announced. Any absolute threshold calibrated on batch 1
becomes invalid when the corpus size changes. The fusion mechanism must be invariant to corpus scale.

**R6 — Metadata is not guaranteed complete.** Some videos may lack a metadata file. Any feature built
on metadata introduces a systematic bias against that group of videos.

### 2.2. Parameters of the deployment phase

The quantities in this section are **not** properties of the problem. They are parameters of the
execution environment, they change when the environment changes, and they are assigned values in
`CONSTRAINTS.md` with a date. This section only defines them and records which decision each one
affects, together with that decision's reversal condition.

| Parameter | Symbol | Decision depending on it | Reversal condition |
|---|---|---|---|
| Compute budget of the preparation phase | $B$ | Which stages sit at index time and which at query time (section 0) | $B \ge N \cdot c$ of the deferred stage |
| Per-unit cost of stage $s$ | $c_s$ | The same decision | A cheaper model or faster hardware |
| Available device memory | $V$ | Which model sizes are available; how many fit at once | $V$ large enough for a bigger model |
| Available disk capacity | $D$ | Store pixels or only vectors; extract or read from archives | $D$ large enough for the costlier representation |

**On $B$ and $c_s$.** These are the governing pair, because the rule in section 0 depends only on the
product $N_s c_s$ and the threshold $B$. The preliminary round is an offline submission over a finite
query set, so $B$ is the *total* budget of the preparation phase, not a per-query latency limit.

**On $V$.** Device memory does not appear in the stage-allocation rule. It acts through two indirect
paths, and those must be kept distinct from the effect of $B$:

1. *Through $c_s$.* $V$ decides which model can be loaded, and the model decides $c_s$. A 7B
   vision-language model at fp16 needs about 16 GB; at 4-bit quantisation a 2B model occupies about
   2 GB and leaves room for the kv-cache. Replacing 7B with a quantised 2B lowers $c_s$, but not
   enough to bring the semantic-description stage's $N_s c_s$ below $B$ (section 11.3), so the
   conclusion in section 0 does not change.
2. *Through concurrency.* When $V$ cannot hold the generative model and the dense index at once, the
   stages must run sequentially with explicit load and release. That is a scheduling constraint, not a
   total-cost one.

The consequence: a statement of the form "constraint $V$ rules out stage $s$" is a reasoning error,
because $V$ does not bound $N_s$. The correct statement always has the form "$N_s c_s > B$".

**On $D$.** The batch 1 corpus is about 107 GiB compressed. Any representation that stores additional
pixels competes directly with the source data itself. The dense index measures 173 MiB
(section 11.3), three orders of magnitude smaller than the corresponding pixels, so the decision to
store vectors and re-decode pixels on demand holds for every reasonable value of $D$, independently of
the current one.

**On bandwidth.** The 173 MiB dense index fits comfortably in system RAM and in device memory under
every configuration considered, so search is a direct matrix multiplication. That size has a further
useful consequence: no approximate index is needed, so the dense channel's recall is exact rather than
an approximation carrying unmeasured error. Reversal condition: when the corpus grows to the point
that the index exceeds RAM, see N6 in section 13.

### 2.3. Operational constraints

The rules impose no per-query latency limit, and the preliminary round is an offline submission over a
finite query set. The quantity to optimise is therefore quality under a total compute budget, not
per-query latency: a query taking 60 seconds is acceptable if it raises $P_1$.

That property is the precondition for the rule in section 0 to work at all. With a latency limit,
reducing $N_s$ by moving a stage to the query phase would no longer be viable, because the cost lands
exactly where the limit applies. On the available measurements: an analysis costing 2.5 seconds per
shot is 83 GPU-hours over 120,000 shots, and about 2 minutes per query over 50 candidate shots.

A consequence worth recording for later rounds: if the final round imposes an interactive latency
limit, the P6 decision must be revisited, because its premise disappears.

### 2.4. Mapping premises to architectural decisions

The third column distinguishes the two kinds of premise: a decision derived from an invariant premise
survives an environment change, one derived from a deployment parameter does not.

| Premise | Architectural decision | Kind |
|---|---|---|
| The objective is coverage at 5 cut-offs under a 100-slot budget | The output layer is submodular maximisation, not ranking | invariant |
| R1 — keyframes are sparse relative to the answer window | A two-tier index: coarse resident, full resolution on demand | invariant |
| R2 — the frame index convention is a silent error source | Validating the convention is a blocking item, done first | invariant |
| R3 — there is no validation set | The internal evaluation set is infrastructure | invariant |
| R4 — the corpus is not uniform; news is 14.6 % of duration | OCR and speech remain first-class channels, on a narrower basis; add a genre prior | measured; see N4 |
| R5 — the corpus will grow | Fuse by rank (RRF), never by an absolute score threshold | invariant |
| $N_s c_s > B$ for the semantic-description stage | Semantic description moves to on-demand verification (P6) | parameter; reverses as $B$ grows |
| $D$ against the size of the pixels | Store vectors, re-decode pixels on demand | parameter, but a three-order margin makes it durable |
| No latency limit | Permits heavy analysis over a narrow candidate set | parameter of this round |

---

## 3. Decomposition

The system decomposes into thirteen sub-problems across four layers. The index layer runs once,
offline. The query layer and the task layer run per query. The output layer is purely algorithmic.

| Code | Sub-problem | Layer | Role in the score |
|---|---|---|---|
| P1 | Frame index normalisation | Index | Blocking — an error here loses all of TRAKE |
| P2 | Temporal segmentation | Index | Fixes the granularity of every channel behind it |
| P3 | Visual representation | Index | The primary retrieval channel |
| P4 | On-screen text | Index | The highest-precision retrieval channel |
| P5 | Speech | Index | The retrieval channel for abstract semantics |
| P6 | Shot semantic description | Query | The verification channel |
| P7 | Query understanding | Query | Routes everything downstream |
| P8 | Retrieval and fusion | Query | Decides $P_{100}$ |
| P9 | Verification and reranking | Query | Decides $P_1$ |
| P10 | Frame localisation within a shot | Task | Decides the KIS score once the video is right |
| P11 | Answer generation | Task | The third failure axis of Q&A |
| P12 | Event sequence alignment | Task | Decides the TRAKE score once the video is right |
| P13 | 100-slot allocation | Output | Multiplies all the work above it |

Dependencies: P1 blocks P2, P10 and P12. P2 fixes the index unit for P3–P6. P7 distributes signal to
P8. P8 blocks P9. P9 blocks P10–P12. P13 consumes the output of all three task branches.

**The organising principle:** series stages multiply loss; parallel channels multiply survival. With
$m$ series stages of recall $r_i$, end-to-end recall is $\prod_i r_i$. With $m$ independent parallel
channels it is $1 - \prod_i (1 - r_i)$. The design must therefore **minimise series depth and maximise
parallel width**.

---

## 4. The index layer

### P1 — Frame index normalisation

**Statement.** Establish a definite correspondence between a moment in the video and the frame index
the organisers score against.

**Solution.** Compare the organiser-supplied `frame_idx` against candidate formulas applied to the
`(pts_time, fps)` they also supply, over **every** keyframe of batch 1. Acceptance condition: exactly
zero mismatches across all videos. Any non-zero mismatch must be traced to its root cause before any
other component is built.

The measured outcome settles the question: `floor(pts_time × fps)` matches 177,321 of 177,321
keyframes, `round` matches 87.07 % and `ceil` 78.19 % (`DATA_AUDIT.md` section 3.1). The convention is
`floor`, and using `round` would be off by one frame on 12.93 % of keyframes.

Validating VFR *genuinely* would need per-frame `pts` from `ffprobe -show_frames` rather than the
nominal frame rate; that remains an open risk because `ffmpeg` is absent (`CONSTRAINTS.md` R2). The
available evidence against VFR is indirect but strong: no video carries two `fps` values, and
`frac(pts × fps)` never lands near 0.5.

**Difficulty.** An error here does not surface as a runtime error. The system runs normally, produces
valid indices, and scores zero. Nothing in the development loop points at the problem unless there is
an explicit cross-check step.

**Edge cases.**
- Variable frame rate: drift accumulates, small early in the video and larger towards the end. A test
  sampling only the first few frames reports a false success.
- Dropped frames or non-monotonic timestamps: `pts` decreases mid-stream.
- Fractional frame rates (29.97 = 30000/1001): rounding accumulates.
- A container declaring a frame rate different from the actual stream.

**Propagation.** TRAKE multiplies its score by zero on the wrong video, and loses individual moments
when off by more than 10 frames. A systematic 15-frame drift in the second half of a video wipes out
the TRAKE score of every query whose event lies in that half, while every internal metric still shows
accurate semantic localisation.

### P2 — Temporal segmentation

**Statement.** Divide each video into index units such that one unit contains a complete describable
event.

**Candidate solutions.**

| Option | Advantage | Disadvantage | Cost |
|---|---|---|---|
| Fixed 10 s windows | Simple, uniform | Cuts events at arbitrary boundaries | Zero |
| Overlapping 10 s / 5 s step | Fewer cuts | Doubles the unit count and every downstream cost | 2× |
| Shot boundaries (TransNetV2) | Matches editorial structure, fewest units | A long shot may contain several events | Cheap, small model |
| Shots plus splitting long shots | Balanced | Adds one hyperparameter | Cheap |

**Decision, as built.** Shot boundaries are **derived from the organiser's own keyframes** rather than
by running TransNetV2, plus forced splitting of any shot longer than 20 seconds into 10-second pieces.
A second index over 30-second windows stepped every 15 seconds is added for queries describing events
spanning several cuts. The two indexes are parallel channels, not series stages.

The reason for the change from the original TransNetV2 plan: the rules state that keyframes are
"extracted from the video", and the measured gap distribution matches the shot-length distribution of
edited content — median 55 frames, strongly skewed, p99 = 183, max = 211. A skewed distribution rules
out uniform sampling, which would make every gap identical. Deriving shots from keyframes therefore
costs **zero** GPU hours, and every shot maps one-to-one onto an existing CLIP vector, so the dense
retrieval layer and the segmentation layer share one index space. See `aic.index.shots`.

**Difficulty.** The corpus has a high and uneven cut density. Shot-based segmentation produces a
strongly skewed length distribution: many shots under 2 seconds, some interview shots lasting minutes.
Index units of non-uniform length skew score comparison across units.

**Edge cases.**
- Cross-dissolve transitions or graphic effects: the detector produces spurious boundaries in bulk.
  Because the boundaries are inherited from the supplied data, the detection threshold cannot be tuned
  here.
- A long static shot (a presenter at a desk, or one continuous cooking step) covering several topics.
- Split screens or picture-in-picture: one unit holding two unrelated visual contents.
- The target event straddling exactly one shot boundary. The 30-second window index exists for this
  case.

**Propagation.** Segmentation granularity is the upper bound on the precision of every channel behind
it. A 60-second shot summarised as a single description loses all the detail inside it.

### P3 — Visual representation

**Statement.** Map each frame into a vector space shared with text, so that retrieval is cosine
similarity.

**Solution, as built.** Use the organiser-supplied `clip-ViT-B-32` features rather than re-encoding:
177,321 vectors of 512 dimensions in fp16, 173 MiB, one per keyframe. At this corpus size, exhaustive
search is faster *and* more accurate than approximate search; an approximate index structure only
becomes necessary past roughly 20 million vectors.

The original plan was to sample at 2 fps and encode with SigLIP or EVA-CLIP. That is still the correct
upgrade path — `ViT-B-32` is a generation behind — and the architecture allows the swap: write out a
different `DenseIndex` with the same row schema. What changed is the starting point: the supplied
features are a free baseline, so re-encoding is an improvement rather than a prerequisite.

**Three intrinsic limitations of this model family**, which must be handled at a different layer rather
than by changing model:

1. **Attribute binding.** The representation behaves close to a bag of words. The query "a person in
   red to the left of a person holding a microphone" and an image of "a person in blue to the right of
   a person holding a microphone" receive similar scores.
2. **Negation.** There is no mechanism for representing negation. The vector for "a scene without a
   hat" sits close to the vector for "a scene with a hat".
3. **Counting.** Three people and five people are not distinguished.

**A consequence of constraint $V$ (section 2.2).** The encoding model and the index are not resident in
device memory at the same time. The schedule must therefore be sequential: the index phase loads the
model and writes vectors to disk in batches; the query phase keeps the index in RAM and loads only the
text encoder onto the device. That is a scheduling constraint and does not change the total cost.

**Edge cases.**
- Dark, backlit or motion-blurred scenes: unstable vectors, noisy similarity.
- Full-screen graphics (data tables, maps): a natural-image model handles these poorly and the OCR
  channel has to carry them.
- Near-duplicate scenes recurring across videos (studio backdrops, title graphics): one query matches
  thousands of near-identical frames and floods the candidate list.
- A query describing an action with no characteristic still frame ("a person tripping"): no single
  frame carries the visual signature of the action.

**Propagation.** This is the widest-coverage retrieval channel. A failure here usually cannot be
rescued by the others unless the query carries a textual or spoken cue.

### P4 — On-screen text

**Statement.** Extract and index all text visible in the frame.

**Basis for the high precision.** In a news bulletin, the lower-third carries names, job titles, place
names, dates and figures — exactly the entity types that appear in identifying queries — and a string
match on a rare proper noun is stronger evidence than a visual similarity score. That argument has only
been verified over the news portion, 14.6 % of corpus duration (R4). Across the cooking and
exam-revision portions the on-screen text is of a different kind — dish and ingredient names, exam
questions and formulas — so it remains useful for identifying queries but by a different mechanism, and
that usefulness has not been measured. The measurement needed is stated in N4.

**Reducing $N_s$ to satisfy the budget condition.** Recognition at 1 fps over the whole corpus gives
$N_s \approx 4.7 \times 10^5$ units, over budget. Three mechanisms for reducing $N_s$, applied in
order:

1. **Deduplicate before computing.** Consecutive frames are near-identical. Filter by vector distance
   (reusing the P3 output) or by perceptual hash, with the threshold set to keep one frame per
   significant content change. The observed reduction factor is 4–8×.
2. **Detect text regions first, recognise second.** A detector is an order of magnitude lighter than a
   recogniser, and most frames contain no significant text.
3. **Recognise per shot, not per frame.** A lower-third is stable for the duration of a shot.

The recogniser must support full Vietnamese diacritics.

**Difficulty.** Recognition quality degrades sharply on scrolling text, text over moving backgrounds,
and text with drop shadows. The output contains character noise. The index must tolerate misspelling:
fuzzy matching over character n-grams rather than exact matching over tokens.

**Edge cases.**
- Dropped or wrong diacritics: "Nguyễn" becomes "Nguyên". Fuzzy matching is mandatory.
- Continuously scrolling text: one sentence is cut into fragments across frames and must be rejoined
  temporally.
- A query naming an entity that is **spoken but never shown**: this channel is silent and the speech
  channel must carry it.
- Text that is itself noise (channel logos, clocks, hashtags) appearing on every frame: needs a
  document-frequency exclusion mechanism, which is what the IDF weighting provides.

### P5 — Speech

**Statement.** Turn narration and interview speech into timestamped text.

**Role.** Speech is the only channel with access to abstract semantics, causal relations, and entities
mentioned without being shown. Queries such as "a covert recording", "a dispute", "a signing ceremony"
often have no clear visual signature and never appear in on-screen text, but they almost certainly
appear in the narration.

**Solution.** Whisper large-v3-turbo or distil-large-v3 at int8 quantisation, run in batches. Keep
word-level timestamps so text can be referred back to a shot. The assumed cost is 25–40× real time,
about 3–4 hours for 130.7 hours of video (section 11.3). Among the channels not yet built, this one has
the highest ratio of semantic coverage to $N_s c_s$, because $N_s$ is counted in hours of audio rather
than in keyframes.

**Difficulty.** Background music and location noise degrade quality. Regional accents and foreign
proper nouns are mis-transcribed. Narration frequently discusses a topic different from the footage on
screen (the b-roll phenomenon).

**Edge cases.**
- **Audio–video mismatch.** The narration discusses event A while the footage is archive material of
  event B. If speech were weighted highly in an intersection-style fusion, it would pull the system to
  the wrong position. This is why each channel must be an independent vote with no veto.
- Long stretches with no speech (music, illustrative footage): the channel is silent.
- Recogniser hallucination over silence: it emits repeated text that was never spoken. Needs filtering
  by probability and by repetition pattern.
- Consistently mis-transcribed proper nouns: handled by fuzzy matching and synonym expansion at the
  query layer.

### P6 — Shot semantic description

**Statement.** Generate a natural-language description of a shot, capturing the action and the
relations that a single-frame vector representation misses.

**The quantitative condition that decides where this module lives.** Apply the rule from section 0. A
4-bit quantised 2B model processes one shot in about 2.5 seconds, so with the $N_s = 176{,}707$ shots
measured on batch 1:

| Corpus | $N_s$ | $N_s c_s$ |
|---|---|---|
| Batch 1 (130.7 h) | 176,707 | ~123 GPU-hours |
| Extrapolated to $H = 300$ | ~400,000 | ~280 GPU-hours |

That exceeds the current $B$ by an order of magnitude (section 11.3), while the sum of every other
index-time stage is about 9.4 hours. This is the entire basis for the decision below; device memory
plays no part in the argument, because it does not bound $N_s$.

**Decision.** Semantic description moves from the index layer to on-demand verification (P9), called on
30–50 filtered candidate shots.

**The trade.** The gain has two parts. First, $N_s c_s$ drops from 123 hours to about 2 minutes per
query. Second, the model is asked a specific question directly related to the query rather than
producing a generic description, so the signal is sharper for the same compute. The loss is that
semantic description no longer serves *retrieval*: an event findable only through an action
description, which simultaneously has no static visual signature, no on-screen entity and no mention in
the narration, will never reach the candidate set. No other mechanism in the design compensates; see N1
in section 13.

**Reversal condition.** The decision reverses at $B \gtrsim 123$ GPU-hours. This task is a one-off
batch job with no interactivity, so it suits a session-based compute environment; at 30 hours per week
the budget needs about four weeks. It is a conditional enhancement, not an assumption of the baseline
design, and not a permanent limit of the architecture.

---

## 5. The query layer

### P7 — Query understanding

**Statement.** Turn a free-form Vietnamese query into the control structure that drives everything
downstream.

**Why this is the highest-blast-radius failure point.** A wrong task classification routes the query
into the wrong branch. A wrong translation breaks the main retrieval channel. Invented keywords inject
noise into the sparse channel.

**Mitigated architecturally, not by model quality.** The **raw, unprocessed** query is always retained
as an independent retrieval channel, running in parallel with the expanded one. If query understanding
mangles a query, the raw channel is still intact. The cost is one extra vector search, about 5 ms.

**Two modes.**

| Mode | Provides | Does not provide |
|---|---|---|
| `RuleParser` | Task classification, entity extraction, negation extraction, moment splitting, domain hints | Translation into English |
| LLM callable | Better translation and better TRAKE moment splitting | Determinism; it needs an external API |

**Measured on the published mock set: the missing translation was the largest single loss in the
system, and it is now routed around rather than fixed.** Embedding the raw Vietnamese with a text
tower trained on English left two of three verifiable targets unreachable — video-rank *beyond 4,000
vectors*, against rank 1 and rank 2 for an English paraphrase of the same query. Rather than adding a
translation step (an external API, a new failure mode, per-query latency), a **second text tower
distilled into the same image space** runs as its own channel (P8). Neither tower dominates: the
English one puts the panna-cotta video at video-rank 5 where the multilingual one puts it at 564, and
the multilingual one puts the charity-club video at rank 1 where the English one never finds it.
Keeping both is the same argument as keeping the raw query as its own channel.

**Moment labels are read, not guessed.** The organisers write TRAKE moments as `E1:`, `E2:` on their
own lines. The heuristic splitter looked for `(1)`, `1.` and `bước 1:`, found fewer than two moments,
and every TRAKE query in the published set fell through to the KIS branch — wrong task, wrong file,
wrong column count. Labelled moments now take precedence, and the label *values* are not trusted: one
published query is numbered `E1, E2, E2, E4` and still has four moments.

**The instruction frame is not content.** Queries arrive wrapped in phrasing shared by almost every
query ("Đoạn clip cần tìm là cảnh…", "Hãy tìm chính xác phân cảnh…"). Those words must not reach the
sparse channel, and IDF cannot remove them: descriptions in `media-info` never say "đoạn clip", so
they are *rare in the corpus* and score a **high** IDF. Corpus IDF measures rarity, not
informativeness. `QUERY_FRAME_WORDS` filters them on the query side only, never at index time.

The rule result is *always* kept as the base; the LLM may only **add** to it, never overwrite a field
the rules already know for certain. If the LLM call fails, the parser degrades to rules rather than
failing the query.

**Negation and spatial relations are never fed into the embedding step.** The vector for "a scene
without a hat" sits close to the vector for "a scene with a hat", so retrieving on a negated constraint
retrieves precisely the wrong thing. They are extracted as exclusion constraints and applied only at
the verification layer (P9).

**Edge cases.**
- A query with no visual content at all ("the bulletin of 01/08/2024"): only the metadata channel can
  answer, via `publish_date`.
- A query mixing a positive and a negative constraint in one sentence: the negation must be split off
  before embedding, not after.
- A TRAKE query where the moments are described in prose rather than enumerated: the moment splitter
  finds fewer than two moments and the branch cannot run. The service raises a `ValueError` here rather
  than guessing, because a wrong moment count is an invalid answer.

### P8 — Retrieval and fusion

**Statement.** Reduce the whole corpus to 300–500 candidate shots.

**This stage sets the ceiling on $P_{100}$.** An answer that does not reach the candidate set is lost
permanently; no later stage can recover it.

Five channels run in parallel and are fused with RRF:

| Channel | Input | Index unit | Weight |
|---|---|---|---|
| `dense_original` | Raw Vietnamese query, English text tower | keyframe | 0.6 |
| `dense_multilingual` | Raw Vietnamese query, multilingual tower in the same image space | keyframe | 1.0 |
| `sparse_title` | Pruned query terms against **titles only**, BM25 | video / shot | 1.4 |
| `sparse_text` | Pruned query terms against title + description + keywords, BM25 | video / shot | 0.8 |
| `entity_fuzzy` | Entities, character n-grams | video / shot | 0.7 |

`dense_translated` stays defined for a future translation step and is empty while P7 does not
translate. The weights are uncalibrated priors (R3): with 3 of 24 queries labelled they can be
reasoned about but not yet tuned.

**Why RRF, and why union.** RRF consumes only *ranks*, never scores, so it is invariant to each
channel's score scale shifting as the corpus grows (constraint R5) — weights tuned on batch 1 still
mean something on batch 1+2. Union guarantees no channel holds a veto: a query where the visual channel
fails completely can still be rescued by OCR, and the reverse. With $n$ channels at recall
$r_1..r_n$, the fused recall is $1 - \prod(1 - r_i)$ under independence — five channels at 0.6 each
give 0.990, whereas their intersection gives 0.08.

That arithmetic assumes independence, and the two dense channels are the case where it is weakest:
they read the same query into the same image space through different text towers. Measured, they
disagree far more than that framing suggests — each finds videos the other misses entirely (P7) —
which is what justifies paying for both rather than picking the better one.

**Two mechanisms this stage requires.**

*Deduplicate within the candidate list, not afterwards.* A query matching a studio backdrop will fill
all 500 candidate slots with near-identical frames.

*But keep every cluster member when content is rebroadcast.* The same report is replayed across several
bulletins, and episodes of one series share a set and title graphics; many videos match legitimately
but only one is the answer. Deduplicating across videos is harmful if it removes the copy that happens
to be the answer. Hence: cluster for ranking, keep every member as a cheap hedge for the lower slot
bands. The submodular coverage function will lower the marginal gain of a cluster's second member on
its own.

**A known noise source.** `media-info` is video-level text, so a BM25 hit says "this video is
relevant", not "this shot is". The score has to be spread across the shots of the video, which is
imprecise by construction. It only goes away once P4 and P5 exist at shot granularity.

**A second noise source, measured on the submitted files: topically broad videos captured the most
valuable slots.** Counted over the 24 files of the first real submission, one travel-show episode
held **slot 1 of four queries** and appeared in the top 5 of **nine**; the six worst offenders were
all episodes of the same series. Slot 1 alone is worth a fifth of a query's score, which made this
the most expensive defect in the retrieval layer. Two causes, both structural rather than
statistical:

1. *BM25 ran on the whole query* — roughly 24 terms, most of them instruction phrasing or generic
   description, which favours whichever document covers the widest range of topics. The query is now
   pruned to its 8 highest-IDF terms (`retrieval.sparse_max_terms`) after the frame words of P7 are
   removed. On the three verifiable queries this moves the correct video to BM25 rank 1.
2. *Title and description were one document.* A hit on a curated title is near-certain evidence; a
   hit inside a subscribe-link wall is close to none. Concatenated they are indistinguishable, and
   the fusion layer can only weight the mixture. Titles are now their own index and their own
   channel, weighted 1.4 against 0.8.

Effect of those two plus the multilingual tower of P7, on the videos whose identity is verifiable:
slot 26 → 3 and slot 10 → 3, with the worst magnet down from 9 of 24 top-5s to 4.

**The defect that dominated everything above: a video-level hit was expanded into the video's title
sequence.** `media-info` is video-level text, so a BM25 hit has to be turned into shots. The first
implementation took `shots.of(video_id)[:max_shots_per_video]` — the **first twelve shots**, which in
this corpus is the series intro. Every episode of a series shares it, so a hit on any episode
contributed a handful of interchangeable credit shots, and the allocator, seeing high-scoring
candidates spread across many videos, distributed slots over them. Measured on the first scored
submission:

| | Before | After |
|---|---|---|
| Submitted rows inside the first 5 s of their video | **37.0 %** (889 of 2,400) | **2.4 %** |
| Most frequent submitted frame ids | 177, 89, 94, 43, 32 | 239, 268, 296, 235, 158 |
| Correct video for `p1-20`, and its frame | rank 3, frame inside the intro → **scores 0** | rank 2, frame on the dish |

The fix follows from what each channel actually knows: a text hit says *which video*, the dense index
says *where inside it*. Shots are now ranked by the best cosine any of their keyframes reaches against
the query vector, and with no encoder the fallback is an even spread over the whole video — still
wrong, but wrong in a way that does not concentrate on the one shot every episode shares. Locked by
`test_a_video_level_hit_does_not_collapse_onto_the_first_shots`.

This is the shape of failure §8 E3 warns about: nothing raised, every internal metric looked normal,
and a third of the submission was spent on frames that could not score.

### P9 — Verification and reranking

**Statement.** From 300–500 candidates, select the 30–50 most probable shots and assign them
calibrated confidence scores.

**A two-step structure.**

Step one, cheap: rerank by cross-matching signals already available (per-channel scores, temporal
overlap between channels, the metadata prior). This reduces 500 to 50.

Step two, expensive: call a 4-bit quantised 2B vision-language model on those 50 shots. It is not asked
to produce a description; it is asked to **answer a binary verification question** constructed from the
query, together with the exclusion constraints from P7. A specific question yields a sharper signal
than a generic description at the same cost.

This is where the limitations of the vector representation — attribute binding, negation, counting —
are handled. They cannot be fixed at the retrieval layer; they are fixed here.

**Cost.** 50 shots × 2.5 seconds ≈ 2 minutes per query. Acceptable in an offline submission setting.

**Difficulty.** The output confidence must be a **calibrated probability**, not a raw score, because
P13 consumes it as a probability. A 2B model tends to answer affirmatively. Calibration by isotonic
regression on the internal evaluation set is mandatory, and its quality is bounded by the quality of
that set.

**Edge cases.**
- A small model answering "yes" to every candidate: the verification score loses all discriminative
  power. Detect it by monitoring the entropy of the score distribution.
- A query requiring an exact count: a 2B model is not trustworthy here. For those queries, count with
  an object detector and treat the model output as a secondary signal.
- A long candidate shot: the model only sees a few sampled frames and may miss the target moment.

---

## 6. The task layer

### P10 — Frame localisation within a shot (KIS)

**Statement.** Given a shot believed to contain the answer, choose which frames to submit.

**Analysis.** This is an interval covering problem, not a point estimation problem. The answer is an
interval $[s,e]$ whose position inside the shot is unknown. Submitting a single frame at the best point
estimate is dominated while slots remain free.

Let $W$ be the shot length in frames, $L$ the answer interval width, and $\Delta$ the sampling step. If
the answer position is uniform within the shot, the coverage probability with $m = W/\Delta$ evenly
spaced samples is

$$\kappa \approx \min\left(1, \frac{mL}{W}\right) = \min\left(1, \frac{L}{\Delta}\right)$$

Choosing $\Delta \le L$ guarantees coverage. For a 10-second shot at 25 fps ($W = 250$) and an $L$ of a
few seconds, a step of $\Delta = 25$ (one second) gives near-complete coverage with 10 slots.

On the measured corpus the numbers are tighter: the median shot is 55 frames, so with
`answer_span.kis = 25` a median shot needs `ceil(55/25) = 3` frames, and the longest shot (211 frames)
needs 9. One hundred slots therefore guarantee coverage of about 30 median-length shots.

**Strategy.** Slot 1 receives the frame with the highest verification score. Slots in the 6–20 and
21–50 bands receive frames spread evenly through the same shot with step $\Delta$. This converts
shot-level confidence into frame-level score, and it is nearly free in compute.

**Edge cases.**
- An unusually narrow answer interval. Covering cost grows linearly in $1/L$: with $L = 10$ and
  $W = 250$, one shot needs 25 slots.
- An event at a shot boundary: sampling must spill into the adjacent shot.
- A very short shot (under a second): one sample suffices and the spare slots move to another
  candidate.

### P11 — Answer generation (Q&A)

**Statement.** Given a localised frame, produce an answer to the accompanying question.

**Analysis.** The Q&A R-Score is a conjunction of three conditions. The answer axis is independent of
the other two and **far cheaper to hedge**. This is where consequence H5 is exploited.

**Strategy.** Generate $k$ candidate answers with probabilities from a vision-language model, read at
high resolution, with the in-frame OCR text supplied as additional context. For the most confident
`(video, frame)` pair, occupy three slots in the 2–5 band with three different answers. That moves
almost all of the answer axis's probability mass into $P_5$ at a cost of two slots in a band that has
spare capacity.

**A condition to confirm with the organisers.** The strategy assumes the scoring system accepts several
answers sharing a `(video_id, frame_id)` with different `answer` values. The rules do not forbid it, but
this is an assumption that should be confirmed first. `hedge_answers=False` provides the safe mode.

**Until the vision-language model exists, answers enter through the same interface, from a human.**
`Engine.solve(answers=[(text, prob), ...])` accepts hypotheses from any source and `aic run
--answers` reads them from a file. This is not a stopgap bolted to the side: the answer axis is one
of three conjuncts, so without it a Q&A row cannot score however good the retrieval — measured, the
three Q&A queries of the first submission were a guaranteed zero, 12.5 % of the total. All three
answers turned out to be **on-screen text** (a commune name on a banner, a couplet beside a bust, a
dish name on a recipe sheet), which is the strongest argument in this document for building P4: the
channel that would have found them automatically is the one not yet built.

**A verified row belongs at the head of the list.** `solve(pins=[(video, frame), ...])` places rows a
human has confirmed by looking at the frame ahead of everything the model proposed. The reason is
arithmetic, not preference: the one Q&A query whose answer the pipeline did find placed it at **slot
29**, keeping 0.4 of the 1.0 it had already earned. Pins carry `source="pinned"` and gain 0 in the
trace, so no report presents them as model predictions.

**Difficulty.** Semantic scoring ($a \equiv GT_a$) means the answer's form is flexible but its content
must match. Answers should be short and direct; a long answer carrying extra content risks being judged
a non-match.

**Edge cases.**
- Counting questions: hedging around the estimate ($n-1, n, n+1$) is a sensible allocation.
- Colour questions: hedge over perceptually adjacent colours.
- Proper-noun questions: the OCR channel is almost always a better source than a generative model.
- The answer may be in Vietnamese or English. Use the language of the query; do not spend slots hedging
  over language until content hedging is exhausted.

### P12 — Event sequence alignment (TRAKE)

**Statement.** Given a video and $N$ ordered moment descriptions, determine $N$ frame indices.

**The constraint that shapes the solution.** The number and content of the moments are defined by the
judges at query time. No training label set exists for these action classes. The solution is therefore
required to be **zero-shot and text-conditioned**.

**Algorithm.**

1. Decode the candidate window at full temporal resolution, extended ±15 seconds around the candidate
   shot.
2. Encode $T$ frames with the same visual encoder used in P3.
3. Build a similarity matrix $S \in \mathbb{R}^{N \times T}$ where $S_{j,t}$ is the similarity between
   moment description $j$ and frame $t$.
4. Find the sequence $t_1 < t_2 < \dots < t_N$ maximising the total similarity under a monotonicity
   constraint:

$$D_{j,t} = S_{j,t} + \max_{t' \le t - \delta} D_{j-1,t'}$$

With a running prefix maximum this recursion runs in $O(NT)$. The parameter $\delta$ is the minimum
separation between two consecutive moments.

**The monotonicity constraint is the direct answer to the repeated-action problem.** If an athlete
performs the movement ten times, a method that picks the independent maximum for each moment may pair
the "run-up" of the first repetition with the "take-off" of the fifth. Dynamic programming with an
ordering constraint and a separation penalty cannot produce that combination.

**Extension for multi-cycle repetition.** When the repetition count is large, add a preprocessing step:
build the self-similarity matrix of the window, detect the period from the maximum of the
autocorrelation, segment into cycles, run the alignment independently within each cycle, and take the
highest-scoring cycle. The runner-up cycles become hedge candidates for the lower slot bands.

**Cost.** Decoding 30 seconds at 25 fps yields 750 frames; encoding them costs roughly 2–3 seconds
under the assumptions of §11.1. The dynamic program is negligible. Total is about 3 seconds per
candidate video, so this branch's $N_s c_s$ scales with the candidate count and stays within the
query-phase budget.

**Difficulty.** There is no way to calibrate $S$ without labels. The similarity between a technical
text description ("the moment the foot leaves the ground entirely at take-off") and a frame is a weak
and noisy signal. The monotonicity constraint compensates in part by ruling out absurd combinations,
but high absolute accuracy within a 10-frame window is not something to expect.

**Edge cases.**
- A moment fully occluded (the camera cuts to another angle): no frame matches, yet the dynamic program
  is still forced to pick a point. A low-confidence flag is needed so P13 knows this is the moment to
  vary.
- Slow motion or a replay: the same action appears twice at different speeds.
- $N$ moments occurring inside one second (a very fast action sequence): $\delta$ must be small, and
  25 fps may not be enough resolution.
- A candidate video containing a similar action performed by a different person.

**Propagation.** The factor $\Pr(v = GT_v)$ dominates everything. Improving alignment accuracy from 0.6
to 0.8 while the probability of the correct video is 0.4 yields less than raising that video
probability from 0.4 to 0.6. The engineering budget for TRAKE should favour the retrieval layer.

---

## 7. The output layer

### P13 — Allocating the 100 slots

**Statement.** Given a candidate set with confidence scores, produce an ordered list of 100 answers
maximising $\mathbb{E}[\text{Final}]$.

**This is the module with the highest score-per-compute ratio in the whole system.** It needs no GPU, no
model, and its effect multiplies the quality of everything above it.

**Algorithm.** Greedy on the band-weighted coverage function, using the hierarchical model of §1.4:

```
S ← ∅
for j = 1 to 100:
    w ← band_weight(j)
    c* ← argmax over remaining candidates of
             w · [ Pr(hit | S ∪ {c}) − Pr(hit | S) ]
    S ← S ∪ {c*}
    place c* at position j
```

With $\Pr(\text{hit}\mid S) = \sum_v \pi_v \kappa_v(S_v)$ the marginal gain is available in closed form
and the whole loop runs in milliseconds. The implementation uses lazy greedy (CELF), which visits far
fewer candidates per step and — as `tests/test_allocator.py` asserts element by element — produces the
identical gain sequence.

**The tie rule that made the loop non-terminating.** Ties are the common case here, not an edge case:
two frames at least L apart cover exactly L start positions each, so their gains are equal by
construction, and the tie is broken on the candidate's own confidence so that slot 1 lands on the
anchor frame of a locus (H3). The first implementation compared a *freshly computed* gain against the
*stale* gain of the next heap entry. A candidate whose true gain was the largest by 3e-18 — inside the
tie tolerance — but whose score lost the tie-break was pushed back unchanged, returned to the top of
the heap, and popped again forever: measured at **399,645 pops of a single key** and 12.3 million
interval-union evaluations without terminating. At a budget of 70 slots the same query finished in
0.08 s, which is why the defect survived until a 100-slot run on the real query set.

The fix is the textbook CELF invariant: an entry is accepted only once its gain has been recomputed
against the **current** selection. A stale gain is an upper bound (coverage is submodular, so gains
only shrink), so a clean entry at the top of the heap is still the true greedy maximum with the same
tie-break; and every iteration either accepts or turns one stale entry clean, bounding a slot at
2·|heap| pops. All three allocators shared the defective rule and now share one implementation.

**The calibration this algorithm assumes, and what happens without it.** Greedy coverage is optimal
for $\mathbb{E}[\text{Final}]$ *given* $\pi_v$. It is not robust to $\pi_v$ being wrong in a
particular way: nearly uniform. RRF scores live in a narrow band, so normalising them linearly gave
the leading video about **6 %** of the mass, and under a posterior that flat, spreading slots across
thirty videos is the correct move — for that posterior. Measured on the eight queries of round 1
whose answer is known:

| | Rank of the correct video |
|---|---|
| in the translated dense channel | **1st on five queries, 2nd on two, 3rd on one** |
| in the fused candidate list | 1st to 15th |
| in the 100 submitted rows | 51st, 71st, or **absent** |

Retrieval was not the failure. The answer was found, fused down, and then hedged away. Two changes
fixed it, both measured on those eight queries (`aic evaluate`, real scoring function):

| Configuration | mean Final | R@100 |
|---|---|---|
| as submitted | 0.125 | 0.375 |
| translated channel weighted 6x, sparse channels halved | 0.225 | 0.625 |
| plus `pi_sharpness = 3` | **0.575** | **1.000** |

`pi_sharpness` raises $\pi_v$ to a power before normalising — monotone, so it changes no ranking,
only how much mass the leader holds and therefore how many slots the allocator commits before it
starts hedging. Pushed too far it fails the other way: at an exponent of 5 or more with the old
weights, everything went to one video and the mean fell to **0.000**. The two knobs are coupled and
neither is safe alone.

Both values are calibrated on **eight** queries. That is enough to see a factor of four and nowhere
near enough to separate 2 from 3.

**Emergent behaviour.** The algorithm needs no hand-written rules; three behaviours arise automatically
from the structure of the problem:

- Slot 1 goes to the candidate with the largest $\pi_v \kappa$.
- Once the leading video's $\kappa_v$ approaches 1 (enough frames spread to cover it), the marginal gain
  of adding another frame to that video collapses and the algorithm moves to the second-ranked video by
  itself.
- The final slots land on videos with low $\pi_v$ that have no coverage at all, because at
  $\kappa_v = 0$ the marginal gain is still the full $\pi_v$.

**The TRAKE variant.** $R$ is continuous there, so $\Pr(\text{hit})$ is replaced by
$\mathbb{E}[\max_i R_i]$, estimated by Monte Carlo. For independent moments, a new tuple's marginal gain
concentrates on the moment where it differs from the tuples already selected. That yields the
"vary the least certain moment" behaviour described in H6 without any special-casing.

**Difficulty.** The whole algorithm consumes $\pi_v$ as a true probability. If the input scores are not
calibrated the ordering is still sensible, but the switch-over points between "add another frame" and
"move to the next video" are placed wrongly. This is the module's weakest link, and it depends on the
internal evaluation set.

**Edge cases.**
- Fewer than 100 distinct candidates: fill the remainder by spreading more frames inside the videos
  already selected, or with candidates from the tail of the RRF list. Never submit short.
- Every candidate from a single video: $\pi$ collapses to one point and there is nothing to diversify.
  That is a symptom of retrieval-layer failure, not a P13 problem.
- For Q&A the candidate space is the three-way product `(video, frame, answer)`. The coverage function
  must carry an extra answer-probability factor.

---

## 8. System-level edge cases

The following cases belong to no single sub-problem; they arise from the interaction between modules.

**E1 — A micro event inside a macro video.** A two-second action inside a one-hour video. Shot
segmentation handles most of these, because a short action usually occupies a shot of its own in edited
material. The remaining case — an action in the middle of a long shot — is handled by the dense channel
at frame granularity, the only channel that passes through no aggregation bottleneck. This is why the
dense channel must keep frame granularity rather than shot granularity. Note that the measured corpus
makes this case more common than a news-heavy corpus would: cooking and exam-revision content
(§`DATA_AUDIT` §3) is edited in long static shots, with a p99 keyframe gap of 183 frames.

**E2 — All channels silent at once.** A query describing a purely visual action, with no on-screen
entity, no mention in the narration, and no distinctive still frame. The dense, OCR and ASR channels all
fail. The current design has no fourth channel to fall back on, because semantic description was pushed
to the verification layer. This is a known gap.

**E3 — Uncertainty propagates asymmetrically.** An error in P1 causes silent, total score loss. An error
in P8 causes detectable score loss (visibly wrong candidates). The test plan must invest
disproportionately in modules of the first kind — hence `aic validate` as a blocking gate.

**E4 — False consensus between channels.** Two channels wrong in the same direction make RRF more
confident than it should be. The typical case: OCR reads a person's name and ASR mentions the same name,
but both occur in a title card or opening segment rather than in the segment containing the event.
Mitigate by logging per-channel scores and manually inspecting high-consensus cases during development.

**E5 — Distribution shift when batch 2 arrives.** Batch 1 is dominated by cooking (57 % of videos) and
exam revision (27.7 % of hours); news is 6.9 % of videos. If batch 2 shifts that mix, the relative value
of the channels shifts with it: a corpus with more on-screen text raises OCR's value, one with more
free-form action raises the dense channel's. The RRF weights need recalibrating, which requires an
internal evaluation set covering both batches.

**E6 — The internal evaluation set carries the bias of whoever built it.** Queries written by the team
tend to reflect what the team knows the system can do. Mitigation: generate a query by first picking a
video segment at random and only then writing a query describing it, rather than thinking of a query and
looking for a matching segment. The order matters, and it is the one rule stated in
[`devset/README.md`](../devset/README.md).

---

## 9. Pipeline

### 9.1. The indexing phase (once, offline)

```
Video / organiser-supplied artefacts
 ├─ map-keyframes + ffprobe → frame index convention check      [P1, blocking]
 ├─ keyframe gaps → shot boundaries → split over-long shots     [P2]
 ├─ clip-features-32 (supplied) → fp16 vector index             [P3]
 ├─ objects/ detections → OCR strings → n-gram index            [P4]
 └─ (absent) ASR → time-stamped transcript → sparse index       [P5]

Products under data/processed/index/:
  vectors.f16        the dense index, fp16
  shots.json         shot boundaries and the frame mapping
  text.idx           BM25 over OCR ∪ ASR ∪ metadata
  ocr_strings.idx    character n-grams for fuzzy matching
```

Two deviations from the original plan are worth stating explicitly, because both were resolved in favour
of the supplied artefacts:

- **P2 uses the organiser's keyframes, not TransNetV2.** `aic.index.shots` derives boundaries from the
  gap distribution of `map-keyframes` at zero GPU cost. The justification is in P2: the measured gap
  distribution (median 55, p99 183, max 211) is far from uniform, so the keyframe positions already
  encode the organiser's own shot decisions.
- **P3 uses `clip-features-32`, not a self-run SigLIP pass.** The features are supplied, and §11 shows a
  self-run pass costs more than the whole query-phase budget.

The branches after P1 are independent and can be run sequentially on one GPU with explicit model
load/release.

### 9.2. The query phase (per query)

```
Vietnamese query
 └─ parse → control structure                                   [P7]
     │   frame words stripped, moments read off E-labels
     │
     ├─ dense(original, EN tower)     ─┐
     ├─ dense(original, multilingual)  │
     ├─ BM25(8 top-IDF terms, titles)  ├─ RRF → ~500 candidate shots   [P8]
     ├─ BM25(8 top-IDF terms, full)    │
     └─ fuzzy(entities)               ─┘
                              │
                    cheap rerank → 50 candidates
                              │
                    2B VLM verification + calibration           [P9]  NOT BUILT
                              │
        ┌─────────────────────┼─────────────────────┐
      KIS                   Q&A                  TRAKE
  frame spread       supplied answers      full-fps + DP
     [P10]            + hedge [P11]       keyframe-only [P12]
        └─────────────────────┼─────────────────────┘
                              │
              band-weighted greedy coverage → 100 slots         [P13]
                              │
              human-verified rows pinned to the head            [P11]
```

### 9.3. Three structural properties

**Serial depth four.** Retrieval, verification, localisation, allocation. Every serial stage multiplies
the loss; four is the minimum needed to solve the problem at all.

**Parallel width four at the recall layer.** This is the only place where the cost of an extra channel is
linear while the gain in recall is multiplicative.

**Exactly one point of heavy on-demand compute.** The whole query-time GPU budget sits in P9 and P12,
both of which run on a set already filtered by two orders of magnitude.

---

## 10. Pipeline analysis

### 10.1. The end-to-end recall budget

Let $\rho_i$ be the probability that the correct answer survives stage $i$:

$$P_{100} = \rho_{\text{index}} \cdot \rho_{\text{retrieve}} \cdot \rho_{\text{verify}} \cdot \rho_{\text{allocate}}$$

| Stage | Source of loss | Target |
|---|---|---|
| $\rho_{\text{index}}$ | The event is recorded by no channel | ≥ 0.95 |
| $\rho_{\text{retrieve}}$ | Not in the top 500 after RRF | ≥ 0.90 |
| $\rho_{\text{verify}}$ | Reranked out of the top 50 | ≥ 0.95 |
| $\rho_{\text{allocate}}$ | Not selected into the 100 slots | ≥ 0.99 |
| **Product** | | **≈ 0.80** |

The table shows where to invest. $\rho_{\text{retrieve}}$ is the weakest term and also the one most
sensitive to the number of channels. $\rho_{\text{allocate}}$ is nearly free to achieve.

### 10.2. Score sensitivity

With $\mathbb{E}[\text{Final}] = \frac{1}{5}(P_1 + P_5 + P_{20} + P_{50} + P_{100})$ the partial
derivative with respect to each $P_k$ is $1/5$. But **the engineering cost of raising each $P_k$ differs
by orders of magnitude**:

| Quantity | Cost of +0.05 |
|---|---|
| $P_{100}$ | Very low — fill every slot, diversify the tail |
| $P_{50}$, $P_{20}$ | Low — a better covering strategy |
| $P_5$ | Medium — better verification |
| $P_1$ | High — requires a better model |

The implementation order therefore runs bottom-up: exhaust the cheap bands before touching model
quality.

### 10.3. Bottleneck analysis

| Stage | Bottleneck | Governing constraint |
|---|---|---|
| Indexing | ASR and OCR | GPU throughput |
| Retrieval | none | tens of milliseconds |
| Verification | 2B VLM over 50 candidates | GPU throughput, ~2 min |
| TRAKE | full-fps decoding | disk I/O and CPU decode |
| Allocation | none | milliseconds |

Total per query: 2–4 minutes for KIS and Q&A, 4–6 minutes for TRAKE. For a query set of a few dozen, the
whole submission run fits in a few hours.

---

## 11. Cost analysis

This section distinguishes two kinds of number: **measured** on batch 1 (sourced from `DATA_AUDIT.md` or
`data/processed/reports/`), and **assumed**, taken from model documentation and not yet verified in this
environment. Only the first kind is used to reach a conclusion; the second is used solely to compare
orders of magnitude.

### 11.1. Per-unit cost $c_s$ — assumed

| Stage | Configuration | Throughput | Source |
|---|---|---|---|
| Video decoding | ffmpeg, with scaling | ~40× real time (CPU-bound) | assumed |
| Shot detection | TransNetV2 | ~100× real time | assumed |
| Visual encoding | CLIP ViT-B/32, fp16, batch 64 | ~400 images/s | assumed |
| OCR | a mobile-class model | ~25 images/s | assumed |
| Speech | Whisper turbo int8 | ~25–40× real time | assumed |
| Captioning | a 2B model, 4-bit, 8 frames | ~2.5 s/shot | assumed |

No value in this table has been measured in the current environment; see `CONSTRAINTS.md` on `torch`
currently being a CPU build. The table is only good enough to sort the stages by order of magnitude of
$c_s$, and that is all the rule in §0 needs.

### 11.2. Corpus scale — measured

| Quantity | Value | Source |
|---|---|---|
| Videos | 873 | measured |
| Keyframes | 177,321 | measured |
| Shots after P2 | 176,707 | measured |
| Duration | 130.7 hours | measured |
| Keyframe gap (median / mean / max) | 55 / 69 / 211 frames | measured |

### 11.3. Indexing cost

Applying the $c_s$ of §11.1 to the scale of §11.2, with a deduplication factor $\gamma = 0.2$ for OCR.
The $H = 130.7$ column is the current corpus; the others extrapolate for batch 2.

| Stage | $H = 130.7$ | $H = 300$ | $N_s c_s \le B$? |
|---|---|---|---|
| Decoding + shot detection | 3.3 h | 7.5 h | satisfied |
| Visual encoding | 0.65 h | 1.5 h | satisfied (and already supplied) |
| Speech | 4.4 h | 10 h | satisfied |
| OCR after deduplication | 1.0 h | 2.4 h | satisfied |
| Total over indexing-phase stages | **~9.4 h** | **~21 h** | |
| Whole-corpus captioning | **~123 h** | ~280 h | **violated** |

The last row is the decisive one. Captioning the whole corpus costs **an order of magnitude** more than
the sum of every other stage, and that is the sole basis of the P6 decision. Moving the stage to a
50-candidate set takes $N_s c_s$ from 123 hours down to roughly 2 minutes per query.

The reversal condition is quantified: the P6 decision flips when $B \gtrsim 123$ GPU-hours. Three weeks
of session-based GPU access at 30 hours a week reaches approximately that threshold, so this is a
conditional upgrade, not a permanent limit of the architecture.

### 11.4. Storage cost

| Item | Value | Source |
|---|---|---|
| Dense index, 177,321 × 512 fp16 | 173 MiB | measured |
| Text index (metadata only; no OCR/speech yet) | 1.4 MiB | measured |
| Shot table and frame mapping | 5.6 MiB | measured |
| Total derived data at present | **~181 MiB** | measured |
| Organiser keyframes, extracted | 28.66 GiB | measured |
| Source video, compressed | 77.3 GiB | measured |

The derived index is three orders of magnitude smaller than the source data. That is why the decision to
"store vectors and re-decode pixels on demand" is robust for any plausible value of $D$: the margin is
too large for a change in disk capacity to reverse it.

The measurements also expose one assumption the original design got wrong. It budgeted for extracting
frames at 2 fps — roughly 940,000 images for 130.7 hours. The organisers supply 177,321 keyframes with
CLIP features already computed, so that stage does not need to run at all; in exchange, the keyframe
density follows shot boundaries rather than a uniform time grid, and that is precisely what creates
constraint R1.

### 11.5. Query-time cost

| Item | Time | Source |
|---|---|---|
| Query understanding (rule-based) | < 0.01 s | measured |
| Three-channel retrieval + RRF + 100-slot allocation | ~0.7 s | measured |
| Vision-language verification over 50 candidates | ~120 s | assumed |
| The TRAKE branch (decode + encode + DP) | ~90 s | assumed |

The measured values come from the current pipeline over the whole batch-1 corpus, with a placeholder
rather than a real text encoder. The 0.7 seconds includes slot allocation and does not change materially
when the encoder is swapped, because the dominant term is the matrix product against the dense index.

---

## 12. Implementation plan

### Stage 0 — Blocking infrastructure

Two items must be finished before anything else.

**Frame index validation.** A script cross-checks the self-computed index against the organiser-supplied
index over every keyframe of batch 1. Acceptance condition: zero absolute deviation on 100 % of samples.
The report breaks the deviation down by relative position within the video, so cumulative drift would be
visible. Status: implemented as `aic validate`; current result 177,321/177,321 with `floor`, 0
mismatches.

**The internal evaluation set.** At minimum 40 KIS queries, 20 Q&A and 15 TRAKE with complete answers.
The procedure is mandatory: pick a video segment at random first, and only then write the query
describing it. The reverse order produces a set biased towards what the system already does well (E6).
For TRAKE the answer window must be marked by hand to better than 10-frame precision.

The evaluation set is the precondition for verifying any number in this document. Until it exists, every
hyperparameter in §2.2 and in `configs/default.json` is a *reasoned* value, not a *calibrated* one; R6 in
`CONSTRAINTS.md` records that state. `aic devset` and [`devset/README.md`](../devset/README.md) provide
the scaffolding.

### Stage 1 — The baseline

The dense and ASR channels, RRF fusion, slot allocation. No OCR, no verification, no task-specific
branches. Measure the Final Score on the internal set. This configuration is about 15 % of the total
effort and should capture most of the final score — it is the reference point every later addition is
compared against.

### Stage 2 — Full slot allocation

Implement P13 with the hierarchical coverage model, together with P10's frame spreading and P11's answer
hedging. This stage needs no GPU, so its $N_s c_s$ does not compete with any other stage for the budget
$B$. §10.2 quantifies its contribution to the score; by that analysis it comes before model improvements
in the implementation order. Status: implemented (`aic.core.allocator`, `aic.core.spread`,
`aic.tasks.qa`).

### Stage 3 — The TRAKE branch

Full-fps decoding and monotonic dynamic programming. Without this step TRAKE contributes approximately
nothing. Status: the alignment is implemented (`aic.core.align`); decoding is blocked on `ffmpeg` (C6).

### Stage 4 — OCR and verification

The OCR channel at the recall layer, the 2B VLM at the verification layer. These two items raise $P_1$
and $P_5$, the most expensive bands.

### Stage 5 — Refinement

Probability calibration by isotonic regression, RRF weights conditioned on query type, clustering of
re-broadcast videos, handling of negation and counting.

### A principle throughout

Every channel's score for every candidate is logged on every query. Without that trace a failure cannot
be attributed to a specific channel, so adjusting the channel weights has no measurement basis.
`RetrievalResult.channel_sizes` and `AllocationTrace` fill that role in the implementation.

---

## 13. Weaknesses of the design

This section lists the weaknesses of the architecture itself. Risks arising from missing information from
the organisers and from the execution environment are recorded separately in `CONSTRAINTS.md`; the two
kinds need different treatment, because only the first kind disappears when the design changes.

**N1 — No recall channel for action semantics.** Moving semantic description to the verification layer
(P6) trades coverage for budget feasibility. A query describing an action, with no static visual
signature, no on-screen entity and no mention in the narration, will not enter the candidate set; no
other mechanism in the design compensates. This is the price of the condition $N_s c_s > B$ at P6, and it
disappears when that condition reverses — unlike N2 and N5, which are intrinsic to the method.

**N2 — Probability calibration is the weakest link.** All of P13 assumes $\pi_v$ is a true probability. In
practice it comes from RRF scores and VLM scores, neither of which is a probability. Calibration requires
a sufficiently large internal evaluation set, and with 75 self-authored queries the variance of the
calibration function will be substantial. The consequence: the switch-over point between "add another
frame to the current video" and "move to another video" is placed wrongly. The estimated impact is losing
a fraction of P13's theoretical value, not all of it, because the relative ordering remains correct.

**N3 — The TRAKE branch has no way to check itself.** A zero-shot dynamic program has no signal telling
it whether it is right, other than a self-annotated evaluation set. If the similarity matrix $S$ is
systematically weak for some class of action, the design cannot detect it until a comparison against
manual labels. With 15 TRAKE queries in the internal set, the statistical resolution is very low.

**N4 — The basis for the two text channels is narrower than the corpus.** The usefulness of the OCR and
speech channels was argued from the characteristics of news broadcasts: tickers, headlines, named people,
narration naming identifiable entities. Per R4, the share of the corpus with those characteristics is
14.6 % of the duration. Over the remainder these two channels are not yet demonstrated to be useful, and
if they weaken together the system falls back to roughly the dense-only baseline.

Unlike the first draft, this is **not** a batch-2 risk but the state of batch 1. The measurement needed
to close this weakness: after building OCR and speech, split recall by L-group and compare the news
portion against the rest.

**N5 — The dense channel does not model time.** The representation is per single frame and carries no
motion information. Any query whose discriminating information lies in temporal dynamics rather than
static content is beyond this channel. A correct fix needs a video model, which changes the encoding
stage's $c_s$ by an order of magnitude; this is a weakness of the chosen representation, not of how the
stages were allocated.

**N6 — Exhaustive search has a scaling limit.** At the current scale the dense index is 173 MiB for
177,321 vectors and exhaustive search is the right choice. Linear extrapolation puts the limit at roughly
$2 \times 10^7$ vectors, about 20 GiB. Beyond that an approximate index is required, and the transition
carries a recall loss that must be re-measured — at present that loss is zero, so this is a debt not yet
incurred rather than a defect.

**N7 — The answer-hedging strategy rests on an unconfirmed assumption.** If the scoring system
deduplicates on `(video_id, frame_id)`, the three slots holding three answers collapse into one and the
strategy is void. This needs confirmation from the organisers. The cost of the assumption being wrong is
three slots in the 2–5 band — not severe — but the expected gain vanishes with it.

**N8 — Query understanding is a single point of failure.** The parallel raw-query channel mitigates a
broken translation, but it does not mitigate misclassifying the task type, because the task type
determines all downstream routing. With three task types that have clear syntactic signatures the risk
is low, but not zero.
