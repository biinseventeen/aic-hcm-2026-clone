# Internal evaluation set

## What this directory is for

The organisers supply **no queries with answers** (constraint R3 in
[`../docs/CONSTRAINTS.md`](../docs/CONSTRAINTS.md)). Without labelled data, every number the
system reports is unverifiable and every hyperparameter is a guess: the RRF channel weights, the
assumed answer-span length `L`, the probability calibration. This directory holds the queries and
ground truth the team writes itself, and it is the only thing that turns a tuning change from a
belief into a measurement.

## Why it sits outside `data/`

All of `data/` is git-ignored: `data/raw` is a link to a 107 GiB corpus and `data/processed` is
reproducible output of `aic build-index`. This set is neither. It is **hand-annotated content** —
as much authored work as the source code — and losing it means redoing hours of frame-accurate
labelling. It therefore lives in version-controlled space.

## How it gets filled

```bash
aic devset                 # blind-samples video segments and writes the template here
# ... a human annotates devset.json ...
aic evaluate               # scores the pipeline against it
```

`aic devset` writes two files here:

| File | Purpose |
|---|---|
| `devset.json` | The template to fill in: one entry per sampled segment, plus ground truth. |
| `annotation_instructions.txt` | Per-target instructions, printed so the order cannot be mistaken. |

## The one rule that matters

**Watch the video segment first, then write the query.** Never the reverse.

Thinking of a query and then hunting for a matching segment produces a set biased towards what the
system already does well (weakness E6 in the design). The internal score then rises while the
competition score does not move. `aic devset` enforces the correct order by sampling blind, before
any human sees the content.

Minimum targets: 40 KIS, 20 Q&A, 15 TRAKE. For TRAKE, each moment needs a frame range annotated to
**under 10 frames** of precision — this is the slow part and it cannot be skipped.

## Reading the numbers honestly

With 15 TRAKE queries, one query is 6.7 percentage points. A small difference between two
configurations is noise, not a result. `aic evaluate` prints this warning itself rather than letting
the number be over-trusted.
