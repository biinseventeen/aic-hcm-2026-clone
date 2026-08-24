# `devset/phase1` — round 1, the set being scored

25 queries: **20 KIS, 4 Q&A, 1 TRAKE**. Generated from `data/query/phase1/*.txt` by
[`scripts/build_query_set.py`](../../scripts/build_query_set.py); the contract for every file here is
in [`docs/PIPELINE_IO.md`](../../docs/PIPELINE_IO.md).

```bash
HF_HUB_OFFLINE=1 uv run aic run devset/phase1/queries.json --out phase1
uv run aic check-submission --dir data/processed/submissions/phase1
```

## State

| | |
|---|---|
| Submission | 25 files, 2,500 rows, **0 errors / 0 warnings** |
| Verified by hand | **8 of 25**, pinned at slot 1 (`pins.json`, evidence per entry) |
| English renderings | 25 of 25 (`english.json`) — feeds `dense_translated` |
| Ground truth | 8 of 25 annotated, so `aic evaluate` measures those eight |
| Q&A answers | none of the four supplied yet: all score zero on the answer axis |
| TRAKE | `p1-16`, N = 3, keyframe-only; `L24_V030` is the likely video (golden dragons kf 057, gong kf 071) but pinning does not support TRAKE rows yet |

Scored by the organisers: **3.2**, which is `0.3200 x 10` — exactly the offline mean. The portal
scale is Final Score x 10, so the eight pins accounted for the entire score and the other seventeen
queries contributed nothing.

## What the eight labelled queries taught

Run without pins, the pipeline scored **0.125** on them while the translated dense channel ranked the
correct video 1st-3rd on **all eight**. The answer was being found and then discarded by the
allocator under a nearly uniform posterior. Re-weighting the channels and sharpening the posterior
took the same eight from 0.125 to **0.575**, with R@100 going from 0.375 to **1.000** — the correct
answer is now somewhere in every one of those lists. Details in `DESIGN.md` P13.

That change applies to all 25 queries, so the seventeen unverified ones now have a real chance where
before they had almost none. How much it is worth cannot be measured without labelling them: the
eight that were labelled are the eight I could identify by eye, which is exactly the population where
retrieval is easiest.

## Notes on the paper

1. **`p1-8` and `p1-14` are the same query, word for word** — the chef arranging bar-shaped and
   flower-cut pieces on a steaming plate. Two ids, one answer; both files must be filled.
2. `p1-16` writes its moments as `E1 ` with no colon, where the mock set wrote `E1:`. Both forms are
   parsed; the label count still fixes N (= 3 here).
3. Several queries name things no metadata in this corpus mentions — London Zoo, a gemstone mine, an
   earthquake map, a named mountain pass. Those are items *inside* the daily `60 Giây` bulletins
   (L21/L22), whose titles carry only a date, so the sparse channels cannot reach them and the whole
   burden falls on the dense channel.
4. Four queries are lecture slides (`p1-22`, `p1-23`) or school-stage scenes (`p1-25`) — L25/L30
   material where the discriminating detail is **on-screen text**, the channel that does not exist
   yet (G2).

## What would move the score

The mock round showed the ordering. Per query verified by hand and pinned, the score gains
`1/25 = 4 %`; the four Q&A queries gain nothing at all until an answer is supplied, however good the
retrieval. `answers.json` and `pins.json` do not exist here yet — create them the way
[`../demo/`](../demo/README.md) does, with the evidence recorded next to each entry.
