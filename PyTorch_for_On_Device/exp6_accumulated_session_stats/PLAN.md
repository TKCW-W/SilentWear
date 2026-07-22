# Exp 6 — accumulated-session BN stats (vs reset-per-batch)

**Date:** 2026-07-22
**Purpose:** Test whether *accumulating* the BN running stats across session-3 batches (a pooled,
growing whole-session estimate) beats *resetting and recollecting per batch* (exp5 streaming). More
data → more robust, outlier-diluted stats, and possibly a smaller gap to the paper in later batches.

## Protocol (streaming, but stats POOL across the session instead of reset-per-batch)
```
W ← base ;  session stats reset at first accumulation ;  S ← pretrained
for b = 1..5:
    zs(b) = classify batch b using (W, S)      # S = pooled stats over batches 1..b-1
    if b < 5:
        accumulate batch b into the running stats (cumulative, momentum=None)   # native PyTorch
        W ← fine-tune W on 30% of batch b (BN frozen at S)
```
Compared to exp5 (reset per batch): at batch b, exp5 uses only batch (b-1)'s stats; exp6 uses the
pooled b1..b-1. n8/3e-4, S01 vocalized, 3 folds. `run_accumulated_adabn.py`.

## Implementation note — a bug that was fixed
The first attempt accumulated per-window via forward-pre-hooks in **eval mode**. That is WRONG: in eval
mode each BN layer normalizes its input with the *stale* upstream running stats, so the deeper layers'
collected stats were computed under the wrong upstream normalization (BN0 matched `collect_bn_stats` to
1e-7, but eval accuracy diverged 87.8 → 79.4). Fix: use PyTorch's native cumulative running-stat update
— one **whole-batch train-mode** forward per batch (each layer normalized by its own fresh stats), not
reset between batches. At batch 1 this is identical to `collect_bn_stats`, so exp6 b2 matches exp5 b2;
accumulation only diverges from b3 on. (The buggy first result, 80.88, is discarded.)

## Results — S01 vocalized, 3 folds

| batch | streaming-reset (exp5) | **accumulate (exp6)** |
|---|---|---|
| 2 | 86.30 | 86.30 |
| 3 | 84.07 | 84.26 |
| 4 | 89.26 | 89.07 |
| 5 | 82.04 | **83.89** |
| **mean b2–5** | **85.42** | **85.88** |

**Accumulation helps a little: +0.46 pp overall, concentrated at batch 5 (+1.85 pp)** — exactly where
the pooled whole-session estimate (~720 windows by b5) is most robust vs a single previous batch (180).
b2 is identical (86.30) because at that point the accumulator holds only b1 (= `collect_bn_stats`),
which confirms the fix. Still below the transductive 87.78 (accumulation uses previous batches, not the
current one). So pooling the session stats is a mild, late-session improvement — worth having, small.

## Progress log
- 2026-07-22: first (buggy, eval-mode hook) run discarded; native-accumulation version rerun for S01.
