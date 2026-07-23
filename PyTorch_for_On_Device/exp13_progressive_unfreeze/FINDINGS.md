# Exp 13 findings — progressive unfreezing: last-block+fc is a real sweet spot (S01)

**Date:** 2026-07-23
**Purpose:** Test the middle ground between head-only (fc only) and full training — unfreeze the LAST K
blocks (+ fc) and fine-tune. Question: does any K beat head-only, or does partial FT just land between
head-only and full? Frozen pretrained BN stats, streaming incremental b1→b5, S01 vocalized, 3 folds,
n_accum 4, 40 ep, 30% data, fixed order; best lr picked per K from {3e-4, 1e-3, 3e-3, 1e-2}.

## Result — a clear peak at K=1

| K (unfrozen) | trainable | best mean b2–5 | best lr | vs head-only 84.68 |
|---|---|---|---|---|
| 0 | fc only (≈ head-only) | 84.95 | 3e-3 | — |
| **1** | **last block + fc** | **86.48** | **1e-3** | **+1.80** |
| 2 | last 2 blocks + fc | 85.60 | 3e-4 | +0.92 |
| 3 | last 3 blocks + fc | 85.05 | 3e-4 | +0.37 |
| 4 | last 4 blocks + fc | 85.56 | 3e-4 | +0.88 |
| 5 | full | 85.32 | 3e-4 | +0.64 |

References: head-only 84.68 ; full frozen-stat 85.65 ; paper 88.24.

## The finding — unfreezing ONLY the last block wins
- **K=1 (last conv block + fc) = 86.48** beats head-only (+1.8), full-training (+1.2), and every other K.
  It nearly **halves the gap to the paper** (88.24): 3.6 pp → 1.8 pp.
- **Not a lucky config:** the three lower lrs for K=1 cluster tightly (86.11 / 86.48 / 86.30), so the
  gain is stable, not a single draw/lr fluke.
- **Why it works:** the last block holds the most *session-specific* high-level features; adapting just
  it captures the feature shift the classifier alone can't, while keeping the other 4 blocks frozen
  avoids the overfitting + activation↔stat mismatch that caps full-training. It threads the needle that
  head-only (too little capacity) and full (too much) both miss.
- **More blocks overfit and get fragile:** K≥2 falls back toward ~85 and the optimization becomes
  brittle — high lr collapses toward chance (K=2 @3e-3→83.9, @1e-2→27.8; K=3 @3e-3→34.7; K≥3 @1e-2→11.1).
- **lr–K trend confirms the capacity/fragility story:** best lr drops monotonically with K
  (K0 3e-3 → K1 1e-3 → K≥2 3e-4); more trainable body ⇒ more fragile ⇒ smaller usable lr. This is why a
  single global lr would have hidden the K=1 peak — it needs its own lr (1e-3).

## Deployability — and it's cheap to realize
- K=1 is deployable on-device **without any graph change** via per-parameter grad-buffer masking: after
  the backward, `memset` every grad-accumulation buffer to zero EXCEPT the last block's conv + fc, then
  run the optimizer (which applies `w -= lr*grad` per parameter, so zeroed grads = exact freeze under our
  plain-SGD-no-wd recipe). See the harness analysis (`deeploytraintest.c` grad-buffer layout /
  `run_optimizer_step`). Cost vs head-only: one extra block's backward (block 4 conv 32×32×7×1); still far
  below full training.

## Caveats before changing the deployment pick
- **S01 only, 3 folds, single stratified draw.** +1.8 pp is meaningful but within the range 3-fold noise
  can inflate; the cross-lr consistency is reassuring but not conclusive.
- **Needs confirmation:** (1) 4-subject (S01–S04) K=1 vs head-only, (2) multi-seed on S01 to check
  draw-robustness. Only if K=1 holds across subjects/seeds should it replace head-only as the deployment
  recipe. Confirmation launched (`exp13b`/4-subject).

## Takeaway
Contrary to the "partial FT just lands between head-only and full" prior, **there is a genuine sweet spot
at last-block+fc** for S01 (+1.8 pp over head-only, closest to the paper of any on-device recipe). If it
generalizes across subjects, it is the new deployment candidate — and it is cheaply deployable via
grad-buffer masking, no graph regeneration.

## Files
`run_progressive_unfreeze.py`, `results/progressive_unfreeze_S01.csv`.
