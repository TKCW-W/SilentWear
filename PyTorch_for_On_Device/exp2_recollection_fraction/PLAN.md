# Exp 2 — how many windows are enough to recollect the BN running stats?

**Status:** IN PROGRESS · started 2026-07-22 · S01 (clean) + S02 (noisy) vocalized · 3 folds

## Question
AdaBN currently recollects BN stats over the whole rest-downsampled batch (180 windows). What is the
**smallest collection size K** whose stats are as good as K=180, and **robust to outliers**? Decide
the most efficient K for the on-device stat-collection pass.

## Method
Recollect-only probe (no fine-tuning, isolates stat quality): for each batch, collect BN stats on K
randomly-drawn windows at the base weights, evaluate balanced accuracy on the whole batch. Sweep
K ∈ {2,5,10,16,32,64,90,180}, **8 random draws (seeds)** per K → mean, std, and **min-draw**
(worst-case = outlier sensitivity). S01 (clean) and S02 (noisy). K=180 reproduces the existing
recollect-only number. `frac_recollect.py`.

## Results — recollect-only balanced acc, mean(b2–5) over 8 random draws (`results/frac_summary.csv`)

**S01 (clean):**

| K (windows) | mean | draw-to-draw std | worst draw |
|---|---|---|---|
| 2 | 59.88 | 5.60 | 51.39 |
| 5 | 75.42 | 4.08 | 66.06 |
| 10 | 80.75 | 1.48 | 78.38 |
| 16 | 83.05 | 0.96 | 81.85 |
| **32** | **83.96** | **0.74** | 83.06 |
| 64 | 84.93 | 0.67 | 84.12 |
| 90 | 84.96 | 0.55 | 84.12 |
| 180 (full) | 85.19 | — | 85.19 |

**S02 (noisy):**

| K | mean | draw std | worst draw |
|---|---|---|---|
| 2 | 45.64 | 3.09 | 41.25 |
| 5 | 55.87 | 3.02 | 49.77 |
| 10 | 59.39 | 1.39 | 57.13 |
| 16 | 61.52 | 0.76 | 60.05 |
| **32** | **62.49** | **0.65** | 61.53 |
| 64 | 63.40 | 0.48 | 62.41 |
| 180 (full) | 63.66 | — | 63.66 |

## Decision — K ≈ 32 windows

**K ≈ 32 is the efficiency sweet spot** on both the clean (S01) and noisy (S02) subjects:
- reaches **within ~1.2 pp** of the full-batch (K=180) mean on both (S01 83.96 vs 85.19; S02 62.49 vs 63.66),
- with **low draw-to-draw variance** (std ≈ 0.7 pp) — i.e. it barely matters *which* 32 windows you draw,
- and it equals the **paper's batch size** (32), ~**18 %** of the 180-window rest-downsampled batch
  (~10 % of the raw 320-window batch).

Below **K=16** accuracy and stability degrade quickly (K=10 std jumps to ~1.5; K=5 to ~4; K=2 to ~5.6
with worst-draws 8–15 pp under the mean). Above K=32 the gain is <1 pp (K=64 ≈ full within 0.3 pp),
so paying for more windows buys little. **Recommendation: collect BN stats on ≈32 windows** (round
up to 64 if you want to be within 0.3 pp of the full-batch stats at ~2× the cost).

Note: this `draw_std` is natural subsampling variance on *clean* data. True outlier robustness (a few
corrupted windows) is Exp 1b — the K decision should be paired with a robust collector (MAD-reject).

## Progress log
- 2026-07-22: runner built; K-sweep launched (S01+S02 × 8 K × 8 draws).
