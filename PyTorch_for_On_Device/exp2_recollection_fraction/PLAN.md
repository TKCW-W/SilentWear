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

## Results — mean(b2–5) ± std over draws, and worst single draw
_(to fill from `results/frac_<subj>_K<K>.csv`)_

## Decision
_(to fill: smallest K where the mean plateaus AND the min-draw stays acceptable, for both subjects.)_

## Progress log
- 2026-07-22: runner built; K-sweep launched (S01+S02 × 8 K × 8 draws).
