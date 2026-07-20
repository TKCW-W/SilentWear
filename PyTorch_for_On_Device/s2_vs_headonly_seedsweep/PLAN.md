# S2 (frozen-stat full-FT) vs head-only — multi-seed sweep

**Status:** IN PROGRESS · S01 vocalized · 3 folds · b1→b5 incremental · 10 seeds · PyTorch host

## Question
Our complete 3-fold/b1→b5 experiment showed **S2** (full model, frozen pretrained BN stats for
train+infer) at mean(b2–5) **85.65** vs **head-only 84.68** — S2 edges head-only by ~+1 pp. But each
was a single seed-42 draw, and the earlier frozen-BN study warned this recipe has high draw-to-draw
variance (std ~3 pp). **Is the ~+1 pp real, or noise?**

## Method
Run BOTH recipes' full incremental (3 folds, b1→b5) for **seeds 0–9**; the seed varies the 30%
stratified FT-subset draw (main variance source) and the training data order. Per seed → per-batch
balanced acc (3-fold mean) for each recipe. Aggregate across seeds → mean ± std, paired diff (S2 −
head per seed), and win rate.

Recipes (both on-device SGD, batch-1 + n_accum 4 SUM, 40 ep, 30% data, frozen pretrained BN stats):
- **head-only**: fc only, lr 0.01 (shipped recipe)
- **S2**: full model, lr 3e-4 (best from `../frozen_stat_full_training`)

## Files
- `seedsweep.py` — per-seed runner (both recipes, full incremental).
- `results/seed_<0..9>.csv` — per-seed per-batch results.
- `results/seedsweep_summary.csv` — aggregate mean±std + paired diff + win rate.

## Results (10 seeds, mean(b2–5), `results/seedsweep_summary.csv`)

| recipe | mean ± std | range |
|---|---|---|
| **head-only** | **85.54 ± 0.81** | [84.0, 86.4] |
| S2 (frozen-stat full-FT) | 84.76 ± 0.70 | [83.6, 85.9] |
| paired diff (S2 − head) | **−0.78 ± 0.81** | S2 wins **2/10** |

Per-seed (S2 − head): −1.67, +0.05, −1.30, −0.09, −1.67, +0.47, −0.61, −1.81, −0.33, −0.88.
Paired t = **−3.07** (df=9); |t| **> 2.26** → the difference **is significant at p=0.05**, in favor
of **head-only**.

Per-batch (mean over seeds): head 86.5/86.3/87.3/82.1 vs S2 85.4/84.9/86.4/82.3 for b2/b3/b4/b5.

## Verdict
**The seed-42 result where S2 beat head-only (+0.97) was a favorable draw — noise, and wrong-signed.**
Over 10 seeds, **head-only is slightly but significantly better** (85.54 vs 84.76, −0.78 pp paired,
wins 8/10, p<0.05). This **vindicates shipping head-only**: on top of matching-or-better accuracy, it
is simpler and cheaper on-device (BN folds away, only fc trains, no full-model backprop) and
lower-variance. Frozen-stat full-model FT is *at best on par* with head-only, not a win — consistent
with the earlier fold-3/b1→b2 conclusion, now confirmed on the full 3-fold/b1→b5 setting over 10 seeds.

Caveat: this is the *frozen pretrained-stat* full-FT (S2). AdaBN-full (which re-collects target-session
stats) is a different, genuinely higher recipe (~87.8) — see `../AdaBN_Full_training.md`.

## Progress log
- 2026-07-20: subfolder + runner created; 10-seed sweep run; head-only 85.54±0.81 vs S2 84.76±0.70
  (paired −0.78±0.81, S2 wins 2/10, t=−3.07 significant) → head-only confirmed ≥ S2.
