# Full-model FT with pretrained BN stats — two settings (on-device SGD recipe)

**Status:** IN PROGRESS · S01 vocalized · 3 folds · b1→b5 incremental · PyTorch host

## Motivation
Compare full-model fine-tuning under two BatchNorm regimes, both using **pretrained** running stats
(no AdaBN re-collection), against head-only / AdaBN-full / paper. Both use our on-device recipe
(SGD no-momentum, batch-1 + n_accum SUM, fixed lr, 40 ep, 30% data); we sweep n_accum × lr and take
the best, then run the full incremental.

## The two settings
**S1 — original BN kernel (live train / frozen infer).** Full model trains. Train-forward uses the
**single batch-1 window's own** statistics (BN in train mode); running stats are **NOT updated**
(momentum=0), so they stay at the pretrained values; **inference uses the frozen pretrained stats**.
→ This is the *faithful on-device naive* case (train/inference normalization mismatch).

**S2 — frozen-stat kernel (frozen train + frozen infer).** Full model trains, but BN normalizes with
the **frozen pretrained** running stats for **both** the training forward and inference (BN in eval
mode throughout; stats never touched). → train ≡ inference normalization. This is the `BN_FROZEN_STATS`
kernel recipe with pretrained stats.

Difference from earlier work:
- vs **head-only** (`../ft_summary_ondevice...`): S1/S2 train the *full model*, not just fc.
- vs **AdaBN-full** (`../ft_summary_adabn_full...`): AdaBN re-collects target-session stats; S1/S2 keep
  the *pretrained* stats.
- vs ablation E3/E3b: those use the paper **Adam** optimizer; S1/S2 use the on-device **SGD** recipe.
- vs old `frozen_bn_kernel_finetune`: that was b1→b2 only, old windowing; here full b1→b5, onset windowing, inter_session_ft.

## Sweep
b1→b2, 3 folds. n_accum ∈ {1,4,8,16,32} × lr ∈ {0.01, 0.003, 0.001, 0.0003}. Best = highest 3-fold
mean balanced acc. Then full b1→b5 incremental at the best config per setting.

## Files (this dir)
- `frozenstat.py` — model/train/eval for S1 & S2 + sweep + incremental runners.
- `run_sweep_s1.sh`, `run_sweep_s2.sh` — parallel n_accum jobs.
- `results/` — sweep CSVs, per-setting SUBJECT_MEAN CSVs, combined comparison.

## Results

**Best configs (b1→b2 sweep, 3-fold mean):**
- **S1** (live train / frozen infer): n_accum=32, lr=1e-3 → 72.2% (all S1 configs ~70–72%, *below*
  zero-shot 74.6). `results/sweep_s1_full.csv`
- **S2** (frozen both): n_accum=4, lr=3e-4 → 86.7%. `results/sweep_s2_full.csv`

**Full incremental (S01 vocalized, 3 folds, b1→b5), balanced acc mean ± std** —
`results/comparison_full_S01_vocalized.csv`:

| batch | zero-shot (no FT) | head-only | **S1 live/frozen** | **S2 frozen both** | AdaBN-full | paper |
|---|---|---|---|---|---|---|
| 2 | 74.63 | 88.52 | 72.22 | 86.67 | 87.96 | 87.22 |
| 3 | 77.04 | 83.33 | 66.67 | 83.33 | 89.63 | 87.96 |
| 4 | 77.96 | 85.93 | 57.96 | 89.44 | 88.89 | 90.37 |
| 5 | 65.56 | 80.93 | 46.11 | 83.15 | 84.63 | 87.41 |
| **mean b2–5** | **73.80** | **84.68** | **60.74** | **85.65** | **87.78** | **88.24** |

## Analysis
- **S1 (original BN kernel) collapses — and compounds.** Full training with live batch-1 stats but
  frozen pretrained inference stats degrades *below* zero-shot and **gets worse every round**
  (72→67→58→46). The conv is trained against per-window normalization but deployed with frozen
  pretrained stats (train/inference mismatch); incrementally carrying that mismatch forward
  accumulates the damage. **This is the faithful naive on-device full-FT case → it does not work.**
- **S2 (frozen pretrained stats, both) works and is stable** — mean 85.65, **beats head-only (84.68)**
  and is only ~2.6 pp under the paper (88.24). Because train and inference use the *same* (pretrained)
  stats, there is no mismatch; batch-1 is irrelevant (BN never looks at the batch). This is the
  `BN_FROZEN_STATS` recipe with pretrained stats.
- **Ordering:** S1 (60.7) ≪ zero-shot (73.8) < head-only (84.7) < S2 (85.7) < AdaBN-full (87.8) <
  paper (88.2). So freezing BN is essential (S2 ≫ S1), full-model helps a bit over head-only
  (S2 > head-only), and re-collecting target-session stats (AdaBN) adds ~2 pp more on top of S2.

## Progress log
- 2026-07-20: subdir + plan + runner created; sweeps run (40 configs); best S1 n32/1e-3, S2 n4/3e-4;
  full b1→b5 3-fold incrementals done; comparison table built.
