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
_(to fill: best config per setting; full b1→b5 3-fold table; comparison vs head-only/AdaBN/paper.)_

## Progress log
- 2026-07-20: subdir + plan + runner created; sweeps launching.
