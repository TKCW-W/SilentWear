# Frozen-pretrained-stat full-model fine-tuning vs head-only — findings

**Date:** 2026-07-22
**Purpose:** This file summarises all findings on **full-model on-device fine-tuning that keeps the
frozen pretrained BatchNorm running stats (from sessions 1+2) for both training and inference** —
including the config search and the multi-seed comparison showing it does **not** outperform our
shipped head-only recipe (S01 vocalized, 3 folds, b1→b5).

---

## 1. The setting ("S2")

Full model trainable (conv + BN γ,β + fc). BatchNorm uses the **frozen pretrained running stats**
(the session-1+2 values baked in the checkpoint) for **both** the training forward and inference —
they are never updated and never recollected. Everything else is the on-device recipe:

| knob | value |
|---|---|
| trainable scope | full model (no BN folding) |
| BN stats | frozen pretrained (train == inference) |
| optimizer | SGD, no momentum, no weight decay |
| learning rate | static (swept) |
| effective batch | 1, gradient accumulation `n_accum` SUM (swept) |
| epochs | 40 (fixed) |
| data | 30 % of each batch (54 stratified windows, 6/class) |
| base model | `inter_session_ft/S01/vocalized/.../leave_one_session_out_fold_{1,2,3}.pt` |
| windowing | onset-anchored (paper-faithful) |

Distinction from the other recipes:
- vs **head-only** (shipped): head-only trains only `fc` with BN folded/frozen; S2 trains the full model.
- vs **AdaBN-full**: AdaBN *re-collects* BN stats on the new session; S2 keeps the **pretrained** stats.
- vs the earlier `TrainDeeploy/.../frozen_bn_kernel_finetune` study: that was fold-3 / b1→b2 only, old
  windowing + `inter_session` weights; here it is the full 3-fold / b1→b5, onset windowing, `inter_session_ft`.

---

## 2. Best config (n_accum × lr sweep, b1→b2, 3 folds)

`frozen_stat_full_training/results/sweep_s2_full.csv` — top configs:

| n_accum | lr | eff_lr | balanced acc (b1→b2) |
|---|---|---|---|
| **4** | **3e-4** | 0.0012 | **86.67 ± 1.67** |
| 1 | 3e-4 | 0.0003 | 86.11 ± 1.67 |
| 8 | 3e-4 | 0.0024 | 85.93 ± 1.28 |
| 1 | 1e-3 | 0.0010 | 84.07 ± 0.64 |

Best config = **n_accum = 4, lr = 3e-4** (very lr-sensitive; eff_lr ≳ 0.02 destabilises).

---

## 3. Full incremental (single seed-42), S01 vocalized, 3 folds, b1→b5

`frozen_stat_full_training/results/ft_summary_s2_S01_vocalized_SUBJECT_MEAN.csv`:

| batch | base (no-FT) | S2 (frozen-stat full-FT) |
|---|---|---|
| 2 | 74.63 | 86.67 |
| 3 | 77.04 | 83.33 |
| 4 | 77.96 | 89.44 |
| 5 | 65.56 | 83.15 |
| **mean b2–5** | **73.80** | **85.65** |

On this single seed, S2 (85.65) edged head-only (84.68) by ~+1 pp — which motivated the multi-seed check.

---

## 4. Multi-seed comparison vs head-only — the decisive result

10 seeds (0–9), each running BOTH recipes' full incremental (3 folds, b1→b5); the seed varies the
30 % stratified FT-subset draw and data order. `s2_vs_headonly_seedsweep/results/seedsweep_summary.csv`.

**Aggregate (mean over b2–5):**

| recipe | mean ± std | range |
|---|---|---|
| **head-only** | **85.54 ± 0.81** | [84.03, 86.44] |
| S2 (frozen-stat full-FT) | 84.76 ± 0.70 | [83.56, 85.88] |
| paired diff (S2 − head) | **−0.78 ± 0.81** | S2 wins **2 / 10** |

Paired **t = −3.07** (df = 9, |t| > 2.26) → the difference is **statistically significant, in favour
of head-only**.

**Per-seed mean(b2–5) — head vs S2 (and S2 − head):**

| seed | head | S2 | S2 − head |
|---|---|---|---|
| 0 | 86.44 | 84.77 | −1.67 |
| 1 | 85.83 | 85.88 | +0.05 |
| 2 | 86.02 | 84.72 | −1.30 |
| 3 | 84.03 | 83.94 | −0.09 |
| 4 | 85.23 | 83.56 | −1.67 |
| 5 | 84.44 | 84.91 | +0.47 |
| 6 | 86.30 | 85.69 | −0.61 |
| 7 | 86.25 | 84.44 | −1.81 |
| 8 | 85.19 | 84.86 | −0.33 |
| 9 | 85.69 | 84.81 | −0.88 |

---

## 5. Verdict

- **The single seed-42 result where S2 beat head-only (+0.97 pp) was a favourable draw — noise, and
  the wrong sign.** Over 10 seeds, **head-only is slightly but significantly better** (85.54 vs 84.76,
  paired −0.78 ± 0.81, wins 8/10, p < 0.05).
- **Full-model fine-tuning with frozen pretrained BN stats does NOT outperform head-only.** It is *at
  best on par*, and on the balance of seeds slightly worse.
- This **confirms the earlier fold-3 / b1→b2 conclusion** ("frozen-stat full-FT does not robustly beat
  head-only") on the full 3-fold / b1→b5 setting and across 10 seeds — the earlier call was
  under-powered but correct in direction.
- **Head-only + BN-fold remains the better on-device choice:** ≥ accuracy, plus it is simpler and
  cheaper to deploy (BN folds away, only `fc` trains, no full-model backprop) and lower variance.

**Scope note:** this is the *frozen pretrained-stat* full-FT. It is a different recipe from **AdaBN-full**
(which re-collects the new session's BN stats and does reach ~87.8, beating head-only) — see
`AdaBN_Full_training.md`. So "full-model FT" is only competitive on-device when the BN stats are
*adapted to the new session* (AdaBN); keeping the pretrained stats does not help over head-only.

---

## 6. Source files

- Config search + single-seed incremental: `frozen_stat_full_training/` (`frozenstat.py`, `PLAN.md`,
  `results/sweep_s2_full.csv`, `results/ft_summary_s2_S01_vocalized_SUBJECT_MEAN.csv`).
- Multi-seed comparison: `s2_vs_headonly_seedsweep/` (`seedsweep.py`, `aggregate.py`, `PLAN.md`,
  `results/seed_0..9.csv`, `results/seedsweep_summary.csv`).
- Head-only reference: `results/ft_summary_ondevice_S01_vocalized_SUBJECT_MEAN.csv`.
- Related: `AdaBN_Full_training.md` (the stat-re-collection recipe that *does* beat head-only).
