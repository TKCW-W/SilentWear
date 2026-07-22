# AdaBN outlier robustness — study summary

**Date:** 2026-07-22
**Purpose:** This file summarises the whole outlier-robustness study for AdaBN's BN-stat recollection —
whether recollected stats are corrupted by outlier windows, how much data is needed, and how to make
the collection robust — consolidating results from `exp1_outlier_impact/`, `exp2_recollection_fraction/`,
and `exp3_loo_sensitivity_winsorize/` (all S01/S02 vocalized, 3 folds, PyTorch host).

---

## Motivation
AdaBN recollects BatchNorm running stats from ≤180 windows of one incoming batch (vs the paper's
session-1+2 EMA on GPU batch-32). A small sample is more exposed to outliers, and EMG is unnormalized,
so a large-amplitude artifact window can inflate a channel's variance and — since BN divides by
√(σ²+ε) — quietly attenuate that channel for the whole session. This study asks: **is that a real
risk, how much data do we actually need, and what is the robust, efficient collection recipe?**

---

## 1. Are there real outliers in the (filtered) data? — YES  (`amplitude_outlier_scan.py`)
Per-window peak amplitude of the **filtered** signal, vs the batch median, across all 30 S01/S02
batches (`results/amplitude_outliers.csv`):

- **All 30/30 batches** contain ≥1 window above median + 3·MAD (typically **5–14 of 180**, ~3–8%).
- **4/30 batches** have an **extreme** window (>10× median); the worst is **55× the batch median**
  (S02 sess2/batch1); others reach 22–38× (S01).
- Two shapes occur: *isolated freak* (one 55× window, rest of tail ≤2.8×) and *cluster* (e.g. 37.7×
  **and** 24.4×, then a tail) — so the number of outliers per batch varies.

**Why filtering doesn't remove them:** the paper's 20 Hz high-pass + 50 Hz notch targets DC drift and
line noise; motion artifacts, electrode pops and saturation are broadband and pass straight through.
So genuine artifact windows survive preprocessing.

## 2. Do outliers actually corrupt the recollected stats? — YES  (`../exp1_outlier_impact/`, 1b)
Inject ×8 amplitude spikes into a fraction f of the collection windows, evaluate on the CLEAN batch
(recollect-only). **Standard** collection degrades roughly linearly with contamination:

| f | S01 standard | S02 standard |
|---|---|---|
| 0 % | 85.19 | 63.66 |
| 5 % | 82.79 | 59.22 |
| 10 % | 80.96 | 52.65 |
| 20 % | 75.33 (−9.9) | 40.17 (−23.5) |

So corrupted windows meaningfully pollute the stats — confirming the risk is real, not just theoretical.

## 3. How influential is a single window on the real data? — LOO  (3a, `loo_sensitivity.py`)
Leave-one-out max |Δσ²/σ²| per BN layer, over folds×batches:

| | typical (median) | worst (max) |
|---|---|---|
| S01 | 3–7 % | 77–83 % |
| S02 | 2–6 % | 73–87 % |

- **Median 2–7 %** → the *typical* window is well diluted (the dilution intuition holds on average).
- **Max 77–87 %** → some batches contain a window that on its own swings a channel's variance ~80 %.
- **Deeper BN layers (BN3/BN4) are more exposed** than BN0 — they pool far fewer observations.
- Caveat: raw Δσ²/σ² is inflated by near-zero-variance channels (an ε-mirror artifact), so 77–87 % is
  an upper bound — but §1's amplitude data shows a real physical cause (artifact windows) underlies it.

## 4. The fix — sample-level MAD rejection (immune); winsorization fails  (1b + 3b)
Three collectors under ×8 injection (recollect-only, eval on clean batch):

| f | S01 std | S01 **reject** | S01 winsorize | S02 std | S02 **reject** | S02 winsorize |
|---|---|---|---|---|---|---|
| 0 % | 85.19 | **85.42** | 68.52 | 63.66 | **63.75** | 44.40 |
| 10 % | 82.51 | **85.27** | 37.14 | 53.19 | **63.56** | 13.11 |
| 20 % | 77.22 | **85.21** | 16.56 | 40.72 | **63.33** | 11.11 |

- **MAD-reject** (drop windows with peak amplitude > median + 3·MAD, then collect) is **immune** — flat
  at the clean level across all contamination, both subjects, and **zero cost on clean data**. The
  threshold catches the 5–14 flagged windows/batch from §1, however many there are.
- **Winsorization** (clip each observation to per-channel [0.5, 99.5] pct) **failed** — it hurt even at
  0 % contamination, because clipping legitimately heavy-tailed EMG activations removes real variance →
  BN over-normalizes. So the observation-level fix is worse than dropping artifact windows.

## 5. How much data is needed? — K ≈ 32 windows  (`../exp2_recollection_fraction/`)
Recollect-only accuracy vs collection size K (8 random draws), mean(b2–5):

| K | S01 mean (draw-std) | S02 mean (draw-std) |
|---|---|---|
| 16 | 83.05 (0.96) | 61.52 (0.76) |
| **32** | **83.96 (0.74)** | **62.49 (0.65)** |
| 64 | 84.93 (0.67) | 63.40 (0.48) |
| 180 (full) | 85.19 | 63.66 |

K≈32 reaches within ~1.2 pp of the full-batch stats with low draw-to-draw variance (~0.7 pp) on both
clean and noisy subjects (= the paper's batch size, ~18 % of the 180-window batch). Below K=16 accuracy
and stability drop quickly.

## 6. Does AdaBN hold on the noisy subject? — YES  (1a)
Full AdaBN incremental on **S02** (noisiest subject), 3 folds: mean b2–5 = **71.20 ≈ paper 71.71**
(+15.3 pp over base). Real-world subject noise alone does not break AdaBN.

---

## Verdict & decided recipe

Outliers in AdaBN's collection set are **real** (genuine artifact windows survive filtering, up to 55×
median, in every batch) and **can** corrupt the stats — but *how much* depends critically on whether
the collection set matches the evaluation set (self-consistency):

- **Injected outliers with a collect/eval MISMATCH** (corrupt the collection set, evaluate on a clean
  set) degrade standard collection badly (−10/−23 pp at 20 %). This is the worst case.
- **Natural outliers in the SELF-CONSISTENT flow** (collect on the batch, evaluate the *same* batch)
  cost almost nothing: removing them changes accuracy by only **+0.2 pp (S01) / +0.09 pp (S02)** — the
  0 %-injection row. Self-consistency cancels the big mismatch term; only a small within-batch
  shared-variance effect remains. **This is why the earlier AdaBN matched the paper without any outlier
  handling — those results are genuinely robust.**

So the dilution argument holds *on average / self-consistent* but **fails under a collect/eval mismatch
or in the rare extreme batch**. The recipe therefore depends on the deployment mode:

> **If fine-tuning on device: use K = full (collect on the whole batch) and recollect per batch for
> BOTH fine-tuning and inference.** The collection pass is forward-only (~4 % of training cost), so
> shrinking it buys almost nothing — and K=full keeps collection = evaluation (self-consistent), which
> is what makes outliers a near-non-issue. **The robust collector is then optional insurance** (for the
> rare 55× batch).
>
> **If inference-only AdaBN** (no training, so the collection pass is the dominant cost): use **K ≈ 32**
> for ~6× cheaper collection — but this partially breaks self-consistency (collect 32, classify 180),
> so **add sample-level MAD rejection** (drop windows with peak amplitude > median + 3·MAD) + an ε floor.

Supporting facts:
- MAD-reject is immune to injected outliers and free on clean data; **winsorization is NOT a substitute**
  (it hurts even at 0 % contamination).
- K≈32 reaches within ~1.2 pp of the full-batch stats; K=full is the reference.
- Self-consistency requires **batch-wise** operation (buffer the batch → recollect → classify); a pure
  single-window streaming mode with no batch to recollect on is the one setting that breaks it.
- AdaBN's **per-session reset** caps any single bad batch's blast radius to that session.
- Validated on a clean (S01) and a noisy (S02) subject; AdaBN matches the paper on both.

## Source files
- §1 `amplitude_outlier_scan.py` (+ `results/amplitude_outliers.csv`)
- §3 `loo_sensitivity.py` (+ `results/loo_S0{1,2}.csv`)
- §4 `winsorize_vs_reject.py` (+ `results/wins_S0{1,2}.csv`); `../exp1_outlier_impact/outlier_injection.py`
- §5 `../exp2_recollection_fraction/` (`frac_recollect.py`, `results/frac_summary.csv`)
- §6/§2 `../exp1_outlier_impact/adabn_incremental_subject.py` (+ `results/adabn_S02_vocalized.csv`)
- Per-experiment detail: each folder's `PLAN.md`.
