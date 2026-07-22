# Exp 8 — does more FT data close the head-only vs full-model gap? (frozen pretrained BN stats)

**Date:** 2026-07-22
**Purpose:** Test the overfitting explanation for why head-only ties/beats full-model FT (see
`../exp7_headonly_4subj/HEADONLY_VS_FULLMODEL_FINDINGS.md`). If the cause is overfitting on tiny data,
then giving full-model FT MORE data should reduce the gap / let it catch head-only.

## Setup
Incremental streaming protocol (classify batch b with carried model FT'd on 1..b-1 + FROZEN pretrained
BN stats, no recollection). Both recipes at data fractions 30/50/70/100 % (per_class 6/10/14/20 of the
180-window batch). head = fc-only (lr 0.01); full = full-model frozen-stat (lr 3e-4). S01, 3 folds.

## Prediction
- head-only: fairly flat across data size (297 params, already well-regularized).
- full-model: rises with data (less overfitting), approaching / catching head-only at 70–100 %.

## Results — S01, 3 folds, single seed (mean b2–5)

| data % | windows/class | head-only | full-model (frozen stats) | full − head |
|---|---|---|---|---|
| 30 | 6 | 84.68 | 85.65 | +0.97 |
| 50 | 10 | 85.93 | 85.23 | −0.70 |
| 70 | 14 | 86.30 | 88.10 | +1.80 |
| 100 | 20 | 86.57 | 86.76 | +0.19 |

## Findings (tentative — single seed, noisy)
- **Both recipes improve with more data:** head-only 84.68 → 86.57 (+1.9), full-model rises too
  (noisier). So more FT data helps regardless of scope.
- **Weak support for the overfitting explanation:** full-model does *relatively* best at 70 % (+1.8),
  and at 100 % the two are essentially equal (86.6 vs 86.8), whereas at 30 % the *10-seed*
  s2_vs_headonly test had head-only slightly ahead (85.54 vs 84.76). So the head↔full gap tends to
  **close as data grows** — consistent with "full-model overfits on tiny data, catches up with more."
- **BUT the single-seed noise (~±1–2 pp) is too large to be conclusive** — the 50 % point (full below
  head) and the non-monotonic full-model curve show the trend is not clean at one seed. The 30 % row
  here (full > head) even contradicts the 10-seed result at 30 %, underscoring the variance.

**Verdict:** directionally consistent with the overfitting story (more data narrows/closes the gap,
they converge to ~equal by 100 %), but not a clean demonstration at single seed. For a firm claim,
re-run with ≥8 seeds per (recipe, data-fraction). Practically: even with 100 % data, full-model only
*ties* head-only — it never clearly beats it — so the deployment conclusion (head-only is the simpler,
equal-or-better choice) stands regardless.

## Progress log
- 2026-07-22: runner built; 8 jobs (head+full × 4 fractions) launched.
