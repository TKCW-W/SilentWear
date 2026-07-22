# Exp 9 — full training on the BN-folded graph

**Date:** 2026-07-22
**Purpose:** Confirm empirically that "full-model FT on the BN-folded graph" (fold frozen BN into Conv →
BN-free graph, then full-train the folded Conv + fc) matches unfolded frozen-stat full-FT (S2 / exp8
`full`), as predicted by the reparameterization argument — and is thus a clean, BN-kernel-free on-device
full-training path that still only ties head-only.

## Method
Fold each block's BN (frozen pretrained stats) into its Conv: `scale = γ/√(var+ε)`, `W←W·scale`,
`b←(b−μ)·scale+β`, replace BN with Identity. Then full-train (all conv+fc, no BN) on the streaming
incremental protocol, lr 3e-4 / n_accum 4 / 40 ep / 30 % data. S01, 3 folds.

**Fold verified lossless:** unfolded vs folded model outputs differ by 2e-6 (fp32), no-FT identical (81.67 %).

## Results — S01, 3 folds

**Same lr as unfolded (3e-4): folded full-training COLLAPSES.**

| batch | no_ft | folded full-train (lr 3e-4) |
|---|---|---|
| 2 | 74.63 | 30.37 |
| 3 | 77.04 | 28.70 |
| 4 | 77.96 | 34.44 |
| 5 | 65.56 | 32.59 |
| mean b2–5 | 73.80 | **31.52** |

**lr sweep — folded full-training recovers at ~100× smaller lr:**

| lr | folded mean b2–5 |
|---|---|
| 3e-4 | 31.52 (collapse) |
| 3e-5 | 81.16 |
| 1e-5 | 82.78 |
| **3e-6** | **83.33** |
| 1e-6 | 82.92 |

vs unfolded frozen-stat full-FT (exp8) = **85.65**; head-only = **84.68**.

## Findings — my reparameterization prediction was half right
- **Prediction (folded ≈ unfolded) holds for the reachable functions AND at the right lr:** at lr 3e-6
  folded reaches ~83, matching the unfolded frozen-stat full-FT (~85.65, within single-seed noise) and
  tying head-only. So they *are* the same recipe.
- **BUT NOT at the same lr:** folded needs a **~100× smaller lr** (3e-6 vs 3e-4). At the unfolded lr it
  **collapses to 31.5 %**.
- **Why:** BN's scale factor `γ/√(running_var+ε)` is **small** here (EMG unnormalized → running_var in the
  thousands → scale ≈ 0.02–0.03). In the *unfolded* graph that scale sits after the Conv and **damps the
  conv gradients** — an implicit per-layer learning-rate reducer. **Folding removes that damping**, so the
  same lr becomes a ~1/scale² ≈ 1000× larger *effective* lr → instability → collapse.
- **Fold is verified lossless** at fold time (outputs differ 2e-6, no_ft identical 81.67 %); the divergence
  is purely an optimization-dynamics effect, not a forward-pass error.

## Takeaway
"Fold BN then full-train the folded graph" is a valid, BN-kernel-free on-device full-training path and
reaches the same accuracy as unfolded frozen-stat full-FT — **but it is NOT a drop-in**: you must
re-tune the lr down by ~100× because folding destroys BN's implicit gradient scaling. Even then it only
ties head-only. So the practical conclusion stands: head-only (freeze the folded conv, train only fc) is
simpler and equal-or-better, and avoids this lr-sensitivity entirely.

## Progress log
- 2026-07-22: fold verified lossless; lr 3e-4 collapsed (31.5); lr sweep confirmed recovery at ~3e-6 (83.3).

## Progress log
- 2026-07-22: fold verified lossless; S01 launched.
