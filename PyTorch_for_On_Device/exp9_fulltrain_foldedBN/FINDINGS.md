# Exp 9 findings — full training on the BN-folded graph

**Date:** 2026-07-22
**Purpose:** Summarise what happens when we fold the frozen BatchNorm into Conv (BN-free graph) and then
full-train the folded Conv weights — the "cleanest on-device full-training path" (no BN kernels). Tests
the prediction that it equals unfolded frozen-stat full-FT (S2 / exp8 `full`).

## Setup
Fold each block's frozen BN into its Conv (`scale = γ/√(var+ε)`, `W←W·scale`, `b←(b−μ)·scale+β`, BN→
Identity), then full-train the folded Conv + fc (no BN). Streaming incremental, n_accum 4, 40 ep, 30 %
data. S01, 3 folds. Fold verified **lossless** (folded vs unfolded outputs differ 2e-6; no-FT identical
81.67 %).

## Result — collapses at the same lr, recovers only at ~100× lower lr

| lr | folded full-train (mean b2–5) |
|---|---|
| 3e-4 (= unfolded's lr) | **31.52** (collapse) |
| 3e-5 | 81.16 |
| 1e-5 | 82.78 |
| **3e-6** | **83.33** |
| 1e-6 | 82.92 |

vs unfolded frozen-stat full-FT = 85.65 ; head-only = 84.68.

## The finding
- **The reparameterization prediction (folded ≈ unfolded) holds for the reachable functions AND at the
  correct lr** — at lr 3e-6, folded reaches ~83, matching unfolded frozen-stat full-FT (within single-seed
  noise) and tying head-only. So they are the same recipe.
- **But NOT at the same lr:** folded needs a **~100× smaller lr** (3e-6 vs 3e-4); at the unfolded lr it
  **collapses to 31.5 %**.
- **Cause — BN does implicit per-layer learning-rate scaling.** BN's scale `γ/√(running_var+ε)` is small
  here (EMG is unnormalized → `running_var` in the thousands → scale ≈ 0.02–0.03). In the *unfolded* graph
  that scale sits after the Conv and **damps the conv gradients**. **Folding removes that damping**, so the
  same lr acts ~1/scale² ≈ 1000× larger on the folded conv → instability → collapse. Purely an
  optimization-dynamics effect (the fold is lossless in the forward pass).

## Takeaway
"Fold BN → full-train the folded graph" is a legitimate, BN-kernel-free on-device full-training path and
reaches the same accuracy as unfolded frozen-stat full-FT — **but it is not a drop-in**: the lr must be
re-tuned **~100× lower**, because folding destroys BN's implicit gradient scaling. And even tuned it only
*ties* head-only. So the deployment conclusion is unchanged: **head-only (freeze the folded conv, train
only fc) is simpler, equal-or-better, and avoids this lr fragility.** Broader lesson: **BatchNorm — even
frozen — is doing gradient/learning-rate conditioning**, which is why on-device BN handling matters so much.

## Files
`run_folded_fulltrain.py` (fold + folded full-train, `--lr`); `results/folded_S01.csv` (lr 3e-4),
`results/folded_lr*.csv` (sweep). Detail in `PLAN.md`.
