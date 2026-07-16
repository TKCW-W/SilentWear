# AdaBN + full-model on-device fine-tuning — does adapting BN stats unlock full training at batch-1?

**Status:** IN PROGRESS · started 2026-07-16 · Subject S01, vocalized, 3 folds · PyTorch host

## Idea
On-device the training BN kernel normalizes each window by its own (degenerate) single-window
stats, and never updates the running stats — so full-model FT breaks and inference uses frozen
pretrained stats (train/inference mismatch). **AdaBN** fixes the normalization without needing a
real batch: collect population stats over many samples in a separate forward-only pass, freeze
them, and train with those.

## Procedure (two passes per FT round — hence a new incremental runner)
For each fine-tuning round on a batch:
1. **Collect (forward-only, label-free):** run all of the batch's windows one at a time,
   accumulate per-channel μ, σ² across them, and write them into the BN running-stat buffers.
   (On host this equals a single population-stat pass over all windows; the one-at-a-time
   accumulation is the on-device mechanism, same result.) Uses the current (pre-training) weights.
2. **Freeze stats, full fine-tune:** hold μ, σ² fixed (BN in eval mode using the collected stats)
   and train **conv + BN γ,β + fc (full model)** with the on-device recipe:
   SGD (no momentum/wd), **fixed lr**, fixed epochs (40), **30% data**, batch-1 + **n_accum** (sum).

No BN folding — BN stays as a layer, but its stats are the frozen collected ones (batch-independent,
so batch-1 is fine). Inference/eval uses those same frozen collected stats.

## What we sweep
On the **b1→b2** transition first (batch 1 as the first step), 3 folds:
- **n_accum** ∈ {1, 4, 8, 16, 32}
- **lr** ∈ {3e-2, 1e-2, 3e-3, 1e-3, 3e-4}
(with sum accumulation, effective step ≈ lr × n_accum, so the grid spans a wide effective-lr range.)

For each config we report two eval variants to expose the staleness of the pre-training stats:
- **ft_frozen** — eval with the stats collected *before* training (the literal procedure).
- **ft_recollect** — re-collect stats on the FT data at the *final* weights before eval
  (deployment-realistic: the inference graph gets stats from a forward pass at the deployed weights).

## References (b1→b2, S01, 3 folds)
- our head-only + BN-fold recipe: **88.5%**
- paper full-model Adam FT: **87.2%**
- idea1 precomputed-stat full FT (mean reduction): 87.6% @ lr 3e-4 (did not beat head-only)
- naive batch-1 full FT with live BN (ablation E3): **59%** (collapse)

## Goal / question
Does AdaBN (collect+freeze+full-train) let batch-1 **full-model** FT reach or beat head-only /
paper? If yes at some (n_accum, lr), full on-device training is viable with a cheap forward-only
stat-collection kernel. Then run the full incremental (all rounds, 3 folds) with the best config.

## Results — b1→b2 (n_accum × lr) sweep, mean±std over 3 folds
Full grid in `results/adabn_sweep_full.csv`. Top configs (by ft_recollect):

| n_accum | lr | eff_lr | ft_frozen | ft_recollect |
|---|---|---|---|---|
| **8** | **3e-4** | 0.0024 | 86.3 | **87.2** |
| 4 | 3e-4 | 0.0012 | 86.1 | 86.1 |
| 1 | 3e-4 | 0.0003 | 86.1 | 86.1 |
| 4 | 1e-3 | 0.0040 | 85.0 | 85.7 |
| 16 | 3e-4 | 0.0048 | 84.1 | 84.3 |
| … eff_lr ≥ 0.04 | | | 11.1 | 11.1 (collapse) |

refs: naive batch-1 full (live BN) = **59** · paper = **87.2** · head-only = **88.5**

## Analysis
- **AdaBN unlocks batch-1 full-model FT:** 59% (naive) → **87.2%** (best), matching the paper's
  full FT exactly. The collect-freeze-train scheme genuinely fixes the batch-1 BN problem.
- **But it does not beat head-only (88.5)** on b1→b2, and it's costlier (full-model training + a
  forward-only stat-collection pass).
- **Best config: n_accum=8, lr=3e-4** (eff_lr ≈ 0.0024). **Very lr-sensitive** — only low effective
  lr (~0.001–0.005) is stable; eff_lr ≳0.02 destabilizes, ≳0.04 collapses to chance (sum-accum lr coupling).
- **Staleness is modest at low lr:** re-collecting stats at the final weights gives ~+1 pp (best:
  86.3→87.2). At high lr the pre-training stats would be badly stale, but those lrs collapse anyway.

## Phase B — full incremental with the best config (n_accum=8, lr=3e-4)
Does full-model AdaBN help on the **harder batches** (b3/b5) where head-only underperformed?
Run the two-pass incremental (`run_inter_session_ft_adabn.py`), 3 folds, all rounds; compare per
batch to our head-only recipe. _(results below)_

## Progress log
- 2026-07-16: plan written; `adabn_full_training.py` (collect + frozen-stat full train + b1→b2 sweep) built; sweep launching.
