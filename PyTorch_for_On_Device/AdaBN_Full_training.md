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
_(to fill: table of ft_frozen and ft_recollect per config; mark best.)_

## Analysis
_(to fill: best config, does it beat head-only? does re-collection matter (staleness)? verdict.)_

## Progress log
- 2026-07-16: plan written; `adabn_full_training.py` (collect + frozen-stat full train + b1→b2 sweep) built; sweep launching.
