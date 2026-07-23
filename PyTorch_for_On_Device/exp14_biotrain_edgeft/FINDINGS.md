# Exp 14 findings — BioTrain "Edge-FT" (GN + momentum-SGD) does NOT beat head-only on our data

**Date:** 2026-07-23
**Purpose:** Test whether BioTrain's Edge-FT deployment recipe (`biotrain_summary.md` §7) — replace every
BatchNorm with GroupNorm, train the FULL network, SGD momentum 0.9 + weight decay 1e-3 + cosine LR
(init 5e-3), 30 epochs, effective batch 8 via gradient accumulation — genuinely improves accuracy over
our head-only recipe. My earlier `idea2_alt_norm` screen showed GN-full (80.0) < BN-head (87.78) but used
*plain* SGD; exp14 applies the true Edge-FT optimizer and ablates its ingredients.

Protocol (matched to idea2_alt_norm for comparability): S01 vocalized, pretrain each norm from scratch on
the 2 non-held-out sessions (LOSO), FT on 30% of batch 1 (6/class, seed 42), eval batch 2; mean±std, 3 folds.

## Results — S01 vocalized, b1→b2, 3 folds

| config | b2 mean ± std |
|---|---|
| **BN head-only (ours)** | **87.78 ± 2.42** |
| GN full, plain-SGD (idea2) | 80.00 ± 4.94 |
| GN Edge-FT (paper: mom0.9, wd1e-3, cosine, lr5e-3, b8) | 80.74 ± 10.32 |
| **GN Edge-FT lr 1e-3** | **87.41 ± 1.16** |
| GN Edge-FT lr 1e-2 | 74.26 ± 5.16 |
| GN Edge-FT no-momentum | 85.56 ± 2.42 |
| GN Edge-FT no-wd | 80.74 ± 9.82 |
| GN Edge-FT no-cosine | 80.56 ± 11.10 |
| GN Edge-FT sum-grad | 44.07 ± 3.90 |

## On the BN baseline (why 87.78, not the official 88.52)
The `BN head-only` row here (87.78) is a **from-scratch** BN, pretrained with the *same* recipe as the GN
model — deliberately, so the BN-vs-GN comparison isolates the **norm** and is not confounded by pretraining
recipe/quality. Our **official deployed head-only** (from the inter-session checkpoint) is **88.52** (b2,
3-fold S01, `recipe_comparison`). The ~0.7 pp gap between the two is exactly that pretraining-recipe
difference. **Against the official 88.52, Edge-FT (best 87.41) loses outright (−1.1 pp)** — so using the
deployed baseline only strengthens the verdict; using the controlled from-scratch BN (87.78) makes it a tie.

## The finding — Edge-FT ties head-only at best, never beats it
- **Paper recipe as-is (lr 5e-3) = 80.74 ± 10.3** — barely above plain GN, very noisy, well below head-only.
- **lr is the whole story:** re-tuning to lr 1e-3 → **87.41 ± 1.16, which TIES head-only (87.78)** and has
  even lower variance. lr 5e-3 is simply too high for SpeechNet/SilentWear. So best-tuned Edge-FT reaches
  head-only level but **does not exceed it**.
- **Ablations = optimizer fragility, not a real gain:** at lr 5e-3, *removing* momentum HELPS (85.56 vs
  80.74) — momentum × high-lr is unstable; wd/cosine are ~neutral (80.7 / 80.6). **sum-grad collapses to
  44%** — our on-device sum-accumulation makes the effective lr 8× too large; Edge-FT needs the mean/÷8
  batch-8 convention.

## Why (consistent with the BioTrain summary's own §9)
BioTrain's full-training win is concentrated in **cross-subject Day-1** (large representational shift,
No-FT near chance). Our protocol is **subject-specific** (inter-session), so that regime does not exist
for us; the analogous BioTrain cell (EOG longitudinal, same-subject, EpiDeNet-family) beat head-only by
only **+0.7 pp**. GN also starts weaker than BN here (GN pretrained-from-scratch zero-shot < BN's), giving
Edge-FT no head-room to exceed head-only.

## Takeaway
On our subject-specific SpeechNet/SilentWear setting, the BioTrain Edge-FT recipe (GN + momentum SGD + wd
+ cosine) **does not genuinely improve accuracy** — best-tuned it ties head-only (87.41 vs 87.78); at the
paper's lr 5e-3 it is *worse* (80.74) and needs re-tuning. And it costs far more: GN-backbone retrain,
full-network backprop, mean-grad accumulation, and lr re-tuning. **Head-only + BN-fold remains the best
deployable recipe** — simpler, cheaper, no GN retrain, and equal-or-better here.

Caveats: b1→b2 single-round, S01 only, 3 folds; GN variants pretrained from scratch (weaker start than the
official BN checkpoint). A cross-subject Day-1 experiment — the regime Edge-FT is designed for — would be
the fair place to see it win, but that regime isn't part of our deployment target.

## Files
`run_edgeft.py`, `results_edgeft_S01.csv`.
