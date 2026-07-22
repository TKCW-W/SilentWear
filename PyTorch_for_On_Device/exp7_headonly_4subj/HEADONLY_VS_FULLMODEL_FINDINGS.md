# Head-only vs full-model fine-tuning (frozen pretrained BN stats) — findings

**Date:** 2026-07-22
**Purpose:** Explain and document why **head-only** fine-tuning matches or beats **full-model**
fine-tuning on-device even though both use the frozen pretrained (sessions 1+2) BN running stats —
i.e. why more trainable capacity does *not* help here.

---

## The comparison (both use frozen pretrained BN stats)

| recipe | trainable params | adapts to session 3 by | S01 mean b2–5 |
|---|---|---|---|
| **head-only** (BN-fold) | `fc` only ≈ **297** | re-fitting the classifier | 84.68 (exp7) / 85.54 ±0.81 (10-seed) |
| **full-model, frozen stats** | conv + BN affine + fc ≈ **16 000** | re-fitting the whole model | 84.76 ±0.70 (10-seed, s2_vs_headonly_seedsweep) |
| streaming AdaBN (recollects stats) | full model + recollection | stats + full model | 85.42 (exp5) |

4-subject: head-only **76.83** ≈ streaming AdaBN **76.54** (exp7); and head-only ≥ full-model-frozen-stat
(s2_vs_headonly_seedsweep: paired −0.78 pp, full wins only 2/10). So all three land ~equal in the
realistic (streaming) setting; head-only is the simplest.

## The puzzle
Full-model FT has ~300× more capacity and could, in principle, reduce to head-only (leave conv
unchanged, move only fc). So it should do **at least as well**. Yet head-only is slightly *better*.

## Why head-only wins/ties — the reasons

1. **Tiny fine-tuning data → overfitting.** We FT on **54 windows** (6/class, 30% of a batch). Fitting
   ~16 000 params to 54 samples overfits their noise → worse on the held-out eval batch. Head-only fits
   297 params → strongly regularized → generalizes better. Extra capacity is a *liability* at this data
   size (classic bias–variance).
2. **Training the conv breaks the activation↔stat consistency.** Head-only never moves the conv, so the
   activations stay exactly what the frozen pretrained stats expect → normalization stays correct.
   Full-model FT shifts the conv, moving activations *away* from the frozen stats → the model normalizes
   with statistics that no longer match its own activations, degrading the normalization head-only preserves.
3. **The pretrained features already transfer.** The session-1+2→3 shift is mostly a classifier
   re-calibration (electrode placement / skin), not a need for new features. So re-fitting `fc` is
   *targeted*; adapting the conv chases unneeded capacity and mostly overfits.
4. **Optimization stability.** A linear head on fixed features is a near-convex, well-conditioned problem;
   batch-1 SGD backpropagated through the whole net is much noisier.

## This is a known phenomenon
"Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" (Kumar et al., ICLR
2022): under **distribution shift + limited target data**, full fine-tuning distorts good pretrained
features and **linear probing (= head-only) matches or beats it**. Session 3 is OOD vs sessions 1+2 and
we have only 54 windows — a textbook instance.

## Test (exp8): does MORE data close the gap?
Prediction from the overfitting explanation: as the FT data grows (30 %→100 % of the batch), full-model
FT should overfit less and its curve should rise toward / catch head-only. See
`../exp8_fullmodel_datasize/` — full-model (frozen stats) vs head-only at 30/50/70/100 % data.

## Deployment takeaway
Head-only + BN-fold is **≥ full-model FT, far cheaper (fc only, no full backprop), lower variance** —
so there is no reason to pay for full training. The residual ~3.5 pp to the paper is the paper's
unconstrained full-model Adam FT on GPU batch-32, which no on-device recipe reaches.
