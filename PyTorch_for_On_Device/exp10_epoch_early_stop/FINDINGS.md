# Exp 10 findings — is streaming AdaBN overfitted at epoch 40? Would fewer epochs help?

**Date:** 2026-07-22
**Purpose:** Test whether the epoch-40 weights in our streaming-AdaBN recipe are over-trained, and
whether manually lowering the epoch count (a fixed global early-stop, chosen like we chose 40) does
better. Deployable form only — Part A (an oracle learning curve that peeks at the eval batch) was
dropped as non-deployable; only a *fixed global epoch count* is a real on-device knob.

## Method
Run the full streaming AdaBN protocol (exp5: recollect stats on batch b, full-train on 30% of batch b,
classify batch b+1 with the carried weights+stats) end-to-end with `EPOCHS` fixed to each value in a
grid. n_accum=8, lr=3e-4, 30% data, S01 vocalized, 3 folds. Report mean b2–5.

## Result — flat, no overfitting peak

| epochs | streaming mean b2–5 |
|---|---|
| 5  | 84.12 |
| 10 | 82.73 |
| **20** | **85.56** (nominal best) |
| 30 | 85.23 |
| **40 (current recipe)** | **85.42** |
| 60 | 84.91 |

## The finding — NO, epoch-40 is not overfitted; early stopping is not a lever
- **Flat from 20→60** (85.6 → 85.2 → 85.4 → 84.9, all within 3-fold noise): **no peak-then-decline**,
  which is the signature of temporal over-training. So the epoch-40 weights are *not* past an
  overfitting cliff.
- **E=20 best but only +0.14 pp over E=40** → noise, not a real gain. Lowering epochs buys nothing.
- **E=60 dips only ~0.5 pp** → at most a whiff of over-training, tiny.
- Low points are at **E=5/10 (under-trained)**, so the real risk is stopping *too early*.

## Two meanings of "overfit" — only one applies
1. **Capacity-vs-data overfitting (real):** 16K params can't productively use 54 windows, so full-model
   FT adds little over the recollection → ties head-only (exp8: more *data* helps). ✅
2. **Temporal over-training (absent):** weights drifting past a sweet spot into memorizing training
   noise, recoverable by early stop. The sweep shows held-out accuracy **plateaus, never declines**. ❌

**Why no temporal overfitting:** small lr (3e-4) + strong pretrained init + sum-accumulation make the
optimizer settle into a shallow minimum near the init and plateau — it fits what little it can and
stops, rather than diverging into noise-memorization. The regime is too gentle to reach the
"held-out collapses" cliff.

## Takeaway
Early stopping (fewer epochs) is **not a lever** for streaming AdaBN — the tie with head-only is the
**session-adaptation ceiling**, not a recoverable over-training artifact. Predicts that adding an
early-stopping brake to the full-model arm would also stay tied (no temporal overfitting for it to
catch); weight decay might shave a hair but won't cross the ceiling without a regime change (more data
+ correct batch-N normalization, both un-portable on-device). Keeping EPOCHS=40 is fine; 20 is
equivalent and ~2× cheaper if compute matters.

## Files
`run_epoch_sweep.py` (Part B only; Part A helper retained but unused/commented in main);
`results/fixed_epoch_sweep_S01.csv`.
