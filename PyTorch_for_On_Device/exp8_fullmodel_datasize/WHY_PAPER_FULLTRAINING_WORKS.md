# Why the paper's full-model FT works (and beats us) despite small data + many params

**Date:** 2026-07-22
**Purpose:** Explain why the paper's *full-model* fine-tuning achieves better results (~80 % vs our
~76.5 %, 4-subj) even though it trains ~16 K params on only 70 % of a batch (126 windows) — i.e. why it
doesn't overfit the way our on-device full-model does. Complements the exp8 data-size finding (on-device
full-model only ties head-only) and the exp7 head-vs-full explanation.

## The paper's FT recipe is heavily regularized (`ft_cfg.json`)
```
Adam, lr 1e-3 ; weight_decay 1e-4 ; ReduceLROnPlateau (patience 2) ;
early_stop_patience 10 (on a 30 % val split) ; up to 50 epochs ; dropout 0.5 ;
batch 32 ; LIVE BatchNorm (batch-32 stats, EMA-updated)
```
Our on-device recipe strips ALL of these: no weight decay, no early stopping, no val set, no dropout,
fixed 40 epochs, static lr, batch-1, frozen BN.

## Three reasons the paper's full-model FT works

**1. Regularization shrinks the *effective* capacity (the user's hypothesis — correct).**
The "16 K params vs 126 windows ⇒ overfit" intuition uses the *raw* parameter count. Weight decay
(L2 prior), dropout 0.5 (halves effective co-adapted capacity + noise), and especially **early stopping
on a real validation split** (the direct anti-overfitting brake — our fixed-40-epoch has none) make the
model behave like a much smaller effective model. Plus fine-tuning from a strong pretrained init
(sessions 1+2) is itself a regularizer. So the full model uses its capacity to adapt features without
memorizing the 126 windows.

**2. Proper batch-32 LIVE BatchNorm (the on-device-impossible ingredient — the fundamental one).**
Batch-32 stats ≈ population, so: (a) the conv adapts in a correctly-normalized regime — our on-device
full-model trains against *frozen* stats, so as conv shifts the activations drift from the frozen stats
(activation↔stat mismatch) and degrade normalization; (b) the running stats EMA-adapt to session 3 and
inference uses them; (c) BN's batch-stat noise is itself a regularizer. Our batch-1 recipe can do none
of this — this is why our on-device full-model only *ties* head-only while the paper's clearly beats both.

**3. Batch-32 gradients are cleaner than batch-1 SGD** → better-behaved optimization, less fitting to
individual noisy windows.

## Where the paper's ~3.5 pp edge comes from
| ingredient | paper | our on-device | portable on-device? |
|---|---|---|---|
| weight decay, dropout, LR schedule | ✅ | ✗ | **Yes** (cheap) |
| early stopping + validation set | ✅ | ✗ | Partly (needs held-out labels) |
| batch-32 live BN (correct norm + stat adapt) | ✅ | ✗ (batch-1) | **No** — fundamental blocker |
| batch-32 clean gradients | ✅ | ✗ (batch-1) | Partly (n_accum sums, doesn't fix BN) |

## Takeaway
Regularization (early stopping + weight decay + dropout) is a **major** reason the paper avoids
overfitting at 70 % data — but on top of that, its **batch-32 live BatchNorm** gives correctly-normalized,
session-adapted training that on-device batch-1 cannot reproduce. The regularizers we *could* port; the
batch-32 BN we cannot. This is exactly why **head-only** (freeze the features, adapt only the classifier)
is our best on-device move — it sidesteps "regularize a full model under bad batch-1 normalization"
entirely. The residual on-device gap to the paper is dominated by the un-portable batch-32-BN advantage.

## Optional follow-up (exp9)
Add weight decay + early-stopping (val split) to the on-device full-model recipe and measure how much of
the gap closes — the remainder isolates the pure (un-portable) batch-32-BN contribution.
