# Exp 11 findings — extended epoch sweep: streaming AdaBN DOES overfit past epoch 40

**Date:** 2026-07-22
**Purpose:** Test whether our streaming-AdaBN full-model recipe is UNDER-trained at 40 epochs (plausible,
since we use plain SGD — no momentum/wd, batch-1, small lr — which converges far slower than the paper's
Adam) or already converged/overfitting. Extends the exp10 grid far past 40 and, crucially, tracks the
final-epoch TRAINING loss alongside held-out accuracy so we can tell "under-trained" from "overfitting".

## Result — monotonic overfitting; SGD is NOT under-trained

| epochs | held-out b2–5 | final train loss |
|---|---|---|
| **40 (current recipe)** | **85.42** | 0.0286 |
| 80 | 84.91 | 0.0123 |
| 120 | 84.26 | 0.0074 |
| 160 | 83.98 | 0.0052 |
| 200 | 83.75 | 0.0039 |

Baselines: no-reg is this table's 85.42 @ep40; head-only = 84.68.

## The finding — the two columns move in OPPOSITE directions = textbook overfitting
- **Train loss falls monotonically toward 0** (0.0286 → 0.0039): SGD is *not* under-trained. It fits the
  54 FT windows ever more perfectly with more epochs. So "plain SGD is slower, give it more epochs" is
  refuted — optimization was never the bottleneck.
- **Held-out accuracy falls monotonically** (85.42 → 83.75, −1.67 pp over +160 epochs): generalization
  *degrades* as the fit tightens. Loss ↓ while held-out ↓ is the textbook signature of temporal
  overfitting — clean and monotone, no noise.
- **Peak generalization is at ~20–40 epochs** (cf. exp10: ep20 = 85.56, ep30 = 85.23, ep40 = 85.42);
  epoch-40 sits at the top of the plateau, and every step past it costs held-out accuracy.

## Correction to exp10
After exp10 (grid 5–60) I called the curve "flat, no temporal overfitting". That was premature — the
grid was too short. Extending to 200 shows a clean monotonic decline; the exp10 ep60 point (84.91),
which I dismissed as noise, is corroborated exactly by ep80 (84.91) and continues down. There IS
temporal overfitting past the peak; it is just mild per-epoch and only visible over a long horizon.

## Takeaway
- **Keep epochs at ~40 (or 20).** More epochs strictly hurt; fewer (down to ~20) are equal and cheaper.
  The generalization ceiling (~85.5, S01) is real and epochs do not cross it — they only move you along
  a plateau that gently declines.
- **The overfitting is real but the cure is not "stop earlier" alone** — the peak (85.5) still only ties
  head-only (84.68). Beating head-only needs a regularizer that raises the ceiling, not just avoids the
  post-peak decline. That is exp12 (portable weight decay / dropout).
- Confirms the recipe's implicit-regularization story from a new angle: strong pretrained init + small lr
  keep the peak early; the model then slowly memorizes the 54 windows if you let it run.

## Files
`run_more_epochs.py` (fixed-epoch streaming sweep + final-train-loss tracking);
`results/more_epochs_S01.csv`.
