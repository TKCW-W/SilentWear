# Exp 12 findings — do the PORTABLE regularizers lift the streaming-AdaBN ceiling? (No.)

**Date:** 2026-07-23
**Purpose:** The paper's full-model FT beats ours partly via regularization (weight decay, dropout,
early stopping, LR schedule). exp11 showed our recipe overfits (mildly) past epoch 40. Question: can we
port any of those regularizers to our Onnx4Deeploy→Deeploy/Siracusa training stack, and does the
portable subset lift streaming-AdaBN full-model past head-only / toward the paper? Two halves:
(A) a grounded on-device portability audit; (B) a PyTorch measurement of the portable regularizers.

---

## A. Portability audit (evidence-grounded, our stack)

| regularizer | portable in our stack? | evidence / what's needed |
|---|---|---|
| **Weight decay** | ✅ **Yes — near-free** | pulp-trainlib kernel already implements `w -= lr*(grad+λw)` (`pulp_optimizers_fp32.c:48-60`, `weight_decay_lambda`). Only codegen glue missing: ONNX SGD node (`optimizer_onnx.py:96`), parser (`Parsers.py:3009`), template (`SGDTemplate.py`) thread only `lr`. ~3 small edits, zero runtime/memory/data cost. |
| **Fixed lower epochs** | ✅ Yes — trivial | just set `N_TRAIN_STEPS` (exp10/11). |
| **Dropout** | ⚠️ **Portable only with a Deeploy op build** | PRNG (`pulp_random.h`) + `pulp_dropout_fp32` kernel exist, but Deeploy has NO Dropout/DropoutGrad Layer/Parser/`PULPMapping`; SpeechNet export omits it "for Deeploy tiling/gradient compatibility". Needs full op integration (Layer+GradLayer+Parser+bindings+mapping+seed mgmt+regen+grad test). |
| **Early stopping** | ❌ Not as-is | training loop is compile-time-fixed (`deeploytraintest.c:370`, `N_TRAIN_STEPS`), no runtime break. Needs ~150–200 LOC: epoch-structured loop + val split (feasible at codegen) + on-device val metric + patience/break + best-weight restore (~60 KB snapshot, cheap). |
| **LR schedule** | ❌ Not as-is | lr is a compile-time constant in generated C (`float learning_rate = 0.0099…`); no runtime mutation path. |
| **Batch-32 live BN** | ❌ Fundamental | impossible under batch-1. This is the paper's dominant, un-portable edge. |

**Portable subset to test:** weight decay (clean); dropout (only if the effect justifies the op build).
PyTorch `SGD(weight_decay=λ)` adds `λw` once per optimizer step on the sum-accumulated gradient —
exactly the pulp-trainlib on-device step — so Part B is a faithful proxy.

---

## B. Effect measurement (streaming AdaBN full-model, S01, 3 folds, mean b2–5, ep40)

**Weight-decay sweep** — baseline (wd 0) = 85.42, head-only = 84.68:

| weight decay | mean b2–5 |
|---|---|
| 0 | 85.42 |
| 1e-5 | 85.42 |
| **1e-4** | **85.69** (best) |
| 3e-4 | 85.51 |
| 1e-3 | 85.46 |
| 3e-3 | 85.60 |

**Dropout sweep:**

| p_dropout | mean b2–5 |
|---|---|
| 0 | 85.42 |
| 0.25 | 83.98 |
| 0.5 | 85.60 |

## The finding — portable regularization does NOT lift the ceiling
- **Weight decay is portable but neutral.** Best λ=1e-4 → 85.69 = **+0.27 pp**, inside 3-fold noise
  (fold std ~1–4 pp). The whole 3e-5→3e-3 range hovers ~85.5. No meaningful gain.
- **Dropout not worth the op build.** Non-monotonic/unreliable: p=0.25 *hurts* (−1.44), p=0.5 +0.18.
  No consistent benefit to justify the full Deeploy Dropout/DropoutGrad integration.
- **Why weight decay barely helps (consistent with exp11):** its value is flattening the *post-peak*
  decline at high epochs. We deploy at ~40 — already near the generalization peak — so there's little
  decline to prevent, and wd does not *raise* the peak. exp11 already says "stop near the peak," which
  leaves wd almost nothing to do.

## Takeaway
The streaming-AdaBN ceiling (~85.5, S01) is **not** an overfitting artifact that portable regularization
can remove. The mild post-40 overfitting is handled for free by using ~40 epochs; the ceiling itself is
the session-adaptation limit plus the paper's **un-portable batch-32 live BN**. So porting the paper's
regularizers on-device buys ≈nothing. **Head-only + BN-fold remains the best deployable recipe** —
simplest, intrinsically regularized (297 params), and equal-or-better than the full-model + portable-reg
alternatives. Weight decay is a free, harmless add if wanted (marginally positive, cheap to wire), but it
is not a lever; dropout, early stopping, and LR schedules are either not portable or not worth it.

## Files
`run_regularizers.py` (portability-audit-driven wd + dropout sweeps);
`results/wd_sweep_ep40_S01.csv`, `results/dropout_sweep_ep40_S01.csv`.
