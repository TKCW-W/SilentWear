# Exp 5 — corrected (streaming) AdaBN incremental setting

**Date:** 2026-07-22
**Purpose:** Run the deployment-realistic AdaBN incremental setting — each batch is classified with the
PREVIOUS round's weights + BN stats (not stats recollected on the batch being classified) — and collect
the accuracy for S01. This is the honest, real-time-deployable, paper-matching number (the earlier
transductive AdaBN was ~1.5–2 pp optimistic; see exp4).

## Protocol
```
W ← base weights ;  S ← pretrained stats
for b = 1..5:
    zs(b) = classify batch b using (W, S)          # previous round's state
    if b < 5:
        S ← recollect BN stats on batch b
        W ← fine-tune W on 30% of batch b (BN frozen at S)
```
- zs(1) = base + pretrained = the no-FT baseline (pure zero-shot).
- zs(2) = weights FT'd on b1 + b1's recollected stats, evaluated on b2.  … etc.
- Only change vs the transductive runner: `collect_bn_stats` runs AFTER the eval (inside b<5).

Config: n_accum=8, lr=3e-4, 40 epochs, 30% data. S01 vocalized, 3 folds. `run_streaming_adabn.py`.

## Results — S01 vocalized, 3 folds (`results/streaming_incr_S01.csv`)

| batch | no-FT | **streaming AdaBN** | (transductive AdaBN) | (paper) |
|---|---|---|---|---|
| 1 | 72.59 | 72.59 (= no-FT, cold start) | 82.04 | 72.59 |
| 2 | 74.63 | 86.30 | 87.96 | 87.22 |
| 3 | 77.04 | 84.07 | 89.63 | 87.96 |
| 4 | 77.96 | 89.26 | 88.89 | 90.37 |
| 5 | 65.56 | 82.04 | 84.63 | 87.41 |
| **mean b2–5** | **73.80** | **85.42** | 87.78 | 88.24 |

**Where it lands:**
- streaming AdaBN **85.42** vs the (optimistic, transductive) 87.78 → the honest deployment number is
  **−2.4 pp**, in line with exp4's ~1.5–2 pp expectation.
- Still **+11.6 pp over the no-FT baseline** (73.80), and still **above head-only (84.68)** by +0.7 pp.
- **−2.8 pp vs the paper (88.24)** — the paper's advantage is its unconstrained full-model Adam FT on
  GPU batch-32; this is our on-device-realizable, real-time-deployable number.
- b1 = no-FT exactly (72.59): correct cold-start behaviour (no session-3 adaptation available yet).

## Decision
**This is the standard AdaBN number to report going forward** — it is real-time deployable (classify
with the previous round's weights + stats), matches the paper's non-transductive protocol, and is honest
(the earlier 87.78 was transductive/optimistic by ~2.4 pp). Net: on-device AdaBN incremental FT recovers
~+11.6 pp over zero-shot for S01 and edges head-only, at ~2.8 pp below the unconstrained paper baseline.

## Progress log
- 2026-07-22: runner built; S01 launched.
