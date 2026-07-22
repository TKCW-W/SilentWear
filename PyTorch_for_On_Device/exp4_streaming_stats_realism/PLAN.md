# Exp 4 — realistic streaming eval: classifying with the previous round's BN stats

**Date:** 2026-07-22
**Purpose:** Quantify the deployment-realism gap in AdaBN: our reported numbers recollect BN stats on
the *same* batch they classify (transductive, needs the whole batch buffered), but real-time
single-sample deployment must classify each incoming window with the *last* recollected stats (from the
previous batch/round). This measures how much accuracy that costs.

## Setup
Recollect-only (no fine-tuning — isolates the stat-staleness effect). S01 + S02 vocalized, 3 folds.
For each batch b (2–5) of the held-out session, evaluate batch b's windows under three BN-stat regimes:
- **pretrained** — frozen session-1+2 stats (no adaptation) — lower bound.
- **streaming** — stats recollected on the **previous** batch (b−1) — the realistic deployment number.
- **transductive** — stats recollected on the **current** batch (b) — what we reported.

**Hypothesis:** AdaBN's win is session-level adaptation (session 3 vs pretraining 1+2); within-session
batch-to-batch drift is small, so `streaming` (previous session-3 batch) should stay close to
`transductive` and well above `pretrained`. If so, real-time deployment keeps most of the AdaBN benefit.

## Results — balanced acc, 3-fold mean (recollect-only, base weights)

**S01:**

| batch | pretrained | streaming (prev batch) | transductive (current) |
|---|---|---|---|
| 2 | 74.63 | 85.18 | 84.81 |
| 3 | 77.04 | 84.63 | 88.15 |
| 4 | 77.96 | 85.74 | 87.41 |
| 5 | 65.55 | 79.26 | 80.37 |
| **mean** | **73.80** | **83.70** | **85.18** |

**S02:**

| batch | pretrained | streaming | transductive |
|---|---|---|---|
| **mean** | **55.93** | **61.62** | **63.66** |

## Findings

**Real-time deployment (classify with the previous round's stats) keeps almost all of the AdaBN benefit:**
- **S01:** streaming recovers **+9.9 pp of the +11.4 pp** total (87%); it trails the transductive number
  by only **1.5 pp**.
- **S02:** recovers +5.7 of +7.7 (74%), trailing transductive by ~2 pp.
- On batch 2, streaming (85.18) even slightly beats transductive (84.81) — the previous batch's stats
  classify the current batch just as well.

**Why:** AdaBN's gain is dominated by **session-level** adaptation (session 3 vs pretraining sessions
1+2). Within-session batch-to-batch drift is small, so the previous session-3 batch's stats are nearly
as good as the current batch's. So **you do NOT need to buffer-and-recollect on the batch you are
classifying** — carrying the last recollected (session-adapted) stats forward is enough.

**Deployment implication:** the transductive number we reported is mildly optimistic (~1.5–2 pp), but
the realistic streaming number is still far above the pretrained baseline. Real-time single-sample
classification is viable: keep the BN stats from the last recollection (end of the previous batch /
FT round) and classify incoming windows with them; recollect when a new batch's worth of unlabeled data
has accumulated. A cold-start window (before any session-3 stats exist) falls back to pretrained stats.

**Caveats:** (1) recollect-only isolates the stat-staleness effect; the streaming−transductive gap
(~1.5–2 pp) is the answer regardless of the carried FT weights. (2) "Previous batch" here is the same
held-out session; if deployment moves to a *new* session with fresh domain shift, the last session's
stats are staler and a re-collection on early new-session data would be needed.

## Progress log
- 2026-07-22: runner built; streaming eval launched (S01, S02).
