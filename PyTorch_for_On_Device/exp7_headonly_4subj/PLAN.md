# Exp 7 — head-only incremental across 4 subjects (vs streaming AdaBN, exp5)

**Date:** 2026-07-22
**Purpose:** Collect the head-only (shipped on-device recipe) incremental results for all 4 subjects
(vocalized), on the same streaming protocol as exp5, to compare head-only vs streaming AdaBN vs paper.

## Recipe
Head-only: train only `fc`, BN folded/frozen at pretrained session-1+2 stats (NO recollection),
SGD lr 0.01, n_accum 4 (sum), 40 epochs, 30% data. Incremental streaming protocol (classify batch b
with the carried model FT'd on batches 1..b-1 + pretrained stats) — inherently deployment-realistic
(no transductive stat peek). S01..S04, 3 folds. `run_headonly.py`, compare via `compare_headonly_vs_adabn.py`.

Head-only has no BN recollection, so it isolates the *weight fine-tuning* contribution; streaming AdaBN
(exp5) adds the per-batch stat recollection on top of full-model FT. The gap = the recollection benefit.

## Results — vocalized, mean b2–5 (3 folds each)  (`results/headonly_vs_adabn_4subj.csv`)

| subject | no-FT | **head-only** | streaming AdaBN (exp5) | paper | AdaBN − head |
|---|---|---|---|---|---|
| S01 | 73.80 | 84.68 | 85.42 | 88.24 | +0.74 |
| S02 | 55.93 | 65.23 | 64.63 | 71.71 | −0.60 |
| S03 | 72.08 | 73.66 | 73.38 | 74.72 | −0.28 |
| S04 | 80.05 | 83.75 | 82.73 | 85.42 | −1.02 |
| **MEAN** | **70.46** | **76.83** | **76.54** | **80.02** | **−0.29** |

Per-batch (4-subj avg): head-only 77.55/76.85/76.44/76.48 vs streaming AdaBN 77.08/75.88/77.36/75.83 (b2–5).

## Findings

**Head-only is on par with — even slightly ahead of — streaming AdaBN, and it's simpler.**
- 4-subject mean: head-only **76.83** vs streaming AdaBN **76.54** (AdaBN −0.29 pp, within noise).
- Head-only **wins on 3 of 4 subjects** (S02/S03/S04); AdaBN wins only on S01 (+0.74).
- Both recover ~+6 pp over no-FT and sit ~3–4 pp below the paper.

**Interpretation.** Once AdaBN is evaluated *realistically* (streaming: classify with the previous
round's stats, exp5), its extra machinery — full-model FT + per-batch BN-stat recollection — **no longer
beats head-only**. The two are just different routes to the same session adaptation: head-only adapts the
**classifier** (fc) with frozen features/stats; AdaBN adapts the **normalization** (recollected stats) +
full model. They reach the same place. The AdaBN advantage we saw earlier (87.78) was the *transductive*
peek at the eval batch, which real-time deployment can't use.

**Deployment implication:** head-only + BN-fold remains the better on-device choice — equal-or-better
accuracy at much lower cost (only `fc` trains, no full-model backprop, no per-batch stat-collection
kernel, no BN-stats-as-graph-inputs). This is consistent with s2_vs_headonly_seedsweep (frozen-stat
full-FT ≈ head-only) — now confirmed for the recollecting (AdaBN) variant too, across 4 subjects.

## Progress log
- 2026-07-22: runner built; S01-S04 launched.
