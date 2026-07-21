# Exp 1 — impact of outliers on AdaBN

**Status:** IN PROGRESS · started 2026-07-22 · vocalized · 3 folds

## Question
AdaBN recollects BN stats from ≤180 windows of one batch — a small, outlier-sensitive sample. How
much do outliers hurt, and can a simple robust collection fix it?

## 1a — AdaBN on S02 (the noisy subject) vs paper
Run the decided AdaBN incremental (collect on batch → eval → full-train 30%, n8/3e-4) for **S02
vocalized**, 3 folds, b1→b5, and compare to the paper reference. Tests AdaBN on real-world noisy data
where recollected stats are most at risk. Paper ref (S02 vocalized): `paper_ft` mean b2–5 = 71.71,
`paper_no_ft` mean b2–5 ≈ 55.9 (note huge cross-fold std ±20). `adabn_incremental_subject.py`.

## 1b — synthetic outlier injection (+ robust fix)
Corrupt a fraction f of the **collection set** with an amplitude spike (×8; on unnormalized EMG this
inflates that window's variance ~×64), collect stats, evaluate on the CLEAN batch. Compare:
- **standard** collection (all windows), vs
- **robust** collection (reject windows with max-abs amplitude > median + 3·MAD, then collect).
Recollect-only (no FT), S01 + S02, f ∈ {0,2,5,10,20}%. `outlier_injection.py`.

## Results
_(to fill: 1a S02 AdaBN table vs paper; 1b accuracy vs f for standard vs robust.)_

## Findings
_(to fill: does AdaBN hold on S02? how fast do outliers degrade standard collection? does robust reject recover it?)_

## Progress log
- 2026-07-22: runners built; launches after the Exp-2 fraction sweep frees the cores.
