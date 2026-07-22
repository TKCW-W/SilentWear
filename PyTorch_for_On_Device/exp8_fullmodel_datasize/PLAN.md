# Exp 8 — does more FT data close the head-only vs full-model gap? (frozen pretrained BN stats)

**Date:** 2026-07-22
**Purpose:** Test the overfitting explanation for why head-only ties/beats full-model FT (see
`../exp7_headonly_4subj/HEADONLY_VS_FULLMODEL_FINDINGS.md`). If the cause is overfitting on tiny data,
then giving full-model FT MORE data should reduce the gap / let it catch head-only.

## Setup
Incremental streaming protocol (classify batch b with carried model FT'd on 1..b-1 + FROZEN pretrained
BN stats, no recollection). Both recipes at data fractions 30/50/70/100 % (per_class 6/10/14/20 of the
180-window batch). head = fc-only (lr 0.01); full = full-model frozen-stat (lr 3e-4). S01, 3 folds.

## Prediction
- head-only: fairly flat across data size (297 params, already well-regularized).
- full-model: rises with data (less overfitting), approaching / catching head-only at 70–100 %.

## Results
_(to fill: head vs full mean b2-5 at 30/50/70/100 %.)_

## Progress log
- 2026-07-22: runner built; 8 jobs (head+full × 4 fractions) launched.
