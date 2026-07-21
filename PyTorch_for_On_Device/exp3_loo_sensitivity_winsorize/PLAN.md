# Exp 3 — LOO variance-influence on real data + winsorize vs MAD-reject

**Status:** IN PROGRESS · started 2026-07-22 · S01 + S02 vocalized · 3 folds

## Motivation
Follow-up to the "outlier dilution" argument. Two open questions it raised:
- Do the **real** (uncorrupted) batches already contain an influential single window (one whose
  removal materially shifts a channel's BN variance)? — the leave-one-out (LOO) diagnostic.
- Does the argument's preferred fix (**per-channel winsorization** in the accumulator) match our
  **sample-level MAD rejection** under injected outliers?

## 3a — LOO variance influence (`loo_sensitivity.py`)
For each batch, compute per-channel BN-input μ_c, σ²_c at all 5 BN layers over all windows, then
recompute leaving out each window, and report `max_c |Δσ²_c/σ²_c|`. Captured on the real, clean data.
- If dropping any window shifts a variance by >5–10 % → an influential/outlier-like window exists.
- If <1 % → pooling has diluted it; nothing to worry about on natural data.
(Note: deeper BN layers pool fewer observations → expected to be more exposed than BN0.)

## 3b — winsorize vs reject (`winsorize_vs_reject.py`)
Inject ×8 spikes into a fraction f of the collection windows (as exp1b), eval on the CLEAN batch,
compare three collectors: `standard`, `reject` (drop windows > median+3·MAD), `winsorize` (clip each
BN-input observation to its per-channel [0.5, 99.5] percentile before the μ/σ² accumulator).

## Results
_(to fill: 3a per-BN-layer max LOO shift on natural data; 3b accuracy vs f for the three methods.)_

## Findings
_(to fill: are there natural influential windows? does winsorize match reject?)_

## Progress log
- 2026-07-22: runners built; LOO (S01,S02) + winsorize (S01,S02) launched.
