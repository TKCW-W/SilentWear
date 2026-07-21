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

### 3a — LOO max |Δσ²/σ²| per BN layer, over folds×batches (`results/loo_*.csv`)

| BN | S01 median | S01 max | S02 median | S02 max |
|---|---|---|---|---|
| 0 | 3.2% | 77.7% | 1.9% | 78.9% |
| 1 | 4.0% | 59.8% | 2.5% | 73.3% |
| 2 | 4.0% | 62.9% | 3.3% | 78.7% |
| 3 | 7.0% | 77.5% | 6.3% | 87.4% |
| 4 | 6.5% | 83.0% | 5.2% | 86.3% |

### 3b — standard vs reject vs winsorize under ×8 injection (`results/wins_*.csv`)

| f | S01 std | S01 reject | S01 wins | S02 std | S02 reject | S02 wins |
|---|---|---|---|---|---|---|
| 0% | 85.19 | 85.42 | 68.52 | 63.66 | 63.75 | 44.40 |
| 5% | 83.85 | 85.42 | 60.07 | 59.14 | 63.73 | 25.87 |
| 10% | 82.51 | 85.27 | 37.14 | 53.19 | 63.56 | 13.11 |
| 20% | 77.22 | 85.21 | 16.56 | 40.72 | 63.33 | 11.11 |

## Findings — verdict on the outlier-dilution argument

- **On average the dilution argument holds:** median single-window LOO influence is only 2–7 %, so a
  typical batch has no influential window. **But it fails in the tail:** the *max* natural influence is
  77–87 % (some batches contain a genuinely influential window), and deeper BN layers (BN3/BN4) are more
  exposed (fewer pooled observations) — refuting "one outlier is diluted to insignificance". Caveat: the
  raw Δσ²/σ² is inflated by near-zero-variance channels (ε-mirror), so 77–87 % is an upper bound — which
  is exactly why an ε floor matters.
- **Sample-level MAD rejection is the right fix** (immune across all injection, both subjects; no cost on
  clean data) — confirms exp1b.
- **Observation-level winsorization ([0.5,99.5] pct, as the argument preferred) does NOT match it — it
  fails.** It hurts even at 0 % contamination (S01 85→69, S02 64→44) because clipping legitimately
  heavy-tailed EMG activations removes real variance → BN over-normalizes. So the argument's preferred
  fix is worse than dropping artifact windows; **keep MAD-reject.** (A much gentler clip might behave
  better, but reject already works perfectly, so there is no reason to tune winsorization.)
- **Net decision unchanged and reinforced:** collect on ≈32 windows (Exp 2) with **sample-level MAD
  rejection** (Exp 1b/3b), plus an **ε floor** (Exp 3a) as cheap insurance for small-variance channels.
  AdaBN's per-session reset already caps any single bad session's blast radius.

## Progress log
- 2026-07-22: runners built; LOO (S01,S02) + winsorize (S01,S02) launched.
