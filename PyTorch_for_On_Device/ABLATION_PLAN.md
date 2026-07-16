# Factor-attribution study: which on-device factor costs the most fine-tuning gain?

**Status:** IN PROGRESS · started 2026-07-16 · Subject S01, vocalized, 3 folds

## Goal
Decompose the fine-tuning-gain gap between the **paper's FT recipe** and **our on-device recipe**
by flipping **one factor at a time** from the paper baseline to its on-device setting, and measuring
how much each flip costs. The flip with the largest drop = the most problematic on-device factor.

## Method
- Incremental inter-session FT protocol (`run_inter_session_ft` style): per fold K, walk batches
  1→5 of the held-out session, carry the model forward, FT on batches 1–4, eval on the whole
  (rest-downsampled, 180-window) batch. Metric = **balanced accuracy**, mean ± std over the 3 folds.
- Headline number per config = **mean over the fine-tuned batches (2–5)**.
- Base = `inter_session_ft/S01/vocalized/.../leave_one_session_out_fold_{1,2,3}.pt`; onset windowing.
- `no_ft` (base model, no FT) is identical across all configs → comparing FT accuracy = comparing gain.
- One parameterized runner (`recipe_runner.py`); each experiment is one config dict.

## Endpoints
- **P0 — paper recipe:** full model · Adam (lr 1e-3, wd 1e-4) · ReduceLROnPlateau · batch 32 ·
  live BatchNorm (train-mode batch stats) · dropout 0.5 · 70/30 split + early stop (patience 10, ≤50 ep) · mean reduction.
- **ODE — our on-device recipe:** head-only · SGD (lr 0.01, no momentum, no wd) · static lr · batch 1 ·
  frozen BN (folded) · no dropout · 30% train, fixed 40 ep, no early stop · (sum accumulation).

## Single-factor flips (each = P0 with exactly ONE aspect set to on-device)
| ID | factor | P0 (paper) | flipped to on-device |
|----|--------|-----------|----------------------|
| E1 | optimizer | Adam lr1e-3 wd1e-4 | SGD lr0.01, no momentum, no wd |
| E2 | LR schedule | ReduceLROnPlateau | static (none) |
| E3 | batch size | 32 | 1 |
| E4 | BatchNorm | live (train-mode batch stats) | frozen pretrained stats (full-model kept) |
| E5 | trainable scope | full model | head-only (fc; conv+BN frozen) |
| E6 | dropout | 0.5 | 0.0 |
| E7 | data & stopping | 70% train + val early-stop (≤50 ep) | 30% train, fixed 40 ep, no val |

Notes / caveats:
- E1 bundles optimizer type + lr + wd + momentum (they come together on-device); a small SGD-lr
  sweep guards against an unfair lr.
- E3 is batch-1 with the paper's *live* BN — this is where batch size actually bites (each window
  normalized by its own stats). No n_accum (pure batch-1); n_accum is our compensation, studied separately if needed.
- E5 (head-only) necessarily freezes BN too; E4 isolates the BN-freeze alone (full model kept).
- Factors interact (batch-1 × live-BN especially) — OAT gives per-factor contribution along the
  paper→on-device path; ODE (all flips) shows the compounded result.

## Results (balanced acc, mean±std over 3 folds; headline = mean over batches 2–5)

| config | b2 | b3 | b4 | b5 | **mean(2–5)** | Δ vs P0 |
|--------|----|----|----|----|---------------|---------|
| P0 (paper) | | | | | | 0 |
| E1 optimizer→SGD | | | | | | |
| E2 schedule→static | | | | | | |
| E3 batch 32→1 | | | | | | |
| E4 BN→frozen | | | | | | |
| E5 scope→head-only | | | | | | |
| E6 dropout→0 | | | | | | |
| E7 data→30%/fixed | | | | | | |
| ODE (all on-device) | | | | | | |

_(official paper ft_summary mean(2–5) ≈ 88.2%; our shipped head-only ≈ 84.7%)_

## Analysis
_(to fill: rank factors by Δ vs P0; identify the most problematic; note interactions.)_

## Progress log
- 2026-07-16: plan written; parameterized runner built; P0 + flips launching.
