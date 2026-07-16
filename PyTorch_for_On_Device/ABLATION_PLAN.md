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
| **P0 (paper)** | 85.9 | 88.5 | 90.6 | 87.6 | **88.15** | 0 |
| E6 dropout→0 | 85.4 | 90.6 | 91.5 | 88.0 | 88.84 | **+0.69** |
| E1 optimizer→SGD | 86.7 | 87.8 | 89.6 | 87.2 | 87.82 | −0.33 |
| E2 schedule→static | 86.3 | 85.9 | 88.9 | 88.3 | 87.36 | −0.79 |
| E7 data→30%/fixed | 88.0 | 88.5 | 86.3 | 85.0 | 86.94 | −1.21 |
| E4 BN→frozen (full) | 83.0 | 87.2 | 89.4 | 85.0 | 86.16 | −1.99 |
| E5 scope→head-only | 85.6 | 85.2 | 88.5 | 80.7 | 85.00 | −3.15 |
| **E3 batch 32→1** | 58.0 | 64.8 | 58.2 | 55.4 | **59.07** | **−29.08** 💥 |
| ODE (all on-device) | 87.6 | 88.0 | 87.2 | 82.0 | 86.20 | −1.95 |
| E3b batch1 + frozen-BN (full) | 80.7 | 86.5 | 86.5 | 83.3 | 84.26 | −3.89 |
| E3c batch1 + head-only | 84.4 | 85.2 | 88.0 | 81.3 | 84.72 | −3.43 |

**Confirmation:** E3 (batch1, live BN) = 59.1, but E3b (batch1, **frozen** BN) = 84.3 and E3c
(batch1, head-only) = 84.7 — freezing BN recovers **~25 pp** of the batch-1 collapse. So the
catastrophe is entirely the **live BatchNorm at batch-1**, not batch-1 gradients or the optimizer.

_(official paper ft_summary mean(2–5) ≈ 88.2%; our shipped head-only w/ n_accum ≈ 84.7%)_

## Analysis
**The single most problematic factor is batch size (32→1): −29 pp.** Every other individual
factor is within ±3 pp (dropout→0 even *helps* slightly). BUT the batch-size catastrophe is an
**interaction, not a standalone effect**: E3 collapses only because it keeps the paper's
**full-model live BatchNorm** — at batch 1 each window is normalized by its own stats, which
corrupts both the forward and the running-stat EMA, wrecking full-model training.

The proof is in **ODE (all flips together) = 86.2, not catastrophic**: ODE is also batch-1, but it
includes **head-only + frozen BN**, which remove BN's dependence on the batch entirely — so batch-1
becomes harmless. In other words, our on-device recipe's head-only + BN-fold choices are precisely
the **mitigation** that defuses the batch-1 bomb.

So the honest decomposition of the ~2–3 pp on-device gap (once batch-1 is survived):
- freezing BN (E4): −1.99 pp  ·  going head-only (E5): −3.15 pp  ← these two are the real "cost",
  and they are *forced by* the batch-1 constraint (you can't run live full-model BN at batch 1).
- optimizer (SGD), static lr, data fraction, dropout: each ≲ 1 pp, some positive → **negligible**.

Ranking (most→least problematic on-device): **batch-1×live-BN (catastrophic) ≫ head-only scope
(−3.2) > frozen BN (−2.0) > data/length (−1.2) > lr schedule (−0.8) > optimizer SGD (−0.3) >
dropout (+0.7, helps)**.

E3b/E3c confirm the causal claim: batch-1 with **frozen** BN (full or head) should recover to ~85+,
proving the collapse is the *live BN at batch 1*, not batch-1 gradients or the optimizer.

## Progress log
- 2026-07-16: plan written; parameterized runner built; P0 + flips launching.
