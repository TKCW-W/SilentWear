# On-device fine-tuning: which factor costs the gain? — findings

**Subject S01, vocalized · 3 folds · complete incremental inter-session FT · PyTorch host**
Base = `inter_session_ft/S01/.../leave_one_session_out_fold_{1,2,3}.pt`, onset windowing.
Metric = balanced accuracy, mean over the 4 fine-tuned batches (b2–b5) averaged across 3 folds.

## Question
The paper fine-tunes with a full-model recipe (Adam, batch 32, live BatchNorm, …); on-device we
use a constrained recipe (SGD, effective batch 1, head-only + folded BN, …). **Which of these
differences actually costs the fine-tuning gain?** We flip **one paper setting at a time** to its
on-device value and measure the drop from the paper baseline (P0). Largest drop = most problematic.

## Recipes
- **P0 (paper):** full model · Adam (lr 1e-3, wd 1e-4) · ReduceLROnPlateau · **batch 32** ·
  **live BatchNorm** · dropout 0.5 · 70/30 split + early stop.
- **ODE (on-device):** head-only · SGD (lr 0.01, no momentum/wd) · static lr · **batch 1** ·
  **frozen/folded BN** · no dropout · 30% data, fixed 40 ep.

## Results (mean over b2–b5, Δ vs P0 = 88.15)

| flip (paper → on-device) | mean(b2–5) | Δ vs P0 |
|---|---|---|
| E6  dropout 0.5 → 0 | 88.84 | **+0.69** (helps) |
| E1  optimizer Adam → SGD (no momentum/wd) | 87.82 | −0.33 |
| E2  lr schedule → static | 87.36 | −0.79 |
| E7  data 70%→30%, early-stop → fixed 40 ep | 86.94 | −1.21 |
| E4  BatchNorm live → frozen (full model) | 86.16 | −1.99 |
| E5  scope full → head-only | 85.00 | −3.15 |
| **E3  batch 32 → 1** (full model, live BN) | **59.07** | **−29.08** 💥 |
| ODE  all flips together | 86.20 | −1.95 |
| — confirmation — | | |
| E3b batch 1 + **frozen** BN (full model) | 84.26 | −3.89 |
| E3c batch 1 + head-only | 84.72 | −3.43 |

## The finding: it's batch-1 destroying BatchNorm — nothing else

1. **Batch size is the only catastrophic factor (−29 pp).** Every other single flip is within ±3 pp;
   dropping dropout even helps slightly.
2. **But the catastrophe is an interaction, not a standalone effect.** E3 collapses only because it
   keeps the paper's **full-model live BatchNorm**: at batch 1 each window is normalized by its own
   statistics, corrupting the forward and the running-stat EMA and wrecking full-model training.
3. **Freezing BN removes the catastrophe** (the causal proof):
   - E3 batch-1 + **live** BN = 59.1
   - E3b batch-1 + **frozen** BN (full) = 84.3  ·  E3c batch-1 + head-only = 84.7
   - → freezing BN recovers **~25 pp**. So it is the *live BN at batch 1*, not batch-1 gradients or
     the optimizer.
4. **ODE (all flips together) = 86.2, not catastrophic** — because ODE already includes head-only +
   frozen BN, which make batch size irrelevant to BN. Our on-device recipe's head-only + BN-fold
   choices are exactly the **mitigation** that defuses the batch-1 bomb.

## Decomposition of the ~2–3 pp residual on-device gap (once batch-1 is survived)
The real cost is the **mitigations forced by batch-1**, not the other recipe differences:
- freeze BN (E4): −1.99 pp · go head-only (E5): −3.15 pp — you cannot run live full-model BN at
  batch 1, so these are mandatory.
- optimizer (SGD) −0.33 · lr schedule −0.79 · data/length −1.21 · dropout **+0.69** → all ≲1 pp,
  **negligible**.

**Ranking (most → least problematic on-device):**
`batch-1 × live-BN (catastrophic) ≫ head-only scope (−3.2) > frozen BN (−2.0) > data/length (−1.2)
> lr schedule (−0.8) > optimizer SGD (−0.3) > dropout (+0.7, helps)`.

## Practical implications
- **Don't waste effort tuning the optimizer, lr schedule, or dropout for on-device** — they cost ~0.
- **The entire problem is BatchNorm at batch-1.** Any fix that removes BN's batch dependence recovers
  the bulk of the gain. Options we evaluated:
  - **head-only + BN-fold** (our shipped recipe): frozen BN, only the head trains → batch-1 safe,
    lands ~2–3 pp under the paper. *Simplest, works.*
  - **frozen-BN full-model FT** (E3b): batch-1 safe, allows full-model adaptation, ~84–86 (see also
    `idea1_bn_stats` — precomputed/AdaBN stats reach paper level at a lower lr).
  - **batch-independent norms** (GroupNorm/LayerNorm/InstanceNorm; `idea2_alt_norm`): no batch stats
    at all, but require retraining and **underperform BN** here (LayerNorm best at 82.2 vs BN 87.8).

## Related studies (same directory)
- `idea1_bn_stats_S01_vocalized.txt` — precomputed/AdaBN BN stats: test-time AdaBN gives +10 pp free;
  precomputed-stat full FT reaches paper level (87.6 @ lr 3e-4) but doesn't beat head-only.
- `idea2_alt_norm_S01_vocalized.txt` — replacing BN with GN/LN/IN: none beats BN+head-only.
- `ablation_results.csv`, `abl_E3b.csv`, `abl_E3c.csv` — raw per-config numbers.
- `../ABLATION_PLAN.md` — full plan, per-batch tables, method.
