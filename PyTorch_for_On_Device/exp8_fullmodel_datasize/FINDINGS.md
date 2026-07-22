# Exp 8 findings — does more fine-tuning data close the head-only vs full-model gap?

**Date:** 2026-07-22
**Purpose:** This file summarises whether giving full-model fine-tuning (frozen pretrained BN stats)
more data lets it catch head-only — testing the overfitting explanation for why head-only ties/beats
full-model on tiny data (see `../exp7_headonly_4subj/HEADONLY_VS_FULLMODEL_FINDINGS.md`).

## Setup
Incremental streaming protocol, frozen pretrained BN stats (no recollection), S01 vocalized, 3 folds,
single seed. Both recipes at data fractions 30/50/70/100 % (6/10/14/20 windows per class). head =
fc-only (lr 0.01); full = full-model frozen-stat (lr 3e-4). Metric: balanced acc, mean over b2–5.

## Results (`results/datasize_summary.csv`)

| data % | head-only | full-model | full − head |
|---|---|---|---|
| 30 | 84.68 | 85.65 | +0.97 |
| 50 | 85.93 | 85.23 | −0.70 |
| 70 | 86.30 | 88.10 | +1.80 |
| 100 | 86.57 | 86.76 | +0.19 |

## Findings
1. **More data helps both recipes** — head-only rises 84.68 → 86.57 (+1.9 pp) with data; full-model
   rises too (noisier).
2. **The head↔full gap tends to close as data grows** — full-model is *relatively* best at 70 % (+1.8)
   and **ties** head-only at 100 % (86.8 vs 86.6), whereas at 30 % the 10-seed s2_vs_headonly test had
   head-only ahead (85.54 vs 84.76). Directionally consistent with **"full-model overfits on tiny data,
   catches up with more"** (Kumar et al. 2022, fine-tuning vs linear probing under OOD + limited data).
3. **Tentative, not conclusive** — single-seed noise (~±1–2 pp) is large: the 50 % point (full < head)
   and the non-monotonic full-model curve show the trend isn't clean; the 30 % single-seed row here even
   flips the sign vs the 10-seed 30 % result. A ≥8-seed re-run per (recipe, fraction) is needed to firm it.

## Verdict / deployment takeaway
The overfitting explanation is **directionally supported** (more data narrows the gap; the two converge
to ~equal by 100 %), but the single-seed noise prevents a firm claim. The **robust** conclusion is that
**even at 100 % data full-model only *ties* head-only — it never clearly beats it** — so head-only
remains the simpler, equal-or-better on-device choice. (And per
`TrainDeeploy/.../ONDEVICE_TRAINING_DATA_MEMORY.md`, 70 %+ data would also need the L3-offload path on
the 4 MB WEIGHTMEM_SRAM, so there's little incentive to push past ~50 % on-device anyway.)

## Follow-up (optional)
Multi-seed (≥8) at 30/70/100 % to make the "gap-closes-with-data" trend statistically clean.
