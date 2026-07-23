# Pre-Deployment Analysis — on-device fine-tuning recipes for SpeechNet (SilentWear)

**Date:** 2026-07-23
**Purpose:** Consolidate the five on-device fine-tuning settings we studied on the PyTorch host, report
each one's *best* configuration and its S01 vocalized accuracy (per batch, averaged over the 3
leave-one-session-out folds) against the paper, pick the setting to deploy, and explain for every other
setting why — despite training more parameters or adapting statistics — it does **not** beat the
simplest recipe (head-only). All numbers: SpeechNet, S01 vocalized, inter-session incremental FT,
30 % of each batch (54 windows, 6/class, seed 42), 40 epochs, plain SGD (no momentum, no weight decay),
effective batch 1 with n_accum SUM accumulation.

---

## Results table — S01 vocalized, per batch, mean of 3 folds (balanced accuracy %)

| batch | no-FT | **1. head-only + BN-fold** | **2. streaming AdaBN** | **3. full, frozen-pretrained stats** | **4. full, BN folded into Conv** | **5. full, batch-1 live-BN** | paper (batch-32 Adam ref) |
|---|---|---|---|---|---|---|---|
| 1 | 72.59 | 72.59 | 72.59 | 72.59 | 72.59 | 72.59 | 72.59 |
| 2 | 74.63 | 88.52 | 86.30 | 86.67 | 84.44 | 72.22 | 87.22 |
| 3 | 77.04 | 83.33 | 84.07 | 83.33 | 83.33 | 66.67 | 87.96 |
| 4 | 77.96 | 85.93 | 89.26 | 89.44 | 88.33 | 57.96 | 90.37 |
| 5 | 65.56 | 80.93 | 82.04 | 83.15 | 77.22 | 46.11 | 87.41 |
| **mean b2–5** | **73.80** | **84.68** | **85.42** | **85.65** | **83.33** | **60.74** | **88.24** |

Batch 1 is the zero-shot (no-FT) accuracy for every setting — FT starts on batch 1's data and is first
evaluated on batch 2. Settings 1–3 land within ~1 pp of each other (a statistical tie at 3-fold noise
±1–4 pp); all sit ~3–4 pp below the paper. Setting 4 is ~1.5 pp lower; setting 5 collapses.

---

## Setting-by-setting

### 1. Head-only + BN-fold  →  **DEPLOYMENT PICK**
- **Config (best):** freeze the whole feature extractor, fold the frozen pretrained (sessions 1+2) BN
  into its Conv (inference-equivalent, BN-free graph), train **only the linear head `fc`** (~297 params).
  SGD **lr = 0.01, n_accum = 4**, 40 epochs, 30 % data.
- **Accuracy:** mean b2–5 = **84.68** (paper 88.24 → −3.56 pp). Reliability: 10-seed sweep
  **85.54 ± 0.81** (`s2_vs_headonly_seedsweep`), and across 4 subjects head-only (76.83) ≥ streaming
  AdaBN (76.54) — so it is not an S01 fluke.
- **What it solves:** the batch-1 × BatchNorm catastrophe (there is no live BN — features and their
  normalization are frozen and folded), overfitting (297 params can't overfit 54 windows), and the
  activation↔stat mismatch (features never move, so the folded stats stay exactly valid). Near-convex,
  well-conditioned optimization → tolerates a high lr, converges fast.
- **What it does NOT solve:** it performs **no feature adaptation** — the conv features are exactly the
  pretrained ones. That is the ~3.5 pp it concedes to the paper (see the cross-cutting analysis below).
- **Why deploy it:** ties or beats every other on-device setting, is by far the cheapest (train one
  linear layer, no BN kernels, no backprop through the conv stack), lowest variance, and needs none of
  the un-portable machinery (recollection kernel, live BN, lr re-tuning).

### 2. Streaming AdaBN (recollect stats per batch, use previous batch's stats at inference)
- **Config (best):** full model trainable; **BN running stats recollected** (forward-only, label-free
  population pass) on each batch and **frozen** during that batch's FT; inference on batch b uses the
  stats recollected on batch b−1 (deployment-realistic — never peeks at b). SGD **lr = 3e-4, n_accum = 8**,
  40 epochs, 30 % data.
- **Accuracy:** mean b2–5 = **85.42** (paper −2.82 pp). (The *transductive* variant — recollect on the
  eval batch itself — is optimistic at 87.78; streaming is the honest number.)
- **What it solves:** the **inference-time** normalization shift — recollecting population stats on the
  target session restores correct normalization at test time (this is the bulk of AdaBN's gain, and it
  is label-free). Marginally edges head-only on S01 because the recollected stats track the session.
- **What it does NOT solve:** the **training-time** normalization. It recollects *before* FT then freezes;
  as the conv moves during FT the activations drift from those frozen stats → the feature adaptation it
  attempts is corrupted. Net: its full-model FT adds almost nothing over recollection, so it **ties**
  head-only (4-subj: 76.54 vs 76.83). Also un-portable as-is: needs an on-device recollect-stats kernel
  and BN stats fed as graph inputs.
- **Why it doesn't beat head-only:** its extra capacity (training the conv) is wasted — see cross-cutting
  analysis. Recollection ≈ head-only's classifier re-fit as a route to session adaptation; they hit the
  same ceiling.

### 3. Full training with frozen pretrained running stats
- **Config (best):** all params (conv + BN affine + fc) trainable; BN **frozen at the pretrained
  (sessions 1+2) stats** for both training and inference (no recollection). SGD **lr = 3e-4, n_accum = 4**
  (eff_lr 0.0012), 40 epochs, 30 % data. (Grid top of `frozen_stat_full_training/sweep_s2_full`: 86.67
  at b1→b2.)
- **Accuracy:** mean b2–5 = **85.65** (paper −2.59 pp) — nominally the best of settings 1–3, but inside
  3-fold noise of head-only, and over 10 seeds it is 84.76 ± 0.70 vs head-only 85.54 ± 0.81 (paired
  −0.78, head-only wins 8/10).
- **What it solves:** nothing head-only doesn't — it keeps normalization stable by never recollecting,
  and can in principle move features.
- **What it does NOT solve:** productive feature adaptation. Training the conv against *frozen* stats
  creates the activation↔stat mismatch, and 16 K params overfit 54 windows. The low optimal lr (3e-4,
  ~33× below head-only's 0.01) is a **symptom** of this fragility, not a fixable cause — raising it
  strictly hurts (eff_lr ≳0.02 → collapses toward chance). So it ties head-only at ~33× the cost.
- **Why it doesn't beat head-only:** more trainable capacity is a *liability* under distribution shift +
  54 samples (Kumar et al., ICLR 2022: full FT distorts good pretrained features; linear probing =
  head-only matches/beats it). Confirmed: exp8 shows it only *catches up* to head-only as data grows.

### 4. Full training with BN folded into Conv
- **Config (best):** fold frozen pretrained BN into Conv (BN-free graph, verified lossless — outputs
  differ 2e-6), then full-train the **folded conv + fc**. SGD **lr = 3e-6** (≈100× lower than setting 3),
  n_accum = 4, 40 epochs, 30 % data.
- **Accuracy:** mean b2–5 = **83.33** (paper −4.91 pp). At the unfolded lr (3e-4) it **collapses to
  31.5**; it only recovers at lr ~3e-6.
- **What it solves:** removes all BN kernels from the *training* graph (cleanest BN-free on-device
  training path) while remaining mathematically the same function class as setting 3.
- **What it does NOT solve:** it is **not a drop-in** — BN, even frozen, provides implicit per-layer
  learning-rate scaling (its scale γ/√(var+ε) ≈ 0.02–0.03 damps conv gradients because EMG is
  unnormalized, var in the thousands). Folding removes that damping → effective lr ~1000× too large →
  collapse unless the lr is re-tuned ~100× lower. Even tuned it only *ties/slightly-trails* head-only.
- **Why it doesn't beat head-only:** it is the same recipe as setting 3 up to reparameterization, so it
  inherits the same feature-adaptation ceiling — plus a fragile lr. No upside over head-only.

### 5. Full training with batch-1 live BN, inference with frozen pretrained stats (the naive on-device baseline)
- **Config (best of a bad lot):** all params trainable; during FT, BN normalizes with the **current
  single-batch (batch-1) statistics** (live BN at batch size 1); inference uses frozen pretrained stats.
  Best knobs SGD lr = 3e-4, n_accum = 8 (80.0 at b1→b2), but streaming it **degrades every batch**.
- **Accuracy:** mean b2–5 = **60.74** (paper −27.5 pp) — a collapse: 72→67→58→46 across batches.
- **What it solves:** nothing — it is the setting that motivated the whole study.
- **What it does NOT solve:** the fundamental **batch-1 × BatchNorm** interaction. A batch of 1 gives a
  degenerate mean/var, so the live-BN forward normalizes with garbage statistics, the gradients are
  meaningless, and each FT round drives the model further from the frozen pretrained stats used at
  inference (train/inference normalization mismatch grows). This is the −29 pp factor that every other
  setting is designed to avoid.
- **Why it's here:** it is the baseline that proves *why* settings 1–4 freeze / fold / recollect BN in
  the first place.

---

## Cross-cutting analysis — why nothing beats head-only, and where the paper's edge is

**Settings 1, 2, 3 tie (~85 %) and all trail the paper (~88 %) by ~3 pp.** The reason is a single
structural fact:

- **Head-only, streaming AdaBN, and frozen-stat full-FT all reach the same session-adaptation ceiling by
  different routes** (classifier re-fit; stat recollection; frozen-stat full FT). That ceiling is "how
  much can you recover from a session shift with 54 windows *without adapting features*." All three hit it.
- **The paper's extra ~3 pp is feature adaptation** — its conv learns new session-specific features. It
  can do this because it uses **batch-32 live BatchNorm**, which does two jobs: (a) correct *inference*
  normalization — which AdaBN also achieves — and (b) correct *training-time* normalization, where the BN
  stats track the shifting activations **live every batch** so the conv adapts in a correctly-normalized
  regime. Job (b) is the un-portable one: batch-1 cannot produce live stats (degenerate), and holding 32
  windows' activations for a batch statistic blows the L1/L2 budget. So on-device we can adapt the
  classifier or the statistics, but **never co-adapt features and normalization jointly** — which is
  exactly the paper's advantage.
- **Regularization cannot close it** (exp12): weight decay (the only cleanly-portable regularizer) gives
  +0.27 pp = noise; dropout is unreliable and needs a Deeploy op build; early-stopping/LR-schedule are
  not portable (compile-time-fixed loop, compile-time-constant lr). The gap is not overfitting — it is a
  structural inability to train features correctly at batch 1.
- **More epochs cannot close it** (exp11): past ~40 epochs the recipe overfits monotonically (train loss
  →0, held-out 85.4→83.8). SGD is not under-trained; the ceiling is generalization, not optimization.

**Conclusion:** head-only + BN-fold is the deployment recipe — it ties the best on-device setting, is the
simplest and cheapest, has the lowest variance, and avoids every un-portable ingredient (live BN,
recollection kernel, lr re-tuning). The residual ~3.5 pp to the paper is the paper's un-portable
batch-32-live-BN feature adaptation, which no batch-1 on-device recipe can reproduce.

## UPDATE (exp13) — a better middle ground: last-block+fc supersedes head-only

After this 5-setting analysis, a progressive-unfreezing sweep (exp13) found a genuine sweet spot
*between* head-only and full training: **unfreeze only the LAST conv block + fc** (frozen pretrained BN,
lr 1e-3, else identical recipe). It beats head-only on **all 4 subjects**:

| | S01 | S02 | S03 | S04 | 4-subj mean | vs paper (~80.02) |
|---|---|---|---|---|---|---|
| head-only (fc only) | 84.95 | 65.88 | 75.00 | 84.17 | 77.50 | −2.5 |
| **last-block + fc (K=1)** | **86.48** | **68.70** | **76.85** | **84.49** | **79.13** | **−0.9** |
| full (K=5) | 85.32 | — | — | — | — | |

- **+1.63 pp over head-only (4-subj mean), winning on every subject**, and it **nearly closes the gap to
  the paper** (~0.9 pp vs head-only's ~2.5 pp) — the best on-device result we have.
- **Why K=1 and not more:** the last block holds the most session-specific features; adapting just it
  captures the feature shift the classifier alone can't, while keeping blocks 0–3 frozen avoids the
  overfitting + activation↔stat mismatch that caps K≥2 and full training (which fall back to ~85/≈77).
- **This revises the "~3.5 pp to the paper is un-portable" conclusion above:** most of that gap was
  *recoverable* by adapting one block — it was a capacity/scope choice, not purely the batch-32-BN wall.
- **Deployable with no graph change** via per-parameter grad-buffer masking (zero every grad-accum buffer
  except block-4 conv + fc before the optimizer step; exact freeze under plain SGD-no-wd), at the cost of
  one extra block's backward. See `exp13_progressive_unfreeze/FINDINGS.md`.

**CORRECTION → FINAL (exp13c+d, 10 seeds/subject):** the K=1 numbers above are single seed-42 draws and
were misleading. Multi-seeded, K1 − K0 is **within noise on 3 of 4 subjects** and a **large robust win on
exactly one**:

| | S01 | S02 | S03 | S04 |
|---|---|---|---|---|
| head-only K0 | 85.91 | 65.58 | 74.79 | 84.01 |
| K1 − K0 (10 seeds) | +0.20±0.57 | **+4.50±0.71 (10/10)** | +0.37±0.60 | +0.25±0.88 |

**The K=1 gain scales inversely with head-only's accuracy:** it rescues +4.5 pp on S02 (head-only only
66% — features genuinely session-shifted) and adds ≈0 where head-only already ≥75%.

**Deployment pick — FINAL: head-only + BN-fold** (simplest, lowest variance, and last-block FT adds
nothing on 3/4 subjects). **Last-block+fc (K=1, lr 1e-3) is a targeted fallback** — enable it only for a
subject where head-only underperforms, where it can recover several pp; cheaply deployable via grad-buffer
masking (no graph change), so an adaptive "if accuracy low, unfreeze last block" policy is realistic.
(Methodological note: S01's +1.53 and S03's +1.85 single-seed "wins" both washed out under 10 seeds — only
S02's survived. Always multi-seed before concluding.)

## Source files
Setting 1: `exp7_headonly_4subj/results/headonly_S01.csv`, `s2_vs_headonly_seedsweep/`. Setting 2:
`exp5_streaming_adabn_incremental/results/streaming_incr_S01.csv`. Settings 1/3/5 per-batch:
`frozen_stat_full_training/results/comparison_full_S01_vocalized.csv` (+ `sweep_s2_full.csv`,
`results/livebn_vs_adabn_S01.csv`). Setting 4: `exp9_fulltrain_foldedBN/results/folded_lr3e-6.csv`.
Supporting: `exp11_extended_epochs/FINDINGS.md`, `exp12_portable_regularization/FINDINGS.md`,
`exp8_fullmodel_datasize/`, `results/ONDEVICE_FACTOR_ATTRIBUTION_FINDINGS.md`.
