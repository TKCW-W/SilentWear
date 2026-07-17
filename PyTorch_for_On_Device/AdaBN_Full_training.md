# AdaBN + full-model on-device fine-tuning — does adapting BN stats unlock full training at batch-1?

**Status:** IN PROGRESS · started 2026-07-16 · Subject S01, vocalized, 3 folds · PyTorch host

## Idea
On-device the training BN kernel normalizes each window by its own (degenerate) single-window
stats, and never updates the running stats — so full-model FT breaks and inference uses frozen
pretrained stats (train/inference mismatch). **AdaBN** fixes the normalization without needing a
real batch: collect population stats over many samples in a separate forward-only pass, freeze
them, and train with those.

## Procedure (two passes per FT round — hence a new incremental runner)
For each fine-tuning round on a batch:
1. **Collect (forward-only, label-free):** run all of the batch's windows one at a time,
   accumulate per-channel μ, σ² across them, and write them into the BN running-stat buffers.
   (On host this equals a single population-stat pass over all windows; the one-at-a-time
   accumulation is the on-device mechanism, same result.) Uses the current (pre-training) weights.
2. **Freeze stats, full fine-tune:** hold μ, σ² fixed (BN in eval mode using the collected stats)
   and train **conv + BN γ,β + fc (full model)** with the on-device recipe:
   SGD (no momentum/wd), **fixed lr**, fixed epochs (40), **30% data**, batch-1 + **n_accum** (sum).

No BN folding — BN stays as a layer, but its stats are the frozen collected ones (batch-independent,
so batch-1 is fine). Inference/eval uses those same frozen collected stats.

## What we sweep
On the **b1→b2** transition first (batch 1 as the first step), 3 folds:
- **n_accum** ∈ {1, 4, 8, 16, 32}
- **lr** ∈ {3e-2, 1e-2, 3e-3, 1e-3, 3e-4}
(with sum accumulation, effective step ≈ lr × n_accum, so the grid spans a wide effective-lr range.)

For each config we report two eval variants to expose the staleness of the pre-training stats:
- **ft_frozen** — eval with the stats collected *before* training (the literal procedure).
- **ft_recollect** — re-collect stats on the FT data at the *final* weights before eval
  (deployment-realistic: the inference graph gets stats from a forward pass at the deployed weights).

## References (b1→b2, S01, 3 folds)
- our head-only + BN-fold recipe: **88.5%**
- paper full-model Adam FT: **87.2%**
- idea1 precomputed-stat full FT (mean reduction): 87.6% @ lr 3e-4 (did not beat head-only)
- naive batch-1 full FT with live BN (ablation E3): **59%** (collapse)

## Goal / question
Does AdaBN (collect+freeze+full-train) let batch-1 **full-model** FT reach or beat head-only /
paper? If yes at some (n_accum, lr), full on-device training is viable with a cheap forward-only
stat-collection kernel. Then run the full incremental (all rounds, 3 folds) with the best config.

## Results — b1→b2 (n_accum × lr) sweep, mean±std over 3 folds
Full grid in `results/adabn_sweep_full.csv`. Top configs (by ft_recollect):

| n_accum | lr | eff_lr | ft_frozen | ft_recollect |
|---|---|---|---|---|
| **8** | **3e-4** | 0.0024 | 86.3 | **87.2** |
| 4 | 3e-4 | 0.0012 | 86.1 | 86.1 |
| 1 | 3e-4 | 0.0003 | 86.1 | 86.1 |
| 4 | 1e-3 | 0.0040 | 85.0 | 85.7 |
| 16 | 3e-4 | 0.0048 | 84.1 | 84.3 |
| … eff_lr ≥ 0.04 | | | 11.1 | 11.1 (collapse) |

refs: naive batch-1 full (live BN) = **59** · paper = **87.2** · head-only = **88.5**

## Analysis
- **AdaBN unlocks batch-1 full-model FT:** 59% (naive) → **87.2%** (best), matching the paper's
  full FT exactly. The collect-freeze-train scheme genuinely fixes the batch-1 BN problem.
- **But it does not beat head-only (88.5)** on b1→b2, and it's costlier (full-model training + a
  forward-only stat-collection pass).
- **Best config: n_accum=8, lr=3e-4** (eff_lr ≈ 0.0024). **Very lr-sensitive** — only low effective
  lr (~0.001–0.005) is stable; eff_lr ≳0.02 destabilizes, ≳0.04 collapses to chance (sum-accum lr coupling).
- **Staleness is modest at low lr:** re-collecting stats at the final weights gives ~+1 pp (best:
  86.3→87.2). At high lr the pre-training stats would be badly stale, but those lrs collapse anyway.

## Phase B — full incremental with the best config (n_accum=8, lr=3e-4)
Two-pass incremental (`run_inter_session_ft_adabn.py`), 3 folds, all rounds. Per round: collect
BN stats on the batch (adapt) → eval → freeze → full-train on 30%. Eval re-collects stats on the
eval batch (test-time AdaBN, forward-only).

| batch | no-FT | head-only | **AdaBN-full** | paper |
|---|---|---|---|---|
| 1 (no FT) | 72.59 | 72.59 | **82.04** | 72.59 |
| 2 | 74.63 | 88.52 | 87.96 | 87.22 |
| 3 | 77.04 | 83.33 | **89.63** | 87.96 |
| 4 | 77.96 | 85.93 | 88.89 | 90.37 |
| 5 | 65.56 | 80.93 | **84.63** | 87.41 |
| **mean b2–5** | | **84.68** | **87.78** | 88.24 |

**Verdict: AdaBN full-model FT unlocks effective batch-1 full training — and across the full session
it beats head-only (+3.1 pp) and nearly matches the paper (−0.46 pp).** It helps most exactly where
head-only was weakest: the hard batches b3 (+6.3) and b5 (+3.7), recovering the feature-adaptation
capacity that head-only lacks.

### Disambiguation — is the win just a free test-time stat refresh? **No — it's a synergy.**
2×2 (mean b2–5, `adabn_disambiguation.py` / `results/adabn_disambiguation_S01_vocalized.csv`):

| | frozen-stats eval | test-time AdaBN eval |
|---|---|---|
| **head-only** | 84.68 (A) | 82.92 (B) |
| **AdaBN-full** | 85.42 (C) | 87.78 (D) |

- test-time AdaBN on head-only **HURTS** (B−A = **−1.76**) — the head was trained against frozen
  pretrained stats, so refreshing them at eval miscalibrates it (cf. idea1 `head_then_adabn`).
- full-model training alone (frozen eval) barely helps (C−A = **+0.74**).
- combined = **+3.10**, but the parts sum to −1.02 → **interaction = +4.13 pp**.

So the +3.1 pp is **not** a cheap free stat refresh: the stat refresh only pays off *because* the
model was **full-trained under the AdaBN regime** (train and eval both normalize with collected batch
stats → they are consistent). Neither piece works alone; they are coupled by design. You cannot get
this by bolting test-time AdaBN onto the existing head-only model (that regresses).

### Attribution vs the BASE model — recollection vs fine-tuning (recollect-only control)
The batch-1 row (no FT yet) already showed **+9.5 pp from stat recollection alone**. To split the
*later*-batch gains, add the control **base model (never FT'd) + stats recollected on each batch**:

| batch | no_ft | recollect-only (no FT) | AdaBN-full (FT+recollect) | recollect gain | FT gain |
|---|---|---|---|---|---|
| 2 | 74.6 | 84.8 | 88.0 | +10.2 | +3.2 |
| 3 | 77.0 | 88.2 | 89.6 | +11.1 | +1.5 |
| 4 | 78.0 | 87.4 | 88.9 | +9.4 | +1.5 |
| 5 | 65.6 | 80.4 | 84.6 | +14.8 | +4.3 |
| **mean b2–5** | 73.8 | **85.2** | 87.8 | **+11.4** | **+2.6** |

**Of the total +14.0 pp over the base model, ≈+11.4 pp (82%) is BN-stat recollection (label-free,
no training) and only ≈+2.6 pp (18%) is the fine-tuning.** Recollect-only (85.2) **already beats our
head-only FT recipe (84.68) with zero training.** So the dominant lever is **adapting BN statistics to
the target session**, not fine-tuning — and recollection helps only when the classifier is consistent
with it (base or full-AdaBN-trained; it *hurts* the head-only-FT model, whose fc was trained against
frozen stats). Practical upshot: the cheapest big on-device win is a forward-only BN-stat-collection
pass; full-model FT adds a modest, real +2.6 pp on top. (`recollect-only` computed inline; see log.)

**Cost vs head-only:** full-model training on-device (more compute/memory + trainable conv/BN),
a forward-only stat-collection pass per round, and careful lr (eff_lr ~0.001–0.005). Head-only is
simpler; AdaBN-full is worth it if the extra ~3 pp (esp. on hard sessions) matters.

### Optimizer-matched naive baseline (live BN) — the "59%" was an Adam artifact
`livebn_full_training.py`: the SAME on-device recipe (SGD, batch-1 + n_accum, fixed lr, 40 ep,
30% data, full model) but **live BN** instead of collected-frozen stats. b1→b2, 3 folds, same grid.

| best config n8/lr3e-4 | balanced acc |
|---|---|
| naive live-BN (SGD, matched) | **80.0** |
| AdaBN frozen | 86.3 |
| AdaBN recollect | 87.2 |

- The earlier **59%** reference was the *Adam paper* recipe at batch-1 (ablation E3) — an aggressive
  config that collapses hard. The **optimizer-matched** naive live-BN baseline is **80.0**, not 59.
  So AdaBN's gain over its *exact* naive counterpart is **80.0 → 87.2 = +7.2 pp** (not +28).
- At high eff_lr, live-BN is *more robust* than AdaBN-frozen (48–77 vs collapse to 11): live BN keeps
  re-normalizing each step, while too-high lr destroys the conv under frozen stats. AdaBN wins only in
  the low-eff-lr regime. (`results/livebn_vs_adabn_S01.csv`)

**CORRECTION — the 80% live-BN above is NOT the on-device case.** That host sim used PyTorch default
BN in train mode, which *updates* the running stats (EMA) and evals with them — a free domain
adaptation the device never does (the on-device BN training kernel leaves running stats frozen at
pretrained; inference uses those). The **faithful device baseline** = per-window batch-1 stats in the
train forward, **running stats frozen at pretrained**, inference with those frozen pretrained stats
(BN `momentum=0`):

| b1→b2, 3-fold | balanced acc |
|---|---|
| base, no FT (zero-shot) | 74.6 |
| **device-live-BN (faithful)** | **~70** (folds 58–82; e.g. n1/3e-4=71.5, n8/3e-4=69.6) |
| AdaBN (collected stats) | 87.2 |

- Faithful naive on-device full FT lands **below zero-shot** — the conv is trained against per-window
  normalization but deployed with frozen pretrained stats (train/inference mismatch, no adaptation).
  As lr→0 it approaches 74.6 (= base); any real training only degrades it. **Naive full-model on-device
  FT cannot beat zero-shot.**
- So there were two *flawed* naive references: **59%** (wrong optimizer — Adam) and **80%** (wrong BN
  semantics — EMA-adapted eval). The correct faithful on-device naive baseline is **~70%, below
  zero-shot**. This strengthens AdaBN's value: it rescues a *harmful* naive FT (~70) into +13 over
  zero-shot (87.2), and the rescue is mostly the collected-stats inference the frozen device path lacks.

## Reproduction — how the AdaBN experiment is implemented (PyTorch host)

Model: `SpeechNetNorm(norm='bn', p_dropout=0.0)` (real `BatchNorm2d`) loaded from the fold
checkpoint. Two PyTorch mechanisms do the work, plus the batch-1 sum-accumulation loop.

**① Collect population BN stats — `collect_bn_stats(model, X)`** (forward-only, label-free)
```python
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()      # zero running_mean/var + num_batches_tracked
        m.momentum = None            # -> cumulative moving average (exact mean over all seen)
model.train()                        # train-mode BN computes batch stats AND writes running buffers
with torch.no_grad():
    model(torch.from_numpy(X))       # ALL M windows in ONE forward -> running = population stats
model.eval()                         # freeze
```
- `momentum=None` → cumulative average, so one forward over all M windows sets `running_mean/var`
  to the **exact population** μ, σ² over M×H×W per channel.
- One batched forward on host == the on-device "run each window one at a time, accumulate Σx, Σx²".

**② Freeze stats + full-train — `full_train(model, X, y, lr, n_accum, epochs=40)`**
```python
model.eval()                                     # BN uses the FROZEN collected stats, and does NOT update them
for p in model.parameters(): p.requires_grad = True   # conv + BN gamma,beta + fc all train
opt = torch.optim.SGD(model.parameters(), lr=lr)      # no momentum, no weight decay
for _ in range(epochs):
    perm = rng.permutation(N); opt.zero_grad(); c = 0
    for j in perm:                               # batch size 1
        crit(model(Xt[j:j+1]), yt[j:j+1]).backward()   # SUM into .grad
        c += 1
        if c % n_accum == 0: opt.step(); opt.zero_grad()   # w <- w - lr * (sum of n_accum grads)
    if c % n_accum: opt.step(); opt.zero_grad()
```
- **Crux: `model.eval()` during training.** eval-mode BN normalizes with the frozen collected running
  stats (not the single-window batch stats) — this decouples normalization from the batch-1 minibatch.
- eval mode does **not** stop `γ, β` getting gradients, so the **full model** (conv + BN affine + fc)
  still trains; only the *statistics* are frozen. Backward = frozen-stat gradient `γ/√(σ²+ε)·dL/dy`.
- **Batch-1 + SUM n_accum:** one window per fwd/bwd, `.grad` accumulates, `opt.step()` every n_accum
  applies `w ← w − lr·Σgrad` (no ÷n_accum).

**Eval — `balanced_accuracy` (`ondevice_ft.py`):** `model.eval()` (frozen running stats), argmax per
window, mean per-class recall over the 9 classes on the 180-window batch.

**Incremental two-pass — `run_inter_session_ft_adabn.py`:** per round on batch b —
(1) `collect_bn_stats(m, Xb)` at current weights → (2) eval on batch b → (3) if b<5: `full_train` on
30 % of batch b; carry the model forward. `no_ft` = base model with pretrained stats.

**On-device mapping.** collect = forward-only kernel accumulating per-channel Σx, Σx² → μ, σ² fed as
BN inputs; freeze+train = the frozen-stat BN training kernel (`BN_FROZEN_STATS`) with those μ, σ² as
fixed inputs, conv/affine/fc via SGD batch-1 + n_accum; inference = BN uses the collected μ, σ².

**Run.**
```bash
python3 adabn_full_training.py 8:0.0003 4:0.001 ... --out adabn_sweep.csv   # b1->b2 (n_accum,lr) sweep
python3 run_inter_session_ft_adabn.py            # full 3-fold incremental at best config (n8, lr3e-4)
python3 adabn_disambiguation.py                  # 2x2 controls (train scope x eval-stat handling)
python3 livebn_full_training.py 8:0.0003 ...     # naive live-BN baseline (see the correction above)
```
Files: `adabn_full_training.py` (collect + frozen full-train + sweep), `run_inter_session_ft_adabn.py`
(incremental), `adabn_disambiguation.py`, `livebn_full_training.py`; model `idea2_alt_norm.SpeechNetNorm`;
data `windowing.py`; eval `ondevice_ft.balanced_accuracy`. All single-threaded (`torch.set_num_threads(1)`).

## On-device deployment (Siracusa / Deeploy) — what it takes

### Pipeline (per FT round on a batch)
```
1. COLLECT    forward-only over all batch windows -> accumulate per-channel Sx, Sx^2 at each BN
              input -> mu = Sx/M, var = Sx^2/M - mu^2 -> write to the BN stat buffers
2. TRAIN      full-model SGD (batch-1 + n_accum sum), BN normalizes with the FROZEN collected
              mu,var (frozen-stat kernel); update conv + BN gamma,beta + fc; dump weights
3. RE-COLLECT (optional) redo step 1 at the final weights -> deploy-time stats
4. INFER      forward-only with the collected mu,var + trained weights
```

### Already exists (reusable)
- **Frozen-stat BN kernel** — `TargetLibraries/PULPOpen/src/BatchNorm.c` (`BN_FROZEN_STATS` /
  `g_bn_frozen_stats`): forward normalizes with passed mean/var, backward `dX = γ·inv_std·dY`.
  Exactly step 2 — it just needs the **collected** stats instead of the pretrained ones.
- Full-model batch-1 on-device training (the `fullfrozen` work), SGD + n_accum sum accumulation,
  weight dump (WDUMP), tiling.

### New to build (ranked)
1. **BN-stat collection kernel/pass** — forward the M windows and accumulate per-channel `Sx`, `Sx²`
   at each BN input, then compute `μ, σ²`. (Must accumulate sums — averaging the per-window variances
   the BN-train kernel produces is wrong; it drops the between-window variance.) Tiny buffers.
2. **Expose BN `mean`/`var` as graph INPUTS** (not baked initializers) — the "pass stats as ONNX
   inputs" change in Onnx4Deeploy, so the collect pass writes them and the train/infer graphs read
   them at runtime (avoids host-side graph regen between passes).
3. **Orchestration** — wire collect → train → re-collect → infer and route the μ,σ² buffers.

Not needed: no BN folding, no new optimizer, no new backward math.

### Two deployment tiers (our attribution says start cheap)
Because recollect-only (85.2, no training) already beats head-only FT (84.68), and full training adds
only ~+2.6:

| tier | on-device work | accuracy |
|---|---|---|
| **AdaBN-inference only** (collect + infer, NO training) | pieces #1 + #2 (forward-only stat kernel + BN-stats-as-inputs). No backprop/optimizer/WDUMP. | ~85% |
| **Full AdaBN training** (collect + full-model FT) | + full-model batch-1 backprop (more L2/compute) | ~87–88% |

**Recommendation: build the AdaBN-*inference* tier first** — dramatically cheaper (forward-only), gets
most of the win; the training tier is a modest +2.6 pp for the full backprop machinery.

### Validation
Bit-exactness host↔GVSoC for collect+infer (stat sums + frozen-stat forward are deterministic); if
pursuing training, verify device loss ≡ ORT and dumped weights ≡ host, as done for the head-only deploy.

## Progress log
- 2026-07-16: plan written; `adabn_full_training.py` (collect + frozen-stat full train + b1→b2 sweep) built; sweep launching.
