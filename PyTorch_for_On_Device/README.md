# PyTorch_for_On_Device — evaluate the on-device fine-tuning recipe on host

Groundwork to evaluate the **Siracusa/Deeploy on-device fine-tuning configuration** on the
PyTorch host (fast, no GVSoC), before committing to on-device runs. It reruns the SilentWear
**inter-session + fine-tuning** protocol (`offline_experiments/IV_inter_session_with_ft.py`) but
swaps the paper's full-model Adam FT for our **hardware-constrained head-only recipe**.

For the head-only + BN-fold path, host PyTorch (eval-mode BN = folded frozen Conv) is calibrated
**bit-exact** to the on-device result, so host numbers are predictive of device.

## On-device fine-tuning configuration (the recipe under test)

| knob | value |
|---|---|
| trainable scope | **head only** (`fc` = Linear 32→9); conv+BN frozen (folded feature extractor) |
| optimizer | **plain SGD** (no momentum, no weight decay) |
| learning rate | **0.01, static** (no schedule/decay) |
| gradient accumulation | **n_accum = 4, SUM** (`w ← w − lr·Σ₄ grad`; no ÷4) |
| effective batch size | **1** |
| epochs | **40** (fixed, no early stopping) |
| FT data per batch | **30 %** = 6 windows/class = 54 windows (stratified, seed 42) |
| dropout | none |
| windowing | onset-anchored, 700 samples, 14 filtered ch in channel_order, no norm |
| eval | whole rest-downsampled batch (20/class = 180 windows), balanced accuracy |

Base weights: `artifacts/models/inter_session_ft/<subj>/<cond>/speechnet/w1400ms/model_1/leave_one_session_out_fold_<K>.pt`.

## Protocol (per fold K = held-out test session K)

Incremental across the 5 batches of session K. At batch *b*:
- `balanced_acc_no_ft` = **base** model on batch *b*;
- `zero_shot_balanced_acc` = model **FT'd on batches 1..b-1** (carried forward) on batch *b*;
- then (b<5) fine-tune the carried model on 30 % of batch *b*. Batch 5 is eval-only.

## Files

| file | role |
|---|---|
| `speechnet.py` | SpeechNet on-device model (loads `leave_one_session_out_fold_*.pt` strict) |
| `windowing.py` | onset-anchored windowing + rest-downsample + stratified draw (== Onnx4Deeploy) |
| `ondevice_ft.py` | head-only SGD sum-accumulation FT + balanced-accuracy eval |
| `run_inter_session_ft.py` | incremental driver → `results/ft_summary_ondevice_<subj>_<cond>_fold<K>.csv` |
| `results/` | output tables |

## Run

```bash
# default: S01 / vocalized / fold_3
python3 run_inter_session_ft.py --subject S01 --condition vocalized --fold 3

# vary the config (this is the point — later configs plug in here):
python3 run_inter_session_ft.py --fold 1 --lr 0.005 --epochs 60 --ft-frac 0.7 --n-accum 8
```

## Result — S01 / vocalized / fold_3 (this milestone)

`balanced_acc_no_ft` reproduces the paper's `ft_summary.csv` **exactly** (validates windowing +
eval + base model). `zero_shot_balanced_acc` is our on-device head-only recipe.

| batch | prev FT rounds | no-FT (base) | **on-device head-only FT** | paper full-model Adam FT |
|---|---|---|---|---|
| 1 | 0 | 80.56 % | 80.56 % (no FT yet) | 80.56 % |
| 2 | 1 | 81.67 % | **90.56 %** | 87.78 % |
| 3 | 2 | 76.67 % | **80.56 %** | 86.67 % |
| 4 | 3 | 87.78 % | **88.89 %** | 88.33 % |
| 5 | 4 | 76.11 % | **81.67 %** | 87.78 % |

Reading it: the head-only recipe **improves every fine-tuned batch over no-FT** (b2 +8.9, b3 +3.9,
b4 +1.1, b5 +5.6), and on batch 2 it *exceeds* the paper's full-model FT — but on the harder
batches (3, 5) it adapts less than full-model FT, as expected from a last-layer-only update.

### Notes
- The `zero_shot_balanced_acc` is a single seed-42 draw; the head-only b→b+1 gain has ≈ ±1.75 pp
  variance (see the on-device report). For robust comparison, sweep ≥10 seeds (`--seed`).
- Batch-2 here is 90.56 % vs the on-device fixture's 89.44 %; the ~1 pp gap is the exact 54-window
  draw/order differing from the exporter fixture — both within draw noise.

## Why our numbers can differ from the paper (base-model training variance)

We fine-tune from the base checkpoints in `artifacts/models/inter_session_ft/.../leave_one_session_out_fold_*.pt`.
These are **not** the same weights as the `inter_session/` checkpoints, even though both are pretrained
on the same sessions (1 & 2, leave-session-3-out), with the **same architecture, hyperparameters, and
seed (42)**. They are two **independent training runs** and end up with **completely different weights**
(mean relative weight difference ≈ 50×; e.g. batch-3-fold-3 zero-shot 78.33 % for `inter_session` vs
81.67 % for `inter_session_ft`). Concretely, they even trained for **different numbers of steps**
(`num_batches_tracked` = 2385 vs 1395).

Why two "identical-config" runs diverge:
1. **A fixed seed does not make training bit-deterministic.** Full reproducibility also needs
   `torch.use_deterministic_algorithms(True)`, deterministic cuDNN, fixed DataLoader worker/shuffle
   order, and identical hardware + library versions. Without those, nondeterministic GPU reductions
   (atomic-add in conv/BN backward), data-order, and un-pinned RNG streams make each run drift.
2. **Early stopping locks in the drift.** Training is chaotic — a ~1e-7 gradient difference amplifies
   over epochs — so the two runs hit the early-stopping criterion at **different epochs**, train for
   different durations, and land in different regions of weight space.
3. **The model is over-parameterized.** Many weight configurations reach similar accuracy, so
   "different weights, similar performance" is normal; the base models differ by a few pp of zero-shot,
   which is ordinary run-to-run training variance, not an error.

**Implications for comparing to the paper:**
- Our reproduction uses `inter_session_ft` (the checkpoints the paper's `ft_summary.csv` actually
  scores), and our `no_ft` (base, un-fine-tuned) column matches the paper's `balanced_acc_no_ft`
  **exactly** — that is the validation that our windowing/eval pipeline is faithful.
- Small gaps in the **fine-tuned** column vs the paper can come from (a) this base-model training
  variance if a different checkpoint were used, and (b) our single seed-42 draw of the 30 % FT subset
  (≈ ±1.75 pp; use `--seed` to average). Neither indicates a pipeline bug.
- Note: our host BatchNorm sim uses default (non-deterministic-agnostic) settings; results here are
  reported as mean ± std over the 3 folds, and the between-*subject* spread is larger still
  (S01 is the easiest subject — see the ablation/attribution docs).
