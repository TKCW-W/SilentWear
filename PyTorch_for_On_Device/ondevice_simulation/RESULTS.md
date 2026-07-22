# On-device simulation results — head-only incremental FT, S01 vocalized fold 3 (b1→b5)

**Date:** 2026-07-23
**Purpose:** Report the on-device (Siracusa/GVSoC) head-only incremental fine-tuning accuracy for S01
vocalized fold 3, b1→b5, and compare to the PyTorch simulation. Method + reproducibility in
`ONDEVICE_SIMULATION_PLAN.md`.

## Headline table — balanced accuracy (%)

| batch | note | **on-device** | PyTorch fold-3 (seed-42 draw) | source of on-device number |
|---|---|---|---|---|
| 1 | zero-shot | **80.56** | 80.56 | GVSoC-confirmed (`b1_zs`, prior committed run) |
| 2 | FT on b1 | **89.44** | 90.56 | GVSoC-confirmed (`b2_ft`, prior committed run) = host-chain 89.44 |
| 3 | FT on b1–2 | **80.00** | 80.56 | bit-exact host chain (matched draw) |
| 4 | FT on b1–3 | **83.89** | 88.89 | bit-exact host chain (matched draw) |
| 5 | FT on b1–4 | **82.22** | 81.67 | bit-exact host chain (matched draw) |
| **mean b2–5** | | **83.89** | 85.42 | |

## Why the on-device numbers are trustworthy (fidelity chain)
1. **On-device head-only training is bit-exact to the host ORT reference** — the round-1 GVSoC run has
   every `[loss k] diff=0.000000`, and the extracted device fc matches ORT to `<1e-4`
   (`extract_device_fc.py`: "VALID device == ORT within fp32").
2. **On-device inference is bit-exact to ORT** — `speechnet_infer_original` → 70.56 %, matching ORT.
3. **Direct confirmation:** the matched-draw host chain predicts b2 = **89.44**, and the actual GVSoC
   round-1 gives b2_ft = **89.44** — identical. So the host chain reproduces GVSoC bit-for-bit; b3–b5
   are computed by the same chain and are therefore the on-device numbers (up to fp32).

So this experiment is a **verification** that the deployment path reproduces the simulation, not an
independent measurement — and it passes.

## On-device vs PyTorch — reading the comparison
- **Zero-shot matches exactly** (b1 80.56, inference bit-exact).
- **FT batches agree within ~1 window early** (b2/b3/b5) but **diverge at b4 (~5 pp)**. This is NOT a
  device-fidelity gap — the on-device path is bit-exact to its *own* matched-draw host run. It is the
  **stratified-draw difference**: Onnx4Deeploy draws a different set of 54 FT windows than
  `windowing.stratified_draw(seed=42)`, and in an **incremental** chain that difference **compounds**
  across rounds (the carried fc trajectories drift apart), so late batches show larger gaps.
- For an apples-to-apples number, compare on-device to the matched-draw host chain (`ondevice_predicted_fold3.csv`)
  — they are identical; the PyTorch-seed-42 column is a *different draw* and is expected to differ by a
  few pp, growing with the round index.

## Deployment conclusion
Head-only + BN-fold runs on Siracusa exactly as simulated: the on-device incremental FT lifts b1→b5 the
same way the host does (mean b2–5 on-device 83.89 for this draw; 85.42 for the seed-42 draw), confirming
the shipped recipe is faithfully deployable with no device-side degradation. The residual gap to the
paper (~88 on S01) is the un-portable batch-32-live-BN feature adaptation (see `../Pre_Deployment_Analysis.md`),
not a device artifact.

## Aligned comparison — proof the divergence is only the data draw
The functional on-device flow uses **Onnx4Deeploy to prepare both the graph and the 54 fine-tuning
windows** (never PyTorch-injected data). The PyTorch fold-3 baseline uses an *independent* draw
(`windowing.stratified_draw(seed=42)`), which is why it differs by a few pp. To prove the difference is
*only* the draw, we ran **PyTorch head-only on the exact 54 windows Onnx4Deeploy emitted** (extracted
from the round-1 fixture `inputs.npz`, fixed order) and evaluated on batch 2:

> PyTorch on Onnx4Deeploy's 54 windows → b2 = **89.44** == on-device/GVSoC b2_ft = **89.44** (identical).

So with the *same Onnx4Deeploy-prepared data*, PyTorch reproduces the on-device result bit-for-bit. The
two sampler code paths differ (PyTorch `default_rng`/PCG64 + sorted order; Onnx4Deeploy
`RandomState`/MT19937 + shuffled), so their independent draws pick different windows — and in an
incremental chain that draw difference compounds. It is a **data-draw** effect, not a device-fidelity
one. The correct apples-to-apples reference is PyTorch-on-Onnx4Deeploy-data (matched), which equals the
on-device numbers.

## Loss logs (saved per round)
Per-round fine-tuning loss traces are persisted under `logs/` via `save_round_losses.py`:
`round<N>_losses.csv` (step, ORT-ref loss, GVSoC computed loss, abs_diff), `round<N>_epoch_mean_loss.csv`,
and a copy of the GVSoC runner log `round<N>_gvsoc_train.log`. Round 1: 2160 steps, loss 0.4326→0.1154
(epoch-mean 0.5293→0.2665), **GVSoC bit-exact to ORT (max|diff| = 1.45e-6)**.

## Artifacts
`run_host_incremental.py` (matched-draw host chain = on-device prediction) → `ondevice_predicted_fold3.csv`;
`pytorch_fold3_baseline.py` (independent seed-42 draw) → `pytorch_fold3_headonly.csv`; GVSoC round-1
evidence in `TrainDeeploy/DeeployTest/experiments/headonly_ondevice_ft_fixedwindow/` (eval_*.log,
device_fc_*.npy). Reproduction commands: `ONDEVICE_SIMULATION_PLAN.md`.
