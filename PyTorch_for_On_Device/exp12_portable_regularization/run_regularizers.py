# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 12 — do the PORTABLE regularizers lift streaming-AdaBN full-model past head-only?

Motivated by exp11 (mild overfitting past epoch 40) + the paper's regularized full-model FT. We audited
which of the paper's regularizers are actually portable to our Onnx4Deeploy->Deeploy/Siracusa training
stack:
  * WEIGHT DECAY  = portable, near-free: pulp-trainlib already implements w -= lr*(grad + lambda*w);
                    only codegen glue is missing. PyTorch SGD(weight_decay=lambda) emulates it exactly
                    (adds lambda*w once per optimizer step, matching our sum-accum on-device step).
  * DROPOUT       = portable only WITH a Deeploy op build (PRNG + pulp_dropout kernel exist, but no
                    Deeploy Dropout/DropoutGrad Layer/mapping; SpeechNet export omits it today). Measured
                    here to decide whether building that op is worth it.
  * EARLY STOP / LR SCHEDULE = NOT portable as-is (compile-time-fixed loop; compile-time-constant lr).
                    Excluded.

Protocol = exp5 streaming AdaBN, full-model arm (recollect stats on batch b, full-train 30% of batch b
with BN frozen, classify batch b+1 with carried weights+stats). S01 vocalized, 3 folds, n_accum=8,
lr=3e-4. Report mean b2-5. Baselines: no-reg 85.42 (ep40), head-only 84.68.

Parts:
  1. weight-decay sweep at 40 epochs      -> does the clean portable reg lift the ceiling / beat head-only?
  2. weight decay at 120 epochs           -> does it RESCUE the exp11 overfitting decline (84.26 @wd0)?
  3. dropout sweep at 40 epochs           -> is dropout worth the Deeploy-op build?

CLI:  python3 run_regularizers.py --subject S01
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from idea2_alt_norm import SpeechNetNorm                                      # noqa: E402
from adabn_full_training import collect_bn_stats                             # noqa: E402
from windowing import load_windows, stratified_draw                          # noqa: E402
from ondevice_ft import balanced_accuracy                                    # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
N_ACCUM, LR = 8, 3e-4


def ckpt_path(subject, cond, fold):
    return (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")


def make_model(ckpt, p_dropout=0.0):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model_state_dict", sd)
    m = SpeechNetNorm(norm="bn", p_dropout=p_dropout); m.load_state_dict(sd); m.eval()
    return m


def full_train_reg(model, X, y, lr, n_accum, epochs, wd=0.0, dropout_active=False, seed=42):
    """Full-model FT with frozen BN; optional weight decay and/or active dropout.
    BN stays in eval (frozen collected stats); only the Dropout module is switched to train mode so
    dropout is applied WITHOUT BN updating its running stats."""
    model.eval()                                             # freeze BN (collected stats)
    if dropout_active:
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()                                  # activate dropout only
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)   # wd == pulp-trainlib lambda
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1   # SUM accumulation
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    model.eval()                                             # dropout off for inference
    return model


def streaming_run(subject, cond, epochs, wd=0.0, p_dropout=0.0):
    byb = {b: [] for b in range(2, 6)}
    for fold in FOLDS:
        m = make_model(ckpt_path(subject, cond, fold), p_dropout=p_dropout)
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
            if b >= 2:
                m.eval()
                byb[b].append(balanced_accuracy(m, Xe, ye))   # classify b with previous round's (W,S)
            if b != 5:
                collect_bn_stats(m, Xe)                        # adapt stats to b (after eval)
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
                full_train_reg(m, Xtr, ytr, LR, N_ACCUM, epochs,
                               wd=wd, dropout_active=(p_dropout > 0))
    return float(np.mean([np.mean(byb[b]) for b in range(2, 6)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01"); ap.add_argument("--cond", default="vocalized")
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    print("=== Part 1: weight-decay sweep, streaming AdaBN full-model, 40 epochs ===", flush=True)
    wd_grid = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    r1 = []
    for wd in wd_grid:
        acc = streaming_run(a.subject, a.cond, 40, wd=wd)
        r1.append(dict(weight_decay=wd, epochs=40, mean_b25=round(acc, 2)))
        print(f"wd={wd:<7g} ep=40  mean b2-5={acc:5.2f}", flush=True)
    pd.DataFrame(r1).to_csv(os.path.join(HERE, f"results/wd_sweep_ep40_{a.subject}.csv"), index=False)

    # Part 2 (weight decay @ ep120) dropped: exp11 showed we would never deploy past ~40 epochs
    # on-device (monotonic overfitting), so an ep120 rescue test is moot.

    print("\n=== Part 2: dropout sweep, streaming AdaBN full-model, 40 epochs ===", flush=True)
    r3 = []
    for p in [0.0, 0.25, 0.5]:
        acc = streaming_run(a.subject, a.cond, 40, p_dropout=p)
        r3.append(dict(p_dropout=p, epochs=40, mean_b25=round(acc, 2)))
        print(f"p_dropout={p:<4g} ep=40  mean b2-5={acc:5.2f}", flush=True)
    pd.DataFrame(r3).to_csv(os.path.join(HERE, f"results/dropout_sweep_ep40_{a.subject}.csv"), index=False)

    print("\nbaselines: no-reg ep40 = 85.42 ; head-only = 84.68", flush=True)


if __name__ == "__main__":
    main()
