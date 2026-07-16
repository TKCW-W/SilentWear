# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
IDEA 1 — precomputed BN statistics as BN inputs (instead of batch-1 single-window stats).

On-device we are stuck at effective batch 1, so the training BN kernel normalizes each window
by its own spatial stats (degenerate) -> we currently fold BN + train head-only. This idea:
precompute mean/var over N (e.g. 32) samples offline and feed them to the BN kernel, so BN
behaves like population/batch BN even at batch-1. On the host this equals frozen-stat BN
(if the stats are the pretrained running stats) or AdaBN (if recomputed on target data).

Question tested: does precomputing BN stats unlock effective FULL-MODEL fine-tuning at batch-1
(more capacity than head-only), and does adapting the stats to the target help at all?

Setup: S01 vocalized, the b1->b2 transition (FT on 30% of batch1, eval whole batch2), 3 folds.
FT optimizer = our on-device recipe (SGD, static lr, eff-batch 1, n_accum 4 summed, 40 epochs).
"""
import copy
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)  # batch-1 loops: single-thread avoids thread-thrash overhead

from speechnet import load_speechnet
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND = "S01", "vocalized"
FOLDS = [1, 2, 3]
LR, N_ACCUM, EPOCHS = 0.01, 4, 40


def recalibrate_bn(model, X):
    """Precompute BN mean/var over the windows X and freeze them (AdaBN). Uses current conv."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None            # cumulative -> exact mean/var over X
    model.train()
    with torch.no_grad():
        model(torch.from_numpy(X.astype(np.float32)))   # one pass -> exact batch stats
    model.eval()
    return model


def finetune(model, X, y, scope="head", lr=LR, n_accum=N_ACCUM, epochs=EPOCHS):
    """SGD sum-accumulation FT. BN kept in eval() (frozen stats). scope: 'head' or 'full'."""
    model.eval()                          # BN uses frozen running stats throughout
    for p in model.parameters():
        p.requires_grad = False
    if scope == "head":
        params = [model.fc.weight, model.fc.bias]
    else:                                 # full: conv + BN affine (gamma,beta) + fc; stats frozen
        params = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                for p in m.parameters(recurse=False):
                    params.append(p)
    for p in params:
        p.requires_grad = True
    opt = torch.optim.SGD(params, lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y)
    for _ in range(epochs):
        opt.zero_grad(); c = 0
        for j in range(N):
            crit(model(Xt[j:j+1]), yt[j:j+1]).backward(); c += 1
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    return model


def evalft(ckpt, Xtr, ytr, Xb2, yb2, scope, lr, calib_train=None, calib_eval=None):
    m = load_speechnet(ckpt)
    if calib_train is not None:
        recalibrate_bn(m, calib_train)              # precompute BN stats before FT
    if scope is not None:
        finetune(m, Xtr, ytr, scope, lr=lr)
    if calib_eval is not None:
        recalibrate_bn(m, calib_eval)               # recompute BN stats at eval (staleness fix)
    return balanced_accuracy(m, Xb2, yb2)


FULL_LRS = [3e-3, 1e-3, 3e-4, 1e-4]


def main():
    variants = ["zs_pretrained", "zs_adabn_b2", "head_pretrained(lr0.01)",
                "head_then_adabn_b2"]
    variants += [f"full_adabn_lr{lr:g}" for lr in FULL_LRS]
    results = {v: [] for v in variants}

    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        sess = fold
        Xb1, yb1 = load_windows(DATA, SUBJECT, sess, 1, COND, downsample_rest=True)
        Xtr, ytr = stratified_draw(Xb1, yb1, per_class=6, seed=42)       # 30% of batch1
        Xb2, yb2 = load_windows(DATA, SUBJECT, sess, 2, COND, downsample_rest=True)

        results["zs_pretrained"].append(evalft(ckpt, Xtr, ytr, Xb2, yb2, None, 0))
        results["zs_adabn_b2"].append(evalft(ckpt, Xtr, ytr, Xb2, yb2, None, 0, calib_eval=Xb2))
        results["head_pretrained(lr0.01)"].append(
            evalft(ckpt, Xtr, ytr, Xb2, yb2, "head", 0.01))
        results["head_then_adabn_b2"].append(
            evalft(ckpt, Xtr, ytr, Xb2, yb2, "head", 0.01, calib_eval=Xb2))
        # FULL-model FT lr sweep, BN precomputed on FT data (b1) + recomputed at eval (b2)
        for lr in FULL_LRS:
            results[f"full_adabn_lr{lr:g}"].append(
                evalft(ckpt, Xtr, ytr, Xb2, yb2, "full", lr, calib_train=Xtr, calib_eval=Xb2))
        print(f"fold {fold} done")

    print("\n=== IDEA 1: precomputed BN stats — S01 vocalized, b1->b2, mean+-std over 3 folds ===")
    print(f"{'variant':<24} {'mean':>7} {'std':>6}   per-fold")
    for v in variants:
        a = np.array(results[v])
        print(f"{v:<24} {a.mean():7.2f} {a.std(ddof=1):6.2f}   {np.round(a,1).tolist()}")
    print("\nref: paper full-model Adam FT b1->b2 (fold-avg) = 87.22%  |  no-FT zero-shot = 74.63%")


if __name__ == "__main__":
    main()
