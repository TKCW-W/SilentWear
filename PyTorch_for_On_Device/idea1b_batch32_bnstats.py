# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
IDEA 1 (faithful) — reproduce the paper's batch-32 BN at on-device batch-1 via PRECOMPUTED stats.

The paper trains the FULL model with batch=32: each BN forward normalizes using that 32-sample
batch's mean/var. On-device at batch-1 the BN kernel uses single-window stats (degenerate), which
is why naive full-model FT fails. Idea: precompute each 32-sample mini-batch's mean/var and feed
them to the BN kernel, so the batch-1 forward normalizes with 32-sample stats -> reproduces the
paper's BN forward. Gradient side is handled by n_accum (aggregate over the 32 windows).

We compare FULL-model FT (same SGD optimizer, mean reduction, lr sweep, 40 epochs, 54 windows,
mini-batch 32) differing ONLY in the BN forward:
  paper_batch32   : real batch-32 BN (train mode) — gradient flows through the batch stats
  precomp32_live  : batch-1 forward w/ that mini-batch's precomputed stats, recomputed each step
                    (faithful forward; gradient treats stats as constant = frozen-stat backward)
  precomp32_once  : precomputed once from the FT set, frozen (literal 'precompute offline')
  naive_bn1       : batch-1 single-window stats (the broken on-device baseline)
  head_only       : our shipped recipe (reference)

NOTE: fixed-stat BN is per-sample independent, so processing a 32-batch at once on the host is
mathematically identical to 32 separate batch-1 forwards with the same precomputed stats — i.e.
this faithfully represents the on-device batch-1 computation. All variants eval on batch2 with BN
stats recomputed on the FT data (deploy-time stats = FT-data stats), held constant across variants.
S01 vocalized, b1->b2, 3 folds.
"""
import copy
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from speechnet import load_speechnet
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND = "S01", "vocalized"
FOLDS = [1, 2, 3]
LRS = [3e-3, 1e-3, 3e-4]
EPOCHS, BS = 40, 32


def set_bn_stats(model, X):
    """Set BN running_mean/var to the exact stats over X (using current conv weights)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad():
        model(torch.from_numpy(X.astype(np.float32)) if isinstance(X, np.ndarray) else X)
    model.eval()


def full_ft(model, X, y, mode, lr, epochs=EPOCHS, bs=BS, seed=42):
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=lr)     # mean reduction, isolate BN forward
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    if mode == "precomp_once":
        set_bn_stats(model, X)
    for _ in range(epochs):
        perm = rng.permutation(N)
        for s in range(0, N, bs):
            b = perm[s:s + bs]; xb, yb = Xt[b], yt[b]
            if mode == "paper_batch32":
                model.train()                             # real batch stats + grad through them
            elif mode == "precomp_live":
                set_bn_stats(model, X[b]); model.eval()   # this batch's stats, frozen backward
            elif mode == "precomp_once":
                model.eval()                              # stats fixed from start
            elif mode == "naive_bn1":
                model.train()                             # will be fed one sample at a time below
            opt.zero_grad()
            if mode == "naive_bn1":
                loss = sum(crit(model(xb[i:i+1]), yb[i:i+1]) for i in range(len(b))) / len(b)
            else:
                loss = crit(model(xb), yb)
            loss.backward(); opt.step()
    set_bn_stats(model, X)                                # deploy-time stats = FT-data stats
    return model


def main():
    modes = ["paper_batch32", "precomp_live", "precomp_once", "naive_bn1"]
    res = {m: {lr: [] for lr in LRS} for m in modes}
    head = []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        Xb1, yb1 = load_windows(DATA, SUBJECT, fold, 1, COND, downsample_rest=True)
        Xtr, ytr = stratified_draw(Xb1, yb1, per_class=6, seed=42)
        Xb2, yb2 = load_windows(DATA, SUBJECT, fold, 2, COND, downsample_rest=True)
        # head-only reference (our recipe)
        from ondevice_ft import finetune_head
        m = load_speechnet(ckpt); finetune_head(m, Xtr, ytr); head.append(balanced_accuracy(m, Xb2, yb2))
        for mode in modes:
            for lr in LRS:
                m = load_speechnet(ckpt)
                full_ft(m, Xtr, ytr, mode, lr)
                res[mode][lr].append(balanced_accuracy(m, Xb2, yb2))
        print(f"fold {fold} done", flush=True)

    print("\n=== IDEA 1 faithful: batch-32 BN via precomputed stats — S01 b1->b2, 3 folds ===")
    h = np.array(head)
    print(f"{'head_only (ours)':<20} {h.mean():6.2f} +- {h.std(ddof=1):4.2f}")
    print(f"{'FULL-model FT:':<20} {'lr=3e-3':>14} {'lr=1e-3':>14} {'lr=3e-4':>14}")
    for mode in modes:
        cells = []
        for lr in LRS:
            a = np.array(res[mode][lr]); cells.append(f"{a.mean():5.1f}+-{a.std(ddof=1):4.1f}")
        print(f"{mode:<20} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")
    print("\nref: paper full-model Adam FT b1->b2 = 87.22% | zero-shot no-FT = 74.63%")
    print("read: does precomp_live ~ paper_batch32 (mechanism reproduces paper BN forward)?")
    print("      does either beat naive_bn1 (fixes batch-1) and head_only (worth the complexity)?")


if __name__ == "__main__":
    main()
