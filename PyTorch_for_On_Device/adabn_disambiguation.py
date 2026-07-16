# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Disambiguate the AdaBN-full advantage: full-model TRAINING vs test-time BN-STAT refresh.

2x2 on the full incremental (S01 vocalized, 3 folds), mean over b2-5:
             eval with frozen stats      eval with test-time AdaBN (re-collect on eval batch)
  head-only        A (= shipped 84.68)         B
  AdaBN-full        C                          D (= 87.78)
We already have A and D; this computes B and C so we can split the two contributions.
"""
import os, copy
import numpy as np
import torch

torch.set_num_threads(1)

from adabn_full_training import make_model, collect_bn_stats, full_train
from ondevice_ft import finetune_head, balanced_accuracy
from windowing import load_windows, stratified_draw

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
import torch.nn as nn


def snapshot_bn(model):
    return {i: (m.running_mean.clone(), m.running_var.clone())
            for i, m in enumerate(x for x in model.modules() if isinstance(x, nn.BatchNorm2d))}


def restore_bn(model, snap):
    for i, m in enumerate(x for x in model.modules() if isinstance(x, nn.BatchNorm2d)):
        m.running_mean.copy_(snap[i][0]); m.running_var.copy_(snap[i][1])


def run_variant(scope, eval_recollect):
    byb = {b: [] for b in range(1, 6)}
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        m = make_model(ckpt)
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
            if eval_recollect:
                snap = snapshot_bn(m); collect_bn_stats(m, Xe)      # test-time AdaBN on eval batch
                zs = balanced_accuracy(m, Xe, ye); restore_bn(m, snap)
            else:
                zs = balanced_accuracy(m, Xe, ye)                    # use carried/frozen stats
            byb[b].append(zs)
            if b != 5:
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
                if scope == "head":
                    finetune_head(m, Xtr, ytr)                       # BN frozen at pretrained
                else:
                    collect_bn_stats(m, Xe); full_train(m, Xtr, ytr, 3e-4, 8)  # AdaBN full-train
    return np.mean([np.mean(byb[b]) for b in range(2, 6)])


def main():
    A = 84.68; D = 87.78          # known (shipped head-only; AdaBN-full+recollect)
    print("computing B (head-only + test-time AdaBN) ...", flush=True)
    B = run_variant("head", True)
    print("computing C (AdaBN-full, frozen-stats eval) ...", flush=True)
    C = run_variant("full", False)
    print("\n=== AdaBN advantage decomposition — S01 vocalized, mean over b2-5 ===")
    print(f"{'':<14}{'frozen-stats eval':>20}{'test-time AdaBN eval':>22}")
    print(f"{'head-only':<14}{A:>20.2f}{B:>22.2f}")
    print(f"{'AdaBN-full':<14}{C:>20.2f}{D:>22.2f}")
    print(f"\ntest-time AdaBN effect (head):  B - A = {B-A:+.2f} pp")
    print(f"full-model training effect (frozen eval):  C - A = {C-A:+.2f} pp")
    print(f"combined (AdaBN-full + recollect):  D - A = {D-A:+.2f} pp")
    print(f"interaction / non-additivity:  (D-A) - (B-A) - (C-A) = {(D-A)-(B-A)-(C-A):+.2f} pp")
    import pandas as pd
    pd.DataFrame([dict(variant="head_frozen", mean_b2_5=A),
                  dict(variant="head_ttadabn", mean_b2_5=round(B, 2)),
                  dict(variant="adabnfull_frozen", mean_b2_5=round(C, 2)),
                  dict(variant="adabnfull_ttadabn", mean_b2_5=D)]).to_csv(
        os.path.join(HERE, "results", "adabn_disambiguation_S01_vocalized.csv"), index=False)
    print("saved -> results/adabn_disambiguation_S01_vocalized.csv")


if __name__ == "__main__":
    main()
