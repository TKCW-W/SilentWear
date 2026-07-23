# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 13b — confirm the K=1 (last-block+fc) sweet spot generalises beyond S01.

Runs the progressive-unfreeze recipe at each K's best lr (from exp13) across ALL 4 subjects
(S01-S04, vocalized), streaming incremental b1->b5, 3 folds. Compares K=0 (head-only), K=1
(last block+fc, the S01 sweet spot), K=2. If K=1 beats head-only across subjects, it becomes the
deployment candidate; if it only wins on S01, it stays an S01 artifact.

CLI:  python3 run_4subj_confirm.py
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model                # noqa: E402
from windowing import load_windows, stratified_draw       # noqa: E402
from ondevice_ft import balanced_accuracy                 # noqa: E402
from run_progressive_unfreeze import train_lastk          # noqa: E402  (reuse exact FT loop)

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
COND, FOLDS = "vocalized", [1, 2, 3]
SUBJECTS = ["S01", "S02", "S03", "S04"]
CONFIGS = [(0, 3e-3), (1, 1e-3), (2, 3e-4)]              # (K, best lr) from exp13


def stream_mean_b25(subject, k, lr):
    byb = {b: [] for b in range(2, 6)}
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{subject}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        m = make_model(ckpt)
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, subject, fold, b, COND, downsample_rest=True)
            if b >= 2:
                byb[b].append(balanced_accuracy(m, Xe, ye))
            if b != 5:
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
                train_lastk(m, Xtr, ytr, lr, k)
    return float(np.mean([np.mean(byb[b]) for b in range(2, 6)]))


def main():
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    hdr = "subj   " + "  ".join(f"K{k}(lr{lr:g})" for k, lr in CONFIGS)
    print(hdr, flush=True)
    for s in SUBJECTS:
        accs = {k: stream_mean_b25(s, k, lr) for k, lr in CONFIGS}
        for (k, lr) in CONFIGS:
            rows.append(dict(subject=s, K=k, lr=lr, mean_b25=round(accs[k], 2)))
        print(f"{s}  " + "  ".join(f"{accs[k]:8.2f}" for k, _ in CONFIGS), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results/progressive_unfreeze_4subj.csv"), index=False)
    print("\n=== 4-subject mean ===", flush=True)
    for (k, lr) in CONFIGS:
        m = df[df.K == k].mean_b25.mean()
        print(f"K={k} (lr {lr:g}): {m:.2f}", flush=True)
    k0 = df[df.K == 0].groupby("subject").mean_b25.mean().mean()
    k1 = df[df.K == 1].groupby("subject").mean_b25.mean().mean()
    print(f"\nK=1 - K=0 (head-only) = {k1 - k0:+.2f} pp (4-subj mean)", flush=True)


if __name__ == "__main__":
    main()
