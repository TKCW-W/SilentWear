# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 13c — multi-seed lock-in for the K=1 (last-block+fc) sweet spot.

The exp13 / exp13b numbers are single seed-42 stratified draws. This varies the 54-window draw seed
(0..9) and reports head-only (K=0) vs last-block+fc (K=1) as mean +- std over draws, on S01 vocalized,
streaming b1->b5, 3 folds. Confirms the +1.6pp gain is draw-robust, not a lucky draw.

CLI:  python3 run_multiseed_k1.py
"""
import os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model            # noqa: E402
from windowing import load_windows, stratified_draw   # noqa: E402
from ondevice_ft import balanced_accuracy             # noqa: E402
from run_progressive_unfreeze import train_lastk      # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
CONFIGS = [(0, 3e-3), (1, 1e-3)]
SEEDS = list(range(10))


def stream_mean_b25(k, lr, seed):
    byb = {b: [] for b in range(2, 6)}
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        m = make_model(ckpt)
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
            if b >= 2:
                byb[b].append(balanced_accuracy(m, Xe, ye))
            if b != 5:
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=seed)
                train_lastk(m, Xtr, ytr, lr, k)
    return float(np.mean([np.mean(byb[b]) for b in range(2, 6)]))


def main():
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for seed in SEEDS:
        r = {"seed": seed}
        for k, lr in CONFIGS:
            r[f"K{k}"] = round(stream_mean_b25(k, lr, seed), 2)
        rows.append(r)
        print(f"seed {seed}: K0={r['K0']:.2f}  K1={r['K1']:.2f}  d={r['K1']-r['K0']:+.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results/multiseed_k1_S01.csv"), index=False)
    k0 = df["K0"].values; k1 = df["K1"].values
    print(f"\nK0 head-only : {k0.mean():.2f} +- {k0.std(ddof=1):.2f}", flush=True)
    print(f"K1 last-blk  : {k1.mean():.2f} +- {k1.std(ddof=1):.2f}", flush=True)
    print(f"K1 - K0      : {(k1-k0).mean():+.2f} +- {(k1-k0).std(ddof=1):.2f}  "
          f"(K1 wins {int((k1>k0).sum())}/{len(SEEDS)})", flush=True)


if __name__ == "__main__":
    main()
