# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Per-batch (b2..b5) results for K=0 (head-only) and K=1 (last-block+fc), averaged across 3 folds AND
4 subjects, at the comparison seed 42 — so exp13 can be added to the per-batch settings comparison in
the same format (per batch, averaged across folds and subjects) as head-only / streaming AdaBN.

Streaming incremental b1->b5, frozen pretrained BN, K0 lr 3e-3 / K1 lr 1e-3, n_accum 4, 40 ep, seed 42.
CLI:  python3 run_perbatch_4subj_seed42.py
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
COND, FOLDS = "vocalized", [1, 2, 3]
SUBJECTS = ["S01", "S02", "S03", "S04"]
CONFIGS = [(0, 3e-3), (1, 1e-3)]
SEED = 42


def perbatch(subject, k, lr):
    """Return {b: [fold accuracies]} for b in 2..5 (streaming, fc/blocks carried)."""
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
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=SEED)
                train_lastk(m, Xtr, ytr, lr, k)
    return byb


def main():
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    # accumulate per (K, batch) across all folds*subjects
    acc = {(k, b): [] for k, _ in CONFIGS for b in range(2, 6)}
    for s in SUBJECTS:
        for k, lr in CONFIGS:
            byb = perbatch(s, k, lr)
            for b in range(2, 6):
                acc[(k, b)] += byb[b]
        print(f"{s} done", flush=True)
    rows = []
    print(f"\n{'batch':>5} {'K0 head-only':>13} {'K1 last-blk+fc':>15}")
    for b in range(2, 6):
        k0 = np.mean(acc[(0, b)]); k1 = np.mean(acc[(1, b)])
        rows.append(dict(batch=b, K0_headonly=round(k0, 2), K1_lastblock=round(k1, 2)))
        print(f"{b:>5} {k0:>13.2f} {k1:>15.2f}", flush=True)
    m0 = np.mean([r["K0_headonly"] for r in rows]); m1 = np.mean([r["K1_lastblock"] for r in rows])
    rows.append(dict(batch="mean_b2_5", K0_headonly=round(m0, 2), K1_lastblock=round(m1, 2)))
    print(f"{'mean':>5} {m0:>13.2f} {m1:>15.2f}   (K1-K0 = {m1-m0:+.2f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results/perbatch_4subj_seed42.csv"), index=False)


if __name__ == "__main__":
    main()
