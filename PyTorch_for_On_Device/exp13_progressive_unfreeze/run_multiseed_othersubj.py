# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 13d — multi-seed the K=1 gain on S02/S03/S04 (the subjects whose single-seed gains were largest).

exp13c showed S01's single-seed +1.53 pp collapsed to +0.20 +- 0.57 (noise) over 10 draws. This checks
whether S02 (+2.82), S03 (+1.85), S04 (+0.32) single-seed gains survive multi-seeding. Decides whether
last-block+fc (K=1) is a real improvement on the harder subjects or just draw noise everywhere.

K0 (lr 3e-3) vs K1 (lr 1e-3), streaming b1->b5, 3 folds, seeds 0..9. Vocalized.
CLI:  python3 run_multiseed_othersubj.py
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
SUBJECTS = ["S02", "S03", "S04"]
CONFIGS = [(0, 3e-3), (1, 1e-3)]
SEEDS = list(range(10))


def stream_mean_b25(subject, k, lr, seed):
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
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=seed)
                train_lastk(m, Xtr, ytr, lr, k)
    return float(np.mean([np.mean(byb[b]) for b in range(2, 6)]))


def main():
    import pandas as pd
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for s in SUBJECTS:
        for seed in SEEDS:
            k0 = stream_mean_b25(s, 0, 3e-3, seed)
            k1 = stream_mean_b25(s, 1, 1e-3, seed)
            rows.append(dict(subject=s, seed=seed, K0=round(k0, 2), K1=round(k1, 2)))
        sub = pd.DataFrame([r for r in rows if r["subject"] == s])
        d = (sub.K1 - sub.K0).values
        print(f"{s}: K0={sub.K0.mean():.2f}+-{sub.K0.std(ddof=1):.2f}  "
              f"K1={sub.K1.mean():.2f}+-{sub.K1.std(ddof=1):.2f}  "
              f"d={d.mean():+.2f}+-{d.std(ddof=1):.2f} (K1 wins {int((d>0).sum())}/10)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results/multiseed_othersubj.csv"), index=False)
    print("\n=== combined S02-S04 (30 draws) ===", flush=True)
    d = (df.K1 - df.K0).values
    print(f"K1 - K0 = {d.mean():+.2f} +- {d.std(ddof=1):.2f}  (K1 wins {int((d>0).sum())}/{len(df)})", flush=True)


if __name__ == "__main__":
    main()
