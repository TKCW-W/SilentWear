# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 5 — the corrected (streaming / deployment-realistic) AdaBN incremental setting.

Difference from the transductive runner (run_inter_session_ft_adabn.py): each batch b is classified
with the state produced by the PREVIOUS round — i.e. the weights fine-tuned on batches 1..b-1 and the
BN stats recollected on batch b-1 — NOT with stats recollected on batch b. This is what real-time
single-sample deployment can do, and it matches the paper's protocol (classify batch b using
adaptation from batches < b, never peeking at batch b).

Protocol (weights W and stats S both carry forward):
    W <- base weights ;  S <- pretrained stats
    for b = 1..5:
        zs(b) = classify batch b using (W, S)          # previous round's weights + stats
        if b < 5:
            S <- recollect BN stats on batch b         # adapt stats to batch b
            W <- fine-tune W on 30% of batch b (BN frozen at S)   # adapt weights
    (zs(1) = base + pretrained = the no-FT baseline; the only change vs transductive is that
     collect_bn_stats now runs AFTER the eval, inside the b<5 block.)

Config: AdaBN best n_accum=8, lr=3e-4, 40 epochs, 30% data. S01 vocalized, 3 folds.
CLI:  python3 run_streaming_adabn.py S01 --out results/streaming_incr_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model, collect_bn_stats, full_train   # noqa: E402
from windowing import load_windows, stratified_draw                       # noqa: E402
from ondevice_ft import balanced_accuracy                                  # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
N_ACCUM, LR, EPOCHS = 8, 3e-4, 40


def run_fold(subject, cond, fold):
    ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = make_model(ckpt)          # never modified -> no_ft reference (base + pretrained stats)
    m = make_model(ckpt)             # carries (W, S) forward
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(m, Xe, ye)          # classify batch b with the PREVIOUS round's (W, S)
        rows.append((b, no_ft, zs))
        if b != 5:
            collect_bn_stats(m, Xe)                # NOW adapt stats to batch b (after the eval)
            Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
            full_train(m, Xtr, ytr, LR, N_ACCUM, epochs=EPOCHS)   # adapt weights (BN frozen at S_b)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject"); ap.add_argument("--cond", default="vocalized"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    byb = {b: {"nf": [], "zs": []} for b in range(1, 6)}
    for fold in FOLDS:
        for b, nf, zs in run_fold(a.subject, a.cond, fold):
            byb[b]["nf"].append(nf); byb[b]["zs"].append(zs)
        print(f"[{a.subject}] fold {fold} done", flush=True)
    rows = []
    for b in range(1, 6):
        nf = np.array(byb[b]["nf"]); zs = np.array(byb[b]["zs"])
        rows.append(dict(batch=b, no_ft_mean=round(nf.mean(), 2), no_ft_std=round(nf.std(ddof=1), 2),
                         streaming_adabn_mean=round(zs.mean(), 2), streaming_adabn_std=round(zs.std(ddof=1), 2)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, a.out), index=False)
    m25 = np.mean([r["streaming_adabn_mean"] for r in rows if r["batch"] >= 2])
    print(df.to_string(index=False))
    print(f"[{a.subject}] streaming-AdaBN mean b2-5 = {m25:.2f}", flush=True)


if __name__ == "__main__":
    main()
