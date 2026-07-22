# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 7 — head-only incremental fine-tuning across all 4 subjects (vocalized), for comparison with the
streaming AdaBN (exp5).

Head-only recipe (our shipped on-device recipe): train only fc, BN folded/frozen at the pretrained
session-1+2 running stats (NO recollection), SGD lr 0.01, n_accum 4 (sum), 40 epochs, 30% data.
Same incremental streaming protocol as exp5: classify batch b with the carried model FT'd on batches
1..b-1 (and frozen pretrained stats) — inherently deployment-realistic (no transductive stat peek).

    W <- base
    for b = 1..5:
        zs(b) = classify batch b using W (weights FT'd on 1..b-1) + frozen pretrained stats
        if b < 5:
            W <- head-only fine-tune W on 30% of batch b

S01..S04, 3 folds each. CLI:  python3 run_headonly.py S01 --out results/headonly_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model              # noqa: E402
from windowing import load_windows, stratified_draw     # noqa: E402
from ondevice_ft import balanced_accuracy, finetune_head  # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
LR, N_ACCUM, EPOCHS = 0.01, 4, 40


def run_fold(subject, cond, fold):
    ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = make_model(ckpt); m = make_model(ckpt)
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(m, Xe, ye)              # carried head-FT'd model + pretrained stats
        rows.append((b, no_ft, zs))
        if b != 5:
            Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
            finetune_head(m, Xtr, ytr, lr=LR, n_accum=N_ACCUM, epochs=EPOCHS)
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
        rows.append(dict(batch=b, no_ft_mean=round(nf.mean(), 2),
                         headonly_mean=round(zs.mean(), 2), headonly_std=round(zs.std(ddof=1), 2)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, a.out), index=False)
    m25 = np.mean([r["headonly_mean"] for r in rows if r["batch"] >= 2])
    print(df.to_string(index=False))
    print(f"[{a.subject}] head-only mean b2-5 = {m25:.2f}", flush=True)


if __name__ == "__main__":
    main()
