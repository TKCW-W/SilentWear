# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 4 — realistic streaming eval: can we classify with the PREVIOUS round's stats?

Our AdaBN numbers are transductive: to classify batch b we recollect stats on batch b itself. In
real-time single-sample deployment you can't do that — you classify each incoming window with the
LAST recollected stats (from the previous batch/round). This experiment quantifies the gap.

Recollect-only (no fine-tuning, isolates the stat-staleness effect). For each batch b (2..5), evaluate
batch b's windows under three BN-stat regimes:
  pretrained  : frozen session-1+2 stats (no adaptation)                     [lower bound]
  streaming   : stats recollected on the PREVIOUS batch (b-1)                [realistic deployment]
  transductive: stats recollected on the CURRENT batch (b)                   [what we reported]

If streaming ≈ transductive, the session-level adaptation is what matters and real-time deployment
keeps the AdaBN benefit. If streaming ≈ pretrained, the benefit needs recollecting on the target data.

CLI:  python3 streaming_eval.py S01 --out results/streaming_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model, collect_bn_stats   # noqa: E402
from windowing import load_windows                             # noqa: E402
from ondevice_ft import balanced_accuracy                      # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]


def run(subject, cond):
    rows = []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        # cache windows per batch of the held-out session (= fold)
        W = {b: load_windows(DATA, subject, fold, b, cond, downsample_rest=True) for b in range(1, 6)}
        for b in range(2, 6):
            Xcur, ycur = W[b]; Xprev, _ = W[b - 1]
            # pretrained (no adaptation)
            m = make_model(ckpt); acc_pre = balanced_accuracy(m, Xcur, ycur)
            # streaming: previous batch's stats
            m = make_model(ckpt); collect_bn_stats(m, Xprev); acc_stream = balanced_accuracy(m, Xcur, ycur)
            # transductive: current batch's stats
            m = make_model(ckpt); collect_bn_stats(m, Xcur); acc_trans = balanced_accuracy(m, Xcur, ycur)
            rows.append(dict(subject=subject, fold=fold, batch=b,
                             pretrained=round(acc_pre, 2), streaming=round(acc_stream, 2),
                             transductive=round(acc_trans, 2)))
        print(f"[{subject}] fold {fold} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject"); ap.add_argument("--cond", default="vocalized"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    df = pd.DataFrame(run(a.subject, a.cond))
    df.to_csv(os.path.join(HERE, a.out), index=False)
    g = df.groupby("batch")[["pretrained", "streaming", "transductive"]].mean()
    print(f"\n=== {a.subject}: BN-stat regime, balanced acc (3-fold mean) ===")
    print(f"{'batch':>5} {'pretrained':>11} {'streaming':>10} {'transductive':>13}")
    for b, r in g.iterrows():
        print(f"{b:>5} {r.pretrained:>11.2f} {r.streaming:>10.2f} {r.transductive:>13.2f}")
    m = df[["pretrained", "streaming", "transductive"]].mean()
    print(f"mean  {m.pretrained:>11.2f} {m.streaming:>10.2f} {m.transductive:>13.2f}", flush=True)


if __name__ == "__main__":
    main()
