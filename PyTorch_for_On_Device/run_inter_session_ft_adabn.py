# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Incremental inter-session FT with AdaBN full-model training (two passes per round).
Best config from the b1->b2 sweep: n_accum=8, lr=3e-4 (see AdaBN_Full_training.md).

Per round on batch b:
  (1) collect population BN stats over ALL of batch b (forward-only) -> running buffers (at current weights);
  (2) eval the carried model on batch b with those adapted stats  (= zero_shot for this batch);
  (3) if b<5: freeze the stats, FULL-train (conv+BN affine+fc) on 30% of batch b (SGD, batch-1, n_accum sum).
no_ft = base model with pretrained stats (paper baseline).  S01 vocalized, 3 folds.

Output: results/ft_summary_adabn_full_S01_vocalized_SUBJECT_MEAN.csv  (+ comparison vs head-only, paper)
"""
import os
import numpy as np
import torch

torch.set_num_threads(1)

from adabn_full_training import make_model, collect_bn_stats, full_train
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
N_ACCUM, LR, EPOCHS = 8, 3e-4, 40


def run_fold(fold):
    ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = make_model(ckpt)
    m = make_model(ckpt)
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)           # base, pretrained stats
        collect_bn_stats(m, Xe)                            # (1) adapt stats to batch b (current weights)
        zs = balanced_accuracy(m, Xe, ye)                  # (2) carried model, adapted stats
        rows.append((b, no_ft, zs))
        if b != 5:
            Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
            full_train(m, Xtr, ytr, LR, N_ACCUM, epochs=EPOCHS)   # (3) freeze + full-train on 30%
    return rows


def main():
    import pandas as pd
    byb = {b: {"nf": [], "zs": []} for b in range(1, 6)}
    for fold in FOLDS:
        for b, nf, zs in run_fold(fold):
            byb[b]["nf"].append(nf); byb[b]["zs"].append(zs)
        print(f"fold {fold} done", flush=True)
    rows = []
    for b in range(1, 6):
        nf = np.array(byb[b]["nf"]); zs = np.array(byb[b]["zs"])
        rows.append(dict(batch=b, no_ft_mean=round(nf.mean(), 2),
                         adabn_full_mean=round(zs.mean(), 2), adabn_full_std=round(zs.std(ddof=1), 2)))
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", f"ft_summary_adabn_full_{SUBJECT}_{COND}_SUBJECT_MEAN.csv")
    df.to_csv(out, index=False)

    # compare to head-only recipe + paper (from the existing SUBJECT_MEAN)
    ref = pd.read_csv(os.path.join(HERE, "results",
                                   f"ft_summary_ondevice_{SUBJECT}_{COND}_SUBJECT_MEAN.csv"))
    print("\n=== AdaBN full-model FT vs head-only vs paper — S01 vocalized, 3 folds ===")
    print(f"{'batch':>5} {'no_ft':>7} {'head_only':>10} {'adabn_full':>11} {'paper':>7}")
    for r in rows:
        b = r["batch"]; rr = ref[ref["batch"] == b].iloc[0]
        print(f"{b:>5} {r['no_ft_mean']:>7.2f} {rr['ft_mean']:>10.2f} "
              f"{r['adabn_full_mean']:>11.2f} {rr['paper_ft_mean']:>7.2f}")
    hb = ref[ref.batch >= 2]["ft_mean"].mean(); pb = ref[ref.batch >= 2]["paper_ft_mean"].mean()
    ab = np.mean([r["adabn_full_mean"] for r in rows if r["batch"] >= 2])
    print(f"\nmean over b2-5:  head_only={hb:.2f}  adabn_full={ab:.2f}  paper={pb:.2f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
