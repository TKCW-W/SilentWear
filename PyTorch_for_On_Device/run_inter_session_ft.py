# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Inter-session + fine-tuning experiment, ON-DEVICE recipe, on PyTorch host.

Replicates the protocol of `offline_experiments/IV_inter_session_with_ft.py` (incremental
leave-one-session-out FT across the held-out session's 5 batches) but swaps the paper's
full-model Adam FT for our **on-device head-only + BN-fold SGD** recipe (see ondevice_ft.py).

Per fold K (base = leave_one_session_out_fold_K.pt, held-out test session K), for batch b=1..5:
  - balanced_acc_no_ft       = BASE model on the whole (rest-downsampled) batch b       [180 windows]
  - zero_shot_balanced_acc   = model FT'd on batches 1..b-1 (carried) on the whole batch b
  - then (b<5) fine-tune the carried model on `ft_frac` of batch b -> carry forward
Batch 5 is eval-only (matches the paper: last batch is not fine-tuned).

Windowing = onset-anchored (windowing.py), identical to Onnx4Deeploy. Eval on the whole
rest-downsampled batch (20/class = 180) reproduces the paper's balanced_acc_no_ft exactly.

Output: results/ft_summary_ondevice_<subject>_<condition>_fold<K>.csv
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd

from speechnet import load_speechnet
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy, finetune_head

HERE = os.path.dirname(os.path.abspath(__file__))


def run_fold(subject, condition, fold, data_path, artifacts_root,
             lr, n_accum, epochs, ft_frac, per_class_total, seed, batches=(1, 2, 3, 4, 5)):
    test_session = fold  # inter-session: fold K -> held-out test session K
    base_ckpt = (f"{artifacts_root}/inter_session_ft/{subject}/{condition}/speechnet/"
                 f"w1400ms/model_1/leave_one_session_out_fold_{fold}.pt")
    base_name = f"leave_one_session_out_fold_{fold}.pt"
    print(f"[{subject}/{condition}/fold_{fold}] base = {base_ckpt}")

    base_model = load_speechnet(base_ckpt)          # never modified (for balanced_acc_no_ft)
    ft_model = load_speechnet(base_ckpt)            # carried & fine-tuned across batches
    per_class_ft = max(1, int(round(per_class_total * ft_frac)))  # 20 * 0.30 -> 6/class

    rows = []
    for b in batches:
        Xe, ye = load_windows(data_path, subject, test_session, b, condition,
                              downsample_rest=True, seed=seed)          # whole batch (180)
        no_ft = balanced_accuracy(base_model, Xe, ye)                   # base, no FT
        zs = balanced_accuracy(ft_model, Xe, ye)                        # adapted (prev rounds)
        prev_rounds = b - batches[0]
        model_to_ft_name = base_name if b == batches[0] else f"fold_{fold}_ft_{b-1}.pt"
        rows.append(dict(
            subject=subject, condition=condition, base_model=base_name,
            test_session=test_session,
            model_to_fine_tune_name=model_to_ft_name,
            zero_shot_test_batch=b, num_prev_ft_rounds=prev_rounds,
            balanced_acc_no_ft=round(no_ft, 4),
            zero_shot_balanced_acc=round(zs, 4),
            new_model_name=f"fold_{fold}_ft_{b}.pt",
        ))
        print(f"  batch {b}: prev_rounds={prev_rounds}  no_ft={no_ft:6.2f}%  "
              f"adapted(zero_shot)={zs:6.2f}%")

        if b != batches[-1]:                        # fine-tune on this batch, carry forward
            Xtr, ytr = stratified_draw(Xe, ye, per_class=per_class_ft, seed=seed)
            print(f"           fine-tuning head on {len(ytr)} windows "
                  f"({per_class_ft}/class, {int(ft_frac*100)}%) ...")
            ft_model = finetune_head(ft_model, Xtr, ytr, lr=lr, n_accum=n_accum, epochs=epochs)

    df = pd.DataFrame(rows)
    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"ft_summary_ondevice_{subject}_{condition}_fold{fold}.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    cols = ["zero_shot_test_batch", "num_prev_ft_rounds", "balanced_acc_no_ft",
            "zero_shot_balanced_acc"]
    print(df[cols].to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01")
    ap.add_argument("--condition", default="vocalized")
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--data-path", default="/app/SilentWear/SilentWear_data/data_raw_and_filt")
    ap.add_argument("--artifacts-root", default="/app/SilentWear/SilentWear/artifacts/models")
    # on-device FT config
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ft-frac", type=float, default=0.30)   # 30% of the batch
    ap.add_argument("--per-class-total", type=int, default=20)  # rest-downsampled class size
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_fold(args.subject, args.condition, args.fold, args.data_path, args.artifacts_root,
             args.lr, args.n_accum, args.epochs, args.ft_frac, args.per_class_total, args.seed)


if __name__ == "__main__":
    main()
