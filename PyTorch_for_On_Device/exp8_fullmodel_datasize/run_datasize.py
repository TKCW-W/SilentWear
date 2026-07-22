# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 8 — does MORE fine-tuning data close the head-only vs full-model gap? (frozen pretrained BN stats)

Tests the overfitting explanation for why head-only ties/beats full-model FT: with tiny data the
full model overfits; with more data it should overfit less and catch up. Incremental streaming
protocol (classify batch b with the carried model FT'd on 1..b-1 + frozen pretrained stats), both
recipes, at data fractions 30/50/70/100 % (per_class 6/10/14/20 of the 180-window batch). NO stat
recollection — both use the frozen pretrained stats throughout.

recipes:  head  = head-only (fc, lr 0.01, n_accum 4)         [finetune_head]
          full  = full-model, frozen stats (lr 3e-4, n_accum 4)  [full_train]  (best frozen-stat cfg)
S01 vocalized, 3 folds. CLI:  python3 run_datasize.py <recipe> <per_class> --out results/<...>.csv
"""
import argparse, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model, full_train        # noqa: E402  (full_train: BN eval/frozen, all params)
from windowing import load_windows, stratified_draw           # noqa: E402
from ondevice_ft import balanced_accuracy, finetune_head      # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]


def ft(m, X, y, recipe):
    if recipe == "head":
        finetune_head(m, X, y, lr=0.01, n_accum=4, epochs=40)
    else:  # full-model, frozen pretrained stats (no recollection)
        full_train(m, X, y, lr=3e-4, n_accum=4, epochs=40)


def run_fold(subject, cond, fold, recipe, per_class):
    ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = make_model(ckpt); m = make_model(ckpt)
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(m, Xe, ye)          # streaming: carried model, frozen pretrained stats
        rows.append((b, no_ft, zs))
        if b != 5:
            Xtr, ytr = stratified_draw(Xe, ye, per_class, seed=42)
            ft(m, Xtr, ytr, recipe)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", choices=["head", "full"]); ap.add_argument("per_class", type=int)
    ap.add_argument("--subject", default="S01"); ap.add_argument("--cond", default="vocalized"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    byb = {b: [] for b in range(1, 6)}
    for fold in FOLDS:
        for b, nf, zs in run_fold(a.subject, a.cond, fold, a.recipe, a.per_class):
            byb[b].append(zs)
        print(f"[{a.recipe} pc={a.per_class}] fold {fold} done", flush=True)
    m25 = np.mean([np.mean(byb[b]) for b in range(2, 6)])
    frac = int(round(a.per_class / 20 * 100))
    pd.DataFrame([dict(recipe=a.recipe, per_class=a.per_class, data_frac=frac,
                       mean_b2_5=round(m25, 2),
                       **{f"b{b}": round(np.mean(byb[b]), 2) for b in range(2, 6)})]).to_csv(
        os.path.join(HERE, a.out), index=False)
    print(f"[{a.recipe} data={frac}% ({a.per_class}/class)] mean b2-5 = {m25:.2f}", flush=True)


if __name__ == "__main__":
    main()
