# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 9 — full training on the BN-FOLDED graph, and confirm it matches unfolded frozen-stat full-FT.

Fold each block's BatchNorm (frozen pretrained stats) into its preceding Conv -> a BN-free
Conv→ReLU→Pool graph, then FULL-train the folded Conv weights (+ fc). This is the cleanest on-device
full-training path: no BN kernels, no batch-1 BN problem. Prediction: mathematically ≈ full-model FT
with a separate frozen-stat BN (S2 / exp8 'full'), up to reparameterization, so accuracy should match.

Fold: for out-channel o,  scale = γ/√(running_var+ε);  W_folded = W·scale;  b_folded = (b−μ)·scale + β;
      then replace BN with Identity.  Full-train all conv+fc (no BN), streaming incremental protocol.

Config lr 3e-4, n_accum 4, 40 ep, 30% data (= exp8 'full'). S01 vocalized, 3 folds.
CLI:  python3 run_folded_fulltrain.py --out results/folded_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model, full_train      # noqa: E402
from windowing import load_windows, stratified_draw          # noqa: E402
from ondevice_ft import balanced_accuracy                     # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
N_ACCUM, EPOCHS, PER_CLASS = 4, 40, 6
LR = 3e-4  # overridden by --lr


def fold_bn(model):
    """Fold each block's frozen BN into its Conv; replace BN with Identity (BN-free graph)."""
    for block in model.blocks:
        conv, bn = block[0], block[1]
        assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
        scale = (bn.weight.data / torch.sqrt(bn.running_var + bn.eps))          # (C,)
        conv.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
        conv.bias.data = (conv.bias.data - bn.running_mean) * scale + bn.bias.data
        block[1] = nn.Identity()                                                 # BN gone
    return model


def run_fold(fold):
    ckpt = (f"{ART}/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = fold_bn(make_model(ckpt))          # folded, frozen — no_ft reference
    m = fold_bn(make_model(ckpt))             # folded, will be full-trained
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, "S01", fold, b, "vocalized", downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(m, Xe, ye)      # streaming: carried folded model
        rows.append((b, no_ft, zs))
        if b != 5:
            Xtr, ytr = stratified_draw(Xe, ye, PER_CLASS, seed=42)
            full_train(m, Xtr, ytr, LR, N_ACCUM, epochs=EPOCHS)   # trains folded conv + fc (no BN)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--lr", type=float, default=3e-4); a = ap.parse_args()
    global LR; LR = a.lr
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
                         folded_full_mean=round(zs.mean(), 2), folded_full_std=round(zs.std(ddof=1), 2)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, a.out), index=False)
    m25 = np.mean([r["folded_full_mean"] for r in rows if r["batch"] >= 2])
    print(df.to_string(index=False))
    print(f"folded-full-train mean b2-5 = {m25:.2f}  (compare exp8 unfolded full 30% = 85.65, head-only 84.68)", flush=True)


if __name__ == "__main__":
    main()
