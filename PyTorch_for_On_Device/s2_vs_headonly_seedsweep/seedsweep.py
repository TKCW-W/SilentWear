# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Seed sweep: head-only vs S2 (frozen pretrained BN stats, full model) — is S2's ~+1 pp real or noise?

For a given seed, run the FULL incremental (S01 vocalized, 3 folds, b1->b5) for BOTH recipes; the seed
varies the 30% stratified FT-subset draw (the main variance source, per the earlier frozen-BN study)
and the training data order. Report per recipe the per-batch balanced acc (3-fold mean) and mean(b2-5).
Aggregate across seeds -> mean +- std + paired diff + win rate.

Recipes (both: on-device SGD, batch-1 + n_accum SUM, 40 ep, 30% data):
  head-only : fc only, BN folded/eval (frozen pretrained stats), lr 0.01, n_accum 4  [shipped recipe]
  S2        : full model, BN eval (frozen pretrained stats, both train+infer), lr 3e-4, n_accum 4

CLI:  python3 seedsweep.py <seed> --out results/seed_<seed>.csv
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from idea2_alt_norm import SpeechNetNorm            # noqa: E402
from windowing import load_windows, stratified_draw  # noqa: E402
from ondevice_ft import balanced_accuracy           # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
EPOCHS = 40


def make_model(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model_state_dict", sd)
    m = SpeechNetNorm(norm="bn", p_dropout=0.0); m.load_state_dict(sd); m.eval()
    return m


def train(model, X, y, recipe, seed):
    """recipe 'head' (fc only, lr0.01) or 's2' (full model, lr3e-4). BN eval throughout (frozen pretrained)."""
    model.eval()                                      # BN uses frozen pretrained stats (both recipes)
    for p in model.parameters():
        p.requires_grad = (recipe == "s2")
    if recipe == "head":
        model.fc.weight.requires_grad = True; model.fc.bias.requires_grad = True
        params = [model.fc.weight, model.fc.bias]; lr = 0.01
    else:
        params = list(model.parameters()); lr = 3e-4
    n_accum = 4
    opt = torch.optim.SGD(params, lr=lr); crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1   # SUM accumulation
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    model.eval(); return model


def run_incremental(recipe, seed):
    byb = {b: [] for b in range(1, 6)}
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        m = make_model(ckpt)
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True, seed=seed)
            byb[b].append(balanced_accuracy(m, Xe, ye))
            if b != 5:
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=seed)      # seed varies the FT subset
                train(m, Xtr, ytr, recipe, seed)
    return {b: float(np.mean(byb[b])) for b in range(1, 6)}   # 3-fold mean per batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    rows = []
    for recipe in ("head", "s2"):
        pb = run_incremental(recipe, a.seed)
        m25 = np.mean([pb[b] for b in range(2, 6)])
        rows.append(dict(seed=a.seed, recipe=recipe,
                         **{f"b{b}": round(pb[b], 2) for b in range(1, 6)},
                         mean_b2_5=round(m25, 2)))
        print(f"seed={a.seed} {recipe:>4}: mean(b2-5)={m25:.2f}  (b2-5: "
              + " ".join(f"{pb[b]:.1f}" for b in range(2, 6)) + ")", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, a.out), index=False)


if __name__ == "__main__":
    main()
