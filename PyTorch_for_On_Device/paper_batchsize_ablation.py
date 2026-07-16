# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
ABLATION (the reasonable one) — the PAPER's FT recipe at batch size 1 vs 32.

To isolate the batch-size contribution to the ours-vs-paper gap, hold the PAPER's recipe fixed
and change ONLY the batch size. This is where batch size actually bites: full-model training with
LIVE BatchNorm, where batch-1 makes each window get normalized by its own stats (the on-device
BN problem). batch-size contribution = paper@32 - paper@1, both run through the same pipeline.

Paper FT recipe (from IV_inter_session_with_ft.py / speechnet ft_cfg):
  full model, Adam lr 1e-3 + weight_decay 1e-4, real BN (train mode), dropout 0.5,
  70/30 stratified split + early stopping (patience 10), up to 50 epochs.
Incremental inter-session protocol, S01 vocalized, 3 folds, onset windowing, inter_session_ft base.

Output: results/ft_summary_paper_batchsize_S01_vocalized.csv  (+ 2x2 comparison print)
"""
import copy, os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from idea2_alt_norm import SpeechNetNorm
from windowing import load_windows
from ondevice_ft import balanced_accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND = "S01", "vocalized"
FOLDS = [1, 2, 3]


def load_paper_model(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model_state_dict", sd)
    m = SpeechNetNorm(norm="bn", p_dropout=0.5)     # paper architecture: real BN + dropout
    m.load_state_dict(sd); m.eval()
    return m


def strat_split(X, y, frac=0.7, seed=42):
    rng = np.random.default_rng(seed); tr, va = [], []
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n = max(1, int(round(len(idx) * frac))); tr += list(idx[:n]); va += list(idx[n:])
    tr, va = np.array(tr), np.array(va if len(va) else tr)
    return X[tr], y[tr], X[va], y[va]


def paper_ft(model, X, y, bs, lr=1e-3, wd=1e-4, epochs=50, patience=10):
    Xtr, ytr, Xva, yva = strat_split(X, y)
    Xtr = torch.from_numpy(Xtr.astype(np.float32)); ytr = torch.from_numpy(ytr.astype(np.int64))
    Xva = torch.from_numpy(Xva.astype(np.float32)); yva = torch.from_numpy(yva.astype(np.int64))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); crit = nn.CrossEntropyLoss()
    N = len(ytr); best, best_sd, bad = 1e9, None, 0
    for _ in range(epochs):
        model.train(); perm = torch.randperm(N)
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            opt.zero_grad(); crit(model(Xtr[b]), ytr[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xva), yva))
        if vl < best - 1e-4:
            best, best_sd, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_sd:
        model.load_state_dict(best_sd)
    model.eval()
    return model


def run_fold(fold, bs):
    ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = load_paper_model(ckpt)
    ft_model = load_paper_model(ckpt)
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(ft_model, Xe, ye)
        rows.append((b, no_ft, zs))
        if b != 5:
            torch.manual_seed(1000 + b)
            ft_model = paper_ft(ft_model, Xe, ye, bs=bs)   # FT on whole batch (paper 70/30 split)
    return rows


def agg(all_rows):
    byb = {}
    for rows in all_rows:
        for b, no_ft, zs in rows:
            byb.setdefault(b, {"nf": [], "zs": []}); byb[b]["nf"].append(no_ft); byb[b]["zs"].append(zs)
    return {b: (np.mean(byb[b]["zs"]), np.std(byb[b]["zs"], ddof=1),
               np.mean(byb[b]["nf"])) for b in sorted(byb)}


def main():
    import pandas as pd
    res = {}
    for bs in (32, 1):
        res[bs] = agg([run_fold(f, bs) for f in FOLDS])
        print(f"paper@bs{bs}: " + "  ".join(f"b{b}={res[bs][b][0]:.1f}" for b in res[bs]), flush=True)

    rows = []
    for b in sorted(res[32]):
        rows.append(dict(batch=b, no_ft_mean=round(res[32][b][2], 2),
                         paper_bs32_mean=round(res[32][b][0], 2), paper_bs32_std=round(res[32][b][1], 2),
                         paper_bs1_mean=round(res[1][b][0], 2), paper_bs1_std=round(res[1][b][1], 2)))
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", f"ft_summary_paper_batchsize_{SUBJECT}_{COND}.csv")
    df.to_csv(out, index=False)

    print("\n=== PAPER recipe @ batch 1 vs 32 — S01 vocalized, 3 folds ===")
    print(f"{'batch':>5} {'no_ft':>7} {'paper@32':>9} {'paper@1':>8} {'32 - 1':>7}")
    for r in rows:
        print(f"{r['batch']:>5} {r['no_ft_mean']:>7.2f} {r['paper_bs32_mean']:>9.2f} "
              f"{r['paper_bs1_mean']:>8.2f} {r['paper_bs32_mean']-r['paper_bs1_mean']:>+7.2f}")
    m32 = np.mean([r['paper_bs32_mean'] for r in rows if r['batch'] >= 2])
    m1 = np.mean([r['paper_bs1_mean'] for r in rows if r['batch'] >= 2])
    print(f"\nmean over batches 2-5:  paper@32={m32:.2f}  paper@1={m1:.2f}")
    print(f"  BATCH-SIZE contribution in the paper recipe (paper@32 - paper@1) = {m32-m1:+.2f} pp")
    print("  (compare: ours@32 - ours@1 = +0.28 pp; our head-only recipe folds BN so batch size barely matters)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
