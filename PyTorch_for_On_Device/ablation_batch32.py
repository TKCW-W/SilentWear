# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
ABLATION — our on-device inter-session FT recipe, but with the paper's batch size (32).

Goal: isolate how much of the ours-vs-paper gain gap is due to the batch-size factor alone.
Everything is kept identical to our recipe (head-only, frozen BN, SGD, 40 epochs, 30% data,
onset windowing, incremental b1->b5) EXCEPT the effective batch size: we train the head with
real mini-batches of 32 (mean reduction) instead of effective-batch-1 / n_accum-4 (sum).

lr is set to match our recipe's effective step (our recipe: sum of n_accum=4 grads x lr 0.01
=> ~0.04 * mean-grad; batch-32 mean at lr 0.04 keeps the step, changing ONLY the batch size).
We also run lr 0.01 for sensitivity.

CAVEAT: head-only folds/freezes BN, so batch size here affects only the head gradient, not BN.
S01 vocalized, 3 folds. Output: results/ft_summary_ondevice_batch32_S01_vocalized_SUBJECT_MEAN.csv
"""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from speechnet import load_speechnet
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND = "S01", "vocalized"
FOLDS = [1, 2, 3]
EPOCHS, BS = 40, 32


def finetune_head_batched(model, X, y, lr, epochs=EPOCHS, bs=BS, seed=42):
    """Head-only FT, frozen BN, real mini-batches of `bs` (mean reduction)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    opt = torch.optim.SGD([model.fc.weight, model.fc.bias], lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(N)
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            opt.zero_grad(); crit(model(Xt[b]), yt[b]).backward(); opt.step()
    return model


def run_fold(fold, lr):
    ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{fold}.pt")
    base = load_speechnet(ckpt)
    ft_model = load_speechnet(ckpt)
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(ft_model, Xe, ye)
        rows.append((b, no_ft, zs))
        if b != 5:
            Xtr, ytr = stratified_draw(Xe, ye, per_class=6, seed=42)   # 30% of batch b
            ft_model = finetune_head_batched(ft_model, Xtr, ytr, lr=lr)
    return rows


def aggregate(all_rows):
    """all_rows: list over folds of [(batch,no_ft,zs)...] -> per-batch mean/std."""
    byb = {}
    for rows in all_rows:
        for b, no_ft, zs in rows:
            byb.setdefault(b, {"no_ft": [], "zs": []})
            byb[b]["no_ft"].append(no_ft); byb[b]["zs"].append(zs)
    out = []
    for b in sorted(byb):
        nf = np.array(byb[b]["no_ft"]); zs = np.array(byb[b]["zs"])
        out.append(dict(batch=b, no_ft_mean=nf.mean(), no_ft_std=nf.std(ddof=1),
                        ft_mean=zs.mean(), ft_std=zs.std(ddof=1)))
    return out


def main():
    import pandas as pd
    results = {}
    for lr in (0.04, 0.01):
        agg = aggregate([run_fold(f, lr) for f in FOLDS])
        results[lr] = agg
        print(f"lr={lr}: " + "  ".join(f"b{r['batch']}={r['ft_mean']:.1f}" for r in agg), flush=True)

    # headline = lr 0.04 (effective-step-matched to our recipe)
    head = results[0.04]
    df = pd.DataFrame(head)
    for c in df.columns:
        if c != "batch":
            df[c] = df[c].round(2)
    outdir = os.path.join(HERE, "results")
    out = os.path.join(outdir, f"ft_summary_ondevice_batch32_{SUBJECT}_{COND}_SUBJECT_MEAN.csv")
    df.to_csv(out, index=False)

    # comparison vs our batch-1 recipe + paper (read the existing SUBJECT_MEAN)
    ref = pd.read_csv(os.path.join(outdir, f"ft_summary_ondevice_{SUBJECT}_{COND}_SUBJECT_MEAN.csv"))
    print("\n=== ABLATION: our recipe with batch size 32 (head-only) — S01 vocalized, 3 folds ===")
    print(f"{'batch':>5} {'no_ft':>7} {'ours_b1':>8} {'ours_b32':>9} {'paper':>7}"
          f"  {'b32-b1':>7} {'paper-b32':>9}")
    for r in head:
        b = r["batch"]; rr = ref[ref["batch"] == b].iloc[0]
        print(f"{b:>5} {rr['no_ft_mean']:>7.2f} {rr['ft_mean']:>8.2f} {r['ft_mean']:>9.2f} "
              f"{rr['paper_ft_mean']:>7.2f}  {r['ft_mean']-rr['ft_mean']:>+7.2f} "
              f"{rr['paper_ft_mean']-r['ft_mean']:>+9.2f}")
    # averages over fine-tuned batches (2-5)
    ft_b32 = np.mean([r["ft_mean"] for r in head if r["batch"] >= 2])
    ft_b1 = ref[ref["batch"] >= 2]["ft_mean"].mean()
    ft_paper = ref[ref["batch"] >= 2]["paper_ft_mean"].mean()
    print(f"\nmean over batches 2-5:  ours_b1={ft_b1:.2f}  ours_b32={ft_b32:.2f}  paper={ft_paper:.2f}")
    print(f"  batch-size contribution (b32 - b1) = {ft_b32-ft_b1:+.2f} pp")
    print(f"  remaining gap to paper (paper - b32) = {ft_paper-ft_b32:+.2f} pp")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
