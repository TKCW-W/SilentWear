# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 3b — does per-channel winsorization match sample-level MAD rejection under outlier injection?

Same injection setup as exp1b (spike ×8 a fraction f of the collection windows, eval on the CLEAN
batch), but comparing THREE BN-stat collection methods:
  standard  : plain population μ,σ² over all windows' activations
  reject    : drop windows with max-abs input amplitude > median + 3·MAD, then standard  (exp1b's fix)
  winsorize : per-channel clip each BN-input observation to its [0.5, 99.5] percentile before the
              μ/σ² accumulator (the argument's preferred fix — bounds leverage, discards no sample)

winsorize/standard are computed on the BN inputs (conv outputs) captured via forward-pre-hooks, then
written into the BN running buffers. Recollect-only, base weights. S01 + S02, f ∈ {0,5,10,20}%.
CLI:  python3 winsorize_vs_reject.py S01 --out results/wins_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from adabn_full_training import make_model, collect_bn_stats   # noqa: E402
from windowing import load_windows                             # noqa: E402
from ondevice_ft import balanced_accuracy                      # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]
BATCHES = [2, 3, 4, 5]
FRACS = [0.0, 0.05, 0.10, 0.20]
SPIKE, N_SEEDS = 8.0, 4
PLO, PHI = 0.5, 99.5


def corrupt(X, frac, seed):
    Xc = X.copy(); n = len(Xc); k = int(round(frac * n))
    if k > 0:
        idx = np.random.RandomState(seed).choice(n, k, replace=False); Xc[idx] = Xc[idx] * SPIKE
    return Xc


def robust_reject(X):
    amp = np.abs(X.reshape(len(X), -1)).max(axis=1)
    med = np.median(amp); mad = np.median(np.abs(amp - med)) + 1e-9
    keep = amp <= med + 3.0 * 1.4826 * mad
    return X[keep] if keep.sum() >= 2 else X


def winsorize_collect(model, X):
    """Set BN running stats from per-channel WINSORIZED activation stats (clip to [PLO,PHI] pct)."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    store = {i: [] for i in range(len(bns))}
    cur = {}
    def mk(i):
        def hook(module, inp):
            cur[i] = inp[0][0].reshape(inp[0].shape[1], -1).numpy()   # (C, H*W)
        return hook
    hooks = [bn.register_forward_pre_hook(mk(i)) for i, bn in enumerate(bns)]
    model.eval()
    with torch.no_grad():
        for w in range(len(X)):
            cur.clear(); model(torch.from_numpy(X[w:w + 1].astype(np.float32)))
            for i in range(len(bns)):
                store[i].append(cur[i])
    for h in hooks:
        h.remove()
    for i, bn in enumerate(bns):
        A = np.concatenate(store[i], axis=1)               # (C, total_obs)
        lo = np.percentile(A, PLO, axis=1, keepdims=True)
        hi = np.percentile(A, PHI, axis=1, keepdims=True)
        Ac = np.clip(A, lo, hi)
        mu = Ac.mean(1); var = Ac.var(1)
        bn.running_mean.copy_(torch.from_numpy(mu.astype(np.float32)))
        bn.running_var.copy_(torch.from_numpy(var.astype(np.float32)))
    model.eval()


def run(subject, cond):
    rows = []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{subject}/{cond}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        for b in BATCHES:
            Xe, ye = load_windows(DATA, subject, fold, b, cond, downsample_rest=True)
            for f in FRACS:
                for s in range(N_SEEDS):
                    Xc = corrupt(Xe, f, seed=3000 + s)
                    for method in ("standard", "reject", "winsorize"):
                        m = make_model(ckpt)
                        if method == "winsorize":
                            winsorize_collect(m, Xc)
                        else:
                            Xcol = robust_reject(Xc) if method == "reject" else Xc
                            collect_bn_stats(m, Xcol)
                        rows.append((fold, b, f, s, method, balanced_accuracy(m, Xe, ye)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject"); ap.add_argument("--cond", default="vocalized"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    df = pd.DataFrame(run(a.subject, a.cond), columns=["fold", "batch", "frac", "seed", "method", "acc"])
    df["subject"] = a.subject; df.to_csv(os.path.join(HERE, a.out), index=False)
    print(f"=== {a.subject}: standard vs reject vs winsorize (recollect-only, eval on clean) ===")
    print(f"{'frac':>5} {'standard':>10} {'reject':>10} {'winsorize':>11}")
    for f in FRACS:
        cells = [df[(df.frac == f) & (df.method == mth)].acc.mean() for mth in ("standard", "reject", "winsorize")]
        print(f"{int(f*100):>4}% {cells[0]:>10.2f} {cells[1]:>10.2f} {cells[2]:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
