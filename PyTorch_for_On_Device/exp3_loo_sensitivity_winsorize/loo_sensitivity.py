# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 3a — leave-one-out (LOO) variance-influence diagnostic on the REAL data.

For each batch we compute per-channel BN input statistics (μ_c, σ²_c) at each of the 5 BatchNorm
layers over all windows, then recompute leaving out each single window, and measure
max_c |Δσ²_c / σ²_c|. This tells us, on the *natural* (uncorrupted) batches, whether dropping any one
window materially shifts any channel's variance — i.e. whether the real data already contains an
influential (outlier-like) window, or whether the pooling has diluted it away.

BN input = the preceding conv output (SpeechNet block = Conv→BN→ReLU→pool), captured via a
forward-pre-hook. S01 (clean) + S02 (noisy), 3 folds, batches 1–5, all 5 BN layers.

CLI:  python3 loo_sensitivity.py S01 --out results/loo_S01.csv
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from adabn_full_training import make_model      # noqa: E402
from windowing import load_windows              # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
FOLDS = [1, 2, 3]


def loo_for_batch(model, X):
    """Return, per BN layer, the max over windows×channels of |Δσ²/σ²| when leaving one window out."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    cur = {}
    def mk(i):
        def hook(module, inp):
            xf = inp[0][0].reshape(inp[0].shape[1], -1)   # (C, H*W)
            cur[i] = (xf.sum(1).numpy(), (xf * xf).sum(1).numpy(), xf.shape[1])
        return hook
    hooks = [bn.register_forward_pre_hook(mk(i)) for i, bn in enumerate(bns)]
    acc = {i: {"s": [], "sq": [], "n": None} for i in range(len(bns))}
    model.eval()
    with torch.no_grad():
        for w in range(len(X)):
            cur.clear()
            model(torch.from_numpy(X[w:w + 1].astype(np.float32)))
            for i in range(len(bns)):
                s, sq, n = cur[i]
                acc[i]["s"].append(s); acc[i]["sq"].append(sq); acc[i]["n"] = n
    for h in hooks:
        h.remove()
    out = {}
    for i in acc:
        S = np.stack(acc[i]["s"]); SQ = np.stack(acc[i]["sq"])   # (W, C)
        n = acc[i]["n"]; W = len(S); Ntot = W * n
        tot_s = S.sum(0); tot_sq = SQ.sum(0)
        var = tot_sq / Ntot - (tot_s / Ntot) ** 2
        # LOO for all windows (vectorised): leave out window w
        s2 = tot_s[None, :] - S; sq2 = tot_sq[None, :] - SQ; N2 = Ntot - n
        var2 = sq2 / N2 - (s2 / N2) ** 2
        shift = np.abs((var2 - var[None, :]) / (np.abs(var[None, :]) + 1e-6))   # (W, C)
        out[i] = float(shift.max())            # worst single window × channel for this BN layer
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--cond", default="vocalized")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    rows = []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{a.subject}/{a.cond}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        m = make_model(ckpt)
        for b in range(1, 6):
            X, y = load_windows(DATA, a.subject, fold, b, a.cond, downsample_rest=True)
            res = loo_for_batch(m, X)
            for bn_i, shift in res.items():
                rows.append(dict(subject=a.subject, fold=fold, batch=b, bn_layer=bn_i,
                                 n_windows=len(X), max_loo_var_shift=round(shift, 4)))
        print(f"[{a.subject}] fold {fold} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, a.out), index=False)
    print(f"\n=== {a.subject} LOO max |Δσ²/σ²| (worst single-window influence) ===")
    print("per BN layer, max over all folds×batches×windows×channels:")
    for bn_i in sorted(df.bn_layer.unique()):
        s = df[df.bn_layer == bn_i].max_loo_var_shift
        print(f"  BN{bn_i}: max={s.max()*100:6.1f}%   median-batch={s.median()*100:5.1f}%")
    overall = df.max_loo_var_shift.max()
    print(f"OVERALL worst single-window variance shift = {overall*100:.1f}%  "
          f"({'INFLUENTIAL (>10%)' if overall > 0.10 else 'diluted (<10%)'})", flush=True)


if __name__ == "__main__":
    main()
